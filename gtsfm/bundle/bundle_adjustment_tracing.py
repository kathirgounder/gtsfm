"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask
import gtsam  # type: ignore
import numpy as np
from dask.delayed import Delayed
from gtsam import BetweenFactorPose3, NonlinearFactorGraph, PriorFactorPoint3, PriorFactorPose3, Values  # type: ignore
from gtsam.noiseModel import Diagonal, Isotropic, Robust, mEstimator  # type: ignore
from gtsam.symbol_shorthand import K, P, X  # type: ignore
from numpy.typing import NDArray

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.reprojection as reprojection_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common import gtsfm_data
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationOptions,
    TriangulationSamplingMode,
)
from gtsfm.data_association.recursive_triangulator import triangulate_track_recursive
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.visualization.pipeline_trace import append_to_manifest, dump_stage

METRICS_GROUP = "bundle_adjustment_metrics"

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"
"""In this file, we use the GTSAM's GeneralSFMFactor2 instead of GeneralSFMFactor because Factor2 enables decoupling
of the camera pose and the camera intrinsics, and hence gives an option to share the intrinsics between cameras.
"""


CAM_POSE3_DOF = 6  # 6 dof for pose of camera
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logger = logger_utils.get_logger()


class RobustBAMode(Enum):
    """Robust BA modes."""

    NONE = "NONE"
    HUBER = "HUBER"
    GMC = "GMC"
    TLS = "TLS"


@dataclass
class BundleAdjustmentOptions:
    """Shared configuration for bundle adjustment across leaf BA, merging BA, etc.

    This dataclass captures the commonly-configured subset of BA parameters
    that vary between call sites (leaf vs. merging). Parameters with fixed
    defaults (e.g. ordering_type, allow_indeterminate_linear_system) are left
    to the ``BundleAdjustmentOptimizer`` constructor.
    """

    robust_ba_mode: Union[RobustBAMode, str] = RobustBAMode.GMC
    shared_calib: bool = False
    use_calibration_prior: bool = False
    use_pose_prior_all_cameras: bool = False
    use_pose_prior_first_camera: bool = False
    use_gnc: bool = False
    gnc_loss: Union[RobustBAMode, str] = RobustBAMode.GMC
    factor_weight_outlier_threshold: float = 0.0
    min_track_length: int = 2
    calibration_prior_focal_sigma: float = 20.0
    calibration_prior_dist_sigma: float | Sequence[float] = 0.1
    calibration_prior_pp_sigma: float = 1e-5
    robust_noise_basin: float = 1.345

    def to_optimizer(self, **overrides) -> "BundleAdjustmentOptimizer":
        """Construct a :class:`BundleAdjustmentOptimizer` from these options.

        Args:
            **overrides: Keyword arguments that override dataclass fields or add
                extra constructor params (e.g. ``min_track_length``,
                ``reproj_error_thresholds``).
        """
        kwargs = dict(
            robust_ba_mode=self.robust_ba_mode,
            shared_calib=self.shared_calib,
            use_calibration_prior=self.use_calibration_prior,
            use_pose_prior_all_cameras=self.use_pose_prior_all_cameras,
            use_pose_prior_first_camera=self.use_pose_prior_first_camera,
            use_gnc=self.use_gnc,
            gnc_loss=self.gnc_loss,
            factor_weight_outlier_threshold=self.factor_weight_outlier_threshold,
            min_track_length=self.min_track_length,
            calibration_prior_focal_sigma=self.calibration_prior_focal_sigma,
            calibration_prior_dist_sigma=self.calibration_prior_dist_sigma,
            calibration_prior_pp_sigma=self.calibration_prior_pp_sigma,
            robust_noise_basin=self.robust_noise_basin,
        )
        kwargs.update(overrides)
        return BundleAdjustmentOptimizer(**kwargs)


def filter_outlier_cameras(data: GtsfmData, min_covisibility_count: int = 5, min_track_len: int = 2) -> GtsfmData:
    """Remove outlier cameras via covisibility clustering (GLOMAP Section 3.4).

    Builds a covisibility graph from shared 3D points, finds strongly connected components
    using an adaptive threshold, and keeps only the largest cluster. Small clusters weakly
    connected to the main reconstruction are merged if possible.

    Args:
        data: GtsfmData with BA-optimized cameras and tracks.
        min_covisibility_count: Minimum shared points to consider an edge in covisibility graph.
        min_track_len: Minimum track length to keep after camera removal.

    Returns:
        New GtsfmData with only the largest camera cluster retained.
    """
    from gtsam import SfmTrack

    camera_dict = data.cameras()
    if len(camera_dict) < 3:
        return data

    # Step 1: Build covisibility graph — count shared 3D points per camera pair.
    covisibility: dict[tuple[int, int], int] = defaultdict(int)
    for track in data.get_tracks():
        cam_idxs = []
        for k in range(track.numberMeasurements()):
            cam_idx, _ = track.measurement(k)
            if cam_idx in camera_dict:
                cam_idxs.append(cam_idx)
        for a in range(len(cam_idxs)):
            for b in range(a + 1, len(cam_idxs)):
                i, j = min(cam_idxs[a], cam_idxs[b]), max(cam_idxs[a], cam_idxs[b])
                covisibility[(i, j)] += 1

    # Discard pairs below minimum count.
    covisibility = {k: v for k, v in covisibility.items() if v >= min_covisibility_count}

    if not covisibility:
        logger.info("Covisibility clustering: no edges with >= %d shared points.", min_covisibility_count)
        return data

    # Step 2: Compute threshold τ = median of remaining pair counts.
    counts = np.array(list(covisibility.values()))
    tau = float(np.median(counts))

    logger.info(
        "Covisibility clustering: %d edges, counts min=%d, med=%d, max=%d, tau=%d",
        len(covisibility), counts.min(), int(np.median(counts)), counts.max(), int(tau),
    )

    # Step 3: Build strong graph (edges with count > τ) and find connected components.
    all_cameras = set(camera_dict.keys())
    strong_adj: dict[int, set[int]] = defaultdict(set)
    for (i, j), count in covisibility.items():
        if count > tau:
            strong_adj[i].add(j)
            strong_adj[j].add(i)

    # BFS to find connected components in strong graph.
    visited = set()
    components = []
    for cam in all_cameras:
        if cam in visited:
            continue
        component = set()
        queue = [cam]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in strong_adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    components.sort(key=len, reverse=True)
    main_component = components[0]

    # Step 4: Try to merge smaller components if they have ≥ 2 edges with count > 0.75τ to main.
    merge_threshold = 0.75 * tau
    for comp in components[1:]:
        strong_edges_to_main = 0
        for cam in comp:
            for main_cam in main_component:
                edge = (min(cam, main_cam), max(cam, main_cam))
                if covisibility.get(edge, 0) > merge_threshold:
                    strong_edges_to_main += 1
        if strong_edges_to_main >= 2:
            main_component = main_component | comp

    logger.info(
        "Covisibility clustering: %d components (sizes: %s). Keeping %d / %d cameras.",
        len(components),
        ", ".join(str(len(c)) for c in components[:5]),
        len(main_component),
        len(all_cameras),
    )

    if len(main_component) == len(all_cameras):
        logger.info("Covisibility clustering: all cameras in one component, no filtering needed.")
        return data

    # Step 5: Rebuild GtsfmData with only the main component.
    filtered = GtsfmData(number_images=data.number_images())
    for cam_idx in main_component:
        filtered.add_camera(cam_idx, camera_dict[cam_idx])
        info = data.get_image_info(cam_idx)
        filtered.set_image_info(cam_idx, name=info.name, shape=info.shape)

    for track in data.get_tracks():
        new_track = SfmTrack(track.point3())
        for k in range(track.numberMeasurements()):
            cam_idx, uv = track.measurement(k)
            if cam_idx in main_component:
                new_track.addMeasurement(cam_idx, uv)
        if new_track.numberMeasurements() >= min_track_len:
            filtered.add_track(new_track)

    logger.info(
        "Covisibility clustering: removed %d cameras, %d tracks remain.",
        len(all_cameras) - len(main_component), filtered.number_tracks(),
    )
    return filtered


def multi_view_retriangulate_from_2d_tracks(
    gtsfm_data: GtsfmData,
    tracks_2d: List["SfmTrack2d"],
    triangulation_options: Optional[TriangulationOptions] = None,
    min_track_length: int = 3,
    use_recursive_splitter: bool = False,
    recursive_max_depth: int = 5,
) -> GtsfmData:
    """GLOMAP-style multi-view retriangulation.

    Re-triangulates the FULL 2D track set with the current (post-BA) cameras. The lift over
    `gtsfm_data`'s existing track set comes from recovering tracks that were dropped between
    union-find (CppDsfTracksEstimator) and BA's filter passes — these are the tracks GLOMAP
    rescues in their retri stage (incremental_triangulator.cc::Retriangulate).

    GLOMAP doesn't re-match features here; it reuses correspondences from the cached match
    graph. We do the same — `tracks_2d` is the union-find output, no re-matching.

    Args:
        gtsfm_data: Scene with optimized cameras (existing tracks ignored; cameras re-used).
        tracks_2d: Full 2D track set from `CppDsfTracksEstimator.run()`.
        triangulation_options: Per-track solve config. Defaults to RANSAC_SAMPLE_UNIFORM,
            reproj_error_threshold=10 px, max_num_hypotheses=100.
        min_track_length: Drop tracks with fewer measurements than this (input + output).
            5e1183e1 found 3 best for British Museum — pure 2-view tracks flood BA with
            weak constraints. Brussels' bimodal scene may benefit from 2 (Phase 2 sweep).
        use_recursive_splitter: When True, use COLMAP-paper Section 4.3 recursive
            triangulation. Each input track may produce *multiple* output 3D points if it
            was a faulty union-find merge. Targets repetitive-feature scenes (Brussels'
            bimodal plaza, BM's colonnade, Pantheon's columns).
        recursive_max_depth: Max recursive splits per input track (only used when
            use_recursive_splitter=True). Default 5.

    Returns:
        New GtsfmData with same cameras and an augmented track set.
    """
    if triangulation_options is None:
        triangulation_options = TriangulationOptions(
            reproj_error_threshold=10.0,
            mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
            max_num_hypotheses=100,
        )

    camera_dict = {i: cam for i, cam in gtsfm_data.cameras().items() if cam is not None}
    initializer = Point3dInitializer(camera_dict, triangulation_options)

    out = GtsfmData(gtsfm_data.number_images())
    for cam_idx, camera in camera_dict.items():
        out.add_camera(cam_idx, camera)

    n_in = len(tracks_2d)
    n_kept = 0
    n_short_input = 0
    n_no_cams = 0
    n_short_output = 0
    n_triangulation_failed = 0
    n_split_into_multiple = 0  # how many input tracks produced > 1 output (recursive only)
    for track_2d in tracks_2d:
        if track_2d.number_measurements() < min_track_length:
            n_short_input += 1
            continue
        valid_measurements = [m for m in track_2d.measurements if m.i in camera_dict]
        if len(valid_measurements) < min_track_length:
            n_no_cams += 1
            continue
        track_2d_filtered = SfmTrack2d(measurements=valid_measurements)

        if use_recursive_splitter:
            new_tracks = triangulate_track_recursive(
                track_2d_filtered, initializer,
                min_consensus_size=min_track_length,
                max_recursion_depth=recursive_max_depth,
            )
            if not new_tracks:
                n_triangulation_failed += 1
                continue
            if len(new_tracks) > 1:
                n_split_into_multiple += 1
            for new_track in new_tracks:
                if new_track.numberMeasurements() < min_track_length:
                    n_short_output += 1
                    continue
                out.add_track(new_track)
                n_kept += 1
        else:
            new_track, _, _ = initializer.triangulate(track_2d_filtered)
            if new_track is None:
                n_triangulation_failed += 1
                continue
            if new_track.numberMeasurements() < min_track_length:
                n_short_output += 1
                continue
            out.add_track(new_track)
            n_kept += 1

    if use_recursive_splitter:
        logger.info(
            "Multi-view retriangulation (recursive): %d kept / %d input (min_len=%d): "
            "%d short_in, %d no_cams, %d tri_failed, %d short_out, %d input tracks split into >1 output.",
            n_kept, n_in, min_track_length, n_short_input, n_no_cams,
            n_triangulation_failed, n_short_output, n_split_into_multiple,
        )
    else:
        logger.info(
            "Multi-view retriangulation: %d kept / %d input (min_len=%d): "
            "%d short_in, %d no_cams, %d tri_failed, %d short_out.",
            n_kept, n_in, min_track_length, n_short_input, n_no_cams,
            n_triangulation_failed, n_short_output,
        )
    return out


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation.

    Due to the process graph requiring separate classes for separate graph objects, this class is a superclass for
    TwoViewBundleAdjustment and GlobalBundleAdjustment (defined in gtsfm/bundle/).
    """

    def __init__(
        self,
        reproj_error_thresholds: Sequence[Optional[float]] = [None],
        robust_ba_mode: RobustBAMode = RobustBAMode.NONE,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
        cam_pose3_prior_noise_sigma: float = 0.1,
        calibration_prior_focal_sigma: float = 20.0,
        calibration_prior_dist_sigma: float | Sequence[float] = 0.1,
        calibration_prior_pp_sigma: float = 1e-5,
        measurement_noise_sigma: float = 2.0,
        allow_indeterminate_linear_system: bool = True,
        print_summary: bool = False,
        ordering_type: str = "METIS",
        save_iteration_visualization: bool = False,
        robust_noise_basin: float = 1.345,
        use_karcher_mean_factor: bool = True,
        use_pose_prior_all_cameras: bool = False,
        use_pose_prior_first_camera: bool = False,
        use_calibration_prior: bool = True,
        use_first_point_prior: bool = False,
        use_gnc: bool = False,
        gnc_loss: RobustBAMode | str = RobustBAMode.GMC,
        factor_weight_outlier_threshold: float = 0.0,
        min_track_length: int = 2,
        num_outer_ba_iterations: int = 1,
        outer_ba_filter_scaling: bool = True,
        outer_ba_min_track_change_frac: float = 0.001,
        positions_only_stage1: bool = False,
        use_multi_view_retriangulation: bool = False,
        mv_retri_min_track_length: int = 3,
        mv_retri_reproj_error_thresh: float = 10.0,
        mv_retri_use_recursive_splitter: bool = False,
        mv_retri_recursive_max_depth: int = 5,
        # Accepted but inert on this branch — kept so the visualizer-tuned yaml can pass
        # them without a TypeError. The 13ada7fc post-BA filter chain that referenced
        # these regressed Brussels (-2.5 AUC@5), so the execution path was reverted to
        # the 4bf2e98c version which uses a single in-BA `filter_landmarks(3px)`.
        mv_retri_two_view_min_tri_angle_deg: float = 0.0,
        mv_retri_two_view_max_count: int = 0,
        filter_max_reproj_error_px: float = 4.0,
        filter_min_tri_angle_deg: float = 1.5,
        densify_with_two_view_tracks: bool = False,
        densify_two_view_max_reproj_error_px: float = 4.0,
        densify_two_view_min_tri_angle_deg: float = 3.0,
        densify_two_view_max_count: int = 10000,
        # Pipeline visualization trace: dump COLMAP-format snapshots at stage boundaries.
        # Off by default — enable only when generating data for the visualizer.
        save_pipeline_trace: bool = False,
        save_pipeline_trace_dir: Optional[str] = None,
    ) -> None:
        """Initializes the parameters for bundle adjustment module.

        Args:
            reproj_error_thresholds (optional): List of reprojection error thresholds used to perform filtering after
                each global bundle adjustment step. Implicitly defines the number of global BA steps, e.g., if
                len(reproj_error_thresholds) == 1, only one step will be performed. If the threshold is None, no
                filtering on output data is performed. Defaults to None.
            robust_ba_mode (optional): Robust BA mode to use, defaults to NONE.
            shared_calib (optional): Flag to enable shared calibration across all cameras. Defaults to False.
            max_iterations (optional): Max number of iterations when optimizing the factor graph. None means no cap.
                Defaults to None.
            cam_pose3_prior_noise_sigma (optional): Camera Pose3 prior noise sigma.
            calibration_prior_focal_sigma (optional): Sigma for the prior on the focal length, defaults to 20.0.
            calibration_prior_dist_sigma (optional): Sigma for the prior on the distortion parameters, defaults to 0.1.
            measurement_noise_sigma (optional): Measurement noise sigma in pixel units.
            allow_indeterminate_linear_system: Reject a two-view measurement if an indeterminate linear system is
                encountered during marginal covariance computation after bundle adjustment.
            ordering_type (optional): The ordering algorithm to use for variable elimination.
            save_iteration_visualization (optional): Save a Plotly animation showing optimization progress.
            robust_noise_basin (optional): Basin to use for the robust noise model.
            use_karcher_mean_factor (optional): Use Karcher mean factor to constrain the camera poses.
            use_pose_prior (optional): Use pose prior to constrain the camera poses. (only used if we use karcher mean)
            use_calibration_prior (optional): Use calibration prior to constrain the camera intrinsics.
            use_first_point_prior (optional): Use first point prior to constrain the scale of the reconstruction.
            use_gnc (optional): Use the GNC optimizer for bundle adjustment.
            gnc_loss (optional): GNC loss to use. Defaults to GMC.
            factor_weight_outlier_threshold (optional): Threshold weight for a reprojection factor to be kept.
            min_track_length: min number of measurements required to keep a track after weight filtering.
        """
        self._reproj_error_thresholds = reproj_error_thresholds
        if isinstance(robust_ba_mode, str):
            self._robust_ba_mode = RobustBAMode[robust_ba_mode]
        else:
            self._robust_ba_mode = robust_ba_mode
        self._shared_calib = shared_calib
        self._max_iterations = max_iterations
        self._cam_pose3_prior_noise_sigma = cam_pose3_prior_noise_sigma
        self._use_calibration_prior = use_calibration_prior
        self._calibration_prior_focal_sigma = calibration_prior_focal_sigma
        self._calibration_prior_dist_sigma = calibration_prior_dist_sigma
        self._calibration_prior_pp_sigma = calibration_prior_pp_sigma
        self._measurement_noise_sigma = measurement_noise_sigma
        self._allow_indeterminate_linear_system = allow_indeterminate_linear_system
        self._ordering_type = ordering_type
        self._print_summary = print_summary
        self._save_iteration_visualization = save_iteration_visualization
        self._robust_noise_basin = robust_noise_basin
        self._use_karcher_mean_factor = use_karcher_mean_factor
        self._use_pose_prior_all_cameras = use_pose_prior_all_cameras
        self._use_pose_prior_first_camera = use_pose_prior_first_camera
        self._use_first_point_prior = use_first_point_prior
        self._use_gnc = use_gnc
        if isinstance(gnc_loss, str):
            self._gnc_loss = RobustBAMode[gnc_loss]
        else:
            self._gnc_loss = gnc_loss
        self._factor_weight_outlier_threshold = factor_weight_outlier_threshold
        self._min_track_length = min_track_length
        self._num_outer_ba_iterations = num_outer_ba_iterations
        self._outer_ba_filter_scaling = outer_ba_filter_scaling
        self._outer_ba_min_track_change_frac = outer_ba_min_track_change_frac
        self._positions_only_stage1 = positions_only_stage1
        # GLOMAP-style multi-view retriangulation (after the outer BA loop converges).
        # See multi_view_retriangulate_from_2d_tracks above + plan section "Phase 1".
        self._use_multi_view_retriangulation = use_multi_view_retriangulation
        self._mv_retri_min_track_length = mv_retri_min_track_length
        self._mv_retri_reproj_error_thresh = mv_retri_reproj_error_thresh
        # Phase 2 recursive splitter (4bf2e98c, opt-in). Default False — flip to test
        # on repetitive-feature scenes (Brussels bimodal, BM colonnade, Pantheon columns).
        self._mv_retri_use_recursive_splitter = mv_retri_use_recursive_splitter
        self._mv_retri_recursive_max_depth = mv_retri_recursive_max_depth
        # Inert kwargs — accepted so the visualizer yaml's filter/densify params don't
        # raise TypeError. Their corresponding execution code paths were the 13ada7fc
        # regression source and have been left unwired on this branch.
        self._mv_retri_two_view_min_tri_angle_deg = mv_retri_two_view_min_tri_angle_deg
        self._mv_retri_two_view_max_count = mv_retri_two_view_max_count
        self._filter_max_reproj_error_px = filter_max_reproj_error_px
        self._filter_min_tri_angle_deg = filter_min_tri_angle_deg
        self._densify_with_two_view_tracks = densify_with_two_view_tracks
        self._densify_two_view_max_reproj_error_px = densify_two_view_max_reproj_error_px
        self._densify_two_view_min_tri_angle_deg = densify_two_view_min_tri_angle_deg
        self._densify_two_view_max_count = densify_two_view_max_count
        # Pipeline visualization trace (None = disabled).
        self._save_pipeline_trace = save_pipeline_trace
        self._save_pipeline_trace_dir = save_pipeline_trace_dir

    def _dump_trace(self, data: GtsfmData, name: str, **extra) -> None:
        """Dump a pipeline_trace stage. No-op when save_pipeline_trace_dir is None."""
        if not self._save_pipeline_trace_dir:
            return
        idx = self._trace_idx
        self._trace_idx += 1
        subdir = f"stage_{idx:03d}_{name}"
        try:
            dump_stage(data, str(Path(self._save_pipeline_trace_dir) / subdir),
                       stage_name=name, iteration=idx, extra_metadata=extra or None)
            append_to_manifest(self._save_pipeline_trace_dir, subdir, name, idx, extra)
        except Exception as e:
            logger.warning("[pipeline_trace] failed to dump stage %s: %s", name, e)

    def _ba_with_iter_capture(
        self,
        prefix: str,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        reproj_error_thresh: Optional[float] = None,
        verbose: bool = False,
        max_dump_frames: int = 6,  # 5 head LM iters + final = the "5+1" mental model.
        **extra_kwargs,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool], float]:
        """Run BA + capture per-LM-iter snapshots into the pipeline_trace.

        TRADE-OFF: this enables save_iteration_visualization=True which forces BA
        through the manual lm.iterate() loop (vs vanilla lm.optimize). On Brussels
        that costs ~3 AUC@5 because LM lambda dynamics differ subtly. Acceptable
        in trace mode — the F yaml is the canonical benchmark, Ft is just for viz.

        Sampling: first `max_dump_frames - 1` LM iters one-for-one (where the most
        dramatic motion happens) plus the absolute final iter for a clean handoff.
        Default 5+1 matches the user's "5 head + final" mental model and keeps the
        animation length proportional to the GP convergence segment.
        """
        capture = bool(self._save_pipeline_trace and self._save_pipeline_trace_dir)
        prev = self._save_iteration_visualization
        if capture:
            self._save_iteration_visualization = True
        try:
            result = self.run_ba_stage_with_filtering(
                initial_data=initial_data,
                absolute_pose_priors=absolute_pose_priors,
                relative_pose_priors=relative_pose_priors,
                reproj_error_thresh=reproj_error_thresh,
                verbose=verbose,
                **extra_kwargs,
            )
        finally:
            self._save_iteration_visualization = prev

        if capture and getattr(self, "_last_values_trace", None):
            trace = self._last_values_trace
            head = list(range(min(max_dump_frames - 1, len(trace))))
            last_idx = len(trace) - 1
            if last_idx not in head:
                head.append(last_idx)
            n_dumped = 0
            for k in head:
                try:
                    iter_data = GtsfmData.from_values(trace[k], initial_data, self._shared_calib)
                    self._dump_trace(iter_data, f"{prefix}_iter_{k:03d}",
                                     num_tracks=iter_data.number_tracks(),
                                     lm_iter=k, total_iters=len(trace))
                    n_dumped += 1
                except Exception as e:
                    logger.warning("[pipeline_trace] %s iter %d dump failed: %s", prefix, k, e)
            logger.info("[pipeline_trace] dumped %d/%d %s LM iter snapshots (5 head + final)",
                        n_dumped, len(trace), prefix)
            self._last_values_trace = None
        return result

    def __map_to_calibration_variable(self, camera_idx: int) -> int:
        return 0 if self._shared_calib else camera_idx

    def __reprojection_factors(
        self, initial_data: GtsfmData, cameras_to_model: List[int], robust_noise_basin: float | None = None
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Generate reprojection factors using the tracks."""
        graph = NonlinearFactorGraph()

        # noise model for measurements -- one pixel in u and v
        measurement_noise = Isotropic.Sigma(IMG_MEASUREMENT_DIM, self._measurement_noise_sigma)
        noise_basin = robust_noise_basin if robust_noise_basin is not None else self._robust_noise_basin
        if self._robust_ba_mode == RobustBAMode.HUBER:
            measurement_noise = Robust(mEstimator.Huber(noise_basin), measurement_noise)
        elif self._robust_ba_mode == RobustBAMode.GMC:
            measurement_noise = Robust(mEstimator.GemanMcClure(noise_basin), measurement_noise)

        # Note: Assumes all calibration types are the same.
        first_camera = initial_data.get_camera(cameras_to_model[0])
        assert first_camera is not None, "First camera in initial data is None"
        sfm_factor_class = gtsfm_types.get_sfm_factor_for_calibration(first_camera.calibration())

        cameras_with_tracks = set()
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            # Retrieve the SfmMeasurement objects.
            for m_idx in range(track.numberMeasurements()):
                # `i` represents the camera index, and `uv` is the 2d measurement
                i, uv = track.measurement(m_idx)
                cameras_with_tracks.add(i)
                # Note use of shorthand symbols `X` and `P`.
                if i not in cameras_to_model:
                    continue
                graph.push_back(
                    sfm_factor_class(
                        uv,
                        measurement_noise,
                        X(i),
                        P(j),
                        K(self.__map_to_calibration_variable(i)),
                    )  # type: ignore
                )

        cameras_without_tracks = {}
        for i in cameras_to_model:
            if i not in cameras_with_tracks:
                cameras_without_tracks[i] = initial_data.get_camera(i)
        if len(cameras_without_tracks) > 0:
            logger.info(f"Cameras without tracks: {cameras_without_tracks.keys()}")

        return graph, cameras_without_tracks

    def _between_factors(
        self, relative_pose_priors: Dict[Tuple[int, int], PosePrior], cameras_to_model: List[int]
    ) -> NonlinearFactorGraph:
        """Generate BetweenFactors on relative poses for pose variables."""
        graph = NonlinearFactorGraph()

        for (i1, i2), i2Ti1_prior in relative_pose_priors.items():
            if i1 not in cameras_to_model or i2 not in cameras_to_model:
                continue

            graph.push_back(
                BetweenFactorPose3(
                    X(i1),
                    X(i2),
                    i2Ti1_prior.value.inverse(),
                    Diagonal.Sigmas(i2Ti1_prior.covariance),
                )
            )

        return graph

    def __rotation_lock_priors(
        self,
        initial_data: GtsfmData,
        cameras_to_model: List[int],
    ) -> NonlinearFactorGraph:
        """Tight rotation-only priors used for GLOMAP-style positions-only stage-1 BA.

        Pins each camera's rotation to its current value (sigma 1e-6 on rotation tangent)
        while leaving translation free (sigma 1e6). Calibration is locked separately via
        the existing tight calibration prior.
        """
        graph = NonlinearFactorGraph()
        sigmas = np.array([1e-6, 1e-6, 1e-6, 1e6, 1e6, 1e6], dtype=float)
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        for camera_idx in cameras_to_model:
            camera_i = initial_data.get_camera(camera_idx)
            assert camera_i is not None, f"Camera {camera_idx} in initial data is None"
            graph.push_back(PriorFactorPose3(X(camera_idx), camera_i.pose(), noise_model))
        return graph

    def __pose_priors(
        self,
        absolute_pose_priors: List[Optional[PosePrior]],
        initial_data: GtsfmData,
        cameras_to_model: List[int],
    ) -> NonlinearFactorGraph:
        """Generate prior factors (in the world frame) on pose variables."""
        graph = NonlinearFactorGraph()

        # TODO(Ayush): start using absolute prior factors.

        if self._use_karcher_mean_factor:
            camera_keys = [X(i) for i in cameras_to_model]
            graph.push_back(gtsam.KarcherMeanFactorPose3(camera_keys, 6, 1000))

        if self._use_pose_prior_all_cameras:
            for camera_idx in cameras_to_model:
                camera_i = initial_data.get_camera(camera_idx)
                assert camera_i is not None, f"Camera {camera_idx} in initial data is None"
                graph.push_back(
                    PriorFactorPose3(
                        X(camera_idx),
                        camera_i.pose(),
                        Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
                    )
                )
        elif self._use_pose_prior_first_camera:
            first_camera = initial_data.get_camera(cameras_to_model[0])
            assert first_camera is not None, "First camera in initial data is None"
            graph.push_back(
                PriorFactorPose3(
                    X(cameras_to_model[0]),
                    first_camera.pose(),
                    Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
                )
            )

        return graph

    def __calibration_priors(self, initial_data: GtsfmData, cameras_to_model: list[int]) -> NonlinearFactorGraph:
        """Generate prior factors on calibration parameters of the cameras."""
        graph = NonlinearFactorGraph()

        # Note: Assumes all calibration types are the same.
        first_valid_camera_idx = cameras_to_model[0]
        first_camera = initial_data.get_camera(first_valid_camera_idx)
        assert first_camera is not None, "First camera in initial data is None"
        calibration_prior_factor_class = gtsfm_types.get_prior_factor_for_calibration(first_camera.calibration())
        calibration_dim = first_camera.calibration().dim()
        noise_model = gtsfm_types.get_noise_model_for_calibration(
            first_camera.calibration(),
            focal_sigma=self._calibration_prior_focal_sigma,
            dist_sigma=self._calibration_prior_dist_sigma,
            pp_sigma=self._calibration_prior_pp_sigma,
            skew_sigma=1e-6,
        )
        if self._shared_calib:
            graph.push_back(
                calibration_prior_factor_class(
                    K(self.__map_to_calibration_variable(first_valid_camera_idx)),
                    gtsfm_data.get_average_calibration(initial_data, cameras_to_model),  # type: ignore
                    noise_model,
                )
            )
        else:
            for i in cameras_to_model:
                camera_i = initial_data.get_camera(i)
                assert camera_i is not None, f"Camera {i} in initial data is None"
                if camera_i.calibration().dim() != calibration_dim:
                    raise ValueError(
                        "BundleAdjustmentOptimizer: Assumption that all calibration types are the same is violated"
                    )
                graph.push_back(
                    calibration_prior_factor_class(
                        K(self.__map_to_calibration_variable(i)), camera_i.calibration(), noise_model  # type: ignore
                    )
                )

        return graph

    def __construct_simple_factor_graph(
        self,
        cameras_to_model: List[int],
        initial_data: GtsfmData,
        robust_noise_basin: float | None = None,
        lock_rotations: bool = False,
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Construct the factor graph with just reprojection factors and calibration priors."""

        graph = NonlinearFactorGraph()
        if not cameras_to_model:
            return graph, {}

        reprojection_graph, cameras_without_tracks = self.__reprojection_factors(
            initial_data=initial_data,
            cameras_to_model=cameras_to_model,
            robust_noise_basin=robust_noise_basin,
        )
        graph.push_back(reprojection_graph)

        graph.push_back(
            self.__pose_priors(absolute_pose_priors=[], initial_data=initial_data, cameras_to_model=cameras_to_model)
        )

        if lock_rotations:
            graph.push_back(self.__rotation_lock_priors(initial_data, cameras_to_model))

        if self._use_first_point_prior and initial_data.number_tracks() > 0:
            graph.push_back(
                PriorFactorPoint3(P(0), initial_data.get_track(0).point3(), Isotropic.Sigma(POINT3_DOF, 0.1))
            )
        if self._use_calibration_prior:
            graph.push_back(self.__calibration_priors(initial_data, cameras_to_model))

        return graph, cameras_without_tracks

    def __construct_factor_graph(
        self,
        cameras_to_model: List[int],
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        robust_noise_basin: float | None = None,
        lock_rotations: bool = False,
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Construct the factor graph with reprojection factors, BetweenFactors, and prior factors."""
        # Create a factor graph.
        graph, cameras_without_tracks = self.__construct_simple_factor_graph(
            cameras_to_model, initial_data, robust_noise_basin, lock_rotations=lock_rotations
        )

        # Add priors
        graph.push_back(
            self._between_factors(relative_pose_priors=relative_pose_priors, cameras_to_model=cameras_to_model)
        )

        return graph, cameras_without_tracks

    def __optimize_factor_graph(
        self, graph: NonlinearFactorGraph, initial_values: Values, ordering_type: str
    ) -> Tuple[Values, Optional[List[Values]], Optional[NDArray[np.float64]]]:
        """Optimize the factor graph, optionally capturing per-iteration values."""
        start_time = time.time()

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR" if not self._print_summary else "SUMMARY")
        params.setOrderingType(ordering_type)
        if self._max_iterations:
            params.setMaxIterations(self._max_iterations)

        if not self._use_gnc:
            lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
        else:
            gnc_params = gtsam.GncLMParams(params)
            if self._gnc_loss == RobustBAMode.GMC:
                gnc_params.setLossType(gtsam.GncLossType.GM)
            elif self._gnc_loss == RobustBAMode.TLS:
                gnc_params.setLossType(gtsam.GncLossType.TLS)
            else:
                raise ValueError(f"Unsupported GNC loss type: {self._gnc_loss}.")
            gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.SUMMARY)
            gnc_params.setAllowNonNoiseModelFactors(True)
            gnc_params.setRelativeCostTol(1e-3)
            lm = gtsam.GncLMOptimizer(graph, initial_values, gnc_params)

        # gnc does not support getAbsoluteErrorTol and getRelativeErrorTol params
        if not self._save_iteration_visualization or self._use_gnc:
            result_values = lm.optimize()
            values_trace = None
        else:
            values_trace = [initial_values]
            # Hard cap manual-loop iters in trace mode. GTSAM's default is 100 but the
            # convergence check rarely fires on dense problems (Brussels: 101/101 every
            # outer BA; Pantheon retri-final: 100 iters × ~12s = ~20 min). Visualizer
            # only samples 5 head + final per BA anyway, so iters past ~30 add zero
            # visual info and just burn wall time. 30 keeps total per-BA wall time
            # under ~5 min on Pantheon-scale (300+ cams, 30K+ tracks).
            TRACE_MODE_MAX_ITERS = 30
            max_iters = self._max_iterations
            if max_iters is None:
                try:
                    max_iters = min(int(params.getMaxIterations()), TRACE_MODE_MAX_ITERS)
                except Exception:
                    max_iters = TRACE_MODE_MAX_ITERS
            else:
                max_iters = min(max_iters, TRACE_MODE_MAX_ITERS)

            abs_tol = float(params.getAbsoluteErrorTol())
            rel_tol = float(params.getRelativeErrorTol())
            prev_error = float(lm.error())

            for _ in range(max_iters):
                lm.iterate()
                values_trace.append(lm.values())
                curr_error = float(lm.error())
                # Match GTSAM's checkConvergence (NonlinearOptimizer.cpp) EXACTLY:
                #   abs_decrease = std::abs(newError - currentError);
                #   if (abs_decrease <= abs_tol) return true;
                #   if (abs_decrease / std::abs(currentError) <= rel_tol) return true;
                # Uses |delta|, not (prev - curr): rejected LM steps (lambda back-off
                # with no error change) count as convergence — that's GTSAM's intended
                # behavior. "No progress = stop." The earlier `abs_decrease > 0` gate
                # I added was wrong — it kept the loop spinning past the natural exit
                # point, accumulating manual-vs-vanilla LM drift on Brussels.
                abs_delta = abs(prev_error - curr_error)
                if abs_delta < abs_tol:
                    logger.info("🚀 Converged (abs_delta=%.2e < abs_tol).", abs_delta)
                    break
                if curr_error > 0 and abs_delta / curr_error < rel_tol:
                    logger.info("🚀 Converged (rel_delta=%.2e < rel_tol).", abs_delta / curr_error)
                    break
                prev_error = curr_error
            result_values = lm.values()

        elapsed_time = time.time() - start_time
        logger.info(f"🚀 Factor graph optimization completed in {elapsed_time:.2f} seconds.")
        # Side-channel: stash the per-iter trace so callers can pick it up without
        # changing the return signature. Used by _run_ba_and_evaluate to dump the
        # first outer BA's per-LM-iter snapshots into the pipeline_trace.
        self._last_values_trace = values_trace
        if self._use_gnc:
            weights = lm.getWeights()
            return result_values, values_trace, weights
        return result_values, values_trace, None

    def get_two_view_ba_pose_graph_keys(self, initial_data: GtsfmData):
        """Retrieves GTSAM keys for camera poses in a 2-view BA problem."""
        valid_camera_indices = initial_data.get_valid_camera_indices()
        return [X(valid_camera_indices[0]), X(valid_camera_indices[1])]

    def is_two_view_ba(self, initial_data: GtsfmData) -> bool:
        """Determines whether two-view bundle adjustment is being executed."""
        return len(initial_data.get_valid_camera_indices()) == 2

    def __optimize_and_recover(
        self, initial_data: GtsfmData, graph: NonlinearFactorGraph, ordering_type: str
    ) -> Tuple[GtsfmData, Values, float]:
        """Optimize the graph, report errors, and convert `Values` back to `GtsfmData`."""
        initial_values = initial_data.to_values(shared_calib=self._shared_calib)
        result_values, _, weights = self.__optimize_factor_graph(graph, initial_values, ordering_type)
        final_error = graph.error(result_values)
        optimized_data = GtsfmData.from_values(result_values, initial_data, self._shared_calib)
        if self._use_gnc and weights is not None and self._factor_weight_outlier_threshold > 0:
            optimized_data = self.__filter_tracks_by_factor_weights(graph, optimized_data, weights)
        return optimized_data, result_values, final_error

    def __filter_tracks_by_factor_weights(
        self, graph: NonlinearFactorGraph, optimized_data: GtsfmData, weights: NDArray[np.float64]
    ) -> GtsfmData:
        """Filter tracks based on the weights of the reprojection factors, if GNC optimization is used."""
        if weights is None:
            logger.error("Weights array is None, cannot filter tracks by factor weights.")
            return optimized_data
        cameras_to_model = sorted(optimized_data.get_valid_camera_indices())
        first_camera = optimized_data.get_camera(cameras_to_model[0])
        assert first_camera is not None, "First camera in optimized factor graph is None"
        sfm_factor_class = gtsfm_types.get_sfm_factor_for_calibration(first_camera.calibration())
        cams_to_remove_per_track: dict[int, set[int]] = defaultdict(set)
        for i in range(graph.nrFactors()):
            if weights[i] >= self._factor_weight_outlier_threshold:
                continue
            factor = graph.at(i)
            if isinstance(factor, sfm_factor_class):
                camera_key, track_key, _ = factor.keys()
                camera_id = int(gtsam.symbolIndex(camera_key))
                track_id = int(gtsam.symbolIndex(track_key))
                cams_to_remove_per_track[track_id].add(camera_id)

        if not cams_to_remove_per_track:
            return optimized_data

        for track_id, camera_ids in cams_to_remove_per_track.items():
            track = optimized_data.get_track(track_id)
            new_measurements = []
            for m_idx in range(track.numberMeasurements()):
                cam_id, _ = track.measurement(m_idx)
                if cam_id not in camera_ids:
                    new_measurements.append(track.measurement(m_idx))
            track.measurements = new_measurements

        length_filtered_tracks = [
            track for track in optimized_data.get_tracks() if track.numberMeasurements() >= self._min_track_length
        ]
        if len(length_filtered_tracks) == optimized_data.number_tracks():
            optimized_data._camera_to_measurement_map = None
            return optimized_data

        image_info = {
            image_id: optimized_data.get_image_info(image_id) for image_id in optimized_data.get_all_image_ids()
        }

        filtered_data = GtsfmData.from_cameras_and_tracks(
            cameras=optimized_data.cameras(),
            tracks=length_filtered_tracks,
            number_images=optimized_data.number_images(),
            image_info=image_info,
            gaussian_splats=optimized_data.get_gaussian_splats(),
        )

        return filtered_data

    def run_simple_ba(
        self, initial_data: GtsfmData, robust_noise_basin: float | None = None
    ) -> Tuple[GtsfmData, float]:
        """Runs bundle adjustment and optionally filters the resulting tracks by reprojection error.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            robust_noise_basin: Robust noise basin to use for the BA, not used if self._robust_ba_mode is NONE.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Final error value of the optimization problem.
        """
        cameras_to_model = sorted(initial_data.get_valid_camera_indices())

        graph, cameras_without_tracks = self.__construct_simple_factor_graph(
            cameras_to_model, initial_data, robust_noise_basin
        )
        if len(cameras_without_tracks) == len(initial_data.cameras()):
            logger.warning("Skipping bundle adjustment because all cameras are without tracks.")
            return initial_data, 0.0
        optimized_data, _, final_error = self.__optimize_and_recover(
            initial_data, graph, self._ordering_type if not cameras_without_tracks else "COLAMD"
        )
        return optimized_data, final_error

    def run_iterative_robust_ba(
        self, initial_data: GtsfmData, robust_noise_basins: List[float]
    ) -> Tuple[GtsfmData, float]:
        """Runs iterative robust bundle adjustment, using different robust noise basins for each iteration."""
        optimized_data = initial_data
        final_error = float("nan")
        for robust_noise_basin in robust_noise_basins:
            optimized_data, final_error = self.run_simple_ba(optimized_data, robust_noise_basin)
        return optimized_data, final_error

    def run_ba_stage_with_filtering(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        reproj_error_thresh: Optional[float],
        verbose: bool = True,
        lock_rotations: bool = False,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool], float]:
        """Runs bundle adjustment and optionally filters the resulting tracks by reprojection error.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: Priors to be used on cameras.
            relative_pose_priors: Priors on the pose between two cameras.
            reproj_error_thresh: Maximum 3D track reprojection error, for filtering tracks after BA.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Optimized camera poses after filtering landmarks (and cameras with no remaining landmarks).
            Valid mask as a list of booleans, indicating for each input track whether it was below the re-projection
                threshold.
            Final error value of the optimization problem.
        """
        logger.info(
            "Input: %d tracks on %d cameras", initial_data.number_tracks(), len(initial_data.get_valid_camera_indices())
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # No cameras or tracks to optimize, so bundle adjustment is not possible, return invalid result.
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return initial_data, initial_data, [False] * initial_data.number_tracks(), 0.0

        cameras_to_model = sorted(initial_data.get_valid_camera_indices())
        graph, cameras_without_tracks = self.__construct_factor_graph(
            cameras_to_model, initial_data, absolute_pose_priors, relative_pose_priors,
            lock_rotations=lock_rotations,
        )
        optimized_data, result_values, final_error = self.__optimize_and_recover(
            initial_data, graph, self._ordering_type if not cameras_without_tracks else "COLAMD"
        )

        if self.is_two_view_ba(initial_data):
            try:
                # Calculate marginal covariances for all two pose variables.
                marginals = gtsam.Marginals(graph, result_values)
                graph_keys = self.get_two_view_ba_pose_graph_keys(initial_data)
                for key in graph_keys:
                    _ = marginals.marginalCovariance(key)

            except RuntimeError:
                if not self._allow_indeterminate_linear_system:
                    logger.error(
                        "BA result discarded due to Indeterminate Linear System (ILS) when computing marginals."
                    )
                    return None, None, None, None

        # Convert the `Values` results to a `GtsfmData` instance.
        # Filter landmarks by reprojection error.
        if reproj_error_thresh is not None:
            if verbose:
                logger.info("[Result] Number of tracks before filtering: %d", optimized_data.number_tracks())
            filtered_result, valid_mask = optimized_data.filter_landmarks(reproj_error_thresh)
            if verbose:
                logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())

        else:
            valid_mask = [True] * optimized_data.number_tracks()
            filtered_result = optimized_data
        return optimized_data, filtered_result, valid_mask, final_error

    def run_ba(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        verbose: bool = True,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool]]:
        """Runs bundle adjustment by forming a factor graph and optimizing it using Levenberg–Marquardt optimization.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: Priors to be used on cameras.
            relative_pose_priors: Priors on the pose between two cameras.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Optimized camera poses after filtering landmarks (and cameras with no remaining landmarks).
            Valid mask as a list of booleans, indicating for each input track whether it was below the re-projection
                threshold.
        """
        num_ba_steps = len(self._reproj_error_thresholds)
        current_data = initial_data
        for step, reproj_error_thresh in enumerate(self._reproj_error_thresholds):
            # Use previous round's filtered result as initial condition for this step.
            (optimized_data, filtered_result, valid_mask, final_error) = self.run_ba_stage_with_filtering(
                current_data,
                absolute_pose_priors,
                relative_pose_priors,
                reproj_error_thresh,
                verbose,
            )
            current_data = filtered_result
            # Print intermediate results.
            if num_ba_steps > 1:
                logger.info(
                    "[BA Step %d/%d] Error: %.2f, Number of tracks: %d"
                    % (step + 1, num_ba_steps, final_error, filtered_result.number_tracks())
                )

        return optimized_data, filtered_result, valid_mask  # type: ignore

    def _run_ba_and_evaluate(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
        verbose: bool = True,
        tracks_2d: Optional[List["SfmTrack2d"]] = None,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool], GtsfmMetricsGroup]:
        """Runs the equivalent of `run_ba()` and `evaluate()` in a single function, to enable time profiling."""
        logger.info(
            "Input: %d tracks on %d cameras", initial_data.number_tracks(), len(initial_data.get_valid_camera_indices())
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # No cameras or tracks to optimize, so bundle adjustment is not possible.
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return (
                initial_data,
                initial_data,
                [False] * initial_data.number_tracks(),
                GtsfmMetricsGroup(METRICS_GROUP, []),
            )
        step_times = []
        start_time = time.time()

        # Pipeline trace bookkeeping. Auto-derive trace dir from `save_dir` when
        # `save_pipeline_trace=True` and no explicit override given. No-op when both off.
        if self._save_pipeline_trace and not self._save_pipeline_trace_dir and save_dir:
            # `save_dir` arriving here is the BA's per-scene save dir (e.g. for
            # plots/metrics) — already inside the scene's output_root. Just nest
            # one more level for the trace; explicit override available if a
            # different layout is wanted.
            self._save_pipeline_trace_dir = str(Path(save_dir) / "pipeline_trace")
        self._trace_idx = 0
        self._dump_trace(initial_data, "ba_input")

        # GLOMAP-style outer BA iteration. Mirrors global_mapper.cc:201-260 exactly:
        #
        #   while ite < num_outer:
        #     run BA   (no filter)
        #     filter_num = 0
        #     while ite < num_outer:
        #       scaling = max(num_outer - ite, 1)             # 3, 2, 1
        #       filter_num += filter @ scaling * max_reproj
        #       if filter_num > 0.1% of tracks:               # significant drop
        #         break inner   → next outer (do BA again)
        #       else:
        #         ite++                                       # tighten without BA
        #     if no significant drop ever happened: break outer (converged)
        #     ite++
        #
        # This means BA count varies from 1 (early termination, tracks already settled)
        # to num_outer (each outer iter actually needed BA after a significant filter pass).
        # We use the LAST element of reproj_error_thresholds as the base threshold;
        # earlier elements are ignored.
        num_outer = max(1, self._num_outer_ba_iterations)
        max_reproj = self._reproj_error_thresholds[-1]
        if max_reproj is None:
            max_reproj = 3.0  # safety: GLOMAP-default tight threshold

        current_data = initial_data
        # Track state across iterations.
        optimized_data = initial_data
        filtered_result = initial_data
        valid_mask: List[bool] = [True] * initial_data.number_tracks()
        final_error = 0.0

        ite = 0
        while ite < num_outer:
            # ── Stage 1: positions-only BA (rotations + intrinsics locked) ──
            # GLOMAP-style: lets translations + landmarks settle before rotations move,
            # which reduces BA basin variance on scenes with mirror-symmetric structure.
            ba_start = time.time()
            if self._positions_only_stage1:
                logger.info("[OuterBA ite=%d] Stage 1: positions-only (rotations locked)", ite)
                (stage1_data, _, _, _) = self.run_ba_stage_with_filtering(
                    initial_data=current_data,
                    absolute_pose_priors=absolute_pose_priors,
                    relative_pose_priors=relative_pose_priors,
                    reproj_error_thresh=None,
                    verbose=verbose,
                    lock_rotations=True,
                )
                current_data = stage1_data

            # ── Stage 2: full BA (no filter — we filter ourselves below) ──
            if self._positions_only_stage1:
                logger.info("[OuterBA ite=%d] Stage 2: full BA", ite)

            # Per-LM-iter capture for EVERY outer BA. The first outer BA shows
            # the dramatic GP→BA coarse-to-fine tightening; subsequent outer BAs
            # show finer refinements after intermediate filter passes. All add
            # smooth animation frames to the visualizer.
            (optimized_data, _, _, final_error) = self._ba_with_iter_capture(
                prefix=f"outer_ba_{ite}",
                initial_data=current_data,
                absolute_pose_priors=absolute_pose_priors,
                relative_pose_priors=relative_pose_priors,
                reproj_error_thresh=None,
                verbose=verbose,
                max_dump_frames=12,
            )
            current_data = optimized_data
            step_times.append(time.time() - ba_start)

            # ── Inner filter loop: tighten scaling, only re-BA if significant filter drop ──
            initial_tracks_for_iter = current_data.number_tracks()
            filter_num = 0
            status = True  # True == "no significant drop yet at any scaling"

            while status and ite < num_outer:
                scaling = max(num_outer - ite, 1)
                thresh = scaling * max_reproj
                before = current_data.number_tracks()
                filtered_result, valid_mask = current_data.filter_landmarks(thresh)
                after = filtered_result.number_tracks()
                dropped = before - after
                filter_num += dropped
                current_data = filtered_result

                logger.info(
                    "[OuterBA ite=%d/%d, scaling=%dx (thresh=%.2fpx)] dropped %d (cumul %d / %.2f%%), tracks=%d",
                    ite, num_outer, scaling, thresh, dropped, filter_num,
                    100 * filter_num / max(initial_tracks_for_iter, 1), after,
                )

                if filter_num > self._outer_ba_min_track_change_frac * initial_tracks_for_iter:
                    # Significant drop — re-run BA next outer iter.
                    status = False
                else:
                    # Few changes — tighten scaling without re-running BA.
                    ite += 1

            self._dump_trace(current_data, f"outer_ba_iter_{ite}", num_tracks=current_data.number_tracks())

            if status:
                logger.info(
                    "[OuterBA] converged after iter ite=%d: filtered %d tracks (<%.3f%% threshold) — exit outer.",
                    ite, filter_num, 100 * self._outer_ba_min_track_change_frac,
                )
                break

            ite += 1  # mirrors GLOMAP's outer for-loop increment

        # ── Multi-view retriangulation + final BA ──
        # Recover tracks dropped between union-find and BA's filter passes by re-triangulating
        # the full 2D track set with post-BA cameras, then re-BA on the augmented track set.
        # See multi_view_retriangulate_from_2d_tracks() and the plan's Phase 1 description.
        if self._use_multi_view_retriangulation:
            if tracks_2d is None:
                logger.warning(
                    "use_multi_view_retriangulation is True but tracks_2d was not "
                    "passed to _run_ba_and_evaluate — skipping retri stage."
                )
            else:
                retri_start = time.time()
                logger.info(
                    "[Retri] Running multi-view retriangulation on %d 2D tracks "
                    "(min_track_len=%d, reproj_error_thresh=%.1fpx)",
                    len(tracks_2d), self._mv_retri_min_track_length,
                    self._mv_retri_reproj_error_thresh,
                )
                retri_options = TriangulationOptions(
                    reproj_error_threshold=self._mv_retri_reproj_error_thresh,
                    mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
                    max_num_hypotheses=100,
                )
                retri_data = multi_view_retriangulate_from_2d_tracks(
                    gtsfm_data=filtered_result,
                    tracks_2d=tracks_2d,
                    triangulation_options=retri_options,
                    min_track_length=self._mv_retri_min_track_length,
                    use_recursive_splitter=self._mv_retri_use_recursive_splitter,
                    recursive_max_depth=self._mv_retri_recursive_max_depth,
                )
                logger.info(
                    "[Retri] %d → %d tracks (%.1f%%) after retriangulation, took %.1fs",
                    filtered_result.number_tracks(), retri_data.number_tracks(),
                    100 * retri_data.number_tracks() / max(filtered_result.number_tracks(), 1),
                    time.time() - retri_start,
                )
                self._dump_trace(retri_data, "retri", num_tracks=retri_data.number_tracks())

                # Final BA on the retri'd track set. Routed through _ba_with_iter_capture
                # so the visualizer gets per-LM-iter snapshots of the final refining stage
                # (5 head iters + final). In trace mode we've already accepted the manual
                # lm.iterate() drift on outer BAs — keeping retri-final consistent here
                # since this yaml is for viz, not benchmark numbers.
                if retri_data.number_tracks() > 0:
                    final_ba_start = time.time()
                    final_thresh = self._reproj_error_thresholds[-1] or 3.0
                    (optimized_data, filtered_result, valid_mask, _) = self._ba_with_iter_capture(
                        prefix="retri_final_ba",
                        initial_data=retri_data,
                        absolute_pose_priors=absolute_pose_priors,
                        relative_pose_priors=relative_pose_priors,
                        reproj_error_thresh=final_thresh,
                        verbose=verbose,
                    )
                    self._dump_trace(filtered_result, "retri_final_ba",
                                     num_tracks=filtered_result.number_tracks())
                    logger.info(
                        "[Retri] Final BA on %d retri'd tracks: filtered to %d (%.1fs).",
                        retri_data.number_tracks(), filtered_result.number_tracks(),
                        time.time() - final_ba_start,
                    )
                    step_times.append(time.time() - retri_start)

        total_time = time.time() - start_time

        metrics = self.evaluate(optimized_data, filtered_result, cameras_gt, save_dir)  # type: ignore
        for i, step_time in enumerate(step_times):
            metrics.add_metric(GtsfmMetric(f"step_{i}_run_duration_sec", step_time))
        metrics.add_metric(GtsfmMetric("total_run_duration_sec", total_time))

        return optimized_data, filtered_result, valid_mask, metrics  # type: ignore

    def evaluate(
        self,
        unfiltered_data: GtsfmData,
        filtered_data: GtsfmData,
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
    ) -> GtsfmMetricsGroup:
        """Computes metrics on the bundle adjustment result, and packages them in a GtsfmMetricsGroup object.

        Args:
            unfiltered_data: Optimized BA result, before filtering landmarks by reprojection error.
            filtered_data: Optimized BA result, after filtering landmarks and cameras.
            cameras_gt: Cameras with GT intrinsics and GT extrinsics.

        Returns:
            Metrics group containing metrics for both filtered and unfiltered BA results.
        """
        ba_metrics = GtsfmMetricsGroup(name=METRICS_GROUP, metrics=unfiltered_data.get_metrics(suffix="_unfiltered"))

        input_image_idxs = list(unfiltered_data._image_info.keys())
        poses_gt = {
            i: cameras_gt[i].pose() for i in input_image_idxs if i < len(cameras_gt) and cameras_gt[i] is not None
        }
        if not poses_gt:
            return ba_metrics

        # Align the sparse multi-view estimate after BA to the ground truth pose graph.
        aligned_filtered_data = filtered_data.align_via_sim3_and_transform(poses_gt)
        ba_pose_error_metrics = metrics_utils.compute_ba_pose_metrics(
            gt_wTi=poses_gt, computed_wTi=aligned_filtered_data.get_camera_poses(), save_dir=save_dir,
            metric_constructed_only=True,
        )
        ba_metrics.extend(metrics_group=ba_pose_error_metrics)

        output_tracks_exit_codes = track_utils.classify_tracks3d_with_gt_cameras(
            tracks=aligned_filtered_data.get_tracks(), cameras_gt=cameras_gt
        )
        output_tracks_exit_codes_distribution = Counter(output_tracks_exit_codes)

        for exit_code, count in output_tracks_exit_codes_distribution.items():
            metric_name = "Filtered tracks triangulated with GT cams: {}".format(exit_code.name)
            ba_metrics.add_metric(GtsfmMetric(name=metric_name, data=count))

        ba_metrics.add_metrics(aligned_filtered_data.get_metrics(suffix="_filtered"))

        logger.info("[Result] Mean track length %.3f", np.mean(aligned_filtered_data.get_track_lengths()))
        logger.info("[Result] Median track length %.3f", np.median(aligned_filtered_data.get_track_lengths()))
        aligned_filtered_data.log_scene_reprojection_error_stats()

        return ba_metrics

    def create_computation_graph(
        self,
        sfm_data_graph: Delayed,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
        tracks_2d: Optional[Delayed] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: An GtsfmData object wrapped up using dask.delayed.
            absolute_pose_priors: Priors on the poses of the cameras (not delayed).
            relative_pose_priors: Priors on poses between cameras (not delayed).
            cameras_gt: Ground truth camera calibration & pose for each image/camera.
            save_dir: Directory where artifacts and plots should be saved to disk.
            tracks_2d: (optional) Delayed list of 2D tracks from CppDsfTracksEstimator.
                Required if `use_multi_view_retriangulation` is enabled — used to retriangulate
                the full union-find track set with post-BA cameras.

        Returns:
            GtsfmData aligned to GT (if provided), wrapped up using dask.delayed
            Metrics group for BA results, wrapped up using dask.delayed
        """

        _, filtered_sfm_data, _, metrics_graph = dask.delayed(self._run_ba_and_evaluate, nout=4)(
            sfm_data_graph,
            absolute_pose_priors,
            relative_pose_priors,
            cameras_gt,
            save_dir=save_dir,
            tracks_2d=tracks_2d,
        )
        return filtered_sfm_data, metrics_graph
