"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""

import logging

import gtsfm.utils.logger as logger_utils
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from dask.delayed import Delayed, delayed
from dask.distributed import Future
from gtsam import Pose3, Rot3, Unit3  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.graph as graph_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.global_ba import GlobalBundleAdjustment
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.data_association.cpp_dsf_tracks_estimator import CppDsfTracksEstimator
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.global_positioner.global_positioner import GlobalPositioner
from gtsfm.products.one_view_data import OneViewData
from gtsfm.products.two_view_result import TwoViewResult
from gtsfm.products.visibility_graph import AnnotatedGraph
from gtsfm.view_graph_estimator.cycle_consistent_rotation_estimator import (
    CycleConsistentRotationViewGraphEstimator,
    EdgeErrorAggregationCriterion,
)
from gtsfm.utils import verification as verification_utils
from gtsfm.view_graph_estimator.view_graph_calibration import calibrate_view_graph
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase

logger = logger_utils.get_logger()


class MultiViewOptimizer:
    @staticmethod
    def _extract_two_view_components(
        two_view_results: AnnotatedGraph[TwoViewResult],
    ) -> tuple[
        Dict[Tuple[int, int], Rot3],
        Dict[Tuple[int, int], Unit3],
        AnnotatedGraph[np.ndarray],
        AnnotatedGraph[TwoViewEstimationReport],
        AnnotatedGraph[PosePrior],
    ]:
        """Split TwoViewResult objects into the pieces needed by downstream modules."""

        i2Ri1_dict: Dict[Tuple[int, int], Rot3] = {}
        i2Ui1_dict: Dict[Tuple[int, int], Unit3] = {}
        v_corr_idxs_dict: AnnotatedGraph[np.ndarray] = {}
        two_view_reports: AnnotatedGraph[TwoViewEstimationReport] = {}
        relative_pose_priors: AnnotatedGraph[PosePrior] = {}

        for ij, result in two_view_results.items():
            i2Ri1_dict[ij] = result.i2Ri1
            i2Ui1_dict[ij] = result.i2Ui1
            v_corr_idxs_dict[ij] = result.v_corr_idxs
            two_view_reports[ij] = result.post_isp_report
            if result.relative_pose_prior is not None:
                relative_pose_priors[ij] = result.relative_pose_prior

        return i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports, relative_pose_priors

    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
        bundle_adjustment_module: GlobalBundleAdjustment,
        view_graph_estimator: Optional[ViewGraphEstimatorBase] = None,
        global_positioner: Optional[GlobalPositioner] = None,
        run_view_graph_calibration: bool = False,
        rotation_outlier_threshold_deg: float = 0.0,
    ) -> None:
        self.view_graph_estimator = view_graph_estimator
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = data_association_module
        self.ba_optimizer = bundle_adjustment_module
        self.global_positioner = global_positioner
        self._run_view_graph_estimator: bool = self.view_graph_estimator is not None
        self._run_view_graph_calibration = run_view_graph_calibration
        self._rotation_outlier_threshold_deg = rotation_outlier_threshold_deg


    def __repr__(self) -> str:
        return f"""
        MultiviewOptimizer:
            ViewGraphEstimator: {self.view_graph_estimator}
            RotationAveraging: {self.rot_avg_module}
            TranslationAveraging: {self.trans_avg_module}
        """

    def create_computation_graph(
        self,
        keypoints_graph: Delayed,
        two_view_results_graph: Delayed,
        one_view_data_dict: Dict[int, OneViewData],
        image_future_map: Mapping[int, Future],
        output_root: Optional[Path] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, list]:
        """Creates a computation graph for multi-view optimization.

        Args:
            keypoints_graph: Delayed task producing padded keypoints for images.
            two_view_results_graph: Delayed task producing valid two-view results for image pairs.
            one_view_data_dict: Per-view data entries keyed by image index.
            image_future_map: Mapping of image index to cached futures for `get_image`.
            output_root: Path where output should be saved.

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            Dict of TwoViewEstimationReports after view graph estimation.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """

        i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports, pose_priors_graph = delayed(
            MultiViewOptimizer._extract_two_view_components, nout=5
        )(two_view_results_graph)

        # Create debug directory.
        debug_output_dir = None
        if output_root:
            debug_output_dir = output_root / "debug"
            os.makedirs(debug_output_dir, exist_ok=True)

        num_images = len(one_view_data_dict)
        all_intrinsics = [one_view_data_dict[idx].intrinsics for idx in one_view_data_dict.keys()]
        if self._run_view_graph_estimator and self.view_graph_estimator is not None:
            (
                viewgraph_i2Ri1_graph,
                viewgraph_i2Ui1_graph,
                viewgraph_v_corr_idxs_graph,
                viewgraph_two_view_reports_graph,
                viewgraph_estimation_metrics,
            ) = self.view_graph_estimator.create_computation_graph(
                i2Ri1_dict,
                i2Ui1_dict,
                all_intrinsics,
                v_corr_idxs_dict,
                keypoints_graph,
                two_view_reports,
                debug_output_dir,
            )

            # (v2 cycle-consistency pass removed — empirically worse than no-v2 on Brussels
            # across all thresholds. GLOMAP-style post-RA iterative filter replaces it below.)
        else:
            viewgraph_i2Ri1_graph = i2Ri1_dict
            viewgraph_i2Ui1_graph = i2Ui1_dict
            viewgraph_v_corr_idxs_graph = v_corr_idxs_dict
            viewgraph_two_view_reports_graph = two_view_reports
            viewgraph_estimation_metrics = delayed(GtsfmMetricsGroup("view_graph_estimation_metrics", []))

        # GLOMAP Stage 2: Re-score inliers using F-matrix Sampson error + orientation signum cheirality.
        # Pixel-space filter, intrinsic-independent. Filters correspondences that passed PoseLib
        # RANSAC but fail the tighter epipolar test.
        viewgraph_v_corr_idxs_graph, viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph = delayed(
            rescore_inliers_fundamental, nout=3
        )(
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph,
            viewgraph_v_corr_idxs_graph, keypoints_graph, all_intrinsics,
            min_inlier_count=30,
            min_inlier_ratio=0.25,
        )

        # View graph calibration: refine focal lengths from F-matrices before global positioning.
        if self._run_view_graph_calibration:
            all_intrinsics, edges_to_remove = delayed(calibrate_view_graph, nout=2)(
                viewgraph_v_corr_idxs_graph, keypoints_graph, all_intrinsics, num_images
            )
            # Remove edges with high calibration error (GLOMAP FilterImagePairs).
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, viewgraph_v_corr_idxs_graph = delayed(
                _filter_edges, nout=3
            )(viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, viewgraph_v_corr_idxs_graph, edges_to_remove)
            # Re-estimate relative poses with refined intrinsics (GLOMAP Section 3.5).
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph = delayed(reestimate_relative_poses, nout=2)(
                viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph,
                viewgraph_v_corr_idxs_graph, keypoints_graph, all_intrinsics,
            )

        # Prune the graph to a single connected component.
        gt_wTi = {k: val.pose_gt for k, val in one_view_data_dict.items()}
        gt_wTi_list = [gt_wTi[i] for i in sorted(list(gt_wTi.keys()))]
        pruned_i2Ri1_graph, pruned_i2Ui1_graph = delayed(graph_utils.prune_to_largest_connected_component, nout=2)(
            viewgraph_i2Ri1_graph, viewgraph_i2Ui1_graph, pose_priors_graph
        )

        # Single rotation averaging pass.
        delayed_wRi, rot_avg_metrics = self.rot_avg_module.create_computation_graph(
            num_images,
            pruned_i2Ri1_graph,
            i1Ti2_priors=pose_priors_graph,
            gt_wTi_list=gt_wTi_list,
            v_corr_idxs=viewgraph_v_corr_idxs_graph,
        )
        # Optional: prune cameras with globally inconsistent rotations (off by default).
        if self._rotation_outlier_threshold_deg > 0:
            delayed_wRi = delayed(prune_rotation_outliers)(
                delayed_wRi, pruned_i2Ri1_graph, self._rotation_outlier_threshold_deg
            )
        tracks2d_graph = delayed(get_2d_tracks)(viewgraph_v_corr_idxs_graph, keypoints_graph)

        absolute_pose_priors = [one_view_data_dict[idx].absolute_pose_prior for idx in range(num_images)]
        cameras_gt = [one_view_data_dict[idx].camera_gt for idx in sorted(list(one_view_data_dict.keys()))]

        if self.global_positioner is not None:
            # Path B: Global positioner replaces trans_avg + data_assoc.
            ba_input_graph, gp_metrics = delayed(self.global_positioner.run, nout=2)(
                num_images, delayed_wRi, tracks2d_graph, all_intrinsics,
                output_root=output_root,
            )
            ta_metrics = gp_metrics
            data_assoc_metrics_graph = delayed(GtsfmMetricsGroup)("data_association_metrics", [])
        else:
            # Path A: Existing pipeline — trans_avg + data_assoc.
            wTi_graph, ta_metrics, ta_inlier_idx_i1_i2 = self.trans_avg_module.create_computation_graph(
                num_images,
                pruned_i2Ui1_graph,
                delayed_wRi,
                tracks2d_graph,
                all_intrinsics,
                absolute_pose_priors,
                pose_priors_graph,
                gt_wTi_list=gt_wTi_list,
            )
            ta_v_corr_idxs_graph = delayed(filter_corr_by_idx)(viewgraph_v_corr_idxs_graph, ta_inlier_idx_i1_i2)
            ta_inlier_tracks_2d_graph = delayed(get_2d_tracks)(ta_v_corr_idxs_graph, keypoints_graph)
            # TODO(akshay-krishnan): update pose priors also with the same inlier indices, right now these are unused.

            init_cameras_graph = delayed(init_cameras)(wTi_graph, all_intrinsics)

            images: List[Future] = [image_future_map[idx] for idx in sorted(list(one_view_data_dict.keys()))]
            ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
                num_images,
                init_cameras_graph,
                ta_inlier_tracks_2d_graph,
                cameras_gt,
                pose_priors_graph,
                images,
            )

        # Use the full union-find 2D track set for retri (recovers tracks dropped by GP/BA).
        # Path A (trans_avg): use the trans-avg-inlier-filtered subset since that's what BA saw.
        # Path B (global_positioner): use the unfiltered viewgraph track set.
        retri_tracks_2d_graph = tracks2d_graph if self.global_positioner is not None else ta_inlier_tracks_2d_graph
        ba_result_graph, ba_metrics_graph = self.ba_optimizer.create_computation_graph(
            ba_input_graph,
            absolute_pose_priors,
            pose_priors_graph,
            cameras_gt,
            save_dir=str(output_root) if output_root else None,
            tracks_2d=retri_tracks_2d_graph,
        )

        multiview_optimizer_metrics_graph = [
            viewgraph_estimation_metrics,
            rot_avg_metrics,
            ta_metrics,
            data_assoc_metrics_graph,
            ba_metrics_graph,
        ]

        # This conversion is OK since the GT poses are expected to be complete.
        gt_wTi_dict: dict[int, Pose3] = {
            i: gt_wTi_list[i] for i in range(len(gt_wTi_list)) if gt_wTi_list[i] is not None
        }

        # Align the sparse multi-view estimate before BA to the ground truth pose graph.
        ba_input_graph = delayed(GtsfmData.align_via_sim3_and_transform)(ba_input_graph, gt_wTi_dict)

        return ba_input_graph, ba_result_graph, viewgraph_two_view_reports_graph, multiview_optimizer_metrics_graph


def filter_edges_by_rotation(
    wRi_list: List[Optional[Rot3]],
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    max_rotation_error_deg: float = 10.0,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Filter edges whose relative rotation is inconsistent with global rotations (GLOMAP FilterRotations).

    For each edge (i1, i2), compares the measured i2Ri1 with the expected wRi2 * wRi1^T.
    Removes edges where the angular error exceeds the threshold.

    Args:
        wRi_list: Global rotations from rotation averaging.
        i2Ri1_dict: Relative rotations from the view graph.
        i2Ui1_dict: Relative translation directions.
        max_rotation_error_deg: Maximum angular error in degrees to keep an edge.

    Returns:
        Filtered relative rotations and translation directions.
    """
    num_images = len(wRi_list)
    filtered_R = {}
    filtered_U = {}
    num_removed = 0

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        wRi1 = wRi_list[i1] if i1 < num_images else None
        wRi2 = wRi_list[i2] if i2 < num_images else None
        if wRi1 is None or wRi2 is None:
            # Can't check — keep the edge.
            filtered_R[(i1, i2)] = i2Ri1
            if (i1, i2) in i2Ui1_dict:
                filtered_U[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            continue
        # Expected relative rotation from global estimates.
        i2Ri1_expected = wRi2.compose(wRi1.inverse())
        error_rot = i2Ri1_expected.compose(i2Ri1.inverse())
        error_deg = abs(error_rot.axisAngle()[1]) * 180.0 / np.pi
        if error_deg <= max_rotation_error_deg:
            filtered_R[(i1, i2)] = i2Ri1
            if (i1, i2) in i2Ui1_dict:
                filtered_U[(i1, i2)] = i2Ui1_dict[(i1, i2)]
        else:
            num_removed += 1

    logger.info(
        "Rotation edge filtering: removed %d / %d edges (threshold=%.1f deg), %d remain.",
        num_removed, len(i2Ri1_dict), max_rotation_error_deg, len(filtered_R),
    )
    return filtered_R, filtered_U


def _filter_by_reprojection_logged(
    gtsfm_data: GtsfmData, reproj_err_thresh: float, min_track_length: int
) -> GtsfmData:
    """Wrap GtsfmData.filter_landmark_measurements with entry/exit logs."""
    logger.info(
        "FilterTracksByReprojection: running on %d tracks (threshold=%.1f px, min_track_len=%d).",
        gtsfm_data.number_tracks(), reproj_err_thresh, min_track_length,
    )
    filtered = gtsfm_data.filter_landmark_measurements(
        reproj_err_thresh=reproj_err_thresh, min_track_length=min_track_length
    )
    logger.info(
        "FilterTracksByReprojection: DONE — %d → %d tracks (%d dropped).",
        gtsfm_data.number_tracks(), filtered.number_tracks(),
        gtsfm_data.number_tracks() - filtered.number_tracks(),
    )
    return filtered


def filter_tracks_by_triangulation_angle(
    gtsfm_data: GtsfmData,
    min_angle_deg: float = 1.0,
) -> GtsfmData:
    """GLOMAP FilterTrackTriangulationAngle: drop tracks with no sufficiently parallaxed observation pair."""
    logger.info(
        "FilterTrackTriangulationAngle: running on %d tracks (threshold=%.2f deg).",
        gtsfm_data.number_tracks(), min_angle_deg,
    )
    cos_min = np.cos(np.radians(min_angle_deg))
    filtered = GtsfmData(gtsfm_data.number_images())
    for cam_idx, camera in gtsfm_data.cameras().items():
        filtered.add_camera(cam_idx, camera)

    num_before = gtsfm_data.number_tracks()
    num_dropped = 0
    num_too_short = 0
    for j in range(num_before):
        track = gtsfm_data.get_track(j)
        point3 = track.point3()
        n_meas = track.numberMeasurements()
        if n_meas < 2:
            num_dropped += 1
            num_too_short += 1
            continue
        dirs = []
        for i in range(n_meas):
            cam_idx, _ = track.measurement(i)
            camera = gtsfm_data.get_camera(cam_idx)
            if camera is None:
                continue
            v = point3 - camera.pose().translation()
            n = float(np.linalg.norm(v))
            if n < 1e-12:
                continue
            dirs.append(v / n)
        found = False
        for i in range(len(dirs)):
            for k in range(i + 1, len(dirs)):
                if float(dirs[i] @ dirs[k]) < cos_min:
                    found = True
                    break
            if found:
                break
        if found:
            filtered.add_track(track)
        else:
            num_dropped += 1

    logger.info(
        "FilterTrackTriangulationAngle: DONE — dropped %d / %d tracks (%d too-short, %d low-parallax), %d remain.",
        num_dropped, num_before, num_too_short, num_dropped - num_too_short, num_before - num_dropped,
    )
    return filtered


def rescore_inliers_fundamental(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: AnnotatedGraph[np.ndarray],
    keypoints_list: List[Keypoints],
    intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
    max_sampson_error_px: float = 4.0,
    min_inlier_count: int = 0,
    min_inlier_ratio: float = 0.0,
) -> Tuple[AnnotatedGraph[np.ndarray], Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Re-score inliers using F-matrix Sampson error + orientation signum cheirality (GLOMAP ScoreErrorFundamental).

    Also filters weak pairs by minimum inlier count and ratio (GLOMAP FilterInlierNum/FilterInlierRatio).

    Args:
        i2Ri1_dict: Relative rotations per edge.
        i2Ui1_dict: Relative translation directions per edge.
        v_corr_idxs_dict: Verified correspondence indices per edge.
        keypoints_list: Keypoints for all images.
        intrinsics: Camera intrinsics (used to compute F from E).
        max_sampson_error_px: Maximum Sampson error in pixels.
        min_inlier_count: Minimum re-scored inliers to keep a pair.
        min_inlier_ratio: Minimum ratio of re-scored inliers to total matches.

    Returns:
        Re-scored correspondence indices per edge.
        Filtered relative rotations (weak pairs removed).
        Filtered relative translations (weak pairs removed).
    """
    import cv2
    from gtsam import EssentialMatrix

    max_sampson_sq = max_sampson_error_px * max_sampson_error_px
    rescored = {}
    total_before = 0
    total_after = 0

    for (i1, i2), v_corr_idxs in v_corr_idxs_dict.items():
        if v_corr_idxs.shape[0] == 0:
            rescored[(i1, i2)] = v_corr_idxs
            continue

        i2Ri1 = i2Ri1_dict.get((i1, i2))
        i2Ui1 = i2Ui1_dict.get((i1, i2))
        if i2Ri1 is None or i2Ui1 is None:
            rescored[(i1, i2)] = v_corr_idxs
            continue

        # Compute F-matrix from relative pose and intrinsics: F = K2^{-T} [t]_x R K1^{-1}
        K1 = intrinsics[i1].K()
        K2 = intrinsics[i2].K()
        E = EssentialMatrix(i2Ri1, i2Ui1).matrix()
        K2_inv_T = np.linalg.inv(K2).T
        K1_inv = np.linalg.inv(K1)
        F = K2_inv_T @ E @ K1_inv

        # Compute epipole for orientation signum cheirality.
        epipole = np.cross(F[0], F[2])
        if np.linalg.norm(epipole) < 1e-12:
            epipole = np.cross(F[1], F[2])

        coords_i1 = keypoints_list[i1].coordinates[v_corr_idxs[:, 0]]
        coords_i2 = keypoints_list[i2].coordinates[v_corr_idxs[:, 1]]

        # Compute Sampson error + orientation signum for all matches.
        keep_mask = np.zeros(len(v_corr_idxs), dtype=bool)
        signums = []
        inlier_indices = []
        inlier_errors = []

        for k in range(len(v_corr_idxs)):
            pt1 = coords_i1[k]
            pt2 = coords_i2[k]
            pt1_h = np.array([pt1[0], pt1[1], 1.0])
            pt2_h = np.array([pt2[0], pt2[1], 1.0])

            # Sampson error in pixel space.
            Fx1 = F @ pt1_h
            Ftx2 = F.T @ pt2_h
            x2Fx1 = pt2_h @ Fx1
            sampson_sq = (x2Fx1 ** 2) / (Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-12)

            if sampson_sq < max_sampson_sq:
                # Orientation signum cheirality (from GC-RANSAC / GLOMAP).
                signum1 = F[0, 0] * pt2[0] + F[1, 0] * pt2[1] + F[2, 0]
                signum2 = epipole[1] - epipole[2] * pt1[1]
                signums.append(signum1 * signum2)
                inlier_indices.append(k)
                inlier_errors.append(sampson_sq)

        if not signums:
            rescored[(i1, i2)] = np.array([], dtype=np.int32).reshape(0, 2)
            total_before += len(v_corr_idxs)
            continue

        # Determine dominant signum direction.
        positive_count = sum(1 for s in signums if s > 0)
        negative_count = len(signums) - positive_count
        is_positive = positive_count > negative_count

        # Keep only matches with consistent cheirality.
        for idx, signum in zip(inlier_indices, signums):
            if (signum > 0) == is_positive:
                keep_mask[idx] = True

        total_before += len(v_corr_idxs)
        total_after += keep_mask.sum()
        rescored[(i1, i2)] = v_corr_idxs[keep_mask]

    # Filter weak pairs by inlier count and ratio (GLOMAP FilterInlierNum + FilterInlierRatio).
    filtered_corr = {}
    filtered_R = {}
    filtered_U = {}
    num_pairs_removed = 0
    for (i1, i2), corr in rescored.items():
        original_count = len(v_corr_idxs_dict.get((i1, i2), []))
        rescored_count = len(corr)
        ratio = rescored_count / max(original_count, 1)
        if rescored_count >= min_inlier_count and ratio >= min_inlier_ratio:
            filtered_corr[(i1, i2)] = corr
            if (i1, i2) in i2Ri1_dict:
                filtered_R[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            if (i1, i2) in i2Ui1_dict:
                filtered_U[(i1, i2)] = i2Ui1_dict[(i1, i2)]
        else:
            num_pairs_removed += 1

    logger.info(
        "F-matrix inlier re-scoring: %d → %d correspondences across %d edges (%.1f%% kept). "
        "Removed %d weak pairs (<%d inliers or <%.0f%% ratio), %d pairs remain.",
        total_before, total_after, len(v_corr_idxs_dict),
        100.0 * total_after / max(total_before, 1),
        num_pairs_removed, min_inlier_count, min_inlier_ratio * 100, len(filtered_corr),
    )
    return filtered_corr, filtered_R, filtered_U


def _filter_edges(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: AnnotatedGraph[np.ndarray],
    edges_to_remove: set,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], AnnotatedGraph[np.ndarray]]:
    """Remove edges flagged by view graph calibration."""
    if not edges_to_remove:
        return i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict
    filtered_R = {k: v for k, v in i2Ri1_dict.items() if k not in edges_to_remove}
    filtered_U = {k: v for k, v in i2Ui1_dict.items() if k not in edges_to_remove}
    filtered_corr = {k: v for k, v in v_corr_idxs_dict.items() if k not in edges_to_remove}
    logger.info("Edge filtering: removed %d edges, %d remain.", len(edges_to_remove), len(filtered_R))
    return filtered_R, filtered_U, filtered_corr


def reestimate_relative_poses(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: AnnotatedGraph[np.ndarray],
    keypoints_list: List[Keypoints],
    intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Re-estimate relative poses using refined intrinsics.

    After view graph calibration improves focal lengths, re-compute E-matrices
    and decompose into relative rotation/translation using the updated intrinsics.

    Args:
        i2Ri1_dict: Current relative rotations.
        i2Ui1_dict: Current relative translation directions.
        v_corr_idxs_dict: Verified correspondence indices per image pair.
        keypoints_list: Keypoints for all images.
        intrinsics: Refined intrinsics from view graph calibration.

    Returns:
        Updated relative rotations and translation directions.
    """
    import cv2

    updated_i2Ri1 = {}
    updated_i2Ui1 = {}
    num_updated = 0
    num_failed = 0

    for (i1, i2) in i2Ri1_dict:
        if (i1, i2) not in v_corr_idxs_dict:
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            continue

        v_corr_idxs = v_corr_idxs_dict[(i1, i2)]
        if v_corr_idxs.shape[0] < 5:
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            continue

        coords_i1 = keypoints_list[i1].coordinates[v_corr_idxs[:, 0]]
        coords_i2 = keypoints_list[i2].coordinates[v_corr_idxs[:, 1]]

        K1 = intrinsics[i1]
        K2 = intrinsics[i2]

        # Estimate F-matrix from verified correspondences.
        F, mask = cv2.findFundamentalMat(coords_i1, coords_i2, method=cv2.FM_8POINT)
        if F is None or F.shape != (3, 3):
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            num_failed += 1
            continue

        # Convert F → E using refined intrinsics, then decompose.
        i2Ei1 = verification_utils.fundamental_to_essential_matrix(F, K1, K2)
        i2Ri1_new, i2Ui1_new = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1, coords_i1, coords_i2, K1, K2,
        )

        if i2Ri1_new is not None and i2Ui1_new is not None:
            updated_i2Ri1[(i1, i2)] = i2Ri1_new
            updated_i2Ui1[(i1, i2)] = i2Ui1_new
            num_updated += 1
        else:
            updated_i2Ri1[(i1, i2)] = i2Ri1_dict[(i1, i2)]
            updated_i2Ui1[(i1, i2)] = i2Ui1_dict[(i1, i2)]
            num_failed += 1

    logger.info(
        "Re-estimated relative poses: %d updated, %d failed, %d total.",
        num_updated, num_failed, len(i2Ri1_dict),
    )
    return updated_i2Ri1, updated_i2Ui1


def prune_rotation_outliers(
    wRi_list: List[Optional[Rot3]],
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    max_median_error_deg: float = 8.0,
) -> List[Optional[Rot3]]:
    """Null out cameras whose estimated global rotation is inconsistent with their pairwise edges.

    For each camera, computes the angular residual against all its relative rotation edges:
        error = angle(wRj * wRi^T, i2Ri1_measured)
    Cameras with median residual above the threshold are set to None.

    Args:
        wRi_list: Global rotations from rotation averaging.
        i2Ri1_dict: Relative rotations from the view graph.
        max_median_error_deg: Threshold in degrees for median edge residual.

    Returns:
        Pruned rotation list with outlier cameras set to None.
    """
    from collections import defaultdict

    num_images = len(wRi_list)
    per_camera_errors: Dict[int, List[float]] = defaultdict(list)

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        wRi1 = wRi_list[i1] if i1 < num_images else None
        wRi2 = wRi_list[i2] if i2 < num_images else None
        if wRi1 is None or wRi2 is None:
            continue
        # Expected relative rotation from global estimates.
        i2Ri1_expected = wRi2.compose(wRi1.inverse())
        # Angular error between expected and measured.
        error_rot = i2Ri1_expected.compose(i2Ri1.inverse())
        error_deg = abs(error_rot.axisAngle()[1]) * 180.0 / np.pi
        per_camera_errors[i1].append(error_deg)
        per_camera_errors[i2].append(error_deg)

    pruned = list(wRi_list)
    num_pruned = 0
    all_median_errors = []
    for cam_idx in range(num_images):
        errors = per_camera_errors.get(cam_idx, [])
        if len(errors) == 0:
            continue
        median_err = float(np.mean(errors))
        all_median_errors.append(median_err)
        if median_err > max_median_error_deg:
            pruned[cam_idx] = None
            num_pruned += 1

    if all_median_errors:
        arr = np.array(sorted(all_median_errors))
        logger.info(
            "Rotation outlier pruning: per-camera median errors (deg): "
            "min=%.2f, p50=%.2f, p75=%.2f, p90=%.2f, p95=%.2f, max=%.2f",
            arr[0], np.median(arr), np.percentile(arr, 75),
            np.percentile(arr, 90), np.percentile(arr, 95), arr[-1],
        )
    logger.info(
        "Rotation outlier pruning: removed %d / %d cameras (threshold=%.1f deg).",
        num_pruned, sum(1 for r in wRi_list if r is not None), max_median_error_deg,
    )
    return pruned


def init_cameras(
    wTi_list: List[Optional[Pose3]],
    intrinsics_list: List[gtsfm_types.CALIBRATION_TYPE],
) -> Dict[int, gtsfm_types.CAMERA_TYPE]:
    """Generate camera from valid rotations and unit-translations.

    Args:
        wTi_list: Estimated global poses for cameras.
        intrinsics_list: Intrinsics for cameras.

    Returns:
        Valid cameras.
    """
    cameras = {}

    camera_class = gtsfm_types.get_camera_class_for_calibration(intrinsics_list[0])
    for idx, (wTi) in enumerate(wTi_list):
        if wTi is None:
            continue
        cameras[idx] = camera_class(wTi, intrinsics_list[idx])

    return cameras


def get_2d_tracks(correspondences: AnnotatedGraph[np.ndarray], keypoints_graph: List[Keypoints]) -> List[SfmTrack2d]:
    tracks_estimator = CppDsfTracksEstimator()
    return tracks_estimator.run(correspondences, keypoints_graph)


def filter_corr_by_idx(correspondences: AnnotatedGraph[np.ndarray], idxs: List[Tuple[int, int]]):
    """Filter correspondences by indices.

    Args:
        correspondences: Correspondences as a dictionary.
        idxs: Indices to filter by.

    Returns:
        Filtered correspondences.
    """
    return {k: v for k, v in correspondences.items() if k in idxs}
