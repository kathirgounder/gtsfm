"""Global positioning of cameras and 3D points via bearing constraints.

Implements the global positioning step from GLOMAP (Pan et al., ECCV 2024),
which jointly estimates camera positions and 3D point positions using
bearing vector constraints from 2D feature tracks.

This module replaces BOTH translation averaging AND data association
(triangulation) in the GTSFM pipeline. It outputs GtsfmData directly —
cameras with full poses + 3D tracks with landmarks — ready for bundle
adjustment.

Pipeline integration:
    BEFORE:  rot_avg → trans_avg → data_association → BA
    AFTER:   rot_avg → GlobalPositioner → BA

The GlobalPositioner takes rotations from rotation averaging, 2D tracks,
and intrinsics, and produces a complete GtsfmData with:
    - Camera objects (Rot3 from RA + estimated Point3 + intrinsics)
    - 3D tracks (estimated Point3 landmarks + 2D measurements)
    - Reprojection-error-based filtering of outlier tracks

Uses GTSAM's native BilinearAngleTranslationFactor (C++) via
TranslationRecovery with use_bilinear_translation_factor=True.
This is the exact BATA residual from GLOMAP: scale * (Tb - Ta) - measurement,
running at full C++ speed with analytical Jacobians and sparse Schur complement.

References:
    Pan, L., Barath, D., Pollefeys, M., Schonberger, J.L.
    "Global Structure-from-Motion Revisited." ECCV 2024.

    Zhuang, B., Tran, Q.H., Ji, P., Cheong, L.F., Chandraker, M.
    "Learning Structure-And-Motion-Aware Rolling Shutter Correction." CVPR 2018.

Authors: Kathir Gounder
"""

import os
import pickle
import time
from typing import List, Optional, Set, Tuple

import gtsam
import numpy as np
from gtsam import (
    BinaryMeasurementUnit3,
    BinaryMeasurementsUnit3,
    Point3,
    Pose3,
    Rot3,
    SfmTrack,
    Symbol,
    TranslationRecovery,
    Unit3,
    Values,
)
from gtsam.symbol_shorthand import A as C  # Camera position variables (Point3)
from gtsam.symbol_shorthand import B as L  # Landmark position variables (Point3)

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

logger = logger_utils.get_logger()


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MIN_TRACK_MEASUREMENTS = 3
DEFAULT_HUBER_LOSS_SCALE = 0.1      # GLOMAP: loss_function_scale = 0.1
DEFAULT_BEARING_NOISE_SIGMA = 0.01  # Matches 1DSfM's NOISE_MODEL_SIGMA
DEFAULT_MAX_ITERATIONS = 100        # GLOMAP: max_num_iterations = 100
DEFAULT_MAX_REPROJ_ERROR = 5.0      # Pixels — for filtering output tracks
DEFAULT_MAX_TRACKS = 0              # 0 = no cap (C++ solver is fast enough)


# ──────────────────────────────────────────────────────────────────────────────
# Bearing computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_world_bearings(
    tracks_2d: List[SfmTrack2d],
    intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
    wRi_list: List[Optional[Rot3]],
    valid_cameras: Set[int],
) -> List[Tuple[int, int, np.ndarray]]:
    """Compute world-frame bearing vectors for all track observations.

    For each (track, camera) pair, computes the bearing direction from the
    camera to the 3D point in world coordinates:
        bearing = wRi * normalize(Ki^{-1} * [u, v, 1]^T)

    This matches the bearing computation in averaging_1dsfm.py's
    _get_landmark_directions() and GLOMAP's AddPoint3DToProblem().

    Returns list of (track_idx, cam_idx, unit_bearing) tuples.
    """
    observations = []
    for track_idx, track in enumerate(tracks_2d):
        for m_idx in range(track.number_measurements()):
            m = track.measurement(m_idx)
            cam_idx, uv = m.i, m.uv
            if cam_idx not in valid_cameras:
                continue
            wRi, Ki = wRi_list[cam_idx], intrinsics[cam_idx]
            if wRi is None or Ki is None:
                continue
            try:
                normalized_pt = Ki.calibrate(uv)
            except RuntimeError:
                continue

            camera_ray = np.array([normalized_pt[0], normalized_pt[1], 1.0])
            camera_ray /= np.linalg.norm(camera_ray)
            world_bearing = wRi.rotate(Point3(*camera_ray))
            wb = np.array([world_bearing[0], world_bearing[1], world_bearing[2]])
            norm = np.linalg.norm(wb)
            if norm < 1e-12:
                continue
            observations.append((track_idx, cam_idx, wb / norm))

    return observations


# ──────────────────────────────────────────────────────────────────────────────
# Track filtering
# ──────────────────────────────────────────────────────────────────────────────


def filter_tracks(tracks_2d, valid_cameras, min_measurements=MIN_TRACK_MEASUREMENTS, max_tracks=DEFAULT_MAX_TRACKS):
    """Keep tracks with >= min_measurements observations from valid cameras.

    If max_tracks > 0 and more tracks survive, subsample by preferring
    longer tracks (they provide stronger constraints).
    """
    scored = []
    for track in tracks_2d:
        valid_count = sum(
            1 for m_idx in range(track.number_measurements())
            if track.measurement(m_idx).i in valid_cameras
        )
        if valid_count >= min_measurements:
            scored.append((valid_count, track))

    # Sort by track length descending — longer tracks constrain the problem better.
    scored.sort(key=lambda x: x[0], reverse=True)

    if max_tracks > 0 and len(scored) > max_tracks:
        scored = scored[:max_tracks]

    return [track for _, track in scored]


# ──────────────────────────────────────────────────────────────────────────────
# Factor graph construction + optimization (native C++)
# ──────────────────────────────────────────────────────────────────────────────


def build_and_solve(
    observations: List[Tuple[int, int, np.ndarray]],
    noise_sigma: float,
    huber_loss_scale: float,
    max_iterations: int,
) -> Tuple[Values, Set[int], Set[int]]:
    """Build bearing measurements and solve via TranslationRecovery with BATA.

    Uses GTSAM's native BilinearAngleTranslationFactor (C++) which implements
    the exact BATA residual: scale * (Tb - Ta) - measurement. This is the same
    formulation as GLOMAP's BATAPairwiseDirectionCostFunctor.

    TranslationRecovery handles internally:
        - BilinearAngleTranslationFactor graph construction
        - Per-observation scale variables initialized to 1.0
        - Gauge prior (first key pinned to origin)
        - Scale prior on first edge
        - Random initialization for all Point3 variables
        - LM optimization with Schur complement

    Returns: (result_values, camera_indices, track_indices)
    """
    measurements, camera_indices, track_indices = _build_measurements(
        observations, noise_sigma, huber_loss_scale
    )

    logger.info(
        "GlobalPositioner: built %d measurements (%d cameras, %d landmarks)",
        len(observations), len(camera_indices), len(track_indices),
    )

    # Use TranslationRecovery with bilinear BATA factor (C++ native).
    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(max_iterations)
    lm_params.setRelativeErrorTol(1e-5)
    lm_params.setVerbosityLM("SUMMARY")

    algorithm = TranslationRecovery(lm_params, use_bilinear_translation_factor=True)
    result_values = algorithm.run(measurements, 1.0)

    return result_values, camera_indices, track_indices


def _build_measurements(
    observations: List[Tuple[int, int, np.ndarray]],
    noise_sigma: float,
    huber_loss_scale: float,
) -> Tuple[BinaryMeasurementsUnit3, Set[int], Set[int]]:
    """Build BinaryMeasurementsUnit3 from bearing observations (shared by both solve functions)."""
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, noise_sigma)
    huber = gtsam.noiseModel.mEstimator.Huber.Create(huber_loss_scale)
    robust_noise = gtsam.noiseModel.Robust.Create(huber, noise_model)

    measurements = BinaryMeasurementsUnit3()
    camera_indices: Set[int] = set()
    track_indices: Set[int] = set()

    for track_idx, cam_idx, bearing in observations:
        measurements.append(BinaryMeasurementUnit3(
            C(cam_idx), L(track_idx), Unit3(Point3(*bearing)), robust_noise,
        ))
        camera_indices.add(cam_idx)
        track_indices.add(track_idx)

    return measurements, camera_indices, track_indices


def build_and_solve_with_trace(
    observations: List[Tuple[int, int, np.ndarray]],
    noise_sigma: float,
    huber_loss_scale: float,
    max_iterations: int,
) -> Tuple[Values, Set[int], Set[int], List[Values], List[float]]:
    """Build graph manually and solve with per-iteration value capture.

    Uses TranslationRecovery.buildGraph() + addPrior() to get the native C++
    factor graph, then iterates manually with LevenbergMarquardtOptimizer to
    capture optimizer.values() at each step for convergence visualization.

    Returns: (result_values, camera_indices, track_indices, values_trace, error_trace)
    """
    measurements, camera_indices, track_indices = _build_measurements(
        observations, noise_sigma, huber_loss_scale
    )

    logger.info(
        "GlobalPositioner (trace): built %d measurements (%d cameras, %d landmarks)",
        len(observations), len(camera_indices), len(track_indices),
    )

    # Build graph via TranslationRecovery (uses native C++ BilinearAngleTranslationFactor)
    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(max_iterations)
    lm_params.setRelativeErrorTol(1e-5)
    lm_params.setVerbosityLM("SILENT")
    tr = TranslationRecovery(lm_params, use_bilinear_translation_factor=True)

    graph = tr.buildGraph(measurements)
    tr.addPrior(measurements, 1.0, gtsam.BinaryMeasurementsPoint3(), graph)

    # Initialize randomly (matching TranslationRecovery C++ code)
    rng = np.random.RandomState(42)
    initial = Values()
    for cam_idx in camera_indices:
        initial.insert(C(cam_idx), Point3(*rng.uniform(-1, 1, 3)))
    for track_idx in track_indices:
        initial.insert(L(track_idx), Point3(*rng.uniform(-1, 1, 3)))
    for i in range(len(observations)):
        initial.insert(Symbol('S', i).key(), np.array([[1.0]]))

    initial_error = graph.error(initial)
    logger.info("GlobalPositioner (trace): initial error = %.2f", initial_error)

    # Iterate manually, capturing values at each step
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm_params)
    values_trace = [initial]
    error_trace = [initial_error]

    for iteration in range(1, max_iterations + 1):
        t0 = time.time()
        optimizer.iterate()
        dt = time.time() - t0
        error = optimizer.error()
        values_trace.append(optimizer.values())
        error_trace.append(error)

        if iteration <= 5 or iteration % 5 == 0:
            logger.info("  iter %3d: error=%.2f  (%.2fs)", iteration, error, dt)

    result = optimizer.values()
    logger.info("GlobalPositioner (trace): final error = %.4f (%d frames captured)", error_trace[-1], len(values_trace))

    return result, camera_indices, track_indices, values_trace, error_trace


# ──────────────────────────────────────────────────────────────────────────────
# GtsfmData construction from optimization results
# ──────────────────────────────────────────────────────────────────────────────


def build_gtsfm_data(
    result_values: Values,
    num_images: int,
    wRi_list: List[Optional[Rot3]],
    intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
    tracks_2d: List[SfmTrack2d],
    camera_indices: Set[int],
    track_indices: Set[int],
    valid_cameras: Set[int],
    max_reproj_error: float = DEFAULT_MAX_REPROJ_ERROR,
) -> Tuple[GtsfmData, int, int]:
    """Build GtsfmData from the optimization results.

    Extracts camera poses and 3D point positions from the optimized Values,
    constructs camera objects (pose + intrinsics), builds 3D tracks (landmark
    + 2D measurements), and filters tracks by reprojection error.

    This replaces what data association / triangulation would normally do —
    the 3D points come directly from the global positioner rather than from
    separate triangulation.
    """
    gtsfm_data = GtsfmData(number_images=num_images)

    # ── Build cameras: Rot3 (from RA) + Point3 (from positioner) + intrinsics ──
    camera_class = gtsfm_types.get_camera_class_for_calibration(intrinsics[0])

    cameras_added = 0
    for cam_idx in camera_indices:
        wRi = wRi_list[cam_idx]
        Ki = intrinsics[cam_idx]
        if wRi is None or Ki is None:
            continue
        try:
            wti = result_values.atPoint3(C(cam_idx))
            pose = Pose3(wRi, wti)
            camera = camera_class(pose, Ki)
            gtsfm_data.add_camera(cam_idx, camera)
            cameras_added += 1
        except RuntimeError:
            logger.warning("GlobalPositioner: failed to build camera %d.", cam_idx)

    # ── Build 3D tracks: Point3 (from positioner) + 2D measurements (from input) ──
    num_tracks_before = 0
    num_tracks_after = 0
    estimated_camera_set = set(gtsfm_data.get_valid_camera_indices())

    for track_idx in sorted(track_indices):
        try:
            point_3d = result_values.atPoint3(L(track_idx))
        except RuntimeError:
            continue

        if track_idx >= len(tracks_2d):
            continue
        track_2d = tracks_2d[track_idx]

        # Build the 3D SfmTrack with measurements from estimated cameras only.
        sfm_track = SfmTrack(point_3d)
        num_valid_measurements = 0
        for m_idx in range(track_2d.number_measurements()):
            m = track_2d.measurement(m_idx)
            cam_idx, uv = m.i, m.uv
            if cam_idx in estimated_camera_set:
                sfm_track.addMeasurement(cam_idx, uv)
                num_valid_measurements += 1

        if num_valid_measurements < 2:
            continue

        num_tracks_before += 1

        # Filter by reprojection error.
        keep_track = True
        for m_idx in range(sfm_track.numberMeasurements()):
            cam_idx_m, uv_m = sfm_track.measurement(m_idx)
            camera_m = gtsfm_data.get_camera(cam_idx_m)
            if camera_m is None:
                continue
            try:
                projected = camera_m.project(point_3d)
                error = np.sqrt((projected[0] - uv_m[0])**2 + (projected[1] - uv_m[1])**2)
                if error > max_reproj_error:
                    keep_track = False
                    break
            except RuntimeError:
                keep_track = False
                break

        if keep_track:
            gtsfm_data.add_track(sfm_track)
            num_tracks_after += 1

    logger.info(
        "GlobalPositioner: built GtsfmData with %d cameras, %d/%d tracks (%.1f%% kept after reproj filter).",
        cameras_added,
        num_tracks_after,
        num_tracks_before,
        100 * num_tracks_after / max(num_tracks_before, 1),
    )

    return gtsfm_data, num_tracks_before, num_tracks_after


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────


class GlobalPositioner:
    """Joint camera + point position estimation using BATA bearing constraints.

    Replaces BOTH translation averaging AND data association (triangulation)
    in the GTSFM pipeline. Takes rotations + tracks + intrinsics and outputs
    GtsfmData ready for bundle adjustment.

    Uses GTSAM's native BilinearAngleTranslationFactor (C++) via
    TranslationRecovery, giving full C++ speed with the exact BATA
    formulation from GLOMAP.

    Pipeline integration:
        BEFORE:  rot_avg → trans_avg → data_assoc → BA
        AFTER:   rot_avg → GlobalPositioner.run() → BA
    """

    def __init__(
        self,
        noise_sigma: float = DEFAULT_BEARING_NOISE_SIGMA,
        huber_loss_scale: float = DEFAULT_HUBER_LOSS_SCALE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        min_track_measurements: int = MIN_TRACK_MEASUREMENTS,
        max_reproj_error: float = DEFAULT_MAX_REPROJ_ERROR,
        max_tracks: int = DEFAULT_MAX_TRACKS,
        save_iteration_visualization: bool = False,
    ) -> None:
        self._noise_sigma = noise_sigma
        self._huber_loss_scale = huber_loss_scale
        self._max_iterations = max_iterations
        self._min_track_measurements = min_track_measurements
        self._max_reproj_error = max_reproj_error
        self._max_tracks = max_tracks
        self._save_iteration_visualization = save_iteration_visualization

    def run(
        self,
        num_images: int,
        wRi_list: List[Optional[Rot3]],
        tracks_2d: List[SfmTrack2d],
        intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
    ) -> Tuple[GtsfmData, GtsfmMetricsGroup]:
        """Run the global positioner and produce GtsfmData for BA.

        Args:
            num_images: Total number of images.
            wRi_list: Global rotations from rotation averaging.
            tracks_2d: 2D feature tracks across images.
            intrinsics: Camera calibration per image.

        Returns:
            gtsfm_data: GtsfmData with cameras + 3D tracks (empty on failure).
            metrics: GtsfmMetricsGroup with positioning stats.
        """
        start_time = time.time()

        # ── Input validation ──
        if not tracks_2d:
            logger.error("GlobalPositioner: no 2D tracks provided.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])
        if intrinsics is None:
            logger.error("GlobalPositioner: no intrinsics provided.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        valid_cameras = {i for i, wRi in enumerate(wRi_list) if wRi is not None}
        if len(valid_cameras) < 2:
            logger.error("GlobalPositioner: need >= 2 cameras with rotations.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        logger.info("GlobalPositioner: %d valid cameras, %d input tracks.",
                     len(valid_cameras), len(tracks_2d))

        # ── Filter tracks ──
        filtered_tracks = filter_tracks(tracks_2d, valid_cameras, self._min_track_measurements, self._max_tracks)
        logger.info("GlobalPositioner: %d tracks after filtering.", len(filtered_tracks))
        if not filtered_tracks:
            logger.error("GlobalPositioner: no tracks survived filtering.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        # ── Compute bearings ──
        t0 = time.time()
        observations = compute_world_bearings(filtered_tracks, intrinsics, wRi_list, valid_cameras)
        dt_bearing = time.time() - t0
        logger.info("GlobalPositioner: %d bearing observations (%.2fs).", len(observations), dt_bearing)

        if not observations:
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        # ── Build factor graph and solve ──
        t0 = time.time()
        if self._save_iteration_visualization:
            result_values, cam_indices, track_indices, values_trace, error_trace = build_and_solve_with_trace(
                observations, self._noise_sigma, self._huber_loss_scale, self._max_iterations,
            )
            # Save trace to disk for visualization script
            trace_path = "results/gp_convergence_trace.pkl"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            trace_data = {
                "values_trace": values_trace,
                "error_trace": error_trace,
                "camera_indices": cam_indices,
                "track_indices": track_indices,
                "wRi_list": [wRi_list[i] for i in sorted(cam_indices)],
                "wRi_indices": sorted(cam_indices),
                "filtered_tracks": filtered_tracks,
            }
            with open(trace_path, "wb") as f:
                pickle.dump(trace_data, f)
            logger.info("GlobalPositioner: saved convergence trace to %s (%d frames)", trace_path, len(values_trace))
        else:
            result_values, cam_indices, track_indices = build_and_solve(
                observations, self._noise_sigma, self._huber_loss_scale, self._max_iterations,
            )
        dt_solve = time.time() - t0
        logger.info("GlobalPositioner: optimization took %.2fs.", dt_solve)

        # ── Build GtsfmData from results ──
        t0 = time.time()
        gtsfm_data, tracks_before, tracks_after = build_gtsfm_data(
            result_values=result_values,
            num_images=num_images,
            wRi_list=wRi_list,
            intrinsics=intrinsics,
            tracks_2d=filtered_tracks,
            camera_indices=cam_indices,
            track_indices=track_indices,
            valid_cameras=valid_cameras,
            max_reproj_error=self._max_reproj_error,
        )
        dt_build = time.time() - t0

        total_duration = time.time() - start_time

        # ── Metrics ──
        metrics = GtsfmMetricsGroup("global_positioning_metrics", [
            GtsfmMetric("num_valid_cameras", len(valid_cameras)),
            GtsfmMetric("num_tracks_used", len(filtered_tracks)),
            GtsfmMetric("num_bearing_observations", len(observations)),
            GtsfmMetric("num_camera_variables", len(cam_indices)),
            GtsfmMetric("num_landmark_variables", len(track_indices)),
            GtsfmMetric("num_cameras_in_output", len(gtsfm_data.get_valid_camera_indices()) if gtsfm_data else 0),
            GtsfmMetric("num_tracks_before_filter", tracks_before),
            GtsfmMetric("num_tracks_after_filter", tracks_after),
            GtsfmMetric("bearing_computation_sec", dt_bearing),
            GtsfmMetric("optimization_sec", dt_solve),
            GtsfmMetric("gtsfm_data_build_sec", dt_build),
            GtsfmMetric("total_duration_sec", total_duration),
        ])

        logger.info("GlobalPositioner: done in %.2fs — %d cameras, %d tracks.",
                     total_duration,
                     len(gtsfm_data.get_valid_camera_indices()) if gtsfm_data else 0,
                     gtsfm_data.number_tracks() if gtsfm_data else 0)

        return gtsfm_data, metrics
