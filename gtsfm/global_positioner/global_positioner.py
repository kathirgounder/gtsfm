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

Implementation follows the GLOMAP C++ source as closely as GTSAM allows:
    - Bilinear BATA residual: e = v_ik - d_ik * (X_k - c_i)
    - Per-observation scale variable d_ik with cheirality constraint
    - Schur complement ordering: scales → points → cameras
    - Huber robust loss for outlier tolerance
    - Random initialization with reliable convergence

References:
    Pan, L., Barath, D., Pollefeys, M., Schonberger, J.L.
    "Global Structure-from-Motion Revisited." ECCV 2024.

Authors: <your name>
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import gtsam
import numpy as np
from gtsam import (
    NonlinearFactorGraph,
    Ordering,
    Point3,
    Pose3,
    Rot3,
    SfmTrack,
    Unit3,
    Values,
)
from gtsam.noiseModel import Isotropic, Robust, mEstimator
from gtsam.symbol_shorthand import A as C  # Camera position variables (Point3)
from gtsam.symbol_shorthand import B as L  # Landmark position variables (Point3)
from gtsam.symbol_shorthand import D as S  # Scale variables (1D per observation)

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

logger = logger_utils.get_logger()


# ──────────────────────────────────────────────────────────────────────────────
# Constants (matching GLOMAP defaults)
# ──────────────────────────────────────────────────────────────────────────────

MIN_TRACK_MEASUREMENTS = 3          # GLOMAP: min_num_view_per_track = 3
DEFAULT_HUBER_LOSS_SCALE = 0.1      # GLOMAP: loss_function_scale = 0.1
DEFAULT_BEARING_NOISE_SIGMA = 0.01
RANDOM_INIT_RANGE = 100.0           # GLOMAP: 100.0 * RandVector3d(-1, 1)
MIN_SCALE = 1e-5                    # GLOMAP: SetParameterLowerBound(&scale, 0, 1e-5)
GAUGE_PRIOR_SIGMA = 1.0
SCALE_PENALTY_WEIGHT = 100.0
DEFAULT_MAX_ITERATIONS = 100        # GLOMAP: max_num_iterations = 100
DEFAULT_MAX_REPROJ_ERROR = 5.0      # Pixels — for filtering output tracks


# ──────────────────────────────────────────────────────────────────────────────
# BATA bearing factor (bilinear form)
#
# Residual: e = v_ik - d_ik * (X_k - c_i)
# Variables: C(cam), L(point), S(obs) — Point3, Point3, Vector1
# Measurement: v_ik (world-frame unit bearing, constant)
# ──────────────────────────────────────────────────────────────────────────────


def make_bata_bearing_factor(camera_key, point_key, scale_key, measured_bearing, noise_model):
    """Create a BATA bearing factor (bilinear form).

    Mirrors GLOMAP's BATAPairwiseDirectionCostFunctor.
    Residual: e = v - d*(X - c)
    Variables: camera position (Point3), landmark (Point3), scale (Vector1)

    Analytical Jacobians:
        de/dc  =  d * I_{3x3}
        de/dX  = -d * I_{3x3}
        de/dd  = -(X - c)   (3x1)
    """
    v = np.asarray(measured_bearing, dtype=np.float64).copy()

    def error_func(this, values, H):
        c = np.array(values.atPoint3(this.keys()[0]))
        X = np.array(values.atPoint3(this.keys()[1]))
        raw_d = float(values.atVector(this.keys()[2])[0])
        d = max(raw_d, MIN_SCALE)

        diff = X - c
        error = v - d * diff

        if H is not None:
            H[0] = d * np.eye(3)
            H[1] = -d * np.eye(3)
            H[2] = -diff.reshape(3, 1)
            if raw_d < MIN_SCALE:
                H[2] = np.zeros((3, 1))

        return error

    return gtsam.CustomFactor(noise_model, [camera_key, point_key, scale_key], error_func)


def make_scale_lower_bound_factor(scale_key, noise_model):
    """Soft cheirality: penalizes d < MIN_SCALE (GTSAM has no native bounds)."""

    def error_func(this, values, H):
        d = float(values.atVector(this.keys()[0])[0])
        violation = max(0.0, MIN_SCALE - d)
        if H is not None:
            H[0] = np.array([[-SCALE_PENALTY_WEIGHT]]) if d < MIN_SCALE else np.array([[0.0]])
        return np.array([violation * SCALE_PENALTY_WEIGHT])

    return gtsam.CustomFactor(noise_model, [scale_key], error_func)


def make_scale_prior_factor(scale_key, target, noise_model):
    """Prior on a 1D scale variable (gauge fixing)."""

    def error_func(this, values, H):
        d = float(values.atVector(this.keys()[0])[0])
        if H is not None:
            H[0] = np.array([[1.0]])
        return np.array([d - target])

    return gtsam.CustomFactor(noise_model, [scale_key], error_func)


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

    Returns list of (track_idx, cam_idx, unit_bearing) tuples.
    Mirrors GLOMAP's AddPoint3DToProblem() bearing computation.
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


def filter_tracks(tracks_2d, valid_cameras, min_measurements=MIN_TRACK_MEASUREMENTS):
    """Keep tracks with >= min_measurements observations from valid cameras."""
    filtered = []
    for track in tracks_2d:
        valid_count = sum(
            1 for m_idx in range(track.number_measurements())
            if track.measurement(m_idx).i in valid_cameras
        )
        if valid_count >= min_measurements:
            filtered.append(track)
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Factor graph construction + optimization
# ──────────────────────────────────────────────────────────────────────────────


def build_and_solve(
    observations: List[Tuple[int, int, np.ndarray]],
    noise_sigma: float,
    huber_loss_scale: float,
    max_iterations: int,
) -> Tuple[Values, NonlinearFactorGraph, Set[int], Set[int], int, float, float]:
    """Build the BATA factor graph and run LM optimization.

    Returns: (result_values, graph, camera_indices, track_indices, num_scales)
    """
    graph = NonlinearFactorGraph()
    initial_values = Values()

    base_noise = Isotropic.Sigma(3, noise_sigma)
    robust_noise = Robust(mEstimator.Huber(huber_loss_scale), base_noise)
    scale_penalty_noise = Isotropic.Sigma(1, 1.0)

    rng = np.random.RandomState(42)
    camera_indices, track_indices = set(), set()

    # One BATA factor + one scale penalty per observation.
    obs_idx = 0
    for track_idx, cam_idx, bearing in observations:
        graph.push_back(make_bata_bearing_factor(C(cam_idx), L(track_idx), S(obs_idx), bearing, robust_noise))
        graph.push_back(make_scale_lower_bound_factor(S(obs_idx), scale_penalty_noise))
        if not initial_values.exists(S(obs_idx)):
            initial_values.insert(S(obs_idx), np.array([1.0]))
        camera_indices.add(cam_idx)
        track_indices.add(track_idx)
        obs_idx += 1

    num_scales = obs_idx

    # Random init for cameras and points (GLOMAP: 100 * rand(-1,1)).
    for cam_idx in camera_indices:
        if not initial_values.exists(C(cam_idx)):
            initial_values.insert(C(cam_idx), Point3(*(RANDOM_INIT_RANGE * rng.uniform(-1, 1, size=3))))
    for track_idx in track_indices:
        if not initial_values.exists(L(track_idx)):
            initial_values.insert(L(track_idx), Point3(*(RANDOM_INIT_RANGE * rng.uniform(-1, 1, size=3))))

    # Gauge fixing: pin first camera + first scale.
    gauge_3d = Isotropic.Sigma(3, GAUGE_PRIOR_SIGMA)
    gauge_1d = Isotropic.Sigma(1, GAUGE_PRIOR_SIGMA)
    sorted_cams = sorted(camera_indices)
    if sorted_cams:
        graph.push_back(gtsam.PriorFactorPoint3(C(sorted_cams[0]), Point3(0, 0, 0), gauge_3d))
    if num_scales > 0:
        graph.push_back(make_scale_prior_factor(S(0), 1.0, gauge_1d))

    # Schur ordering: scales → points → cameras.
    ordering = Ordering()
    for i in range(num_scales):
        ordering.push_back(S(i))
    for t in sorted(track_indices):
        ordering.push_back(L(t))
    for c in sorted(camera_indices):
        ordering.push_back(C(c))

    # Solve.
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setRelativeErrorTol(1e-5)
    params.setVerbosityLM("ERROR")
    params.setOrdering(ordering)

    initial_error = graph.error(initial_values)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()
    final_error = graph.error(result)

    logger.info("GlobalPositioner: error %.4f -> %.4f (%.1fx reduction)",
                initial_error, final_error, initial_error / max(final_error, 1e-10))

    return result, graph, camera_indices, track_indices, num_scales, initial_error, final_error


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

    Args:
        result_values: Optimized GTSAM Values from build_and_solve().
        num_images: Total number of images in the scene.
        wRi_list: Global rotations from rotation averaging.
        intrinsics: Camera calibration per image.
        tracks_2d: Input 2D tracks (provides the 2D measurements for each 3D track).
        camera_indices: Camera indices present in the optimization.
        track_indices: Track indices present in the optimization.
        valid_cameras: Set of cameras with valid rotations.
        max_reproj_error: Maximum reprojection error (pixels) to keep a track.

    Returns:
        gtsfm_data: Complete GtsfmData ready for bundle adjustment.
        num_tracks_before_filter: Track count before reprojection error filtering.
        num_tracks_after_filter: Track count after filtering.
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
    #
    # For each track in the optimization, we:
    #   1. Get the estimated 3D point position from result_values
    #   2. Get the 2D measurements from the input track
    #   3. Filter measurements to only include valid cameras that were estimated
    #   4. Create an SfmTrack (3D) with the point + measurements
    num_tracks_before = 0
    num_tracks_after = 0
    estimated_camera_set = set(gtsfm_data.get_valid_camera_indices())

    for track_idx in sorted(track_indices):
        try:
            point_3d = result_values.atPoint3(L(track_idx))
        except RuntimeError:
            continue

        # Get the 2D measurements from the corresponding input track.
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

        # Filter by reprojection error: check that the 3D point projects
        # close to the 2D observations in each camera.
        max_error_this_track = 0.0
        keep_track = True
        for m_idx in range(sfm_track.numberMeasurements()):
            cam_idx_m, uv_m = sfm_track.measurement(m_idx)
            camera_m = gtsfm_data.get_camera(cam_idx_m)
            if camera_m is None:
                continue
            try:
                projected = camera_m.project(point_3d)
                error = np.sqrt((projected[0] - uv_m[0])**2 + (projected[1] - uv_m[1])**2)
                max_error_this_track = max(max_error_this_track, error)
                if error > max_reproj_error:
                    keep_track = False
                    break
            except RuntimeError:
                # Projection behind camera — skip this track.
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
    ) -> None:
        self._noise_sigma = noise_sigma
        self._huber_loss_scale = huber_loss_scale
        self._max_iterations = max_iterations
        self._min_track_measurements = min_track_measurements
        self._max_reproj_error = max_reproj_error

    def run(
        self,
        num_images: int,
        wRi_list: List[Optional[Rot3]],
        tracks_2d: List[SfmTrack2d],
        intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
    ) -> Tuple[Optional[GtsfmData], GtsfmMetricsGroup]:
        """Run the global positioner and produce GtsfmData for BA.

        Args:
            num_images: Total number of images.
            wRi_list: Global rotations from rotation averaging.
            tracks_2d: 2D feature tracks across images.
            intrinsics: Camera calibration per image.

        Returns:
            gtsfm_data: GtsfmData with cameras + 3D tracks, or None on failure.
            metrics: GtsfmMetricsGroup with positioning stats.
        """
        start_time = time.time()

        # ── Input validation ──
        if not tracks_2d:
            logger.error("GlobalPositioner: no 2D tracks provided.")
            return None, GtsfmMetricsGroup("global_positioning_metrics", [])
        if intrinsics is None:
            logger.error("GlobalPositioner: no intrinsics provided.")
            return None, GtsfmMetricsGroup("global_positioning_metrics", [])

        valid_cameras = {i for i, wRi in enumerate(wRi_list) if wRi is not None}
        if len(valid_cameras) < 2:
            logger.error("GlobalPositioner: need >= 2 cameras with rotations.")
            return None, GtsfmMetricsGroup("global_positioning_metrics", [])

        logger.info("GlobalPositioner: %d valid cameras, %d input tracks.",
                     len(valid_cameras), len(tracks_2d))

        # ── Filter tracks ──
        filtered_tracks = filter_tracks(tracks_2d, valid_cameras, self._min_track_measurements)
        logger.info("GlobalPositioner: %d tracks after filtering.", len(filtered_tracks))
        if not filtered_tracks:
            logger.error("GlobalPositioner: no tracks survived filtering.")
            return None, GtsfmMetricsGroup("global_positioning_metrics", [])

        # ── Compute bearings ──
        t0 = time.time()
        observations = compute_world_bearings(filtered_tracks, intrinsics, wRi_list, valid_cameras)
        dt_bearing = time.time() - t0
        logger.info("GlobalPositioner: %d bearing observations (%.2fs).", len(observations), dt_bearing)

        if not observations:
            return None, GtsfmMetricsGroup("global_positioning_metrics", [])

        # ── Build factor graph and solve ──
        t0 = time.time()
        result_values, graph, cam_indices, track_indices, num_scales, initial_error, final_error = build_and_solve(
            observations, self._noise_sigma, self._huber_loss_scale, self._max_iterations,
        )
        dt_solve = time.time() - t0

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
            GtsfmMetric("num_scale_variables", num_scales),
            GtsfmMetric("num_cameras_in_output", len(gtsfm_data.get_valid_camera_indices()) if gtsfm_data else 0),
            GtsfmMetric("num_tracks_before_filter", tracks_before),
            GtsfmMetric("num_tracks_after_filter", tracks_after),
            GtsfmMetric("bearing_computation_sec", dt_bearing),
            GtsfmMetric("optimization_sec", dt_solve),
            GtsfmMetric("gtsfm_data_build_sec", dt_build),
            GtsfmMetric("total_duration_sec", total_duration),
            GtsfmMetric("initial_graph_error", initial_error),
            GtsfmMetric("final_graph_error", final_error),
        ])

        logger.info("GlobalPositioner: done in %.2fs — %d cameras, %d tracks.",
                     total_duration,
                     len(gtsfm_data.get_valid_camera_indices()) if gtsfm_data else 0,
                     gtsfm_data.number_tracks() if gtsfm_data else 0)

        return gtsfm_data, metrics