"""Utilities for 2D and 3D tracks.

Authors: Ayush Baid, Travis Driver
"""

import itertools
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from gtsam import PinholeCameraCal3Bundler, Point3, Rot3, SfmTrack  # type: ignore

import gtsfm.common.types as gtsfm_types
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationExitCode,
    TriangulationOptions,
    TriangulationSamplingMode,
)
from gtsfm.densify.mvs_utils import calculate_triangulation_angle_in_degrees


def classify_tracks2d_with_gt_cameras(
    tracks: List[SfmTrack2d], cameras_gt: List[PinholeCameraCal3Bundler], reproj_error_thresh_px: float = 3
) -> List[TriangulationExitCode]:
    """Classifies the 2D tracks w.r.t ground truth cameras by performing triangulation and collecting exit codes.

    Args:
        tracks: list of 2d tracks.
        cameras_gt: cameras with GT params.
        reproj_error_thresh_px (optional): Reprojection error threshold (in pixels) for a track to be considered an
                                           all-inlier one. Defaults to 3.

    Returns:
        The triangulation exit code for each input track, as list of the same length as of tracks.
    """
    # do a simple triangulation with the GT cameras
    cameras_dict: Dict[int, PinholeCameraCal3Bundler] = {i: cam for i, cam in enumerate(cameras_gt)}
    point3d_initializer = Point3dInitializer(
        track_camera_dict=cameras_dict,
        options=TriangulationOptions(
            reproj_error_threshold=reproj_error_thresh_px, mode=TriangulationSamplingMode.NO_RANSAC
        ),
    )

    exit_codes: List[TriangulationExitCode] = []
    for track in tracks:
        _, _, triangulation_exit_code = point3d_initializer.triangulate(track_2d=track)
        exit_codes.append(triangulation_exit_code)

    return exit_codes


def classify_tracks3d_with_gt_cameras(
    tracks: List[SfmTrack], cameras_gt: List[PinholeCameraCal3Bundler], reproj_error_thresh_px: float = 3
) -> List[TriangulationExitCode]:
    """Classifies the 3D tracks w.r.t ground truth cameras by performing triangulation and collecting exit codes.

    Args:
        tracks: list of 3d tracks, of length J.
        cameras_gt: cameras with GT params.
        reproj_error_thresh_px (optional): Reprojection error threshold (in pixels) for a track to be considered an
                                           all-inlier one. Defaults to 3.

    Returns:
        The triangulation exit code for each input track, as list of length J (same as input).
    """
    # convert the 3D tracks to 2D tracks
    tracks_2d: List[SfmTrack2d] = []
    for track_3d in tracks:
        num_measurements = track_3d.numberMeasurements()

        measurements: List[SfmMeasurement] = []
        for k in range(num_measurements):
            i, uv = track_3d.measurement(k)

            measurements.append(SfmMeasurement(i, uv))

        tracks_2d.append(SfmTrack2d(measurements))

    return classify_tracks2d_with_gt_cameras(tracks_2d, cameras_gt, reproj_error_thresh_px)


def filter_tracks_by_measurements(
    tracks_2d: List[SfmTrack2d],
    valid_cameras: Set[int],
    min_measurements: int = 3,
    max_tracks: int = 0,
) -> List[SfmTrack2d]:
    """Filter 2D tracks by minimum count of measurements from valid cameras.

    Tracks are sorted longest-first; if `max_tracks > 0` only the top-K survive.
    Used by global positioning to drop short / orphan tracks before solving.
    """
    scored: List[Tuple[int, SfmTrack2d]] = []
    for track in tracks_2d:
        valid_count = sum(
            1
            for m_idx in range(track.number_measurements())
            if track.measurement(m_idx).i in valid_cameras
        )
        if valid_count >= min_measurements:
            scored.append((valid_count, track))
    scored.sort(key=lambda x: x[0], reverse=True)
    if max_tracks > 0 and len(scored) > max_tracks:
        scored = scored[:max_tracks]
    return [track for _, track in scored]


def compute_world_directions(
    tracks_2d: List[SfmTrack2d],
    intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
    wRi_list: List[Optional[Rot3]],
    valid_cameras: Optional[Set[int]] = None,
) -> List[Tuple[int, int, np.ndarray]]:
    """Compute world-frame unit directions from each camera to each observed track point.

    For every (track, measurement) pair, the function:
      1. Calibrates the pixel observation via the camera's intrinsics.
      2. Lifts the calibrated point to a unit ray in the camera frame.
      3. Rotates the ray into the world frame using `wRi`.

    Used by the global positioner (BATA factor graph). The 1DSfM translation-averaging
    module computes the same quantity in averaging_1dsfm.py:_get_landmark_directions —
    TODO(akshay-krishnan): unify those two implementations behind this util.

    Args:
        tracks_2d: 2D tracks across images.
        intrinsics: Per-camera calibration; entries may be None for invalid cameras.
        wRi_list: Per-camera rotations to world frame; entries may be None.
        valid_cameras: If given, only measurements from these cameras are returned.
            Measurements from cameras with None rotation/intrinsics are always skipped.

    Returns:
        List of `(track_idx, cam_idx, world_direction)` triplets where `world_direction`
        is a unit-norm `np.ndarray` of shape (3,).
    """
    observations: List[Tuple[int, int, np.ndarray]] = []
    for track_idx, track in enumerate(tracks_2d):
        for m_idx in range(track.number_measurements()):
            m = track.measurement(m_idx)
            cam_idx, uv = m.i, m.uv
            if valid_cameras is not None and cam_idx not in valid_cameras:
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
            world_dir = wRi.rotate(Point3(*camera_ray))
            wd = np.array([world_dir[0], world_dir[1], world_dir[2]])
            norm = np.linalg.norm(wd)
            if norm < 1e-12:
                continue
            observations.append((track_idx, cam_idx, wd / norm))
    return observations


def get_max_triangulation_angle(track3d: SfmTrack, cameras: Dict[int, gtsfm_types.CAMERA_TYPE]) -> float:
    """Get the angle (in degrees) subtended by the cameras at the 3D landmark of the track.

    Args:
        track3d: the track with the landmark.
        cameras: cameras which have been used to triangulate the landmark.

    Returns:
        The maximum triangulation angle over all pairs of cameras associated with the track.
    """
    assert cameras is not None
    camera_ind: List[int] = []
    for k in range(track3d.numberMeasurements()):
        i, _ = track3d.measurement(k)
        camera_ind.append(i)

    angles: List[float] = []
    for i1, i2 in itertools.combinations(camera_ind, 2):
        angles.append(calculate_triangulation_angle_in_degrees(cameras[i1], cameras[i2], track3d.point3()))

    return max(angles)
