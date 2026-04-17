"""Utilities for verification stage of the frontend.

Authors: Ayush Baid
"""

from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import scipy  # type: ignore
from gtsam import EssentialMatrix, Pose3, Rot3, Unit3  # type: ignore

import gtsfm.utils.features as feature_utils
from gtsfm.common.types import CALIBRATION_TYPE
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

EPS = 1e-8  # constant used to prevent division by zero error.


def decompose_camera_projection_matrix(M: np.ndarray) -> tuple[np.ndarray, Pose3]:
    """Recover a camera pose from a 3x4 projection matrix.

    Reference: https://johnwlambert.github.io/cam-calibration/#decomposition

    Args:
        M: array of shape (3,4) representing camera projection matrix.
            Let M = [m1 | m2 | m3 | m4] = [ Q | m4]

    Returns:
        K: 3x3 upper triangular camera intrinsics matrix. The matrix diagonal is
            guaranteed to contain all positive entries.
        wTc: camera pose in world frame.
    """
    m4 = M[:, 3]
    Q = M[:3, :3]
    wtc: np.ndarray = np.linalg.inv(-Q) @ m4

    K, cRw = scipy.linalg.rq(Q)

    # multiply columns of K by -1 if K(i,i) entry on diagonal is negative.
    # T is a matrix that scales by -1 or 1
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T

    # scale rows of cRw by T, to keep a valid decomposition. T is its own inverse.
    wRc = Rot3(T @ cRw).inverse()
    wTc = Pose3(wRc, wtc)

    return K, wTc


def recover_relative_pose_from_essential_matrix(
    i2Ei1: Optional[np.ndarray],
    verified_coordinates_i1: np.ndarray,
    verified_coordinates_i2: np.ndarray,
    camera_intrinsics_i1: CALIBRATION_TYPE,
    camera_intrinsics_i2: CALIBRATION_TYPE,
) -> Tuple[Optional[Rot3], Optional[Unit3]]:
    """Recovers the relative rotation and translation direction from essential matrix and verified correspondences
    using opencv's API.

    Args:
        i2Ei1: essential matrix as a numpy array, of shape 3x3.
        verified_coordinates_i1: coordinates of verified correspondences in image i1, of shape Nx2.
        verified_coordinates_i2: coordinates of verified correspondences in image i2, of shape Nx2.
        camera_intrinsics_i1: intrinsics for image i1.
        camera_intrinsics_i2: intrinsics for image i2.

    Returns:
        relative rotation i2Ri1, or None if the input essential matrix is None.
        relative translation direction i2Ui1, or None if the input essential matrix is None.
    """
    if i2Ei1 is None:
        return None, None

    # obtain points in normalized coordinates using intrinsics.
    normalized_coordinates_i1 = feature_utils.normalize_coordinates(verified_coordinates_i1, camera_intrinsics_i1)
    normalized_coordinates_i2 = feature_utils.normalize_coordinates(verified_coordinates_i2, camera_intrinsics_i2)

    # use opencv to recover pose
    _, i2Ri1, i2ti1, _ = cv.recoverPose(i2Ei1, normalized_coordinates_i1, normalized_coordinates_i2)
    i2Ri1 = Rot3(i2Ri1)
    i2Ui1 = Unit3(i2ti1.squeeze())
    i2Ei1_reconstructed = EssentialMatrix(i2Ri1, i2Ui1).matrix()

    # normalizing the two essential matrices
    i2Ei1_normalized = i2Ei1 / np.linalg.norm(i2Ei1, axis=None)
    i2Ei1_reconstructed_normalized = i2Ei1_reconstructed / np.linalg.norm(i2Ei1_reconstructed, axis=None)

    # TODO(ayushbaid): we frequently fail to recover E from the estimated R, t. This needs more investigation.
    if not np.allclose(i2Ei1_normalized, i2Ei1_reconstructed_normalized):
        logger.debug("Recovered R, t cannot create the input Essential Matrix")

    return i2Ri1, i2Ui1


def fundamental_to_essential_matrix(
    i2Fi1: np.ndarray, camera_intrinsics_i1: CALIBRATION_TYPE, camera_intrinsics_i2: CALIBRATION_TYPE
) -> np.ndarray:
    """Converts the fundamental matrix to essential matrix using camera intrinsics.

    Args:
        i2Fi1: fundamental matrix which maps points in image #i1 to lines in image #i2.
        camera_intrinsics_i1: intrinsics for image #i1.
        camera_intrinsics_i2: intrinsics for image #i2.

    Returns:
        Estimated essential matrix i2Ei1 as numpy array of shape (3x3).
    """
    return camera_intrinsics_i2.K().T @ i2Fi1 @ camera_intrinsics_i1.K()


def essential_to_fundamental_matrix(
    i2Ei1: EssentialMatrix, camera_intrinsics_i1: CALIBRATION_TYPE, camera_intrinsics_i2: CALIBRATION_TYPE
) -> np.ndarray:
    """Converts the essential matrix to fundamental matrix using camera intrinsics.

    Args:
        i2Ei1: essential matrix which maps points in image #i1 to lines in image #i2.
        camera_intrinsics_i1: intrinsics for image #i1.
        camera_intrinsics_i2: intrinsics for image #i2.

    Returns:
        Fundamental matrix i2Fi1 as numpy array of shape (3x3).
    """
    return np.linalg.inv(camera_intrinsics_i2.K().T) @ i2Ei1.matrix() @ np.linalg.inv(camera_intrinsics_i1.K())


def compute_epipolar_distances_sq_sed(
    coordinates_i1: np.ndarray, coordinates_i2: np.ndarray, i2Fi1: np.ndarray
) -> Optional[np.ndarray]:
    """Compute symmetric point-line epipolar squared distance between corresponding coordinates in two images. The
    squared SED is the geometric point-line distance and is an over-estimate of the gold-standard reprojection error.
    The over-estimate can reject correspondences which are actually inliers.

    References:
    1 "Fathy et al., Fundamental Matrix Estimation: A Study of Error Criteria", https://arxiv.org/abs/1706.07886
    2 "Hartley, R.~I. et al. Multiple View Geometry in Computer Vision.. Cambridge University Press, Pg 288"

    Algorithm:
    - l2 = i2Fi1 @ x1
    - l1 = x2.T @ i2Fi1 = [a1, b1, c1]
    - n1 = EuclideanNorm(Normal(l1)) = sqrt(a1^2 + b1^2)
    - n2 = EuclideanNorm(Normal(l2))
    - SED^2 = ( l1 @ x1 )^2 * (1/n1^2 + 1/n2^2)

    Args:
        coordinates_i1: coordinates in image i1, of shape Nx2.
        coordinates_i2: corr. coordinates in image i2, of shape Nx2.
        i2Fi1: fundamental matrix between two images.

    Returns:
        Symmetric epipolar point-line distances for each row of the input, of shape N.
    """

    if coordinates_i1 is None or coordinates_i1.size == 0 or coordinates_i2 is None or coordinates_i2.size == 0:
        return None

    epipolar_lines_i2 = feature_utils.convert_to_epipolar_lines(coordinates_i1, i2Fi1)  # Ex1
    epipolar_lines_i1 = feature_utils.convert_to_epipolar_lines(coordinates_i2, i2Fi1.T)  # Etx2

    numerator = np.square(feature_utils.point_line_dotproduct(coordinates_i1, epipolar_lines_i1))

    line_sq_norms_i1 = np.sum(np.square(epipolar_lines_i1[:, :2]), axis=1)
    line_sq_norms_i2 = np.sum(np.square(epipolar_lines_i2[:, :2]), axis=1)

    return numerator * (1 / line_sq_norms_i1 + 1 / line_sq_norms_i2)


def compute_epipolar_distances_sq_sampson(
    coordinates_i1: np.ndarray, coordinates_i2: np.ndarray, i2Fi1: np.ndarray
) -> Optional[np.ndarray]:
    """Compute the sampson distance between corresponding coordinates in two images. Sampson distance is the first
    order approximation of the reprojection error, and well-estimates the gold standard reprojection error at low
    values of the reprojection error.

    The squared Sampson distance is computed in OpenCV (Ref 3) as well COLMAP (Ref 4). OpenCV uses the squared sampson
    error to find inliers in RANSAC (Ref 5).

    References:
    1 "Fathy et al., Fundamental Matrix Estimation: A Study of Error Criteria", https://arxiv.org/abs/1706.07886
    2 "Hartley, R.~I. et al. Multiple View Geometry in Computer Vision.. Cambridge University Press, Pg 288"
    3 https://github.com/opencv/opencv/blob/29fb4f98b10767008d0751dd064023727d9d3e5d/modules/calib3d/src/five-point.cpp#L375 # noqa: E501
    4 https://github.com/colmap/colmap/blob/9f3a75ae9c72188244f2403eb085e51ecf4397a8/src/estimators/utils.cc#L87
    5 https://github.com/opencv/opencv/blob/29fb4f98b10767008d0751dd064023727d9d3e5d/modules/calib3d/src/ptsetreg.cpp#L85 # noqa: E731

    Algorithm:
    - l2 = i2Fi1 @ x1
    - l1 = x2.T @ i2Fi1 = [a1, b1, c2]
    - n1 = EuclideanNorm(Normal(l1)) = sqrt(a1^2 + b1^2)
    - n2 = EuclideanNorm(Normal(l2))
    - Sampson^2 = ( l1 @ x1 )^2 / (n1^2 + n2^2)

    Args:
        coordinates_i1: coordinates in image i1, of shape Nx2.
        coordinates_i2: corr. coordinates in image i2, of shape Nx2.
        i2Fi1: fundamental matrix between two images.

    Returns:
        Sampson distance for each row of the input, of shape N.
    """

    if coordinates_i1 is None or coordinates_i1.size == 0 or coordinates_i2 is None or coordinates_i2.size == 0:
        return None

    epipolar_lines_i2 = feature_utils.convert_to_epipolar_lines(coordinates_i1, i2Fi1)  # Ex1
    epipolar_lines_i1 = feature_utils.convert_to_epipolar_lines(coordinates_i2, i2Fi1.T)  # Etx2
    line_sq_norms_i1 = np.sum(np.square(epipolar_lines_i1[:, :2]), axis=1)
    line_sq_norms_i2 = np.sum(np.square(epipolar_lines_i2[:, :2]), axis=1)

    numerator = np.square(feature_utils.point_line_dotproduct(coordinates_i1, epipolar_lines_i1))
    denominator = line_sq_norms_i1 + line_sq_norms_i2

    return numerator / denominator


def filter_correspondences_for_track_quality(
    keypoints_i1: "Keypoints",
    keypoints_i2: "Keypoints",
    v_corr_idxs: np.ndarray,
    camera_intrinsics_i1: CALIBRATION_TYPE,
    camera_intrinsics_i2: CALIBRATION_TYPE,
    i2Ri1: Rot3,
    i2Ui1: Unit3,
    epipole_threshold_px: float = 20.0,
    min_triangulation_angle_deg: float = 1.0,
) -> np.ndarray:
    """Filter correspondences by epipole proximity, cheirality, and triangulation angle.

    Implements GLOMAP Section 3.1 filtering to remove matches that would create
    poorly-constrained tracks: matches near epipoles (near-zero parallax),
    matches that fail cheirality (triangulate behind cameras), and matches with
    small triangulation angles.

    Args:
        keypoints_i1: Keypoints from image i1.
        keypoints_i2: Keypoints from image i2.
        v_corr_idxs: Verified correspondence indices, shape (N, 2).
        camera_intrinsics_i1: Intrinsics for image i1.
        camera_intrinsics_i2: Intrinsics for image i2.
        i2Ri1: Relative rotation from camera i1 to i2.
        i2Ui1: Relative translation direction from camera i1 to i2.
        epipole_threshold_px: Remove matches within this distance of epipole.
        min_triangulation_angle_deg: Minimum triangulation angle in degrees.

    Returns:
        Filtered correspondence indices, shape (M, 2) where M <= N.
    """
    import gtsam

    if v_corr_idxs.shape[0] == 0:
        return v_corr_idxs

    coords_i1 = keypoints_i1.coordinates[v_corr_idxs[:, 0]]
    coords_i2 = keypoints_i2.coordinates[v_corr_idxs[:, 1]]

    # --- Epipole filtering ---
    # Compute F-matrix from E = [t]_x R and intrinsics.
    i2Ei1 = EssentialMatrix(i2Ri1, i2Ui1)
    K1_inv = np.linalg.inv(camera_intrinsics_i1.K())
    K2_inv_T = np.linalg.inv(camera_intrinsics_i2.K()).T
    F = K2_inv_T @ i2Ei1.matrix() @ K1_inv

    # Epipoles via SVD of F.
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]  # epipole in image 1
    if abs(e1[2]) > 1e-10:
        e1 = e1[:2] / e1[2]
    else:
        e1 = None

    U, _, _ = np.linalg.svd(F)
    e2 = U[:, -1]  # epipole in image 2
    if abs(e2[2]) > 1e-10:
        e2 = e2[:2] / e2[2]
    else:
        e2 = None

    keep_mask = np.ones(len(v_corr_idxs), dtype=bool)

    if e1 is not None:
        dist_to_e1 = np.linalg.norm(coords_i1 - e1[np.newaxis, :], axis=1)
        keep_mask &= dist_to_e1 > epipole_threshold_px

    if e2 is not None:
        dist_to_e2 = np.linalg.norm(coords_i2 - e2[np.newaxis, :], axis=1)
        keep_mask &= dist_to_e2 > epipole_threshold_px

    # --- Cheirality + triangulation angle filtering ---
    # Build poses and shared calibration for triangulation.
    try:
        i2Ti1 = Pose3(i2Ri1, i2Ui1.point3())
        wTi1 = Pose3()  # camera 1 at origin
        wTi2 = i2Ti1.inverse()  # camera 2 in world frame
        poses = [wTi1, wTi2]
    except Exception:
        return v_corr_idxs[keep_mask]

    cam1_center = np.zeros(3)
    cam2_center = wTi2.translation()
    min_angle_rad = np.radians(min_triangulation_angle_deg)

    for idx in range(len(v_corr_idxs)):
        if not keep_mask[idx]:
            continue
        try:
            # Normalize pixel coordinates to camera coordinates for triangulation.
            uv1_norm = feature_utils.normalize_coordinates(
                coords_i1[idx:idx+1], camera_intrinsics_i1
            )
            uv2_norm = feature_utils.normalize_coordinates(
                coords_i2[idx:idx+1], camera_intrinsics_i2
            )
            # Use DLT triangulation via bearing vectors.
            ray1_cam = np.array([uv1_norm[0, 0], uv1_norm[0, 1], 1.0])
            ray2_cam = np.array([uv2_norm[0, 0], uv2_norm[0, 1], 1.0])
            ray1_world = wTi1.rotation().rotate(ray1_cam)
            ray2_world = wTi2.rotation().rotate(ray2_cam)

            # Midpoint triangulation.
            A = np.column_stack([ray1_world, -ray2_world])
            b = cam2_center - cam1_center
            result = np.linalg.lstsq(A, b, rcond=None)
            t_vals = result[0]

            # Cheirality: both t values must be positive.
            if t_vals[0] < 0 or t_vals[1] < 0:
                keep_mask[idx] = False
                continue

            pt_3d = cam1_center + t_vals[0] * ray1_world

            # Triangulation angle check.
            vec1 = pt_3d - cam1_center
            vec2 = pt_3d - cam2_center
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-12)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            if np.arccos(cos_angle) < min_angle_rad:
                keep_mask[idx] = False
        except Exception:
            keep_mask[idx] = False

    num_filtered = len(v_corr_idxs) - np.sum(keep_mask)
    if num_filtered > 0:
        logger.debug(
            "Track quality filter: removed %d / %d correspondences (epipole + cheirality + angle).",
            num_filtered, len(v_corr_idxs),
        )

    return v_corr_idxs[keep_mask]
