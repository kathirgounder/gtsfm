"""View graph calibration: estimate camera focal lengths from fundamental matrices.

Uses Bougnoux's formula to estimate focal lengths from F-matrices computed from
verified correspondences. Per-camera focal estimates are aggregated via median
for robustness.

Reference: Hartley & Zisserman, Multiple View Geometry, Section 19.4.
GLOMAP paper Section 3.5 — view graph calibration before global positioning.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
from gtsfm.common.keypoints import Keypoints

logger = logging.getLogger(__name__)


def estimate_fundamental_from_correspondences(
    coords_i1: np.ndarray,
    coords_i2: np.ndarray,
) -> Optional[np.ndarray]:
    """Estimate F-matrix from verified correspondences using 8-point algorithm (no RANSAC needed).

    Args:
        coords_i1: (N, 2) pixel coordinates in image 1.
        coords_i2: (N, 2) pixel coordinates in image 2.

    Returns:
        3x3 fundamental matrix, or None if estimation fails.
    """
    if len(coords_i1) < 8:
        return None
    F, mask = cv2.findFundamentalMat(coords_i1, coords_i2, method=cv2.FM_8POINT)
    if F is None or F.shape != (3, 3):
        return None
    return F


def estimate_focal_from_fundamental(
    F: np.ndarray,
    pp1: np.ndarray,
    pp2: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate focal lengths from fundamental matrix using Bougnoux's formula.

    Given F and principal points, estimates the squared focal lengths:
        f1^2 = -pp2^T F e1 e1^T F^T pp2_x / (pp2_x^T F e1 e1^T F^T pp2_x)
        f2^2 = -pp1^T F^T e2 e2^T F pp1_x / (pp1_x^T F^T e2 e2^T F pp1_x)

    where e1, e2 are the epipoles (null spaces of F^T and F respectively),
    and pp_x is the skew-symmetric matrix of the principal point.

    Reference: Bougnoux, "From Projective to Euclidean Space under any Practical
    Situation", ICCV 1998. Also Hartley & Zisserman Section 19.4.

    Args:
        F: 3x3 fundamental matrix.
        pp1: (2,) principal point of camera 1 (typically image center).
        pp2: (2,) principal point of camera 2.

    Returns:
        (f1, f2) estimated focal lengths. Either may be None if estimation is degenerate.
    """
    # Compute epipoles: e1 = null(F^T), e2 = null(F)
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]  # right null vector of F (epipole in image 1)
    e1 = e1 / e1[2] if abs(e1[2]) > 1e-10 else e1

    U, _, _ = np.linalg.svd(F)
    e2 = U[:, -1]  # left null vector of F (epipole in image 2)
    e2 = e2 / e2[2] if abs(e2[2]) > 1e-10 else e2

    # Homogeneous principal points.
    p1 = np.array([pp1[0], pp1[1], 1.0])
    p2 = np.array([pp2[0], pp2[1], 1.0])

    # Skew-symmetric (cross-product) matrix of a 3-vector.
    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Bougnoux formula for f1^2:
    # f1^2 = - (p2^T F e1) (e1^T F^T p2) / (p2_x^T F e1) (e1^T F^T p2_x)
    # where p2_x = skew(p2) selects the "cross" part
    # Simplified: use the diagonal element form from the epipolar constraint.

    # Following the formulation from Hartley & Zisserman (Result 19.2):
    # Let e' = e2 (epipole in image 2), e = e1 (epipole in image 1)
    # f1^2 = - (e'^T [p2]_x F diag(1,1,0) F^T p2) / (e'^T [p2]_x F diag(0,0,1) F^T p2)
    # f2^2 = - (e^T [p1]_x F^T diag(1,1,0) F p1) / (e^T [p1]_x F^T diag(0,0,1) F p1)

    D_xy = np.diag([1.0, 1.0, 0.0])
    D_z = np.diag([0.0, 0.0, 1.0])

    # Estimate f2 (focal length of camera 2)
    skew_p2 = skew(p2)
    A = skew_p2 @ F
    num_f1 = -(e2 @ A @ D_xy @ F.T @ p2)
    den_f1 = -(e2 @ A @ D_z @ F.T @ p2)

    # Estimate f1 (focal length of camera 1)
    skew_p1 = skew(p1)
    B = skew_p1 @ F.T
    num_f2 = -(e1 @ B @ D_xy @ F @ p1)
    den_f2 = -(e1 @ B @ D_z @ F @ p1)

    f1, f2 = None, None

    if abs(den_f1) > 1e-10:
        f1_sq = num_f1 / den_f1
        if f1_sq > 0:
            f1 = np.sqrt(f1_sq)

    if abs(den_f2) > 1e-10:
        f2_sq = num_f2 / den_f2
        if f2_sq > 0:
            f2 = np.sqrt(f2_sq)

    return f1, f2


def calibrate_view_graph(
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints_list: List[Keypoints],
    initial_intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
    num_images: int,
    min_correspondences: int = 30,
) -> List[gtsfm_types.CALIBRATION_TYPE]:
    """Refine camera focal lengths from view graph F-matrices using Bougnoux's formula.

    For each edge in the view graph, estimates an F-matrix from verified correspondences
    and extracts focal length estimates via Bougnoux. Per-camera estimates are aggregated
    via median.

    Args:
        v_corr_idxs_dict: Verified correspondence indices per image pair.
        keypoints_list: Keypoints for all images.
        initial_intrinsics: Initial intrinsics (e.g., from EXIF or heuristic).
        num_images: Total number of images.
        min_correspondences: Minimum correspondences to attempt F estimation.

    Returns:
        Refined intrinsics list (same length as initial_intrinsics).
    """
    # Collect focal length estimates per camera.
    focal_estimates: Dict[int, List[float]] = defaultdict(list)

    for (i1, i2), v_corr_idxs in v_corr_idxs_dict.items():
        if v_corr_idxs.shape[0] < min_correspondences:
            continue

        coords_i1 = keypoints_list[i1].coordinates[v_corr_idxs[:, 0]]
        coords_i2 = keypoints_list[i2].coordinates[v_corr_idxs[:, 1]]

        F = estimate_fundamental_from_correspondences(coords_i1, coords_i2)
        if F is None:
            continue

        # Principal points from initial intrinsics.
        K1 = initial_intrinsics[i1].K()
        K2 = initial_intrinsics[i2].K()
        pp1 = np.array([K1[0, 2], K1[1, 2]])
        pp2 = np.array([K2[0, 2], K2[1, 2]])

        f1_est, f2_est = estimate_focal_from_fundamental(F, pp1, pp2)

        if f1_est is not None and _is_reasonable_focal(f1_est, initial_intrinsics[i1]):
            focal_estimates[i1].append(f1_est)
        if f2_est is not None and _is_reasonable_focal(f2_est, initial_intrinsics[i2]):
            focal_estimates[i2].append(f2_est)

    # Build refined intrinsics using median focal estimate per camera.
    refined = list(initial_intrinsics)
    num_refined = 0
    for cam_idx in range(num_images):
        estimates = focal_estimates.get(cam_idx, [])
        if len(estimates) < 2:
            # Not enough estimates — keep original.
            continue
        median_focal = float(np.median(estimates))
        old_K = initial_intrinsics[cam_idx]
        if isinstance(old_K, Cal3Bundler):
            refined[cam_idx] = Cal3Bundler(
                fx=median_focal,
                k1=old_K.k1(),
                k2=old_K.k2(),
                u0=old_K.px(),
                v0=old_K.py(),
            )
            num_refined += 1

    logger.info(
        "View graph calibration: refined %d / %d cameras (%.1f%%). "
        "Total focal estimates collected: %d across %d edges.",
        num_refined,
        num_images,
        100.0 * num_refined / max(num_images, 1),
        sum(len(v) for v in focal_estimates.values()),
        len(v_corr_idxs_dict),
    )

    return refined


def _is_reasonable_focal(focal: float, intrinsics: gtsfm_types.CALIBRATION_TYPE) -> bool:
    """Check if estimated focal length is within a reasonable range.

    Rejects estimates that are wildly different from the initial value —
    likely degenerate configurations.
    """
    initial_focal = intrinsics.K()[0, 0]
    if initial_focal <= 0:
        return focal > 0
    ratio = focal / initial_focal
    # Accept if within [0.1x, 10x] of initial estimate.
    return 0.1 < ratio < 10.0
