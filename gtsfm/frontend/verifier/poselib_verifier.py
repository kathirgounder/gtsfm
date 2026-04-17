"""PoseLib-based verifier using 5-point E-matrix with LO-RANSAC and local BA.

PoseLib's estimate_relative_pose provides:
- 5-point essential matrix solver (more constrained than 8-point F-matrix)
- Locally Optimized RANSAC (refines hypotheses on inlier sets)
- Local bundle adjustment within RANSAC loop
- Integrated pose estimation (no separate F→E→pose decomposition chain)

This matches GLOMAP's two-view geometry estimation pipeline.

Reference: PoseLib (https://github.com/PoseLib/PoseLib)
"""

from typing import Optional, Tuple

import numpy as np
import poselib
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()


class PoseLibVerifier(VerifierBase):
    """Verifier using PoseLib's estimate_relative_pose with 5-point solver + LO-RANSAC + local BA."""

    def __init__(
        self,
        estimation_threshold_px: float = 4.0,
        max_iterations: int = 100000,
        confidence: float = 0.999999,
    ) -> None:
        # PoseLib always uses intrinsics internally (5-point E-matrix solver).
        super().__init__(use_intrinsics_in_verification=True, estimation_threshold_px=estimation_threshold_px)
        self._max_iterations = max_iterations
        self._confidence = confidence

    def _cal3bundler_to_poselib_camera(self, K: CALIBRATION_TYPE, width: int, height: int) -> poselib.Camera:
        """Convert GTSAM calibration to PoseLib Camera."""
        if isinstance(K, Cal3Bundler):
            fx = K.fx()
            k1 = K.k1()
            k2 = K.k2()
            cx = K.px()
            cy = K.py()
            if abs(k1) < 1e-10 and abs(k2) < 1e-10:
                return poselib.Camera("PINHOLE", [fx, fx, cx, cy], width, height)
            else:
                return poselib.Camera("SIMPLE_RADIAL", [fx, cx, cy, k1], width, height)
        else:
            # Fallback: extract from K matrix.
            K_mat = K.K()
            fx = K_mat[0, 0]
            cx = K_mat[0, 2]
            cy = K_mat[1, 2]
            return poselib.Camera("PINHOLE", [fx, fx, cx, cy], width, height)

    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: CALIBRATION_TYPE,
        camera_intrinsics_i2: CALIBRATION_TYPE,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, float]:
        """Estimate relative pose using PoseLib's 5-point solver with LO-RANSAC.

        Args:
            keypoints_i1: Detected features in image #i1.
            keypoints_i2: Detected features in image #i2.
            match_indices: Match indices, shape (N, 2).
            camera_intrinsics_i1: Intrinsics for image #i1.
            camera_intrinsics_i2: Intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None.
            Estimated unit translation i2Ui1, or None.
            Verified correspondence indices, shape (M, 2).
            Inlier ratio.
        """
        if match_indices.shape[0] < self._min_matches:
            return self._failure_result

        # Extract matched pixel coordinates.
        pts1 = keypoints_i1.coordinates[match_indices[:, 0]]
        pts2 = keypoints_i2.coordinates[match_indices[:, 1]]

        # Build PoseLib cameras.
        # Estimate image dimensions from keypoint range + principal point.
        K1 = camera_intrinsics_i1.K()
        K2 = camera_intrinsics_i2.K()
        w1 = int(2 * K1[0, 2])
        h1 = int(2 * K1[1, 2])
        w2 = int(2 * K2[0, 2])
        h2 = int(2 * K2[1, 2])

        cam1 = self._cal3bundler_to_poselib_camera(camera_intrinsics_i1, w1, h1)
        cam2 = self._cal3bundler_to_poselib_camera(camera_intrinsics_i2, w2, h2)

        # RANSAC and bundle adjustment options.
        ransac_opt = {
            "max_reproj_error": self._estimation_threshold_px,
            "max_iterations": self._max_iterations,
            "min_iterations": 100,
            "success_prob": self._confidence,
        }
        bundle_opt = {
            "max_iterations": 25,
        }

        try:
            pose, info = poselib.estimate_relative_pose(
                pts1, pts2, cam1, cam2,
                ransac_opt=ransac_opt,
                bundle_opt=bundle_opt,
            )
        except Exception as e:
            logger.debug("PoseLib estimate_relative_pose failed: %s", e)
            return self._failure_result

        # Extract inlier mask from info dict.
        inlier_mask = np.array(info.get("inliers", []), dtype=bool)
        if len(inlier_mask) == 0 or inlier_mask.sum() < self._min_matches:
            return self._failure_result

        # Convert PoseLib pose to GTSAM Rot3/Unit3.
        # PoseLib quaternion is [w, x, y, z], rotation matrix is cam2_from_cam1.
        R = np.array(pose.R)
        t = np.array(pose.t)

        i2Ri1 = Rot3(R)
        if np.linalg.norm(t) < 1e-12:
            return self._failure_result
        i2Ui1 = Unit3(t)

        # Build verified correspondence indices.
        inlier_idxs = np.where(inlier_mask)[0]
        v_corr_idxs = match_indices[inlier_idxs]
        inlier_ratio = float(inlier_mask.sum()) / len(inlier_mask)

        return i2Ri1, i2Ui1, v_corr_idxs, inlier_ratio
