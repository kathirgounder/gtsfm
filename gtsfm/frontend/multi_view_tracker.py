"""Multi-view tracker that uses geometry transformer output to produce tracks and GtsfmData."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gtsam import Point2, Point3
from torch.amp import autocast as amp_autocast  # type: ignore

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.geometry_transformer import GeometryTransformerOutput
from gtsfm.utils import logger as logger_utils
from gtsfm.utils import torch as torch_utils

PathLike = Union[str, Path]

logger = logger_utils.get_logger()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Per-worker cache for DINOv2 model to avoid reloading per cluster/task.
_DINO_V2_MODEL_CACHE: dict[tuple[str, str], Any] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrackingConfig:
    """Configuration for multi-view tracking and GtsfmData assembly."""

    tracking: bool = True
    max_query_pts: int = 2048
    query_frame_num: int = 3
    use_all_frames_forward_only: bool = False
    track_vis_thresh: float = 0.05
    track_conf_thresh: float = 0.2
    keypoint_extractor: str = "aliked+sp+sift"
    vggt_max_reproj_error: float = 14.0
    min_triangulation_angle: float = 10.0
    min_track_length: int = 2
    ba_track_patch_grid_size: int = 8
    enable_ba_track_patching: bool = True
    ba_use_undistorted_camera_model: bool = False


# ---------------------------------------------------------------------------
# Tracking result
# ---------------------------------------------------------------------------


@dataclass
class VGGTTrackingResult:
    """Container for the optional VGGT tracking pipeline outputs.

    Attributes:
        tracks: Array shaped ``(num_frames, num_tracks, 2)`` giving per-frame pixel coordinates.
        visibilities: Array shaped ``(num_frames, num_tracks)`` with per-frame visibility scores.
        confidences: Optional array containing per-track confidence values (may be ``None``).
        points_3d: Optional array of per-track 3D points (may be ``None``).
        colors: Optional array of per-track RGB colors in ``uint8`` range ``[0, 255]`` (may be ``None``).
    """

    tracks: np.ndarray
    visibilities: np.ndarray
    confidences: Optional[np.ndarray]
    points_3d: Optional[np.ndarray]
    colors: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Track selection helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrackSelectionCandidate:
    """Track candidate ranked by geometric support and reprojection error."""

    track_id: int
    track_length: int
    mean_reprojection_error: float
    patches_by_image: dict[int, tuple[int, int]]


def _compute_image_patch(
    u: float,
    v: float,
    image_width: float,
    image_height: float,
    patch_grid_size: int,
) -> tuple[int, int]:
    """Map a 2D pixel measurement to an image patch in an ``NxN`` grid."""
    if patch_grid_size <= 1:
        return (0, 0)
    col = int(np.floor(np.clip(u, 0.0, image_width - 1e-6) / image_width * patch_grid_size))
    row = int(np.floor(np.clip(v, 0.0, image_height - 1e-6) / image_height * patch_grid_size))
    return (row, col)


def _select_track_ids_for_ba_coverage(
    candidates: Sequence[_TrackSelectionCandidate],
    min_track_length: int,
    max_reproj_error: float,
) -> set[int]:
    """Select tracks to maximize patch coverage using ranking by track geometry quality."""
    if not candidates:
        return set()

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (-candidate.track_length, candidate.mean_reprojection_error, candidate.track_id),
    )

    selected_track_ids: set[int] = set()
    covered_patches: set[tuple[int, int, int]] = set()
    for candidate_track in ordered_candidates:
        if (
            candidate_track.track_length < min_track_length
            or candidate_track.mean_reprojection_error > max_reproj_error
        ):
            continue
        candidate_patch_keys = {
            (image_idx, patch_id[0], patch_id[1]) for image_idx, patch_id in candidate_track.patches_by_image.items()
        }
        if not candidate_patch_keys:
            continue
        if candidate_patch_keys.issubset(covered_patches):
            continue
        selected_track_ids.add(candidate_track.track_id)
        covered_patches.update(candidate_patch_keys)

    return selected_track_ids


def _is_point_in_front_of_camera(camera, point_xyz: np.ndarray, *, epsilon: float = 1e-6) -> bool:
    """Return True if ``point_xyz`` projects in front of ``camera``."""
    if camera is None:
        return False
    try:
        x, y, z = (float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]))
        cam_point = camera.pose().transformTo(Point3(x, y, z))
    except Exception:
        return False
    z_val = cam_point[2] if isinstance(cam_point, np.ndarray) else cam_point.z()
    return float(z_val) > epsilon


# ---------------------------------------------------------------------------
# VGGT tracking internals
# ---------------------------------------------------------------------------


def _import_vggsfm_utils():
    """Return the vendored vggsfm utilities module from the VGGT submodule."""
    from gtsfm.frontend.vggt_geometry_transformer import _USING_FASTVGGT, _import_from_vanilla_vggt

    try:
        from vggt.dependency import vggsfm_utils as _vggsfm_utils  # type: ignore
    except ImportError as exc:
        if _USING_FASTVGGT:
            try:
                tracker_module = _import_from_vanilla_vggt("dependency.vggsfm_utils")
                logger.info("Using vggsfm utilities from the vanilla VGGT submodule.")
                return tracker_module
            except ImportError as fallback_exc:
                exc = fallback_exc
        hint = (
            "Could not import VGGT tracker utilities. Ensure the 'vggt' submodule is checked out by "
            "running `git submodule update --init --recursive`."
        )
        if _USING_FASTVGGT:
            hint += " FastVGGT does not bundle the tracker code, so the vanilla VGGT submodule must remain accessible."
        raise ImportError(hint) from exc
    return _vggsfm_utils


def generate_rank_by_dino(
    images, query_frame_num, image_size=336, model_name="dinov2_vitb14_reg", device="cuda", spatial_similarity=False
):
    """Generate a ranking of frames using DINO ViT features.

    Args:
        images: Tensor of shape (S, 3, H, W) with values in range [0, 1]
        query_frame_num: Number of frames to select
        image_size: Size to resize images to before processing
        model_name: Name of the DINO model to use
        device: Device to run the model on
        spatial_similarity: Whether to use spatial token similarity or CLS token similarity

    Returns:
        List of frame indices ranked by their representativeness
    """
    vggsfm_utils = _import_vggsfm_utils()

    images = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)

    device_str = str(device)
    cache_key = (model_name, device_str)
    if cache_key in _DINO_V2_MODEL_CACHE:
        dino_v2_model = _DINO_V2_MODEL_CACHE[cache_key]
    else:
        logger.info("⏳ Loading DINOv2 model (%s)...", model_name)
        dino_v2_model = torch.hub.load("facebookresearch/dinov2", model_name)
        dino_v2_model.eval()
        dino_v2_model = dino_v2_model.to(device)
        _DINO_V2_MODEL_CACHE[cache_key] = dino_v2_model
        logger.info("✅ DINOv2 model loaded successfully.")

    imagenet_mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    images_imagenet_norm = (images - imagenet_mean) / imagenet_std

    with torch.no_grad():
        frame_feat = dino_v2_model(images_imagenet_norm, is_training=True)

    if spatial_similarity:
        frame_feat = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
        similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat = frame_feat["x_norm_clstoken"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        similarity_matrix = torch.mm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))

    distance_matrix = 100 - similarity_matrix.clone()
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)
    most_common_frame_index = torch.argmax(similarity_sum).item()
    fps_idx = vggsfm_utils.farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    del frame_feat, frame_feat_norm, similarity_matrix, distance_matrix
    return fps_idx


def _run_vggt_head_tracking(
    geo_output: GeometryTransformerOutput,
    *,
    model: Any,
    config: TrackingConfig,
) -> VGGTTrackingResult:
    """Generate dense feature tracks using the VGGT track head."""
    vggsfm_utils = _import_vggsfm_utils()

    device = geo_output.device
    if device.type != "cuda":
        raise RuntimeError(
            "VGGT tracking requires a CUDA-capable GPU because DINO relies on flash attention. "
            "Re-run the pipeline with CUDA available."
        )

    images = geo_output.images
    if images.device != device or images.dtype != torch.float32:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)

    frame_num = images.shape[0]
    if config.use_all_frames_forward_only:
        query_frame_indexes = list(range(frame_num))
    else:
        query_frame_indexes = generate_rank_by_dino(
            images,
            query_frame_num=config.query_frame_num,
            image_size=518,
            model_name="dinov2_vitb14_reg",
            device=device,
            spatial_similarity=False,
        )
        if 0 in query_frame_indexes:
            query_frame_indexes.remove(0)
        query_frame_indexes = [0, *query_frame_indexes]

    extractors = vggsfm_utils.initialize_feature_extractors(
        max_query_num=config.max_query_pts,
        extractor_method=config.keypoint_extractor,
        device=device,
    )

    dense_points = geo_output.dense_points
    depth_confidence = geo_output.depth_confidence
    height, width = images.shape[-2:]

    pred_tracks = []
    pred_vis_scores = []
    pred_conf_scores = []
    pred_world_points = []
    pred_world_points_conf = []
    pred_colors = []

    for query_index in query_frame_indexes:
        query_image = images[query_index]
        query_points = vggsfm_utils.extract_keypoints(query_image, extractors, round_keypoints=True)
        if query_points is None or query_points.shape[1] == 0:
            continue

        query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]
        if query_points.shape[1] > config.max_query_pts:
            query_points = query_points[:, : config.max_query_pts]

        query_points_round = query_points.squeeze(0).round().long()
        query_points_round[:, 0] = query_points_round[:, 0].clamp(0, width - 1)
        query_points_round[:, 1] = query_points_round[:, 1].clamp(0, height - 1)

        pred_color = (
            images[query_index][:, query_points_round[:, 1], query_points_round[:, 0]].permute(1, 0).cpu().numpy()
            * 255.0
        ).astype(np.uint8)

        pred_point_3d = dense_points[query_index][query_points_round[:, 1], query_points_round[:, 0]]

        pred_conf = None
        if depth_confidence is not None:
            pred_conf = depth_confidence[query_index][query_points_round[:, 1], query_points_round[:, 0]]

        if query_points.shape[1] == 0:
            continue

        reorder_index = vggsfm_utils.calculate_index_mappings(query_index, frame_num, device=device)
        reorder_images = vggsfm_utils.switch_tensor_order([images], reorder_index, dim=0)[0]

        if device.type == "cuda":
            autocast_ctx: Any = amp_autocast("cuda", dtype=geo_output.dtype)
        else:
            autocast_ctx = nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                aggregated_tokens_list, ps_idx = model.aggregator(reorder_images[None])
                track_list, vis_scores, conf_scores = model.track_head(
                    aggregated_tokens_list,
                    reorder_images[None],
                    ps_idx,
                    query_points=query_points,
                )

        pred_track = track_list[-1]
        pred_track = pred_track.squeeze(0)
        vis_scores = vis_scores.squeeze(0)
        conf_scores = conf_scores.squeeze(0)
        reordered = vggsfm_utils.switch_tensor_order([pred_track, vis_scores, conf_scores], reorder_index, dim=0)
        pred_track, pred_vis, pred_conf_score = reordered

        if config.use_all_frames_forward_only:
            valid_frames_mask = torch.arange(frame_num, device=device) >= query_index
            pred_vis = pred_vis * valid_frames_mask[:, None].to(dtype=pred_vis.dtype)
            if pred_conf_score is not None:
                pred_conf_score = pred_conf_score * valid_frames_mask[:, None].to(dtype=pred_conf_score.dtype)

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        if pred_conf_score is not None:
            pred_conf_scores.append(pred_conf_score)
        pred_world_points.append(pred_point_3d)
        if pred_conf is not None:
            pred_world_points_conf.append(pred_conf)
        pred_colors.append(pred_color)

    if not pred_tracks:
        empty_tracks = np.zeros((frame_num, 0, 2), dtype=np.float32)
        empty_vis = np.zeros((frame_num, 0), dtype=np.float32)
        empty_conf = np.zeros((0,), dtype=np.float32) if depth_confidence is not None else None
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        return VGGTTrackingResult(
            tracks=empty_tracks,
            visibilities=empty_vis,
            confidences=empty_conf,
            points_3d=empty_points,
            colors=empty_colors,
        )

    tracks = torch.cat(pred_tracks, dim=1)
    vis_scores = torch.cat(pred_vis_scores, dim=1)
    confidences = torch.cat(pred_conf_scores, dim=1) if pred_conf_scores else None
    points_3d = torch.cat(pred_world_points, dim=0) if pred_world_points else None
    points_3d_conf = torch.cat(pred_world_points_conf, dim=0) if pred_world_points_conf else None
    colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    if points_3d_conf is not None and points_3d is not None:
        filtered_flag = points_3d_conf > 1.5
        if int(filtered_flag.sum().item()) > config.max_query_pts // 2:
            tracks = tracks[:, filtered_flag]
            vis_scores = vis_scores[:, filtered_flag]
            if confidences is not None:
                confidences = confidences[:, filtered_flag]
            points_3d = points_3d[filtered_flag]
            points_3d_conf = points_3d_conf[filtered_flag]
            if colors is not None:
                colors = colors[filtered_flag.cpu().numpy()]

    return VGGTTrackingResult(
        tracks=tracks.float().cpu().numpy(),
        visibilities=vis_scores.float().cpu().numpy(),
        confidences=confidences.float().cpu().numpy() if confidences is not None else None,
        points_3d=points_3d.float().cpu().numpy() if points_3d is not None else None,
        colors=colors,
    )


# ---------------------------------------------------------------------------
# MultiViewTracker
# ---------------------------------------------------------------------------


class MultiViewTracker:
    """Produces feature tracks and assembles GtsfmData from geometry transformer output."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def run_tracking(
        self,
        geo_output: GeometryTransformerOutput,
        *,
        model: Any = None,
        config: TrackingConfig | None = None,
    ) -> VGGTTrackingResult:
        """Generate dense feature tracks using the VGGT tracking backend.

        Args:
            geo_output: Output from a geometry transformer.
            model: Loaded VGGT model (required for track head).
            config: Optional override config.

        Returns:
            :class:`VGGTTrackingResult` with tracks, visibilities, and 3D points.
        """
        cfg = config or self.config
        if model is None:
            raise ValueError("VGGT tracking requires a loaded VGGT model.")
        return _run_vggt_head_tracking(geo_output, model=model, config=cfg)

    def build_gtsfm_data(
        self,
        geo_output: GeometryTransformerOutput,
        original_coords: torch.Tensor,
        *,
        image_indices: Sequence[int],
        image_names: Optional[Sequence[str]] = None,
        tracking_result: VGGTTrackingResult | None = None,
        points_3d: np.ndarray | None = None,
        points_rgb: np.ndarray | None = None,
        config: TrackingConfig | None = None,
        cluster_label: Optional[str] = None,
    ) -> GtsfmData:
        """Convert geometry transformer output + optional tracks into GtsfmData.

        This creates cameras from extrinsics/intrinsics and optionally inserts
        tracked feature observations as SfmTracks.

        Args:
            geo_output: Output from a geometry transformer.
            original_coords: Tensor shaped ``(N, 6)`` with crop/pad metadata.
            image_indices: Global image indices for each frame.
            image_names: Optional image filenames.
            tracking_result: Optional tracking result to convert into tracks.
            points_3d: Optional dense point cloud (used for track colors).
            points_rgb: Optional per-point RGB colors.
            config: Optional override config.
            cluster_label: Optional cluster name for log messages.

        Returns:
            :class:`GtsfmData` with cameras and optionally tracks.
        """
        cfg = config or self.config

        extrinsic_np = geo_output.extrinsic.to(torch.float32).cpu().numpy()
        intrinsic_np = geo_output.intrinsic.to(torch.float32).cpu().numpy()
        original_coords_np = original_coords.to(torch.float32).cpu().numpy()
        image_names_str = [str(name) for name in image_names] if image_names is not None else None

        gtsfm_data = GtsfmData(number_images=len(image_indices))

        for local_idx, global_idx in enumerate(image_indices):
            image_width = float(original_coords_np[local_idx, 4])
            image_height = float(original_coords_np[local_idx, 5])
            scaled_intrinsic = intrinsic_np[local_idx]

            camera = torch_utils.camera_from_matrices(
                extrinsic_np[local_idx],
                scaled_intrinsic,
                crop_coords=original_coords_np[local_idx],
                use_cal3_bundler=not cfg.ba_use_undistorted_camera_model,
            )
            gtsfm_data.add_camera(global_idx, camera)  # type: ignore[arg-type]
            gtsfm_data.set_image_info(
                global_idx,
                name=image_names_str[local_idx] if image_names_str is not None else None,
                shape=(int(image_height), int(image_width)),
            )

        if tracking_result:
            self._insert_tracks(
                gtsfm_data,
                tracking_result,
                original_coords_np=original_coords_np,
                image_indices=image_indices,
                points_rgb=points_rgb,
                config=cfg,
                cluster_label=cluster_label,
            )

        return gtsfm_data

    def _insert_tracks(
        self,
        gtsfm_data: GtsfmData,
        tracking_result: VGGTTrackingResult,
        *,
        original_coords_np: np.ndarray,
        image_indices: Sequence[int],
        points_rgb: np.ndarray | None,
        config: TrackingConfig,
        cluster_label: Optional[str] = None,
    ) -> None:
        """Insert tracked features as SfmTracks into gtsfm_data (in-place)."""
        import gtsfm.utils.reprojection as reprojection_utils
        from gtsfm.common.sfm_track import SfmMeasurement

        track_mask = tracking_result.visibilities > config.track_vis_thresh

        if tracking_result.confidences is not None and tracking_result.confidences.size > 0:
            confidence_threshold = min(
                config.track_conf_thresh,
                float(np.mean(tracking_result.confidences) - np.std(tracking_result.confidences)),
            )
            track_mask = np.logical_and(track_mask, tracking_result.confidences > confidence_threshold)

        inlier_num = track_mask.sum(0)
        valid_mask = inlier_num >= config.min_track_length
        valid_idx = np.nonzero(valid_mask)[0]
        cameras = gtsfm_data.cameras()

        candidate_tracks: list[tuple[int, Any]] = []
        selection_candidates: list[_TrackSelectionCandidate] = []

        for valid_id in valid_idx:
            rgb: np.ndarray
            if tracking_result.colors is not None and valid_id < tracking_result.colors.shape[0]:
                rgb = tracking_result.colors[valid_id]
            elif points_rgb is not None and valid_id < points_rgb.shape[0]:
                rgb = points_rgb[valid_id]
            else:
                rgb = np.zeros(3, dtype=np.uint8)
            point_xyz = tracking_result.points_3d[valid_id]
            per_track_measurements: list[tuple[int, float, float]] = []
            patches_by_image: dict[int, tuple[int, int]] = {}
            frame_idx = np.where(track_mask[:, valid_id])[0]
            for local_id in frame_idx:
                global_idx = image_indices[local_id]
                u, v = tracking_result.tracks[local_id, valid_id]
                u = u + original_coords_np[local_id, 0]
                v = v + original_coords_np[local_id, 1]

                camera = gtsfm_data.get_camera(global_idx)
                if not _is_point_in_front_of_camera(camera, point_xyz):
                    continue
                per_track_measurements.append((global_idx, u, v))
                patches_by_image[global_idx] = _compute_image_patch(
                    u=u,
                    v=v,
                    image_width=original_coords_np[local_id, 4],
                    image_height=original_coords_np[local_id, 5],
                    patch_grid_size=config.ba_track_patch_grid_size,
                )

            if len(per_track_measurements) < config.min_track_length:
                continue

            sfm_measurements = [
                SfmMeasurement(i=global_idx, uv=np.array([float_u, float_v], dtype=np.float64))
                for global_idx, float_u, float_v in per_track_measurements
            ]
            reproj_errors, _ = reprojection_utils.compute_point_reprojection_errors(
                cameras, point_xyz, sfm_measurements
            )
            valid_reproj_errors = reproj_errors[~np.isnan(reproj_errors)]
            if valid_reproj_errors.size < config.min_track_length:
                continue
            track_length = int(valid_reproj_errors.size)
            mean_reprojection_error = float(np.mean(valid_reproj_errors))

            track = torch_utils.colored_track_from_point(point_xyz, rgb)
            for global_idx, float_u, float_v in per_track_measurements:
                track.addMeasurement(global_idx, Point2(float_u, float_v))
            min_triangulation_angle = config.min_triangulation_angle
            if min_triangulation_angle > 0.0:
                import gtsfm.utils.tracks as track_utils

                if track_utils.get_max_triangulation_angle(track, cameras) < min_triangulation_angle:
                    continue

            candidate_tracks.append((valid_id, track))
            selection_candidates.append(
                _TrackSelectionCandidate(
                    track_id=valid_id,
                    track_length=track_length,
                    mean_reprojection_error=mean_reprojection_error,
                    patches_by_image=patches_by_image,
                )
            )

        if config.enable_ba_track_patching:
            selected_track_ids = _select_track_ids_for_ba_coverage(
                selection_candidates, config.min_track_length, config.vggt_max_reproj_error
            )
        else:
            selected_track_ids = {track_id for track_id, _ in candidate_tracks}
        for track_id, track in candidate_tracks:
            if track_id in selected_track_ids:
                gtsfm_data.add_track(track)

        logger.info(
            "num valid tracks after filtering%s: %d out of %d",
            " and patching" if config.enable_ba_track_patching else " (patching disabled)",
            gtsfm_data.number_tracks(),
            len(valid_idx),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MultiViewTracker",
    "TrackingConfig",
    "VGGTTrackingResult",
    "generate_rank_by_dino",
]
