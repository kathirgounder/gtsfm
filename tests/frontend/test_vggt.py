"""Unit tests for VGGT glue.

Authors: Xinan Zhang and Frank Dellaert
"""

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms  # type: ignore

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.multi_view_tracker import (
    MultiViewTracker,
    TrackingConfig,
    _TrackSelectionCandidate,
    _select_track_ids_for_ba_coverage,
)
from gtsfm.frontend.vggt_geometry_transformer import (
    VggtGeometryConfig,
    VggtGeometryTransformer,
    default_dtype,
    high_confidence_pointcloud,
    load_model,
    offload_vggt_model,
)
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.utils import torch as torch_utils
from gtsfm.utils.tree import Tree  # PreOrderIter

LocalScene = tuple[Path, GtsfmData]
SceneTree = Tree[LocalScene]

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"
MAX_TRACKS_TO_DRAW = 200
MAX_POINTS_PER_FRAME = 200


def _vibrant_bgr_from_index(index: int) -> tuple[int, int, int]:
    """Generate a visually distinct BGR color using a hashed palette."""

    golden_ratio_hash = 0x9E3779B9
    hash_val = (index * golden_ratio_hash + 0xB5297A4D) & 0xFFFFFFFF

    def _component(shift: int) -> int:
        raw = (hash_val >> shift) & 0xFF
        return 64 + (raw * 191) // 255

    r = _component(16)
    g = _component(8)
    b = _component(0)
    return (b, g, r)


def _restore_images_to_original_scale(square_images: torch.Tensor, original_coords: torch.Tensor) -> torch.Tensor:
    """Crop padded square VGGT inputs back to their native aspect ratios."""

    if square_images.ndim != 4:
        raise ValueError(f"Expected square_images with 4 dims, got shape {tuple(square_images.shape)}")
    if original_coords.ndim != 2 or original_coords.shape[1] != 6:
        raise ValueError(f"original_coords must have shape (N,6); received {tuple(original_coords.shape)}")

    coords = original_coords.to(torch.float32)
    widths = coords[:, 4].round().clamp(min=1).to(torch.int64)
    heights = coords[:, 5].round().clamp(min=1).to(torch.int64)
    max_h = int(torch.max(heights).item())
    max_w = int(torch.max(widths).item())

    num_frames, num_channels, square_h, square_w = square_images.shape
    restored_frames: list[torch.Tensor] = []

    for idx in range(num_frames):
        x1, y1, x2, y2 = coords[idx, :4]

        x1i = int(torch.clamp(torch.floor(x1), 0, square_w - 1).item())
        y1i = int(torch.clamp(torch.floor(y1), 0, square_h - 1).item())
        x2i = int(torch.clamp(torch.ceil(x2), x1i + 1, square_w).item())
        y2i = int(torch.clamp(torch.ceil(y2), y1i + 1, square_h).item())

        crop = square_images[idx : idx + 1, :, y1i:y2i, x1i:x2i]
        if crop.numel() == 0:
            crop = square_images[idx : idx + 1]

        target_h = int(heights[idx].item())
        target_w = int(widths[idx].item())
        resized = F.interpolate(crop, size=(target_h, target_w), mode="bilinear", align_corners=False)

        canvas = torch.zeros((1, num_channels, max_h, max_w), dtype=square_images.dtype, device=square_images.device)
        canvas[:, :, :target_h, :target_w] = resized
        restored_frames.append(canvas)

    return torch.cat(restored_frames, dim=0).clamp(0.0, 1.0)


def run_vggt(
    image_batch: torch.Tensor,
    image_indices: list[int],
    original_coords,
    seed=42,
    conf_threshold_value=5.0,
    max_query_pts=1000,
    query_frame_num=4,
    vis_thresh=0.2,
    max_reproj_error=8.0,
) -> GtsfmData:
    """Run VGGT on the given image keys and return GtsfmData."""

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Setting seed as: {seed}")

    device = torch_utils.default_device()
    dtype = default_dtype(device)
    print(f"Using device: {device.type}")
    print(f"Using dtype: {dtype}")

    model = load_model(device=device)
    print("Model loaded")

    image_batch = image_batch.to(device)
    print("image_batch: ", image_batch.shape)
    original_coords = original_coords.to(device)

    geo_config = VggtGeometryConfig(confidence_threshold=conf_threshold_value)
    track_config = TrackingConfig(
        tracking=True,
        max_query_pts=max_query_pts,
        query_frame_num=query_frame_num,
        track_vis_thresh=vis_thresh,
        vggt_max_reproj_error=max_reproj_error,
    )

    transformer = VggtGeometryTransformer(geo_config)
    geo_output = transformer.predict(image_batch, model=model)

    tracker = MultiViewTracker(track_config)
    tracking_result = tracker.run_tracking(geo_output, model=model)
    if geo_output.device.type == "cuda":
        offload_vggt_model(model)

    points_3d, points_rgb = high_confidence_pointcloud(
        geo_output,
        confidence_threshold=geo_config.confidence_threshold,
        max_num_points=geo_config.max_num_points,
    )

    gtsfm_data = tracker.build_gtsfm_data(
        geo_output,
        original_coords,
        image_indices=image_indices,
        image_names=[f"image_{idx}" for idx in image_indices],
        tracking_result=tracking_result,
        points_3d=points_3d,
        points_rgb=points_rgb,
    )

    if points_3d.size == 0:
        print("VGGT produced no confident 3D structure.")

    return gtsfm_data


TEST_DATA = Path(__file__).parent.parent / "data"
PALACE = TEST_DATA / "palace-fine-arts-281"
DOOR = TEST_DATA / "set1_lund_door"


class TestVGGTTrackSelection(unittest.TestCase):
    """Unit tests for spatially distributed BA track selection."""

    def test_selects_only_tracks_that_introduce_new_patches(self) -> None:
        candidates = [
            _TrackSelectionCandidate(
                track_id=0, track_length=5, mean_reprojection_error=5.0, patches_by_image={0: (0, 0)}
            ),
            _TrackSelectionCandidate(
                track_id=1, track_length=4, mean_reprojection_error=3.0, patches_by_image={0: (0, 0)}
            ),
            _TrackSelectionCandidate(
                track_id=2, track_length=3, mean_reprojection_error=2.0, patches_by_image={0: (0, 1)}
            ),
        ]

        selected = _select_track_ids_for_ba_coverage(candidates, min_track_length=3, max_reproj_error=14.0)

        self.assertEqual(selected, {0, 2})

    def test_sorts_by_track_length_then_mean_reprojection_descending(self) -> None:
        candidates = [
            _TrackSelectionCandidate(
                track_id=0, track_length=5, mean_reprojection_error=1.0, patches_by_image={0: (0, 0)}
            ),
            _TrackSelectionCandidate(
                track_id=1, track_length=5, mean_reprojection_error=4.0, patches_by_image={0: (0, 0)}
            ),
            _TrackSelectionCandidate(
                track_id=2, track_length=4, mean_reprojection_error=3.0, patches_by_image={0: (0, 1)}
            ),
        ]

        selected = _select_track_ids_for_ba_coverage(candidates, min_track_length=3, max_reproj_error=14.0)

        # Track 0 wins over track 1 on tie-breaker (lower mean reprojection error), then track 2 adds a new patch.
        self.assertEqual(selected, {0, 2})

    def test_selects_track_if_any_observation_adds_new_patch(self) -> None:
        candidates = [
            _TrackSelectionCandidate(
                track_id=0,
                track_length=5,
                mean_reprojection_error=2.0,
                patches_by_image={0: (0, 0), 1: (0, 0)},
            ),
            _TrackSelectionCandidate(
                track_id=1,
                track_length=4,
                mean_reprojection_error=1.0,
                patches_by_image={0: (0, 0), 1: (0, 1)},
            ),
            _TrackSelectionCandidate(
                track_id=2,
                track_length=3,
                mean_reprojection_error=0.5,
                patches_by_image={0: (0, 0), 1: (0, 1)},
            ),
        ]

        selected = _select_track_ids_for_ba_coverage(candidates, min_track_length=3, max_reproj_error=14.0)

        self.assertEqual(selected, {0, 1})


class TestVGGT(unittest.TestCase):

    def setUp(self) -> None:
        pass

    @unittest.skip("Skipping VGGT end-to-end test for now since it is slow and requires GPU.")
    def test_run_vggt_on_some_images(self):
        """Load four door images using Olsson loader and run vggt on them."""

        img_load_original_resolution = 760
        img_load_resolution = 1024
        loader = OlssonLoader(dataset_dir=str(DOOR), max_resolution=img_load_original_resolution)
        indices = [4, 11, 8, 2]

        # resize_transform = None
        resize_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.from_numpy(x)),
                transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [H,W,C] → [C,H,W]
                transforms.Resize(size=(img_load_resolution, img_load_resolution), antialias=True),  # Expects [C,H,W]
            ]
        )
        # Transform 2: Convert to float32 and normalize to [0, 1]
        batch_transform = transforms.Lambda(lambda x: x.type(torch.float32) / 255.0)

        image_batch, original_coords = loader.load_image_batch_vggt(
            indices,
            img_load_resolution,
            resize_transform,
            batch_transform,
        )

        # image_batch, original_coords = loader.load_and_preprocess_images_square_vggt(indices, img_load_resolution)

        print("image_batch: ", image_batch.shape)

        with torch.no_grad():

            gtsfm_data = run_vggt(image_batch, indices, original_coords)

        self.assertIsNotNone(gtsfm_data)
        self.assertEqual(gtsfm_data.number_images(), len(indices))
        self.assertCountEqual(gtsfm_data.get_valid_camera_indices(), indices)



if __name__ == "__main__":
    unittest.main()
