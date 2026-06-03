"""Dump GtsfmData snapshots in COLMAP format for the pipeline visualizer.

One stage = one folder with cameras.txt, images.txt, points3D.txt, metadata.json.
The pipeline visualizer reads these as keyframes and animates between them.

Usage:
    from gtsfm.visualization.pipeline_trace import dump_stage
    dump_stage(gtsfm_data, trace_dir / "stage_03_retri", stage_name="retri", iteration=0)

For colored points: set the env var GTSFM_PIPELINE_TRACE_IMAGES_DIR to the image
directory. The default sampler then reads each track's measurements, samples the
median pixel color, and writes RGB into points3D.txt. Without the env var, points
default to grey (128, 128, 128).

Authors: Kathir Gounder
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from thirdparty.colmap.scripts.python.read_write_model import (
    Camera as ColmapCamera,
    Image as ColmapImage,
    Point3D as ColmapPoint3D,
    rotmat2qvec,
    write_cameras_text,
    write_images_text,
    write_points3D_text,
)

logger = logger_utils.get_logger()


# ── Image-RGB color sampler (ported from scripts/visualize_gp_convergence.py) ──
class ImageColorSampler:
    """Eager-loads all images keyed by camera index 0..N (sorted filename order).
    Samples median per-track RGB from measurements. UV scale factor auto-detected
    on first use (handles pipelines that downscale images vs. raw uvs).
    """

    def __init__(self, images_dir: str | Path):
        self.images_dir = Path(images_dir)
        self._images: dict[int, np.ndarray] = {}
        # Per-image scale: keypoints from different pipelines (COLMAP DB at
        # max_image_size=1600 vs GTSFM SIFT at max_resolution=760) and across
        # crowdsourced datasets with mixed image dimensions all need their OWN
        # uv → image-pixel scale factor. A single global scale gives blue-sky
        # bleed when applied to images with different dims (e.g. Thanjavur
        # ranges from 2256×1504 to 6000×4000).
        self._scale_per_image: dict[int, tuple[float, float]] = {}
        self._max_uv_per_image: dict[int, tuple[float, float]] = {}
        try:
            from PIL import Image as PILImage
            files = sorted(
                f.name for f in self.images_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            )
            for i, fname in enumerate(files):
                try:
                    self._images[i] = np.array(PILImage.open(self.images_dir / fname).convert("RGB"))
                except Exception:
                    pass
            logger.info("[pipeline_trace] loaded %d images for color sampling", len(self._images))
        except Exception as e:
            logger.warning("[pipeline_trace] image loading failed: %s", e)

    def prime_scale(self, tracks, max_tracks: int = 5000):
        """Compute per-image UV→image scale by aggregating max-uv per camera index
        across many tracks. Each image gets its own scale factor based on the keypoint
        coordinate range observed in tracks that include that image."""
        if not self._images:
            return
        n_seen = 0
        for t in tracks:
            for k in range(t.numberMeasurements()):
                i, uv = t.measurement(k)
                cam_idx = int(i)
                cur_u, cur_v = self._max_uv_per_image.get(cam_idx, (0.0, 0.0))
                self._max_uv_per_image[cam_idx] = (
                    max(cur_u, float(uv[0])),
                    max(cur_v, float(uv[1])),
                )
            n_seen += 1
            if n_seen >= max_tracks:
                break
        # Compute per-image scale: image_dim / max_uv_for_that_image (clamp to 1.0
        # if uvs already span >80% of image — already in pixel coords).
        for cam_idx, (max_u, max_v) in self._max_uv_per_image.items():
            img = self._images.get(cam_idx)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale_u = w / max(max_u, 1) if max_u < w * 0.8 else 1.0
            scale_v = h / max(max_v, 1) if max_v < h * 0.8 else 1.0
            # If uvs already roughly match image dims, no scaling needed.
            if scale_u < 1.5 and scale_v < 1.5:
                scale_u, scale_v = 1.0, 1.0
            self._scale_per_image[cam_idx] = (scale_u, scale_v)
        logger.info(
            "[pipeline_trace] per-image uv→image scale primed for %d images (sampled %d tracks). "
            "Scale range: u=[%.2f, %.2f], v=[%.2f, %.2f]",
            len(self._scale_per_image), n_seen,
            min(s[0] for s in self._scale_per_image.values()) if self._scale_per_image else 0,
            max(s[0] for s in self._scale_per_image.values()) if self._scale_per_image else 0,
            min(s[1] for s in self._scale_per_image.values()) if self._scale_per_image else 0,
            max(s[1] for s in self._scale_per_image.values()) if self._scale_per_image else 0,
        )

    def sample_track(self, track) -> list[int]:
        """Median RGB (0-255 ints) across all measurements. Grey if no samples."""
        colors = []
        for k in range(track.numberMeasurements()):
            i, uv = track.measurement(k)
            cam_idx = int(i)
            img = self._images.get(cam_idx)
            if img is None:
                continue
            scale_u, scale_v = self._scale_per_image.get(cam_idx, (1.0, 1.0))
            u = int(round(float(uv[0]) * scale_u))
            v = int(round(float(uv[1]) * scale_v))
            if 0 <= v < img.shape[0] and 0 <= u < img.shape[1]:
                colors.append(img[v, u][:3])
        if not colors:
            return [128, 128, 128]
        return np.median(np.stack(colors), axis=0).astype(int).tolist()


# Module-level cache so all dumps in one process share the same sampler+image cache.
_default_sampler: Optional[ImageColorSampler] = None
_default_sampler_resolved = False
_registered_images_dir: Optional[str] = None


def register_images_dir(path: str | Path) -> None:
    """Auto-detect entry point: runner calls this after instantiating the loader,
    so dumps get colored points without the user needing to set env vars manually.
    Env var GTSFM_PIPELINE_TRACE_IMAGES_DIR takes precedence if set."""
    global _registered_images_dir, _default_sampler_resolved
    _registered_images_dir = str(path)
    _default_sampler_resolved = False  # force re-resolution next call


def get_default_sampler() -> Optional[ImageColorSampler]:
    """Return a process-cached ImageColorSampler. Resolution order:
    1. GTSFM_PIPELINE_TRACE_IMAGES_DIR env var (manual override)
    2. register_images_dir(...) call from the runner (auto-detect from loader)
    Otherwise None (callers fall back to grey points)."""
    global _default_sampler, _default_sampler_resolved
    if _default_sampler_resolved:
        return _default_sampler
    _default_sampler_resolved = True
    images_dir = os.environ.get("GTSFM_PIPELINE_TRACE_IMAGES_DIR") or _registered_images_dir
    if images_dir and Path(images_dir).is_dir():
        _default_sampler = ImageColorSampler(images_dir)
        logger.info("[pipeline_trace] color sampler enabled, images_dir=%s", images_dir)
    return _default_sampler


def dump_stage(
    gtsfm_data: GtsfmData,
    stage_dir: Path | str,
    stage_name: str,
    iteration: int = 0,
    extra_metadata: Optional[dict] = None,
) -> None:
    """Dump a GtsfmData snapshot to `stage_dir` as COLMAP-format files.

    Args:
        gtsfm_data: scene with cameras + 3D tracks.
        stage_dir: target directory (created if not exists). One stage = one folder.
        stage_name: human-readable name (e.g. "gp_iter_001", "outer_ba_iter_2", "retri").
        iteration: per-stage iteration index (used for ordering inside the stage).
        extra_metadata: optional dict merged into metadata.json.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    cameras: dict[int, ColmapCamera] = {}
    images: dict[int, ColmapImage] = {}
    points3D: dict[int, ColmapPoint3D] = {}

    # Cameras + images. We use SIMPLE_PINHOLE [f, cx, cy] for compactness; this is
    # a viz-only export, no need to round-trip the full distortion model.
    for cam_idx, cam in gtsfm_data.cameras().items():
        if cam is None:
            continue
        info = gtsfm_data.get_image_info(cam_idx)
        width = int(info.shape[1]) if info.shape else 1
        height = int(info.shape[0]) if info.shape else 1
        K = cam.calibration()
        fx = float(K.fx())
        cx = float(K.px())
        cy = float(K.py())
        cameras[cam_idx] = ColmapCamera(id=cam_idx, model="SIMPLE_PINHOLE",
                                        width=width, height=height,
                                        params=np.array([fx, cx, cy], dtype=float))

        # Image pose. COLMAP stores cam_from_world; gtsam Pose3 is world_from_cam.
        wTc = cam.pose()
        cTw = wTc.inverse()
        R = cTw.rotation().matrix()
        t = cTw.translation()
        qvec = rotmat2qvec(R)
        name = info.name if info.name else f"image_{cam_idx:05d}.jpg"
        images[cam_idx] = ColmapImage(
            id=cam_idx, qvec=qvec, tvec=np.asarray(t, dtype=float),
            camera_id=cam_idx, name=name,
            xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), dtype=int),
        )

    # 3D points. If the env-driven color sampler is available, sample per-track
    # median RGB from images; otherwise default grey.
    sampler = get_default_sampler()
    # Prime UV→image scale on first stage if not already done (uses up to 200 tracks).
    if sampler is not None and not sampler._scale_per_image and gtsfm_data.number_tracks() > 0:
        sampler.prime_scale((gtsfm_data.get_track(j) for j in range(gtsfm_data.number_tracks())))
    for j in range(gtsfm_data.number_tracks()):
        track = gtsfm_data.get_track(j)
        xyz = np.asarray(track.point3(), dtype=float)
        if sampler:
            rgb = np.array(sampler.sample_track(track), dtype=int)
        else:
            # SfmTrack has r/g/b attrs but they default to 0 when unset → black
            # points in viz. Prefer grey when no sampler is available.
            rgb = np.array([128, 128, 128], dtype=int)
        image_ids = []
        point2D_idxs = []
        for k in range(track.numberMeasurements()):
            i, _ = track.measurement(k)
            image_ids.append(int(i))
            point2D_idxs.append(0)  # placeholder; viz doesn't need real 2D indices
        points3D[j] = ColmapPoint3D(
            id=j, xyz=xyz, rgb=rgb, error=0.0,
            image_ids=np.asarray(image_ids, dtype=int),
            point2D_idxs=np.asarray(point2D_idxs, dtype=int),
        )

    write_cameras_text(cameras, str(stage_dir / "cameras.txt"))
    write_images_text(images, str(stage_dir / "images.txt"))
    write_points3D_text(points3D, str(stage_dir / "points3D.txt"))

    metadata = {
        "stage_name": stage_name,
        "iteration": iteration,
        "num_cameras": len(cameras),
        "num_tracks": len(points3D),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with (stage_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def dump_gp_iter(
    cam_positions: dict[int, np.ndarray],
    wRi_matrices: dict[int, np.ndarray],
    landmark_positions: np.ndarray | dict[int, np.ndarray],
    stage_dir: Path | str,
    stage_name: str,
    iteration: int = 0,
    extra_metadata: Optional[dict] = None,
    image_names: Optional[dict[int, str]] = None,
    track_colors: Optional[dict[int, list[int]]] = None,
) -> None:
    """Dump a GP per-iteration snapshot in COLMAP format from raw numpy arrays.

    GP doesn't have full intrinsics or image_info readily — this writes the COLMAP
    files directly with placeholder intrinsics (SIMPLE_PINHOLE f=500, cx=320, cy=240,
    width=640, height=480) so the visualizer renders consistently across iterations
    without needing real K. Frustums will be rendered at uniform size in the viewer
    using these placeholders; the camera POSITIONS and ROTATIONS are the meaningful
    geometric content per iter.

    Args:
        cam_positions: {camera_idx: (3,) array} — per-camera world position
        wRi_matrices: {camera_idx: (3, 3) array} — world-from-camera rotation
        landmark_positions: (M, 3) array of 3D point positions
        stage_dir: target directory (created if not exists)
        stage_name, iteration, extra_metadata: as in dump_stage
        image_names: optional {camera_idx: filename} — defaults to image_NNNNN.jpg
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    if image_names is None:
        image_names = {}

    # Cameras: placeholder SIMPLE_PINHOLE intrinsics (uniform across all cameras).
    cameras: dict[int, ColmapCamera] = {}
    images: dict[int, ColmapImage] = {}
    for cam_idx, pos in cam_positions.items():
        cameras[cam_idx] = ColmapCamera(
            id=cam_idx, model="SIMPLE_PINHOLE", width=640, height=480,
            params=np.array([500.0, 320.0, 240.0], dtype=float),
        )
        # COLMAP stores cam_from_world; we have world_from_cam (wRi + position).
        # cam_from_world = (wRi.T, -wRi.T @ position).
        wRi = wRi_matrices.get(cam_idx)
        if wRi is None:
            continue
        cRw = wRi.T
        cTw = -cRw @ pos
        qvec = rotmat2qvec(cRw)
        name = image_names.get(cam_idx, f"image_{cam_idx:05d}.jpg")
        images[cam_idx] = ColmapImage(
            id=cam_idx, qvec=qvec, tvec=np.asarray(cTw, dtype=float),
            camera_id=cam_idx, name=name,
            xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), dtype=int),
        )

    # 3D points. Accepts landmark_positions as either an array (sequential ids)
    # or a dict {track_idx: xyz} so we can look up precomputed colors by track id.
    # track_colors is a precomputed {track_idx: [r,g,b]} from a sampler — lets GP
    # iters share image-sampled colors without per-iter measurement access.
    points3D: dict[int, ColmapPoint3D] = {}
    if isinstance(landmark_positions, dict):
        items = landmark_positions.items()
    else:
        items = enumerate(landmark_positions)
    for j, xyz in items:
        if track_colors and j in track_colors:
            rgb = np.array(track_colors[j], dtype=int)
        else:
            rgb = np.array([128, 128, 128], dtype=int)
        points3D[int(j)] = ColmapPoint3D(
            id=int(j), xyz=np.asarray(xyz, dtype=float),
            rgb=rgb, error=0.0,
            image_ids=np.zeros((0,), dtype=int),
            point2D_idxs=np.zeros((0,), dtype=int),
        )

    write_cameras_text(cameras, str(stage_dir / "cameras.txt"))
    write_images_text(images, str(stage_dir / "images.txt"))
    write_points3D_text(points3D, str(stage_dir / "points3D.txt"))

    metadata = {"stage_name": stage_name, "iteration": iteration,
                "num_cameras": len(cameras), "num_tracks": len(points3D)}
    if extra_metadata:
        metadata.update(extra_metadata)
    with (stage_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def append_to_manifest(
    trace_dir: Path | str,
    stage_subdir: str,
    stage_name: str,
    iteration: int,
    extra: Optional[dict] = None,
) -> None:
    """Append a stage entry to `<trace_dir>/manifest.json` (creates the file if missing).

    The manifest is the playlist the visualizer reads — ordered list of stage subdirs.
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = trace_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
    else:
        manifest = {"stages": []}
    entry = {"subdir": stage_subdir, "stage_name": stage_name, "iteration": iteration}
    if extra:
        entry.update(extra)
    manifest["stages"].append(entry)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
