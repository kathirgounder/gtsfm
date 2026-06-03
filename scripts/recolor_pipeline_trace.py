"""Re-color points3D.txt files in a pipeline_trace directory using per-image scale.

The dump_stage code throws away the original 2D UV measurements (point2D_idx is
written as a placeholder 0). To recover colors without re-running the pipeline,
we re-project each 3D point through each observing camera, then sample RGB at
the projected pixel using a per-image scale factor (image-on-disk dims vs
camera-frame dims as recorded in cameras.txt).

Usage:
    python scripts/recolor_pipeline_trace.py \
        --trace-dir benchmark_results/thanjavur-magnum-opus-trace/results/plots/pipeline_trace \
        --images-dir benchmarks/thanjavur/images_532

This rewrites points3D.txt in every stage_*/ subdir, leaving cameras.txt /
images.txt / metadata.json untouched.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage


# ── COLMAP-format minimal parsers (just the fields we need) ───────────────


def parse_cameras_txt(path: Path) -> dict[int, dict]:
    """Return {camera_id: {'width','height','model','params'}} from a COLMAP cameras.txt."""
    cams: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        cams[cam_id] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": np.array([float(x) for x in parts[4:]], dtype=float),
        }
    return cams


def parse_images_txt(path: Path) -> dict[int, dict]:
    """Return {image_id: {'qvec','tvec','camera_id','name'}} from a COLMAP images.txt.

    Detects pose rows by structure (10+ tokens, integer image_id) rather than
    iterating every-other-line. dump_stage emits empty POINTS2D rows that the
    every-other-line approach mis-counts after empty-line filtering.
    """
    images: dict[int, dict] = {}
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
        except ValueError:
            continue  # POINTS2D row (floats), skip
        images[image_id] = {
            "qvec": np.array([float(x) for x in parts[1:5]], dtype=float),  # qw qx qy qz
            "tvec": np.array([float(x) for x in parts[5:8]], dtype=float),
            "camera_id": int(parts[8]),
            "name": parts[9],
        }
    return images


def parse_points3D_txt(path: Path) -> list[dict]:
    """Return list of {'id','xyz','rgb','error','track': [(image_id, point2D_idx), ...]}."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        track = []
        for k in range(8, len(parts), 2):
            track.append((int(parts[k]), int(parts[k + 1])))
        out.append({
            "id": int(parts[0]),
            "xyz": np.array([float(x) for x in parts[1:4]], dtype=float),
            "rgb": np.array([int(x) for x in parts[4:7]], dtype=int),
            "error": float(parts[7]),
            "track": track,
        })
    return out


def write_points3D_txt(path: Path, points: list[dict]) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for p in points:
        track_str = " ".join(f"{i} {idx}" for (i, idx) in p["track"])
        x, y, z = p["xyz"]
        r, g, b = p["rgb"]
        lines.append(f"{p['id']} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {p['error']:.6f} {track_str}")
    path.write_text("\n".join(lines) + "\n")


# ── Geometry: project 3D point through a SIMPLE_PINHOLE camera ────────────


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def project(xyz_world: np.ndarray, image: dict, camera: dict) -> Optional[tuple[float, float]]:
    """Project a world-frame 3D point into a SIMPLE_PINHOLE camera. Returns (u, v) in
    camera-frame pixel coords, or None if behind camera."""
    R = qvec_to_rotmat(image["qvec"])
    t = image["tvec"]
    p_cam = R @ xyz_world + t
    if p_cam[2] <= 0:
        return None
    if camera["model"] == "SIMPLE_PINHOLE":
        f, cx, cy = camera["params"][:3]
        u = f * (p_cam[0] / p_cam[2]) + cx
        v = f * (p_cam[1] / p_cam[2]) + cy
    elif camera["model"] == "PINHOLE":
        fx, fy, cx, cy = camera["params"][:4]
        u = fx * (p_cam[0] / p_cam[2]) + cx
        v = fy * (p_cam[1] / p_cam[2]) + cy
    else:
        return None  # SIMPLE_RADIAL etc not handled — skip
    return float(u), float(v)


# ── Image cache ────────────────────────────────────────────────────────────


class ImageCache:
    def __init__(self, images_dir: Path):
        self.images_dir = images_dir
        self.cache: dict[str, np.ndarray] = {}

    def get(self, name: str) -> Optional[np.ndarray]:
        if name in self.cache:
            return self.cache[name]
        path = self.images_dir / Path(name).name
        if not path.is_file():
            self.cache[name] = None  # type: ignore
            return None
        try:
            arr = np.array(PILImage.open(path).convert("RGB"))
            self.cache[name] = arr
            return arr
        except Exception:
            self.cache[name] = None  # type: ignore
            return None


# ── Recolor a single stage ────────────────────────────────────────────────


def recolor_stage(
    stage_dir: Path, image_cache: ImageCache, image_id_to_filename: dict[int, str]
) -> tuple[int, int]:
    cameras = parse_cameras_txt(stage_dir / "cameras.txt")
    images = parse_images_txt(stage_dir / "images.txt")
    points = parse_points3D_txt(stage_dir / "points3D.txt")

    n_recolored = 0
    n_grey = 0
    for p in points:
        colors: list[np.ndarray] = []
        for (image_id, _) in p["track"]:
            img_meta = images.get(image_id)
            if img_meta is None:
                continue
            cam = cameras.get(img_meta["camera_id"])
            if cam is None:
                continue
            # Workaround for the bug where dump_stage writes 1×1 placeholder dims
            # when info.shape is None (GP-output path doesn't populate shape).
            # Derive camera-frame dims from intrinsics: principal point ≈ image
            # center, so width ≈ 2*cx, height ≈ 2*cy.
            if cam["width"] <= 1 or cam["height"] <= 1:
                params = cam["params"]
                if cam["model"] == "SIMPLE_PINHOLE":
                    _, cx, cy = params[:3]
                else:  # PINHOLE: fx, fy, cx, cy
                    cx, cy = params[2], params[3]
                cam_w = max(int(round(2 * cx)), 2)
                cam_h = max(int(round(2 * cy)), 2)
            else:
                cam_w, cam_h = cam["width"], cam["height"]

            uv = project(p["xyz"], img_meta, cam)
            if uv is None:
                continue

            # Workaround for placeholder image names (`image_00000.jpg` etc).
            # image_id was assigned as the sorted-filename position, so we can map
            # back via the images_dir listing.
            real_name = img_meta["name"]
            if real_name.startswith("image_") and not (image_cache.images_dir / real_name).exists():
                real_name = image_id_to_filename.get(image_id, real_name)

            img_arr = image_cache.get(real_name)
            if img_arr is None:
                continue
            # Per-image scale: on-disk image dims vs camera-frame dims.
            scale_u = img_arr.shape[1] / cam_w
            scale_v = img_arr.shape[0] / cam_h
            u_px = int(round(uv[0] * scale_u))
            v_px = int(round(uv[1] * scale_v))
            if 0 <= v_px < img_arr.shape[0] and 0 <= u_px < img_arr.shape[1]:
                colors.append(img_arr[v_px, u_px][:3])

        if colors:
            p["rgb"] = np.median(np.stack(colors), axis=0).astype(int)
            n_recolored += 1
        else:
            p["rgb"] = np.array([128, 128, 128], dtype=int)
            n_grey += 1

    write_points3D_txt(stage_dir / "points3D.txt", points)
    return n_recolored, n_grey


def propagate_clean_colors(
    trace_dir: Path,
    color_source_stage: str,
    target_stage_prefixes: list[str],
    nn_max_dist: float = 0.05,
) -> None:
    """NN-propagate colors from `color_source_stage` to all stages whose name
    starts with any prefix in `target_stage_prefixes`.

    Use case: retri stages and (some) BA stages process the raw union-find track
    set, which includes sky-projecting tracks that have wrong (sky-blue) sampled
    colors. After the late-stage BA filters drop those bad tracks, the surviving
    points are clean. So we use the final, fully-filtered stage as the canonical
    color source and NN-propagate it back to the dirty earlier stages.
    """
    from scipy.spatial import cKDTree

    src_path = trace_dir / color_source_stage / "points3D.txt"
    if not src_path.is_file():
        print(f"  [propagate] missing {src_path}, skipping")
        return
    src_pts = parse_points3D_txt(src_path)
    src_xyz = np.stack([p["xyz"] for p in src_pts])
    src_rgb = np.stack([p["rgb"] for p in src_pts])
    print(f"  [propagate] color source: {color_source_stage} ({len(src_pts):,} pts)")
    tree = cKDTree(src_xyz)

    target_stages = sorted(
        d for d in trace_dir.iterdir()
        if d.is_dir()
        and any(d.name.startswith(prefix) for prefix in target_stage_prefixes)
        and (d / "points3D.txt").exists()
        and d.name != color_source_stage
    )
    print(f"  [propagate] applying color map to {len(target_stages)} stages")
    for stage in target_stages:
        pts = parse_points3D_txt(stage / "points3D.txt")
        if not pts:
            continue
        anchor_xyz = np.stack([p["xyz"] for p in pts])
        # 5-NN median: averages over a local color neighborhood to smooth out
        # single-wrong-neighbor flicker (e.g., a sky-projection outlier in the
        # source happens to be the closest, but its 4 neighbors are sandstone →
        # median is sandstone, single-NN would be sky). Critical for retri
        # stages whose re-triangulated positions drift from canonical positions.
        dists, nn_idxs = tree.query(anchor_xyz, k=5, distance_upper_bound=nn_max_dist)
        n_recolored = 0
        n_grey = 0
        grey = np.array([128, 128, 128], dtype=int)
        for p, d_arr, ni_arr in zip(pts, dists, nn_idxs):
            valid_idxs = [int(ni) for d, ni in zip(d_arr, ni_arr)
                          if np.isfinite(d) and ni < len(src_rgb)]
            if valid_idxs:
                colors = src_rgb[valid_idxs]
                p["rgb"] = np.median(colors, axis=0).astype(int)
                n_recolored += 1
            else:
                p["rgb"] = grey
                n_grey += 1
        write_points3D_txt(stage / "points3D.txt", pts)
        print(f"    {stage.name}: recolored {n_recolored:,} / {len(pts):,} pts ({n_grey:,} grey fallback)")


def propagate_gp_colors(
    trace_dir: Path,
    color_source_stage: str,
    gp_anchor_stage: str = "stage_gp_020",
) -> None:
    """Color stage_gp_* dirs via NN lookup against a recolored BA stage.

    GP dumps have empty track fields (dump_gp_iter doesn't store measurements)
    so per-track re-projection coloring fails. Workaround: the FINAL GP iter is
    geometrically close to BA-input (same union-find tracks, before BA filtering),
    so we NN-match by 3D position. GP point IDs are stable across iters, so the
    color map derived at gp_020 applies to every gp_iter stage by ID.
    """
    from scipy.spatial import cKDTree  # local import to keep base script importable

    src_path = trace_dir / color_source_stage / "points3D.txt"
    anchor_path = trace_dir / gp_anchor_stage / "points3D.txt"
    if not src_path.is_file() or not anchor_path.is_file():
        print(f"  [gp-propagate] missing {src_path.name} or {anchor_path.name}, skipping")
        return

    src_pts = parse_points3D_txt(src_path)
    anchor_pts = parse_points3D_txt(anchor_path)
    src_xyz = np.stack([p["xyz"] for p in src_pts])
    src_rgb = np.stack([p["rgb"] for p in src_pts])
    anchor_xyz = np.stack([p["xyz"] for p in anchor_pts])
    anchor_ids = np.array([p["id"] for p in anchor_pts])

    print(
        f"  [gp-propagate] color source: {color_source_stage} "
        f"({len(src_pts):,} pts), anchor: {gp_anchor_stage} ({len(anchor_pts):,} pts)"
    )
    tree = cKDTree(src_xyz)
    # 5-NN median: smooths out single-wrong-neighbor flicker (see propagate_clean_colors)
    dists, nn_idxs = tree.query(anchor_xyz, k=5, distance_upper_bound=0.3)
    color_by_id: dict[int, np.ndarray] = {}
    n_close = 0
    for pt_id, d_arr, ni_arr in zip(anchor_ids, dists, nn_idxs):
        valid_idxs = [int(ni) for d, ni in zip(d_arr, ni_arr)
                      if np.isfinite(d) and ni < len(src_rgb)]
        if valid_idxs:
            colors = src_rgb[valid_idxs]
            color_by_id[int(pt_id)] = np.median(colors, axis=0).astype(int)
            n_close += 1
    print(f"  [gp-propagate] {n_close:,}/{len(anchor_pts):,} GP points found a source neighborhood within 0.3")

    gp_stages = sorted(
        d for d in trace_dir.iterdir()
        if d.is_dir() and d.name.startswith("stage_gp_") and (d / "points3D.txt").exists()
    )
    print(f"  [gp-propagate] applying color map to {len(gp_stages)} gp stages")
    grey = np.array([128, 128, 128], dtype=int)
    for gp_stage in gp_stages:
        pts = parse_points3D_txt(gp_stage / "points3D.txt")
        n_recolored = 0
        n_grey = 0
        # Idempotent reset: every point gets EITHER the matched source color OR
        # explicit grey. Wipes stale colors from prior propagation runs.
        for p in pts:
            rgb = color_by_id.get(p["id"])
            if rgb is not None:
                p["rgb"] = rgb
                n_recolored += 1
            else:
                p["rgb"] = grey
                n_grey += 1
        write_points3D_txt(gp_stage / "points3D.txt", pts)
        print(f"    {gp_stage.name}: recolored {n_recolored:,} / {len(pts):,} pts ({n_grey:,} grey)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-dir", type=Path, required=True, help="pipeline_trace dir containing stage_* subdirs")
    parser.add_argument("--images-dir", type=Path, required=True, help="Source images dir")
    parser.add_argument(
        "--gp-color-source",
        type=str,
        default="stage_005_outer_ba_iter_0",
        help="Recolored BA stage to use as color source for GP+retri propagation. "
             "Defaults to outer_ba_iter_0 (post first BA outer iter, ~117K pts) which "
             "has both the cleanest re-projected colors AND the most coverage — final "
             "filtered stages are over-restricted, leaving too many GP points unmatched.",
    )
    parser.add_argument(
        "--gp-anchor",
        type=str,
        default="stage_gp_020",
        help="Final GP iter stage; used to match GP point IDs to BA NN colors",
    )
    parser.add_argument(
        "--skip-ba-recolor",
        action="store_true",
        help="Skip per-stage re-projection recolor and only run GP NN propagation",
    )
    args = parser.parse_args()

    if not args.trace_dir.is_dir():
        raise SystemExit(f"trace-dir not found: {args.trace_dir}")
    if not args.images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {args.images_dir}")

    cache = ImageCache(args.images_dir)

    # Workaround for the placeholder-name bug in dump_stage: image_id is assigned as
    # the GTSFM cam_idx, which equals sorted-filename position in the loader. So we
    # can map image_id → real filename via the images_dir listing.
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    sorted_names = sorted(
        f.name for f in args.images_dir.iterdir()
        if f.suffix.lower() in image_exts
    )
    image_id_to_filename = {i: name for i, name in enumerate(sorted_names)}
    print(f"Built image_id → filename map for {len(image_id_to_filename)} images")

    # GP stages have empty tracks → re-projection recolor produces 100% grey.
    # Skip them in the per-stage pass; they're handled separately via NN propagation
    # from a recolored BA stage below.
    stages = sorted(
        d for d in args.trace_dir.iterdir()
        if d.is_dir() and (d / "points3D.txt").exists() and not d.name.startswith("stage_gp_")
    )
    print(f"Found {len(stages)} non-GP stages in {args.trace_dir}")

    if not args.skip_ba_recolor:
        for i, stage in enumerate(stages):
            n_rec, n_grey = recolor_stage(stage, cache, image_id_to_filename)
            print(f"  [{i+1:3d}/{len(stages)}] {stage.name}: recolored {n_rec:,} pts, {n_grey:,} grey fallback")
        print(f"\n{len(cache.cache)} images cached during BA recolor.")

    # Retri stages process the raw union-find track set, which includes sky-
    # projection tracks BA later drops. The re-projection recolor catches sky-
    # blue at those track measurements. Override with NN propagation from the
    # final clean BA stage (which already had its sky points filtered out).
    print("\n── retri stage propagation (overrides re-projection sky bleed) ──────")
    propagate_clean_colors(
        trace_dir=args.trace_dir,
        color_source_stage=args.gp_color_source,
        target_stage_prefixes=["stage_032_retri", "stage_033_retri_final_ba", "stage_034_retri_final_ba",
                               "stage_035_retri_final_ba", "stage_036_retri_final_ba",
                               "stage_037_retri_final_ba", "stage_038_retri_final_ba"],
        nn_max_dist=0.05,
    )

    print("\n── GP propagation ────────────────────────────────────────")
    propagate_gp_colors(args.trace_dir, args.gp_color_source, args.gp_anchor)
    print("\nDone.")


if __name__ == "__main__":
    main()
