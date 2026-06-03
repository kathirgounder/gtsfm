"""Unified recoloring: every track gets ONE consistent color across ALL stages.

Architecture (replaces the multi-step propagation hacks):

    1. Recolor a canonical stage via re-projection from images (uses real RGB
       sampled from the cameras observing each track). This is the gold-standard
       color source.

    2. Build a "track identity → color" map from the canonical stage. Identity
       is the SET of cameras observing the track (a frozenset of image_ids).
       This identity is stable across BA refinement — BA only drops whole
       tracks, never modifies which cameras observe a surviving track.

    3. For every other non-GP stage: look up each track's camera-set in the map.
       If found, use that color. If not (e.g., a re-triangulated track from
       union-find that doesn't exist in canonical), fall back to position-based
       5-NN median.

    4. For GP stages: dump_gp_iter doesn't write track measurements (camera-sets
       are empty), so fingerprint matching can't apply. Use by-ID propagation
       via gp_anchor=stage_gp_020 — GP point IDs are stable across all GP iters.

Usage:
    python scripts/recolor_unified.py \\
        --trace-dir benchmark_results/<run>/results/plots/pipeline_trace \\
        --images-dir benchmarks/thanjavur/images_532

Or import the function from another script:
    from recolor_unified import unified_recolor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from recolor_pipeline_trace import (
    ImageCache,
    parse_points3D_txt,
    write_points3D_txt,
    recolor_stage,
)


GREY = np.array([128, 128, 128], dtype=int)


def build_fingerprint_map(canonical_pts: list[dict]) -> dict[frozenset, np.ndarray]:
    """Build {camera_set_fingerprint → rgb} from a recolored stage."""
    fp_map: dict[frozenset, np.ndarray] = {}
    for p in canonical_pts:
        fp = frozenset(im_id for im_id, _ in p["track"])
        if fp:  # skip tracks with empty observation list
            fp_map[fp] = p["rgb"]
    return fp_map


def unified_recolor_stage(
    stage_dir: Path,
    fp_map: dict[frozenset, np.ndarray],
    canonical_xyz: np.ndarray,
    canonical_rgb: np.ndarray,
    canonical_tree: cKDTree,
    nn_max_dist: float = 0.5,
) -> tuple[int, int, int]:
    """Recolor a single stage. Returns (n_fingerprint, n_position_fallback, n_grey)."""
    pts = parse_points3D_txt(stage_dir / "points3D.txt")
    if not pts:
        return 0, 0, 0

    xyz = np.stack([p["xyz"] for p in pts])
    _, nn_idxs = canonical_tree.query(xyz, k=5, distance_upper_bound=nn_max_dist)

    n_fp, n_pos, n_grey = 0, 0, 0
    for p, ni_arr in zip(pts, nn_idxs):
        fp = frozenset(im_id for im_id, _ in p["track"])
        color = fp_map.get(fp) if fp else None
        if color is not None:
            p["rgb"] = color
            n_fp += 1
            continue
        # Fingerprint miss → 5-NN median position fallback
        valid = [int(ni) for ni in ni_arr if ni < len(canonical_rgb)]
        if valid:
            p["rgb"] = np.median(canonical_rgb[valid], axis=0).astype(int)
            n_pos += 1
        else:
            p["rgb"] = GREY
            n_grey += 1
    write_points3D_txt(stage_dir / "points3D.txt", pts)
    return n_fp, n_pos, n_grey


def propagate_gp_by_id(
    trace_dir: Path,
    canonical_stage: str,
    fp_map: dict[frozenset, np.ndarray],
    canonical_xyz: np.ndarray,
    canonical_rgb: np.ndarray,
    canonical_tree: cKDTree,
    gp_anchor_stage: str = "stage_gp_020",
    nn_max_dist: float = 0.5,
) -> None:
    """GP stages have empty track measurements, so fingerprint matching can't
    apply. Build a {gp_id → color} map at gp_anchor (final GP iter, where
    positions are close to canonical) using position NN, then propagate by ID
    to all GP iters since GP IDs are stable across iters."""
    anchor_pts = parse_points3D_txt(trace_dir / gp_anchor_stage / "points3D.txt")
    if not anchor_pts:
        return
    anchor_xyz = np.stack([p["xyz"] for p in anchor_pts])
    _, nn_idxs = canonical_tree.query(anchor_xyz, k=5, distance_upper_bound=nn_max_dist)

    color_by_id: dict[int, np.ndarray] = {}
    for p, ni_arr in zip(anchor_pts, nn_idxs):
        valid = [int(ni) for ni in ni_arr if ni < len(canonical_rgb)]
        if valid:
            color_by_id[int(p["id"])] = np.median(canonical_rgb[valid], axis=0).astype(int)
    print(f"  [gp] anchor={gp_anchor_stage}: built {len(color_by_id):,}/{len(anchor_pts):,} id→color map")

    gp_stages = sorted(
        d for d in trace_dir.iterdir()
        if d.is_dir() and d.name.startswith("stage_gp_") and (d / "points3D.txt").exists()
    )
    for gp_stage in gp_stages:
        pts = parse_points3D_txt(gp_stage / "points3D.txt")
        n_match, n_grey = 0, 0
        for p in pts:
            rgb = color_by_id.get(int(p["id"]))
            if rgb is not None:
                p["rgb"] = rgb
                n_match += 1
            else:
                p["rgb"] = GREY
                n_grey += 1
        write_points3D_txt(gp_stage / "points3D.txt", pts)
        print(f"  [gp] {gp_stage.name}: {n_match:,}/{len(pts):,} via id ({n_grey:,} grey)")


def unified_recolor(
    trace_dir: Path,
    images_dir: Path,
    canonical_stage: str = "stage_005_outer_ba_iter_0",
    gp_anchor_stage: str = "stage_gp_020",
    nn_max_dist: float = 0.5,
) -> None:
    """Single-source-of-truth recoloring of all pipeline_trace stages."""
    print(f"── Unified recolor ──")
    print(f"  trace_dir:       {trace_dir}")
    print(f"  canonical stage: {canonical_stage}")
    print()

    # ── Step 1: re-projection recolor of canonical ─────────────────────────
    print(f"Step 1: re-projection recolor of canonical from {images_dir}")
    cache = ImageCache(images_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    sorted_names = sorted(
        f.name for f in images_dir.iterdir() if f.suffix.lower() in image_exts
    )
    image_id_to_filename = {i: name for i, name in enumerate(sorted_names)}
    n_rec, n_grey = recolor_stage(trace_dir / canonical_stage, cache, image_id_to_filename)
    print(f"  canonical: recolored {n_rec:,} pts, {n_grey:,} grey fallback")

    # ── Step 2: build fingerprint map + position-NN tree from canonical ────
    canonical_pts = parse_points3D_txt(trace_dir / canonical_stage / "points3D.txt")
    fp_map = build_fingerprint_map(canonical_pts)
    canonical_xyz = np.stack([p["xyz"] for p in canonical_pts])
    canonical_rgb = np.stack([p["rgb"] for p in canonical_pts])
    canonical_tree = cKDTree(canonical_xyz)
    print(f"\nStep 2: built fingerprint map ({len(fp_map):,} unique camera-sets)"
          f" + position kdtree ({len(canonical_pts):,} pts)")

    # ── Step 3: recolor every non-GP, non-canonical stage ──────────────────
    print(f"\nStep 3: unified recolor of non-GP stages")
    other_stages = sorted(
        d for d in trace_dir.iterdir()
        if d.is_dir()
        and (d / "points3D.txt").exists()
        and not d.name.startswith("stage_gp_")
        and d.name != canonical_stage
    )
    total_fp = total_pos = total_grey = total_pts = 0
    for stage in other_stages:
        n_fp, n_pos, n_grey = unified_recolor_stage(
            stage, fp_map, canonical_xyz, canonical_rgb, canonical_tree, nn_max_dist
        )
        n_pts = n_fp + n_pos + n_grey
        total_fp += n_fp
        total_pos += n_pos
        total_grey += n_grey
        total_pts += n_pts
        print(f"  {stage.name}: {n_fp:>6,} fp + {n_pos:>5,} pos + {n_grey:>4,} grey = {n_pts:>7,}")
    if total_pts:
        print(f"  TOTAL: fp={total_fp:,} ({100*total_fp/total_pts:.1f}%)  "
              f"pos={total_pos:,} ({100*total_pos/total_pts:.1f}%)  "
              f"grey={total_grey:,} ({100*total_grey/total_pts:.1f}%)")

    # ── Step 4: GP stages by ID via gp_anchor ──────────────────────────────
    print(f"\nStep 4: GP stage propagation (by id via {gp_anchor_stage})")
    propagate_gp_by_id(
        trace_dir, canonical_stage, fp_map, canonical_xyz, canonical_rgb, canonical_tree,
        gp_anchor_stage, nn_max_dist,
    )
    print(f"\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--canonical-stage", type=str, default="stage_005_outer_ba_iter_0")
    parser.add_argument("--gp-anchor", type=str, default="stage_gp_020")
    parser.add_argument("--nn-max-dist", type=float, default=0.5)
    args = parser.parse_args()
    unified_recolor(
        trace_dir=args.trace_dir,
        images_dir=args.images_dir,
        canonical_stage=args.canonical_stage,
        gp_anchor_stage=args.gp_anchor,
        nn_max_dist=args.nn_max_dist,
    )


if __name__ == "__main__":
    main()
