"""Identify cameras the SfM pipeline placed inside the temple structure (prakara/
hallway), which is unphysical for tourist photos and indicates a doppelganger
front<->back fold.

Method: for each camera position in the final BA stage, look at the nearest N 3D
points and compute how widely they spread in azimuth (XY plane) around the
camera. Outdoor cameras see structure in front of them only — bearings cluster
in a cone (typically <180°). Cameras wrongly placed *inside* the prakara see
structure on all sides — bearings cover ~360°.

Outputs:
  - <audit_dir>/suspect_cameras.csv : ranked by surroundedness
  - <audit_dir>/suspect_cameras/    : symlinks to the source image files for
                                       hand-inspection. Open these in Finder
                                       to confirm they really are back-of-temple
                                       shots that the pipeline mis-folded.

Usage:
    python scripts/audit_suspect_cameras.py \
        --trace-dir benchmark_results/thanjavur-magnum-opus-trace/results/plots/pipeline_trace \
        --stage stage_039_retri_final_ba \
        --images-dir benchmarks/thanjavur/images_532 \
        --audit-dir benchmark_results/thanjavur-magnum-opus-trace/audit \
        --top-k 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Reuse the parsers from the recolor script — same COLMAP txt format.
from recolor_pipeline_trace import parse_cameras_txt, parse_images_txt, parse_points3D_txt, qvec_to_rotmat


def camera_position_world(image: dict) -> np.ndarray:
    """COLMAP stores cam_from_world (qvec, tvec). Camera center in world coords
    is C = -R^T t where R is the rotation from cam_from_world."""
    R = qvec_to_rotmat(image["qvec"])
    t = image["tvec"]
    return -R.T @ t


def surroundedness_score(cam_xyz: np.ndarray, point_xyz: np.ndarray, k: int = 200) -> tuple[float, float]:
    """Returns (azimuth_spread_deg, median_dist) for the k nearest points around
    the camera.

    azimuth_spread_deg is the angular range (in the XY plane) covered by those
    nearby points relative to the camera position. We use a histogram-based
    coverage metric (count of populated 10° bins) to avoid being thrown off by
    a single outlier point on the back-side: a camera that has 1 stray back-of-
    head point gets the same score as a fully outdoor one this way.

    median_dist is the median Euclidean distance to those k nearest points.
    """
    tree = cKDTree(point_xyz)
    dists, idxs = tree.query(cam_xyz, k=min(k, len(point_xyz)))
    nbrs = point_xyz[idxs]
    rel = nbrs - cam_xyz
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))  # XY-plane azimuth
    az_mod = (az + 360.0) % 360.0
    # Bin into 10° bins, count how many bins have >=2 points (single-point bins
    # could be noise from a far stray point). Then convert bin-coverage back to
    # degrees: 36 bins total → spread = filled_bins * 10°.
    bins = np.zeros(36, dtype=int)
    for a in az_mod:
        bins[int(a // 10) % 36] += 1
    filled = int(np.sum(bins >= 2))
    spread_deg = float(filled * 10)
    return spread_deg, float(np.median(dists))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--stage", type=str, default="stage_039_retri_final_ba",
                        help="Stage subdir to audit (default: final retri+BA stage)")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True,
                        help="Output dir for CSV + symlinked suspect images")
    parser.add_argument("--top-k", type=int, default=60, help="How many top suspects to symlink")
    parser.add_argument("--k-nearest-points", type=int, default=200,
                        help="Number of nearest 3D points to analyze per camera")
    parser.add_argument("--max-dist", type=float, default=2.0,
                        help="Cameras with median nearest-point distance above this are NOT considered suspect "
                             "(they're far enough from structure to be legit outdoor cameras)")
    parser.add_argument("--density-radius", type=float, default=0.3,
                        help="Radius (scene units) for counting nearby cameras as collapsed-cluster signal")
    parser.add_argument("--density-threshold", type=int, default=30,
                        help="Cameras with at least this many neighbors within density-radius are flagged as suspect")
    args = parser.parse_args()

    stage_dir = args.trace_dir / args.stage
    if not stage_dir.is_dir():
        raise SystemExit(f"Stage dir not found: {stage_dir}")

    print(f"[audit] reading {stage_dir}")
    cameras = parse_cameras_txt(stage_dir / "cameras.txt")
    images = parse_images_txt(stage_dir / "images.txt")
    points = parse_points3D_txt(stage_dir / "points3D.txt")
    point_xyz = np.stack([p["xyz"] for p in points])
    print(f"[audit] {len(images)} cameras, {len(point_xyz):,} points")

    # image_id → real filename: dump_stage's bug writes "image_NNNNN.jpg" placeholders
    # when info.name is unset, so fall back to sorted-filename position (image_id == cam_idx).
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    sorted_names = sorted(f.name for f in args.images_dir.iterdir() if f.suffix.lower() in image_exts)
    id_to_real = {i: name for i, name in enumerate(sorted_names)}

    # First pass: collect all camera positions so we can compute camera-density.
    # The doppelganger failure mode shows up as MANY cameras collapsed onto a tiny
    # region (because feature matches all agree on "this is the front of the
    # gopuram" when really half the photos are of the back). Camera density
    # >> typical tourist clustering is the signature.
    cam_positions = []
    cam_meta = []
    for image_id, img_meta in images.items():
        cam_xyz = camera_position_world(img_meta)
        cam_positions.append(cam_xyz)
        cam_meta.append((image_id, img_meta))
    cam_positions = np.stack(cam_positions)
    cam_tree = cKDTree(cam_positions)
    densities = np.array([len(cam_tree.query_ball_point(p, r=args.density_radius)) for p in cam_positions])

    rows = []
    for (image_id, img_meta), cam_xyz, density in zip(cam_meta, cam_positions, densities):
        spread, med_dist = surroundedness_score(cam_xyz, point_xyz, k=args.k_nearest_points)
        recorded_name = img_meta["name"]
        real_name = id_to_real.get(image_id, recorded_name)
        if recorded_name.startswith("image_") and not (args.images_dir / recorded_name).exists():
            display_name = real_name
        else:
            display_name = recorded_name
        rows.append({
            "image_id": image_id,
            "filename": display_name,
            "cam_x": cam_xyz[0],
            "cam_y": cam_xyz[1],
            "cam_z": cam_xyz[2],
            "azimuth_spread_deg": spread,
            "median_nearest_dist": med_dist,
            "density": int(density),
        })

    # Rank by density (collapsed-cluster signature) descending, with embedded-in-
    # structure as a tiebreaker.
    rows.sort(key=lambda r: (-r["density"], -r["azimuth_spread_deg"], r["median_nearest_dist"]))

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.audit_dir / "suspect_cameras.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[audit] wrote ranked CSV: {csv_path}")

    suspects_dir = args.audit_dir / "suspect_cameras"
    if suspects_dir.exists():
        for old in suspects_dir.iterdir():
            if old.is_symlink() or old.is_file():
                old.unlink()
    suspects_dir.mkdir(parents=True, exist_ok=True)

    # Suspect = density above threshold (collapsed cluster).
    suspects = [r for r in rows if r["density"] >= args.density_threshold]
    print(f"[audit] {len(suspects)}/{len(rows)} cameras have density >= {args.density_threshold} "
          f"in radius {args.density_radius}u (collapsed-cluster suspects)")

    top = suspects[: args.top_k]
    for rank, r in enumerate(top):
        src = (args.images_dir / r["filename"]).resolve()
        if not src.is_file():
            continue
        dst = suspects_dir / (
            f"rank{rank:02d}_dens{r['density']:03d}_spread{int(r['azimuth_spread_deg']):03d}_{r['filename']}"
        )
        try:
            dst.symlink_to(src)
        except OSError:
            pass
    print(f"[audit] symlinked top {len(top)} suspects to: {suspects_dir}")
    print(f"[audit] open in Finder:  open {suspects_dir}")

    print("\nTop 20 suspects (by camera-density):")
    print(f"{'rank':>4}  {'density':>7}  {'spread°':>7}  {'med_dist':>8}  {'filename'}")
    print("-" * 60)
    for i, r in enumerate(top[:20]):
        print(f"{i:>4}  {r['density']:>7}  {r['azimuth_spread_deg']:>7.0f}  {r['median_nearest_dist']:>8.2f}  {r['filename']}")


if __name__ == "__main__":
    main()
