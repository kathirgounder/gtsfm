"""MegaLoc score → geometric-verifiability calibration analysis.

Treats MegaLoc as a probabilistic verifiability oracle and asks:
  • At score s, what fraction of pairs ARE GLOMAP-verifiable? (calibration curve)
  • What score threshold guarantees ≥95% / ≥90% / ≥80% precision?
  • Where does precision break down on each scene?
  • Can we skip verification on high-confidence pairs to save compute?

Uses GLOMAP's verified two-view edges (colmap_shared.db two_view_geometries) as
ground truth for "is verifiable". This is a proxy — GLOMAP itself drops some
verifiable pairs — but it's the strongest signal we have without an oracle.

Outputs (per dataset):
  • Calibration curve (20 score bins → P(verified | bin), P(covisible | bin), count)
  • PR curve from pure score thresholds (no num_matched cap)
  • Wide 2D (num_matched, min_score) sweep
  • Adaptive-strategy table: trust ≥ T_high, verify [T_low, T_high), drop < T_low
  • Markdown report at notes/megaloc_pr_<dataset>.md

Cross-dataset summary at notes/megaloc_pr_summary.md.

Usage:
    python scripts/megaloc_pr_analysis.py                      # all datasets
    python scripts/megaloc_pr_analysis.py --datasets brussels  # one
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

import thirdparty.colmap.scripts.python.read_write_model as colmap_io


REPO = Path(__file__).resolve().parent.parent

# Probe known config dirs in priority order — first match wins.
CONFIG_DIRS = [
    "F-megaloc_sift_gp_single_pt",
    "Ft-megaloc_sift_gp_single_pt_trace",
]

DATASET_DIRS = {
    "brussels": "grand_place_brussels",
    "sacre_coeur": "sacre_coeur",
    "pantheon": "pantheon_exterior",
    "british_museum": "british_museum",
}


def discover_paths(dataset: str) -> Optional[dict[str, Path]]:
    """Find the similarity matrix + ground-truth files for a dataset, or None if missing."""
    canonical = DATASET_DIRS[dataset]
    sim_path = None
    for cfg in CONFIG_DIRS:
        for ds_alias in (dataset, canonical):
            p = REPO / "benchmark_results" / cfg / ds_alias / "results" / "plots" / "similarity_matrix.txt"
            if p.exists():
                sim_path = p
                break
        if sim_path:
            break
    if sim_path is None:
        return None
    glomap_dir = None
    for sub in ("full", "ba_no_retri", "gp_only"):
        cand = REPO / "benchmark_results" / "GLOMAP" / canonical / sub / "0"
        if (cand / "points3D.bin").exists():
            glomap_dir = cand
            break
    return {
        "similarity": sim_path,
        "colmap_db": REPO / "benchmarks" / canonical / "colmap_shared.db",
        "glomap_dir": glomap_dir,
        "images_dir": REPO / "benchmarks" / canonical / "images",
    }


def load_similarity_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",")


def get_image_fnames(images_dir: Path) -> list[str]:
    return sorted(p.name for p in images_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def load_glomap_verified_edges(colmap_db: Path, image_fnames: list[str]) -> set[tuple[int, int]]:
    """COLMAP's two_view_geometries table = edges that passed geometric verification.

    Each row's pair_id encodes (image_id1, image_id2) via the COLMAP convention:
      pair_id = image_id1 * MAX_IMAGE_ID + image_id2  (image_id1 < image_id2)
    where MAX_IMAGE_ID = 2^31 - 1.
    """
    if not colmap_db.exists():
        return set()
    MAX_IMAGE_ID = 2**31 - 1
    name_to_idx = {Path(fn).name: i for i, fn in enumerate(image_fnames)}
    edges: set[tuple[int, int]] = set()
    conn = sqlite3.connect(str(colmap_db))
    try:
        # image_id → name lookup from images table
        id_to_name = {row[0]: row[1] for row in conn.execute("SELECT image_id, name FROM images")}
        # `config` field: 0 == DEGENERATE / failed verification; non-zero == verified.
        # Without this filter we'd count attempted-but-failed pairs as verified, which
        # inflates the verified set to nearly exhaustive on dense scenes.
        for pair_id, cfg in conn.execute("SELECT pair_id, config FROM two_view_geometries"):
            if cfg == 0:
                continue
            id1 = pair_id // MAX_IMAGE_ID
            id2 = pair_id % MAX_IMAGE_ID
            n1, n2 = id_to_name.get(id1), id_to_name.get(id2)
            if n1 is None or n2 is None:
                continue
            i = name_to_idx.get(Path(n1).name)
            j = name_to_idx.get(Path(n2).name)
            if i is None or j is None or i == j:
                continue
            edges.add((min(i, j), max(i, j)))
    finally:
        conn.close()
    return edges


def load_glomap_covisibility(glomap_dir: Optional[Path], image_fnames: list[str]) -> set[tuple[int, int]]:
    """Pairs of cameras sharing at least one 3D track in GLOMAP's reconstruction."""
    if glomap_dir is None or not (glomap_dir / "points3D.bin").exists():
        return set()
    name_to_idx = {fn: i for i, fn in enumerate(image_fnames)}
    images = colmap_io.read_images_binary(str(glomap_dir / "images.bin"))
    points = colmap_io.read_points3D_binary(str(glomap_dir / "points3D.bin"))
    img_id_to_idx = {im_id: name_to_idx.get(im.name) for im_id, im in images.items()}
    edges: set[tuple[int, int]] = set()
    for pt in points.values():
        cams = sorted({img_id_to_idx.get(im_id) for im_id in pt.image_ids
                       if img_id_to_idx.get(im_id) is not None})
        for k in range(len(cams)):
            for l in range(k + 1, len(cams)):
                edges.add((cams[k], cams[l]))
    return edges


def calibration_curve(
    scores: np.ndarray, in_verified: np.ndarray, in_covisible: np.ndarray, n_bins: int = 20,
) -> list[dict]:
    """For each score bucket, fraction of pairs that are verified / covisible.

    Returns one dict per bucket: lo, hi, count, frac_verified, frac_covisible.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for k in range(n_bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        mask = (scores >= lo) & (scores < hi if k < n_bins - 1 else scores <= hi)
        n = int(mask.sum())
        out.append({
            "score_lo": lo,
            "score_hi": hi,
            "count": n,
            "frac_verified": float(in_verified[mask].mean()) if n else 0.0,
            "frac_covisible": float(in_covisible[mask].mean()) if n else 0.0,
        })
    return out


def threshold_sweep(
    scores: np.ndarray, in_verified: np.ndarray, n_thresholds: int = 41,
) -> list[dict]:
    """Pure score-threshold sweep — no num_matched cap. Returns precision + recall + count."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    n_total_verified = int(in_verified.sum())
    out = []
    for t in thresholds:
        keep = scores >= t
        kept = int(keep.sum())
        tp = int(in_verified[keep].sum())
        out.append({
            "threshold": float(t),
            "kept": kept,
            "verified_recall_pct": 100 * tp / max(n_total_verified, 1),
            "verified_precision_pct": 100 * tp / max(kept, 1),
        })
    return out


def threshold_at_precision(sweep: list[dict], target_precision: float) -> Optional[float]:
    """Lowest score threshold where precision ≥ target. None if never achieved."""
    for row in sweep:
        if row["verified_precision_pct"] >= target_precision and row["kept"] > 0:
            return row["threshold"]
    return None


def two_d_sweep(
    W: np.ndarray, verified_edges: set[tuple[int, int]],
    nm_grid: list[int], ms_grid: list[float],
) -> dict[tuple[int, float], dict]:
    """Wide 2D (num_matched, min_score) sweep."""
    N = W.shape[0]
    n_total_verified = max(len(verified_edges), 1)
    n_exhaustive = N * (N - 1) // 2
    out: dict[tuple[int, float], dict] = {}
    for nm in nm_grid:
        for ms in ms_grid:
            kept = compute_kept_pairs(W, nm, ms)
            n_kept = max(len(kept), 1)
            tp = len(kept & verified_edges)
            out[(nm, ms)] = {
                "num_pairs": len(kept),
                "pct_exhaustive": 100 * len(kept) / max(n_exhaustive, 1),
                "verified_recall_pct": 100 * tp / n_total_verified,
                "verified_precision_pct": 100 * tp / n_kept,
            }
    return out


def compute_kept_pairs(W: np.ndarray, num_matched: int, min_score: float) -> set[tuple[int, int]]:
    """Mirror of similarity_retriever.py's pair selection: kNN per query AND score ≥ min_score."""
    N = W.shape[0]
    Wf = W.copy()
    np.fill_diagonal(Wf, -np.inf)
    kept: set[tuple[int, int]] = set()
    for i in range(N):
        scores = Wf[i]
        if num_matched < N - 1:
            top = np.argpartition(-scores, num_matched)[:num_matched]
        else:
            top = np.arange(N)
            top = top[top != i]
        for j in top:
            if scores[j] >= min_score:
                kept.add((min(i, int(j)), max(i, int(j))))
    return kept


def adaptive_strategy_table(
    scores: np.ndarray, in_verified: np.ndarray, t_low_grid: list[float], t_high_grid: list[float],
) -> list[dict]:
    """For each (T_low, T_high) bucket: count pairs in {trust, verify, drop} bucket and the
    'true verifiability' rate of each. The trust bucket needs HIGH precision (we skip verif).
    The verify bucket carries cost but should have moderate precision (worth verifying).
    The drop bucket should have LOW precision (cheap to drop).
    """
    n_total = len(scores)
    n_total_verified = int(in_verified.sum())
    out = []
    for t_high in t_high_grid:
        for t_low in t_low_grid:
            if t_low >= t_high:
                continue
            trust = scores >= t_high
            verify = (scores >= t_low) & (scores < t_high)
            drop = scores < t_low
            n_trust, n_verify, n_drop = int(trust.sum()), int(verify.sum()), int(drop.sum())
            tp_trust = int(in_verified[trust].sum())
            tp_verify = int(in_verified[verify].sum())
            tp_drop = int(in_verified[drop].sum())
            recovered = tp_trust + tp_verify  # missed = tp_drop
            out.append({
                "t_low": t_low,
                "t_high": t_high,
                "n_trust": n_trust, "trust_precision_pct": 100 * tp_trust / max(n_trust, 1),
                "n_verify": n_verify, "verify_precision_pct": 100 * tp_verify / max(n_verify, 1),
                "n_drop": n_drop, "missed_verified": tp_drop,
                "recall_pct": 100 * recovered / max(n_total_verified, 1),
                "compute_savings_pct": 100 * (n_trust + n_drop) / max(n_total, 1),
            })
    return out


def write_dataset_report(dataset: str, paths: dict, results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"# MegaLoc PR analysis — {dataset}\n\n")
        f.write(f"- Images: **{results['n_images']}**\n")
        f.write(f"- Possible pairs: **{results['n_exhaustive']:,}**\n")
        f.write(f"- GLOMAP-verified pairs: **{results['n_verified']:,}**\n")
        f.write(f"- GT covisible pairs: **{results['n_covisible']:,}**\n")
        f.write(f"- Similarity matrix: `{paths['similarity'].relative_to(REPO)}`\n\n")

        f.write("## 1. Calibration curve — P(verified | score in bucket)\n\n")
        f.write("This is the central question: **does a high MegaLoc score guarantee verifiability?**\n\n")
        f.write("| Score bucket | # pairs | P(verified) | P(covisible) |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in results["calibration"]:
            if row["count"] == 0:
                continue
            f.write(f"| `[{row['score_lo']:.2f}, {row['score_hi']:.2f})` | "
                    f"{row['count']:,} | "
                    f"{100 * row['frac_verified']:.1f}% | "
                    f"{100 * row['frac_covisible']:.1f}% |\n")

        f.write("\n## 2. Pure threshold sweep (no num_matched cap)\n\n")
        f.write("How does precision/recall trade off as we vary the score threshold ALONE?\n\n")
        f.write("| Threshold | # kept | Verified recall | Verified precision |\n")
        f.write("|---:|---:|---:|---:|\n")
        for row in results["threshold_sweep"][::2]:  # every other for brevity
            f.write(f"| {row['threshold']:.2f} | {row['kept']:,} | "
                    f"{row['verified_recall_pct']:.1f}% | "
                    f"{row['verified_precision_pct']:.1f}% |\n")

        f.write("\n### Precision floors\n\n")
        for tgt in [99.0, 95.0, 90.0, 80.0, 70.0]:
            t = threshold_at_precision(results["threshold_sweep"], tgt)
            f.write(f"- Score threshold for ≥ **{tgt:.0f}%** verified-precision: "
                    f"{'`%.2f`' % t if t is not None else '**unreachable**'}\n")

        f.write("\n## 3. 2D sweep — (num_matched × min_score)\n\n")
        f.write("Cell shows `verified_recall% / verified_precision% / #kept-pairs (% of exhaustive)`.\n\n")
        ms_grid = sorted({k[1] for k in results["sweep_2d"]})
        nm_grid = sorted({k[0] for k in results["sweep_2d"]})
        header = "| nm \\ ms |" + "|".join(f" {ms:.2f} " for ms in ms_grid) + "|\n"
        sep = "|---|" + "|".join("---:" for _ in ms_grid) + "|\n"
        f.write(header); f.write(sep)
        for nm in nm_grid:
            row_cells = []
            for ms in ms_grid:
                cell = results["sweep_2d"][(nm, ms)]
                row_cells.append(
                    f" {cell['verified_recall_pct']:.0f}/{cell['verified_precision_pct']:.0f}/"
                    f"{cell['num_pairs']:,} ({cell['pct_exhaustive']:.0f}%) "
                )
            f.write(f"| {nm} |" + "|".join(row_cells) + "|\n")

        f.write("\n## 4. Adaptive verification strategy\n\n")
        f.write("`trust ≥ T_high` (skip verifier), `verify ∈ [T_low, T_high)` (run verifier), "
                "`drop < T_low` (cheap reject). Goal: high trust precision (≥95%), "
                "moderate verify precision, low drop precision (≪20%).\n\n")
        f.write("| T_low | T_high | #trust (prec) | #verify (prec) | #drop (missed verif) | "
                "Recall | Compute saved |\n")
        f.write("|---|---|---|---|---|---:|---:|\n")
        for row in results["adaptive"]:
            f.write(
                f"| {row['t_low']:.2f} | {row['t_high']:.2f} | "
                f"{row['n_trust']:,} ({row['trust_precision_pct']:.0f}%) | "
                f"{row['n_verify']:,} ({row['verify_precision_pct']:.0f}%) | "
                f"{row['n_drop']:,} ({row['missed_verified']:,}) | "
                f"{row['recall_pct']:.1f}% | "
                f"{row['compute_savings_pct']:.0f}% |\n"
            )


def write_summary(per_dataset: dict[str, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# MegaLoc PR analysis — cross-dataset summary\n\n")
        f.write("Score thresholds at fixed precision floors (lower = better — means MegaLoc "
                "is more discriminative on this scene):\n\n")
        f.write("| Dataset | N | Verified pairs | T@99% | T@95% | T@90% | T@80% | T@70% |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for ds, r in per_dataset.items():
            sweep = r["threshold_sweep"]
            cells = [
                "%.2f" % t if (t := threshold_at_precision(sweep, p)) is not None else "—"
                for p in [99.0, 95.0, 90.0, 80.0, 70.0]
            ]
            f.write(f"| {ds} | {r['n_images']} | {r['n_verified']:,} | "
                    + " | ".join(cells) + " |\n")
        f.write("\n## Reading the table\n\n")
        f.write("- `T@95%` = lowest score threshold where ≥95% of kept pairs are GLOMAP-verified.\n")
        f.write("- A LOW `T@95%` means MegaLoc scores are well-calibrated on this scene "
                "(score acts as a reliable verifiability oracle even at moderate values).\n")
        f.write("- A HIGH `T@95%` means even high MegaLoc scores have many unverifiable pairs — "
                "the scene has visual aliases / repetitive structure.\n")
        f.write("- `—` means precision never reaches the target at any threshold.\n\n")

        f.write("## Pairs/recall in current production configs (reference)\n\n")
        f.write("| Dataset | Current (nm, ms) | recall | precision | #kept | %exhaustive |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        production_configs = {
            "brussels": (100, 0.15),
            "sacre_coeur": (100, 0.25),
            "pantheon": (100, 0.45),
            "british_museum": (75, 0.20),
        }
        for ds, r in per_dataset.items():
            nm, ms = production_configs.get(ds, (100, 0.20))
            cell = r["sweep_2d"].get((nm, ms))
            if cell is None:
                # find nearest available
                continue
            f.write(f"| {ds} | ({nm}, {ms:.2f}) | "
                    f"{cell['verified_recall_pct']:.1f}% | "
                    f"{cell['verified_precision_pct']:.1f}% | "
                    f"{cell['num_pairs']:,} | "
                    f"{cell['pct_exhaustive']:.0f}% |\n")


def analyze_one(dataset: str, paths: dict) -> dict:
    print(f"\n=== {dataset} ===")
    W = load_similarity_matrix(paths["similarity"])
    N = W.shape[0]
    image_fnames = get_image_fnames(paths["images_dir"])
    if len(image_fnames) != N:
        print(f"  ⚠️  loader image count {len(image_fnames)} ≠ matrix N {N} — index mapping may be wrong.")
    verified = load_glomap_verified_edges(paths["colmap_db"], image_fnames)
    covisible = load_glomap_covisibility(paths["glomap_dir"], image_fnames)
    print(f"  N={N}, verified={len(verified):,}, covisible={len(covisible):,}")

    iu = np.triu_indices(N, k=1)
    pair_idx = list(zip(iu[0].tolist(), iu[1].tolist()))
    scores = W[iu]
    in_verified = np.array([(i, j) in verified for i, j in pair_idx])
    in_covisible = np.array([(i, j) in covisible for i, j in pair_idx])

    calibration = calibration_curve(scores, in_verified, in_covisible, n_bins=20)
    threshold_sweep_data = threshold_sweep(scores, in_verified, n_thresholds=41)
    nm_grid = [25, 50, 75, 100, 150, 200, 300, 500]
    ms_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    sweep_2d = two_d_sweep(W, verified, nm_grid, ms_grid)
    t_low_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    t_high_grid = [0.40, 0.50, 0.60, 0.70]
    adaptive = adaptive_strategy_table(scores, in_verified, t_low_grid, t_high_grid)

    return {
        "n_images": N,
        "n_exhaustive": N * (N - 1) // 2,
        "n_verified": len(verified),
        "n_covisible": len(covisible),
        "calibration": calibration,
        "threshold_sweep": threshold_sweep_data,
        "sweep_2d": sweep_2d,
        "adaptive": adaptive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_DIRS),
                        help="Subset of datasets to analyze.")
    parser.add_argument("--out-dir", type=Path, default=REPO / "notes",
                        help="Where to write per-dataset reports + summary.")
    args = parser.parse_args()

    per_dataset = {}
    paths_by_ds = {}
    for ds in args.datasets:
        if ds not in DATASET_DIRS:
            print(f"⚠️  unknown dataset '{ds}', skipping.")
            continue
        paths = discover_paths(ds)
        if paths is None:
            print(f"⚠️  no similarity matrix for '{ds}' — skip. Run F config first.")
            continue
        paths_by_ds[ds] = paths
        per_dataset[ds] = analyze_one(ds, paths)
        out_path = args.out_dir / f"megaloc_pr_{ds}.md"
        write_dataset_report(ds, paths, per_dataset[ds], out_path)
        print(f"  wrote {out_path.relative_to(REPO)}")

    if per_dataset:
        summary_path = args.out_dir / "megaloc_pr_summary.md"
        write_summary(per_dataset, summary_path)
        print(f"\nwrote summary: {summary_path.relative_to(REPO)}")
    else:
        print("\nNo datasets analyzed (no similarity matrices found).")


if __name__ == "__main__":
    main()
