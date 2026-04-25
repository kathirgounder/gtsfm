"""Deep dive on MegaLoc retrieval scores. Generalizes Brussels-only analysis.

Usage:
    python scripts/megaloc_deep_dive.py --dataset brussels
    python scripts/megaloc_deep_dive.py --dataset british_museum --num-matched 100
    python scripts/megaloc_deep_dive.py --dataset pantheon

Auto-discovers per-dataset:
  similarity matrix  → benchmark_results/F-megaloc_sift_gp_single_pt/<ds>/results/plots/similarity_matrix.txt
  GLOMAP recon       → benchmark_results/GLOMAP/<ds_canonical>/{full,ba_no_retri,gp_only}/0/
  COLMAP shared DB   → benchmarks/<ds_canonical>/colmap_shared.db
  image dir          → benchmarks/<ds_canonical>/images/

Outputs:
  - Console summary (score distribution, recall sweep, cross-cluster survival).
  - benchmark_results/F-megaloc_sift_gp_single_pt/<ds>/results/plots/megaloc_pr_curves.png
  - benchmark_results/F-megaloc_sift_gp_single_pt/<ds>/results/plots/megaloc_cross_cluster.png

The cross-cluster analysis is the key piece for catching the Brussels-style
bimodal-basin boost: spectral 2-cluster on GLOMAP-full covisibility, then sweep
min_score to surface where cross-cluster verified edges get cut.

Authors: Kathir Gounder
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import thirdparty.colmap.scripts.python.read_write_model as colmap_io


REPO = Path(__file__).resolve().parent.parent

# Dataset name aliases — user-facing → canonical (matches benchmarks/ + GLOMAP/ paths).
DATASET_ALIASES: dict[str, str] = {
    "brussels": "grand_place_brussels",
    "british_museum": "british_museum",
    "pantheon": "pantheon_exterior",
    "sacre_coeur": "sacre_coeur",
}


def resolve_dataset(name: str) -> str:
    canonical = DATASET_ALIASES.get(name, name)
    if not (REPO / "benchmarks" / canonical).exists():
        raise FileNotFoundError(
            f"Dataset '{name}' (canonical '{canonical}') not found under benchmarks/. "
            f"Known aliases: {list(DATASET_ALIASES)}"
        )
    return canonical


def discover_paths(dataset: str, config_name: str) -> dict[str, Path]:
    canonical = resolve_dataset(dataset)
    plots_dir = REPO / "benchmark_results" / config_name / dataset / "results" / "plots"

    # GLOMAP recon: prefer full > ba_no_retri > gp_only.
    glomap_root = REPO / "benchmark_results" / "GLOMAP" / canonical
    glomap_dir = None
    for sub in ("full", "ba_no_retri", "gp_only"):
        candidate = glomap_root / sub / "0"
        if (candidate / "points3D.bin").exists():
            glomap_dir = candidate
            break

    return {
        "similarity_txt": plots_dir / "similarity_matrix.txt",
        "plots_dir": plots_dir,
        "colmap_db": REPO / "benchmarks" / canonical / "colmap_shared.db",
        "glomap_dir": glomap_dir,
        "images_dir": REPO / "benchmarks" / canonical / "images",
    }


def load_similarity_matrix(path: Path) -> np.ndarray:
    W = np.loadtxt(path, delimiter=",")
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def get_image_fnames(images_dir: Path) -> list[str]:
    """Mirrors GTSFM's Olsson loader: sorted list of image basenames."""
    return sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_glomap_full_covisibility(glomap_dir: Path | None, image_fnames: list[str]) -> set[tuple[int, int]]:
    if glomap_dir is None:
        return set()
    cameras, images, points3d = colmap_io.read_model(path=str(glomap_dir), ext=".bin")
    fname_to_loader = {f: i for i, f in enumerate(image_fnames)}
    colmap_to_loader = {img.id: fname_to_loader[Path(img.name).name]
                        for img in images.values() if Path(img.name).name in fname_to_loader}
    edges: set[tuple[int, int]] = set()
    for p in points3d.values():
        loader_ids = sorted({colmap_to_loader[i] for i in p.image_ids if i in colmap_to_loader})
        for i, j in combinations(loader_ids, 2):
            edges.add((i, j))
    return edges


def load_glomap_verified_edges(colmap_db: Path, image_fnames: list[str]) -> set[tuple[int, int]]:
    if not colmap_db.exists():
        return set()
    fname_to_loader = {f: i for i, f in enumerate(image_fnames)}
    conn = sqlite3.connect(str(colmap_db))
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    db_to_loader = {db_id: fname_to_loader[Path(name).name]
                    for db_id, name in cur.fetchall() if Path(name).name in fname_to_loader}
    cur.execute("SELECT pair_id, config FROM two_view_geometries")
    edges: set[tuple[int, int]] = set()
    for pair_id, cfg in cur.fetchall():
        if cfg == 0:
            continue
        img1 = int(pair_id) // 2147483647
        img2 = int(pair_id) - img1 * 2147483647
        if img1 in db_to_loader and img2 in db_to_loader:
            a, b = db_to_loader[img1], db_to_loader[img2]
            if a != b:
                edges.add((min(a, b), max(a, b)))
    conn.close()
    return edges


def spectral_2cluster(W: np.ndarray, edges: set[tuple[int, int]] | None = None) -> np.ndarray:
    """Fiedler-vector clustering. If `edges` is given, builds the affinity matrix
    from just those edges (e.g. GLOMAP-full covisibility) — gives the cleanest
    'true' cluster split. Else uses the dense MegaLoc matrix.
    """
    N = W.shape[0]
    if edges is not None:
        A = np.zeros((N, N), dtype=np.float64)
        for i, j in edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
    else:
        A = W.copy()
        np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    L = np.diag(deg) - A
    Dinv_sqrt = np.diag(1.0 / np.sqrt(deg_safe))
    Lnorm = Dinv_sqrt @ L @ Dinv_sqrt
    eigvals, eigvecs = np.linalg.eigh(Lnorm)
    fiedler = eigvecs[:, 1]
    # Standard spectral cut: sign of the Fiedler vector. Splits at the natural
    # community boundary (not 50/50 like a median split would force).
    labels = (fiedler > 0).astype(int)
    # Guard against degenerate cases (all-same-sign): fall back to median.
    if labels.min() == labels.max():
        labels = (fiedler > np.median(fiedler)).astype(int)
    return labels


def compute_kept_pairs(W: np.ndarray, num_matched: int, min_score: float) -> set[tuple[int, int]]:
    """Mirror the production retriever: top-K per node, then min_score cut."""
    N = W.shape[0]
    kept: set[tuple[int, int]] = set()
    for i in range(N):
        row = W[i].copy()
        row[i] = -np.inf
        top_k_idx = np.argpartition(-row, min(num_matched, N - 1))[:num_matched]
        for j_idx in top_k_idx:
            j = int(j_idx)
            if row[j] >= min_score and j != i:
                a, b = (min(i, j), max(i, j))
                kept.add((a, b))
    return kept


def histo_summary(scores: np.ndarray, label: str) -> None:
    if scores.size == 0:
        print(f"  {label}: (none)")
        return
    pct = lambda q: float(np.quantile(scores, q))
    print(
        f"  {label}: n={scores.size}, "
        f"min={scores.min():.3f}, p10={pct(0.10):.3f}, p25={pct(0.25):.3f}, "
        f"median={pct(0.50):.3f}, p75={pct(0.75):.3f}, max={scores.max():.3f}"
    )


def plot_pr_curves(
    W: np.ndarray,
    verified_edges: set[tuple[int, int]],
    covis_edges: set[tuple[int, int]],
    num_matched: int,
    out_path: Path,
    dataset: str,
) -> None:
    sweep = np.arange(0.05, 0.41, 0.025)
    rows = []
    for ms in sweep:
        kept = compute_kept_pairs(W, num_matched, float(ms))
        n = max(len(kept), 1)
        v_inter = len(kept & verified_edges)
        c_inter = len(kept & covis_edges)
        rows.append({
            "min_score": float(ms),
            "n_kept": len(kept),
            "v_precision": v_inter / n,
            "v_recall": v_inter / max(len(verified_edges), 1),
            "c_precision": c_inter / n,
            "c_recall": c_inter / max(len(covis_edges), 1),
        })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    xs = [r["min_score"] for r in rows]
    ax.plot(xs, [r["v_recall"] for r in rows], "o-", label="recall vs GLOMAP-verified", color="tab:blue")
    ax.plot(xs, [r["v_precision"] for r in rows], "s--", label="precision vs GLOMAP-verified", color="tab:blue", alpha=0.5)
    ax.plot(xs, [r["c_recall"] for r in rows], "o-", label="recall vs GT covisible", color="tab:orange")
    ax.plot(xs, [r["c_precision"] for r in rows], "s--", label="precision vs GT covisible", color="tab:orange", alpha=0.5)
    ax.set_xlabel("min_score")
    ax.set_ylabel("rate")
    ax.set_title(f"{dataset}: P/R vs min_score (num_matched={num_matched})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    ax2 = axes[1]
    ax2.plot([r["v_recall"] for r in rows], [r["v_precision"] for r in rows], "o-",
             color="tab:blue", label="vs GLOMAP-verified")
    ax2.plot([r["c_recall"] for r in rows], [r["c_precision"] for r in rows], "o-",
             color="tab:orange", label="vs GT covisible")
    for r in rows:
        if abs(r["min_score"] - 0.15) < 1e-3 or abs(r["min_score"] - 0.20) < 1e-3 or abs(r["min_score"] - 0.25) < 1e-3:
            ax2.annotate(f"{r['min_score']:.2f}", (r["v_recall"], r["v_precision"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points", color="tab:blue")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title("PR curve (annotated by min_score at 0.15, 0.20, 0.25)")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → wrote {out_path.relative_to(REPO)}")


def plot_cross_cluster_survival(
    W: np.ndarray,
    verified_edges: set[tuple[int, int]],
    covis_edges: set[tuple[int, int]],
    labels: np.ndarray,
    num_matched: int,
    out_path: Path,
    dataset: str,
) -> None:
    """For each min_score, count how many cross-cluster verified/covisible edges
    survive both the top-K cut and the threshold."""
    sweep = np.arange(0.05, 0.41, 0.025)
    cross_verified_total = sum(1 for i, j in verified_edges if labels[i] != labels[j])
    cross_covis_total = sum(1 for i, j in covis_edges if labels[i] != labels[j])

    cross_v_survival = []
    cross_c_survival = []
    n_kept_list = []
    for ms in sweep:
        kept = compute_kept_pairs(W, num_matched, float(ms))
        cv = sum(1 for e in kept & verified_edges if labels[e[0]] != labels[e[1]])
        cc = sum(1 for e in kept & covis_edges if labels[e[0]] != labels[e[1]])
        cross_v_survival.append(cv / max(cross_verified_total, 1))
        cross_c_survival.append(cc / max(cross_covis_total, 1))
        n_kept_list.append(len(kept))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep, cross_v_survival, "o-", label=f"cross-cluster verified ({cross_verified_total} total)",
            color="tab:red")
    ax.plot(sweep, cross_c_survival, "s-", label=f"cross-cluster covisible ({cross_covis_total} total)",
            color="tab:purple")
    ax.set_xlabel("min_score")
    ax.set_ylabel("survival rate")
    ax.set_title(f"{dataset}: cross-cluster bridge survival (num_matched={num_matched})")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    for ms_target in (0.15, 0.20, 0.25):
        idx = int(np.argmin(np.abs(sweep - ms_target)))
        ax.annotate(f"{cross_v_survival[idx]:.0%}", (sweep[idx], cross_v_survival[idx]),
                    fontsize=8, xytext=(0, 8), textcoords="offset points", ha="center", color="tab:red")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → wrote {out_path.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. brussels, british_museum, pantheon).")
    parser.add_argument("--num-matched", type=int, default=100, help="kNN cap (default 100).")
    parser.add_argument("--config", default="F-megaloc_sift_gp_single_pt",
                        help="Config dir under benchmark_results/ (default F-megaloc_sift_gp_single_pt).")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing PNG plots.")
    args = parser.parse_args()

    paths = discover_paths(args.dataset, args.config)
    print(f"Dataset: {args.dataset}")
    for k, v in paths.items():
        existence = "✓" if (v is not None and v.exists()) else "✗"
        print(f"  [{existence}] {k}: {v}")
    if not paths["similarity_txt"].exists():
        raise FileNotFoundError(
            f"\nNo similarity matrix found at {paths['similarity_txt']}.\n"
            f"Run the {args.config} config on '{args.dataset}' first."
        )

    print(f"\nLoading similarity matrix ...")
    W = load_similarity_matrix(paths["similarity_txt"])
    N = W.shape[0]
    print(f"  shape: {W.shape}, nonzero edges: {(W > 0).sum() // 2}")

    image_fnames = get_image_fnames(paths["images_dir"])
    if len(image_fnames) != N:
        print(f"  ⚠️  loader image count {len(image_fnames)} ≠ matrix N {N} — index mapping may be wrong.")

    print("Loading GLOMAP-verified pose-graph edges ...")
    verified_edges = load_glomap_verified_edges(paths["colmap_db"], image_fnames)
    print(f"  {len(verified_edges)} verified pairs")

    print("Loading GT covisibility from GLOMAP reconstruction ...")
    covis_edges = load_glomap_full_covisibility(paths["glomap_dir"], image_fnames)
    print(f"  {len(covis_edges)} covisible pairs")

    iu = np.triu_indices(N, k=1)
    pair_idx = list(zip(iu[0].tolist(), iu[1].tolist()))
    scores = W[iu]
    in_verified = np.array([(i, j) in verified_edges for i, j in pair_idx])
    in_covis = np.array([(i, j) in covis_edges for i, j in pair_idx])

    print(f"\n=== 1. Score distribution per category ===")
    histo_summary(scores, "all pairs")
    histo_summary(scores[in_verified], "GLOMAP-verified")
    histo_summary(scores[in_covis], "GT covisible")
    histo_summary(scores[~in_verified & ~in_covis], "neither (noise)")

    exhaustive_pairs = N * (N - 1) // 2
    print(f"\n=== 2. Recall sweep at num_matched={args.num_matched} (GLOMAP exhaustive baseline = {exhaustive_pairs} pairs) ===")
    print(f"  {'min_score':<10} {'#pairs':>8} {'%exh':>7} {'verif_rec':>10} {'covis_rec':>10} {'verif_prec':>11} {'covis_prec':>11}")
    print(f"  {'-' * 76}")
    for ms in [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35]:
        kept = compute_kept_pairs(W, args.num_matched, ms)
        n = max(len(kept), 1)
        pct_exh = 100 * len(kept) / max(exhaustive_pairs, 1)
        v_rec = (len(kept & verified_edges) / max(len(verified_edges), 1)) * 100
        c_rec = (len(kept & covis_edges) / max(len(covis_edges), 1)) * 100
        v_prec = (len(kept & verified_edges) / n) * 100
        c_prec = (len(kept & covis_edges) / n) * 100
        print(f"  {ms:<10.2f} {len(kept):>8d} {pct_exh:>6.1f}% {v_rec:>9.1f}% {c_rec:>9.1f}% {v_prec:>10.1f}% {c_prec:>10.1f}%")

    # 2D joint sweep: (num_matched, min_score) → verified recall + efficiency.
    print(f"\n=== 2b. 2D sweep: num_matched × min_score ===")
    print(f"  Matrix cells = '<#pairs> (<%exh>) | verif_rec=<r>%'")
    nm_grid = [25, 50, 75, 100, 150, 200]
    ms_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    header = "num_matched ↓ / min_score →" + "".join(f"   ms={ms:.2f}".rjust(20) for ms in ms_grid)
    print(f"  {header}")
    for nm in nm_grid:
        row_cells = []
        for ms in ms_grid:
            kept = compute_kept_pairs(W, nm, ms)
            v_rec = (len(kept & verified_edges) / max(len(verified_edges), 1)) * 100
            pct_exh = 100 * len(kept) / max(exhaustive_pairs, 1)
            row_cells.append(f"{len(kept):>5d} ({pct_exh:>4.1f}%) {v_rec:>4.1f}%")
        print(f"  nm={nm:<3d}                  " + "  ".join(row_cells))

    # Run cross-cluster analysis with TWO clustering targets:
    #   (a) GLOMAP-verified pairs — sparser, only direct geometric verifications
    #   (b) GLOMAP-full covisibility — denser, includes transitive covis through 3D points
    # The two splits typically agree on coarse community structure but differ on
    # which "marginal" cameras land on which side; the Brussels-style boost is
    # consistent if both clusterings flag the same low-score-tail problem.
    cluster_targets: list[tuple[str, set[tuple[int, int]]]] = []
    if verified_edges:
        cluster_targets.append(("GLOMAP-verified two-view graph", verified_edges))
    if covis_edges:
        cluster_targets.append(("GLOMAP-full reconstruction covisibility", covis_edges))

    cross_cluster_results: dict[str, np.ndarray] = {}  # source → labels
    for cluster_label, cluster_edges in cluster_targets:
        section_idx = len(cross_cluster_results) + 3
        print(f"\n=== {section_idx}. Cross-cluster analysis (spectral 2-cluster on {cluster_label}) ===")
        labels = spectral_2cluster(W, edges=cluster_edges)
        cross_cluster_results[cluster_label] = labels
        sz_a = int((labels == 0).sum())
        sz_b = int((labels == 1).sum())
        print(f"  cluster sizes: A={sz_a}, B={sz_b}")

        cross_mask = np.array([labels[i] != labels[j] for i, j in pair_idx])
        within_mask = ~cross_mask
        print(f"\n  Score distribution by cluster relationship:")
        histo_summary(scores[within_mask & in_verified], "within-cluster + verified")
        histo_summary(scores[cross_mask & in_verified], "cross-cluster + verified  ← critical")

        cross_verified_idx = np.where(cross_mask & in_verified)[0]
        cross_verified_scores = scores[cross_verified_idx]
        print(f"\n  Cross-cluster verified bridge survival (at num_matched={args.num_matched}):")
        for thresh in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
            survive = (cross_verified_scores >= thresh).sum()
            pct = (survive / max(cross_verified_idx.size, 1)) * 100
            print(f"    at min_score={thresh:.2f}: {survive}/{cross_verified_idx.size} survive ({pct:.1f}%)")

        # Boost-opportunity flag — calibrated against Brussels:
        # Brussels lost 14.5% of cross-cluster verified at min_score=0.20, gained
        # +14 AUC@5 by dropping to 0.15. Threshold of 10% catches that pattern.
        bridges_below_020 = int((cross_verified_scores < 0.20).sum())
        bridges_below_015 = int((cross_verified_scores < 0.15).sum())
        bridges_below_010 = int((cross_verified_scores < 0.10).sum())
        bridges_below_025 = int((cross_verified_scores < 0.25).sum())
        cv_loss_at_020 = bridges_below_020 / max(cross_verified_idx.size, 1)
        print(f"\n  Recoverable cross-cluster verified bridges by dropping min_score:")
        print(f"    0.25 → 0.20: +{bridges_below_025 - bridges_below_020} bridges")
        print(f"    0.20 → 0.15: +{bridges_below_020 - bridges_below_015} bridges")
        print(f"    0.15 → 0.10: +{bridges_below_015 - bridges_below_010} bridges")
        if cv_loss_at_020 > 0.10:
            print(f"\n  🔔 BOOST OPPORTUNITY: at num_matched={args.num_matched}, min_score=0.20 cuts "
                  f"{cv_loss_at_020 * 100:.1f}% ({bridges_below_020}) cross-cluster verified bridges.")
            print(f"     Dropping to 0.15 recovers {bridges_below_020 - bridges_below_015}. "
                  f"Brussels saw +14 AUC@5 from this exact change.")
        else:
            print(f"\n  ℹ️  min_score=0.20 cuts only {cv_loss_at_020 * 100:.1f}% of cross-cluster verified — already healthy.")

    # Cluster-label agreement between the two clusterings (sanity check).
    if len(cross_cluster_results) == 2:
        keys = list(cross_cluster_results.keys())
        a = cross_cluster_results[keys[0]]
        b = cross_cluster_results[keys[1]]
        # Account for the binary label ambiguity (0/1 vs 1/0).
        agree_direct = (a == b).sum() / N
        agree_flipped = (a != b).sum() / N
        agreement = max(agree_direct, agree_flipped)
        print(f"\n=== Cluster-labelling agreement between the two targets ===")
        print(f"  {agreement * 100:.1f}% of cameras land in the same community.")
        if agreement > 0.90:
            print(f"  → Strong agreement. The two cluster splits identify the same scene structure.")
        else:
            print(f"  → Disagreement on >10% of cameras. Inspect both cross-cluster reports.")

    if not args.no_plots:
        print(f"\n=== 4. Writing plots ===")
        paths["plots_dir"].mkdir(parents=True, exist_ok=True)
        plot_pr_curves(W, verified_edges, covis_edges, args.num_matched,
                       paths["plots_dir"] / "megaloc_pr_curves.png", args.dataset)
        # Plot cross-cluster survival once per clustering source.
        for cluster_label, cluster_labels in cross_cluster_results.items():
            slug = "covis" if "covisibility" in cluster_label else "verified"
            plot_cross_cluster_survival(W, verified_edges, covis_edges, cluster_labels, args.num_matched,
                                        paths["plots_dir"] / f"megaloc_cross_cluster_{slug}.png",
                                        f"{args.dataset} ({cluster_label})")


if __name__ == "__main__":
    main()
