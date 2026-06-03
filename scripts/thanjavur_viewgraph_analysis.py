"""Viewgraph analysis on a GTSFM run with no GT (e.g. Thanjavur).

Without GLOMAP-verified ground truth there's no precision/recall, but we CAN
characterize the actual viewgraph that came out of the frontend:

  1. Verifier pass rate as a function of MegaLoc score (calibration on this scene)
  2. Connected components of the post-verifier graph (fragmentation diagnosis)
  3. Per-camera degree distribution (which cameras are barely connected → at risk
     of getting dropped by the back-end)
  4. Spectral structure of the verified graph (does it factor into weakly-connected
     sub-communities → symptoms of disjoint hemispheres / view clusters)

Reads from a GTSFM run output dir. Defaults assume the F config layout.

Usage:
    python scripts/thanjavur_viewgraph_analysis.py --run-dir benchmark_results/thanjavur-300-test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_similarity(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",")


def load_verified_pairs(report_path: Path) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict]]:
    """Return (pairs_list, per_pair_metadata) for all post-verifier pairs."""
    with report_path.open() as f:
        entries = json.load(f)
    pairs = []
    meta = {}
    for e in entries:
        i, j = int(e["i1"]), int(e["i2"])
        a, b = (i, j) if i < j else (j, i)
        pairs.append((a, b))
        meta[(a, b)] = {
            "num_inliers": e.get("num_inliers_est_model", 0),
            "inlier_ratio": e.get("inlier_ratio_est_model", 0.0),
        }
    return pairs, meta


def calibration_curve_on_thanjavur(
    W: np.ndarray, verified_set: set[tuple[int, int]], n_bins: int = 20,
) -> list[dict]:
    """For each MegaLoc score bucket, what fraction of pairs were verified?"""
    iu = np.triu_indices(W.shape[0], k=1)
    pair_idx = list(zip(iu[0].tolist(), iu[1].tolist()))
    scores = W[iu]
    in_verified = np.array([(i, j) in verified_set for i, j in pair_idx])
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
        })
    return out


def connected_components(n_cams: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    """Union-find on the verified-pair graph."""
    parent = list(range(n_cams))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for i, j in edges:
        union(i, j)
    groups = defaultdict(set)
    for x in range(n_cams):
        groups[find(x)].add(x)
    return sorted(groups.values(), key=len, reverse=True)


def degree_distribution(n_cams: int, edges: list[tuple[int, int]]) -> dict[int, int]:
    """Per-camera count of verified neighbors."""
    deg = Counter()
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    # Include cameras with zero verified neighbors
    for c in range(n_cams):
        if c not in deg:
            deg[c] = 0
    return dict(deg)


def spectral_structure(W: np.ndarray, edges: list[tuple[int, int]]) -> dict:
    """Spectral 2-cluster the verified subgraph (weighted by inlier count would be ideal,
    but we use binary edges + MegaLoc scores as similarity for analysis)."""
    n = W.shape[0]
    A = np.zeros((n, n), dtype=float)
    for i, j in edges:
        s = max(W[i, j], 0.01)
        A[i, j] = s
        A[j, i] = s
    deg = A.sum(axis=1)
    keep = deg > 0
    if keep.sum() < 4:
        return {"note": "graph too sparse for spectral analysis"}
    A_sub = A[np.ix_(keep, keep)]
    deg_sub = A_sub.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg_sub))
    L_norm = np.eye(A_sub.shape[0]) - D_inv_sqrt @ A_sub @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L_norm)
    fiedler = eigvecs[:, 1]
    labels_sub = (fiedler >= 0).astype(int)
    labels = np.full(n, -1, dtype=int)
    labels[keep] = labels_sub
    n_a = int((labels == 0).sum())
    n_b = int((labels == 1).sum())
    n_iso = int((labels == -1).sum())
    # Cross-cluster verified edges (the bridges)
    n_cross = sum(1 for i, j in edges if labels[i] != labels[j] and labels[i] != -1 and labels[j] != -1)
    n_within = len(edges) - n_cross - sum(1 for i, j in edges if labels[i] == -1 or labels[j] == -1)
    return {
        "fiedler_eigenvalue": float(eigvals[1]),
        "cluster_a_size": n_a,
        "cluster_b_size": n_b,
        "isolated_cameras": n_iso,
        "within_cluster_verified_edges": n_within,
        "cross_cluster_verified_edges": n_cross,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="GTSFM run output dir (contains results/plots/similarity_matrix.txt + results/metrics/).")
    parser.add_argument("--n-cams", type=int, default=None,
                        help="Override camera count. If None, inferred from similarity matrix shape.")
    args = parser.parse_args()

    run = args.run_dir
    sim_path = run / "results" / "plots" / "similarity_matrix.txt"
    report_path = run / "results" / "metrics" / "two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    retriever_path = run / "results" / "metrics" / "retriever_metrics.json"
    if not sim_path.exists() or not report_path.exists():
        raise SystemExit(f"Missing files: {sim_path.exists()=} {report_path.exists()=}")

    W = load_similarity(sim_path)
    n_cams = args.n_cams or W.shape[0]
    pairs, meta = load_verified_pairs(report_path)
    verified_set = set(pairs)

    n_retrieved = json.load(retriever_path.open())["retriever_metrics"]["num_retrieved_image_pairs"]
    n_exhaustive = n_cams * (n_cams - 1) // 2

    print(f"=== Viewgraph audit: {run.name} ===")
    print(f"Cameras (similarity matrix): {n_cams}")
    print(f"Exhaustive pairs:            {n_exhaustive:,}")
    print(f"MegaLoc-retrieved pairs:     {int(n_retrieved):,}  ({100*n_retrieved/n_exhaustive:.1f}% of exhaustive)")
    print(f"Verified pairs (post-2view): {len(pairs):,}  ({100*len(pairs)/n_retrieved:.1f}% of retrieved, "
          f"{100*len(pairs)/n_exhaustive:.1f}% of exhaustive)")

    print("\n— Verifier inlier stats")
    inliers = np.array([m["num_inliers"] for m in meta.values()])
    ratios = np.array([m["inlier_ratio"] for m in meta.values()])
    print(f"  num_inliers: median={int(np.median(inliers))}, p25={int(np.percentile(inliers,25))}, p75={int(np.percentile(inliers,75))}, max={int(inliers.max())}")
    print(f"  inlier_ratio: median={np.median(ratios):.3f}, p25={np.percentile(ratios,25):.3f}, p75={np.percentile(ratios,75):.3f}")

    print("\n— Verifier pass rate by MegaLoc score bucket (CALIBRATION)")
    print("  score range  | candidate pairs | verified |  pass rate")
    print("  -------------|----------------:|---------:|----------:")
    for row in calibration_curve_on_thanjavur(W, verified_set):
        if row["count"] < 5:
            continue
        n_v = int(row["count"] * row["frac_verified"])
        print(f"  [{row['score_lo']:.2f}, {row['score_hi']:.2f}) | "
              f"{row['count']:>15,} | {n_v:>8,} | {100*row['frac_verified']:>9.1f}%")

    print("\n— Connected components of verified graph")
    comps = connected_components(n_cams, pairs)
    print(f"  {len(comps)} components; sizes (top 5): {[len(c) for c in comps[:5]]}")
    print(f"  largest component covers {len(comps[0])} / {n_cams} cameras ({100*len(comps[0])/n_cams:.1f}%)")
    isolated = [i for i, c in enumerate(comps) if len(c) == 1]
    if comps[-1] and len(comps[-1]) == 1:
        n_isolated = sum(1 for c in comps if len(c) == 1)
        print(f"  fully isolated cameras (degree 0): {n_isolated}")

    print("\n— Per-camera degree distribution")
    deg = degree_distribution(n_cams, pairs)
    deg_arr = np.array(list(deg.values()))
    print(f"  median={int(np.median(deg_arr))}, p10={int(np.percentile(deg_arr,10))}, p25={int(np.percentile(deg_arr,25))}, "
          f"p75={int(np.percentile(deg_arr,75))}, max={int(deg_arr.max())}")
    print(f"  cameras with ≤2 verified neighbors: {int((deg_arr <= 2).sum())} / {n_cams} (high drop risk)")
    print(f"  cameras with ≤5 verified neighbors: {int((deg_arr <= 5).sum())} / {n_cams}")
    print(f"  cameras with 0 verified neighbors:  {int((deg_arr == 0).sum())} / {n_cams}")

    print("\n— Spectral structure of verified graph")
    spec = spectral_structure(W, pairs)
    if "note" in spec:
        print(f"  {spec['note']}")
    else:
        print(f"  Fiedler eigenvalue: {spec['fiedler_eigenvalue']:.4f}  (smaller = more bimodal)")
        print(f"  Cluster A: {spec['cluster_a_size']} cams, Cluster B: {spec['cluster_b_size']} cams, isolated: {spec['isolated_cameras']}")
        print(f"  Within-cluster verified edges: {spec['within_cluster_verified_edges']:,}")
        print(f"  Cross-cluster verified edges:  {spec['cross_cluster_verified_edges']:,}")
        ratio = spec['cross_cluster_verified_edges'] / max(spec['within_cluster_verified_edges'], 1)
        print(f"  Cross/within ratio: {ratio:.3f}  (>0.3 = well-bridged, <0.1 = brittle)")


if __name__ == "__main__":
    main()
