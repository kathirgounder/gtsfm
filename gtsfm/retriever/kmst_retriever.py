"""Retriever that augments kNN pair selection with multi-MST connectivity edges.

Extends SimilarityRetriever with a tiered-threshold strategy:
  - Strong threshold (min_score, e.g. 0.4): kNN selects high-precision pairs
  - Weak floor (mst_min_score, e.g. 0.25): k MSTs add structurally important
    bridge edges that guarantee global connectivity

This prevents the graph fragmentation that causes METIS to amputate scene
regions when the strong threshold is too aggressive. On datasets like Tower
of London, the bridge/fort/walkway may have cross-region MegaLoc similarities
in the 0.25-0.4 range — too low for kNN but structurally critical. The MST
is forced to include these bridge edges because otherwise it's not spanning.

The MST edges are a small, bounded addition: at most k*(N-1) edges total,
and most overlap with kNN pairs. The actual number of MST-only edges is
typically just the handful of bridge edges between weakly connected regions.

Authors: Kathir Gounder
"""

import os
from pathlib import Path
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.similarity_retriever import SimilarityRetriever

logger = logger_utils.get_logger()


class _UnionFind:
    """Disjoint-set / union-find with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Returns True if x and y were in different components (merge happened)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def _build_mst_kruskal(
    edges: List[Tuple[int, int]],
    weights: np.ndarray,
    num_nodes: int,
    masked: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Build a minimum spanning forest via Kruskal's algorithm.

    Args:
        edges: List of (i, j) candidate edges with i < j.
        weights: 1-D array of edge weights, same length as edges.
        num_nodes: Total number of nodes in the graph.
        masked: Set of edges to skip (already selected in previous MSTs).

    Returns:
        List of edges in the MST (forest if graph is disconnected).
    """
    order = np.argsort(weights, kind="stable")
    uf = _UnionFind(num_nodes)
    mst_edges: List[Tuple[int, int]] = []

    for idx in order:
        e = edges[idx]
        if e in masked:
            continue
        if np.isinf(weights[idx]):
            continue
        i, j = e
        if uf.union(i, j):
            mst_edges.append(e)
            if len(mst_edges) == num_nodes - 1:
                break

    return mst_edges


class KMstRetriever(SimilarityRetriever):
    """kNN + multi-MST retriever with tiered thresholds.

    Augments the parent SimilarityRetriever's kNN pair selection with
    connectivity edges from k minimum spanning trees. The kNN pairs use
    a strong similarity threshold (min_score) for high precision, while
    the MSTs use a weaker floor (mst_min_score) to bridge regions that
    the strong threshold would disconnect.

    This is designed for the VGGT hierarchical merging pipeline where
    METIS drops all components except the largest — MST edges prevent
    scene regions from being amputated.
    """

    def __init__(
        self,
        num_matched: int = 20,
        min_score: float = 0.4,
        mst_min_score: float = 0.25,
        num_msts: int = 2,
        blocksize: int = 50,
    ) -> None:
        """
        Args:
            num_matched: Number of top-K matches per query for kNN selection.
            min_score: Strong similarity threshold for kNN pairs (high precision).
            mst_min_score: Weak floor for MST candidate edges. Edges with similarity
                           in [mst_min_score, min_score) can be selected by the MST
                           for connectivity but would not be selected by kNN.
            num_msts: Number of MSTs to build. k=2 provides redundancy so no
                      single edge failure disconnects the graph.
            blocksize: Block size for similarity matrix computation (inherited).
        """
        super().__init__(num_matched=num_matched, min_score=min_score, blocksize=blocksize)
        self._mst_min_score = mst_min_score
        self._num_msts = num_msts
        self._mst_edge_counts: List[int] = []
        self._knn_pair_count: int = 0
        self._mst_only_pair_count: int = 0

    def __repr__(self) -> str:
        return (
            f"KMstRetriever(num_matched={self._num_matched}, min_score={self._min_score}, "
            f"mst_min_score={self._mst_min_score}, num_msts={self._num_msts})"
        )

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Compute image pairs via kNN + multi-MST with tiered thresholds.

        1. Compute similarity matrix (inherited).
        2. Get kNN pairs using the strong threshold (min_score).
        3. Build k MSTs over the wider candidate pool (mst_min_score).
        4. Return the union of both sets.
        """
        if global_descriptors is None:
            raise ValueError("Global descriptors need to be provided")

        sim = self.compute_similarity_matrix(global_descriptors)
        self._latest_similarity_matrix = sim.detach().cpu().clone()

        # Step 1: kNN pairs using strong threshold.
        # Must clone sim because compute_pairs_from_similarity_matrix mutates it in-place.
        knn_pairs = self.compute_pairs_from_similarity_matrix(
            sim.clone(), image_fnames, plots_output_dir=None
        )

        # Step 2: MST connectivity edges using weak floor.
        mst_edges = self._build_mst_edges(sim.cpu().numpy())

        # Step 3: Union, deduplicate, sort.
        pair_set = set()
        for i, j in knn_pairs:
            pair_set.add((min(i, j), max(i, j)))
        knn_set = set(pair_set)  # snapshot before adding MST edges

        for i, j in mst_edges:
            pair_set.add((min(i, j), max(i, j)))

        all_pairs = sorted(pair_set)

        self._knn_pair_count = len(knn_set)
        self._mst_only_pair_count = len(pair_set - knn_set)

        logger.info(
            "KMstRetriever: %d kNN pairs + %d MST-only pairs = %d total.",
            self._knn_pair_count, self._mst_only_pair_count, len(all_pairs),
        )
        return all_pairs

    def _build_mst_edges(self, sim: np.ndarray) -> List[Tuple[int, int]]:
        """Build k MSTs over edges above the weak floor threshold.

        Args:
            sim: (N, N) similarity matrix (upper-triangular from compute_similarity_matrix).

        Returns:
            List of (i, j) edges from all MSTs, with i < j.
        """
        num_nodes = sim.shape[0]
        self._mst_edge_counts = []

        if num_nodes <= 1:
            return []

        sim_full = np.maximum(sim, sim.T)

        edges: List[Tuple[int, int]] = []
        sims: List[float] = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                s = sim_full[i, j]
                if s >= self._mst_min_score:
                    edges.append((i, j))
                    sims.append(s)

        if len(edges) == 0:
            logger.warning("No candidate edges above mst_min_score=%.2f", self._mst_min_score)
            return []

        weights = 1.0 - np.array(sims, dtype=np.float64)

        selected: Set[Tuple[int, int]] = set()
        all_mst_edges: List[Tuple[int, int]] = []

        for m in range(self._num_msts):
            mst = _build_mst_kruskal(edges, weights, num_nodes, masked=selected)
            self._mst_edge_counts.append(len(mst))

            for e in mst:
                selected.add(e)
                all_mst_edges.append(e)

            logger.info("MST %d/%d: %d edges", m + 1, self._num_msts, len(mst))

            if len(mst) == 0:
                break

        return all_mst_edges

    def evaluate(self, num_images: int, visibility_graph: VisibilityGraph) -> GtsfmMetricsGroup:
        """Evaluate retriever output with graph topology metrics."""
        metrics = super().evaluate(num_images, visibility_graph)

        if len(visibility_graph) == 0:
            return metrics

        G = nx.Graph()
        G.add_nodes_from(range(num_images))
        G.add_edges_from(visibility_graph)

        num_components = nx.number_connected_components(G)
        metrics.add_metric(GtsfmMetric("num_connected_components", num_components))

        if num_components == 1:
            diameter = nx.diameter(G)
            avg_path_len = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            diameter = nx.diameter(subG)
            avg_path_len = nx.average_shortest_path_length(subG)

        metrics.add_metric(GtsfmMetric("graph_diameter", diameter))
        metrics.add_metric(GtsfmMetric("avg_shortest_path_length", round(avg_path_len, 2)))

        degrees = [d for _, d in G.degree()]
        metrics.add_metric(GtsfmMetric("avg_node_degree", round(sum(degrees) / len(degrees), 2)))

        num_bridges = len(list(nx.bridges(G)))
        metrics.add_metric(GtsfmMetric("num_bridge_edges", num_bridges))

        logger.info(
            "Graph metrics: components=%d, diameter=%d, avg_path=%.2f, avg_degree=%.2f, bridges=%d",
            num_components, diameter, avg_path_len, sum(degrees) / len(degrees), num_bridges,
        )
        return metrics

    def save_diagnostics(
        self, image_fnames: List[str], pairs: VisibilityGraph, plots_output_dir: Optional[Path]
    ) -> None:
        """Save similarity matrix diagnostics (from parent) plus per-MST statistics."""
        super().save_diagnostics(image_fnames, pairs, plots_output_dir)

        if plots_output_dir is None:
            return

        os.makedirs(plots_output_dir, exist_ok=True)
        stats_path = plots_output_dir / "kmst_stats.txt"

        with open(stats_path, "w") as f:
            f.write(f"num_matched: {self._num_matched}\n")
            f.write(f"min_score: {self._min_score}\n")
            f.write(f"mst_min_score: {self._mst_min_score}\n")
            f.write(f"num_msts: {self._num_msts}\n")
            f.write(f"total_pairs: {len(pairs)}\n")
            f.write(f"knn_pairs: {self._knn_pair_count}\n")
            f.write(f"mst_only_pairs: {self._mst_only_pair_count}\n")
            for m, count in enumerate(self._mst_edge_counts):
                f.write(f"mst_{m + 1}_edges: {count}\n")

        logger.info("Saved KMst stats to %s", stats_path)
