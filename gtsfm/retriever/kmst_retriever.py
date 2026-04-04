"""Retriever that replaces kNN pair selection with multi-MST + distance modulation.

Drop-in replacement for SimilarityRetriever. Reuses the same similarity matrix
computation but selects pairs via iterative minimum spanning trees with
connectivity-aware score modulation, following:

    Wei et al., "Global-Aware Edge Prioritization for Pose Graph Initialization", CVPR 2026.

The key insight: kNN selects edges *locally* (each image picks its best neighbors
independently), which produces dense cliques within visually similar clusters but
thin, fragile connections between clusters. MST selects edges *globally* -- it is
forced to bridge disconnected components, guaranteeing connectivity. Multiple MSTs
add redundancy, and distance modulation explicitly reinforces weak regions of the
graph by boosting edges between nodes that are far apart in the current topology.

Authors: Kathir Gounder (integration), based on Tong Wei et al.'s algorithm
"""

import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    # Sort by weight, break ties by edge index for determinism.
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


def _compute_hop_distances(adj: Dict[int, Set[int]], num_nodes: int) -> np.ndarray:
    """BFS from every node to compute all-pairs hop distances.

    Args:
        adj: Adjacency list mapping node -> set of neighbors.
        num_nodes: Total number of nodes.

    Returns:
        (num_nodes, num_nodes) int array. -1 means unreachable.
    """
    dist = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
    for src in range(num_nodes):
        if src not in adj:
            dist[src, src] = 0
            continue
        d = dist[src]
        d[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj.get(u, ()):
                if d[v] == -1:
                    d[v] = d[u] + 1
                    queue.append(v)
    return dist


class KMstRetriever(SimilarityRetriever):
    """Multi-MST retriever with connectivity-aware distance modulation.

    Replaces kNN pair selection from SimilarityRetriever with iterative
    minimum spanning trees. Each successive MST masks previously selected
    edges and (optionally) modulates scores to reinforce weak graph regions.

    Default parameters are tuned for 1dSfM-scale datasets (50-6000 images,
    landmark scenes with repeated structures and wide baselines).
    """

    def __init__(
        self,
        num_matched: int = 20,
        num_msts: int = 3,
        lambda_modulation: float = 0.3,
        min_rank_threshold: float = 0.5,
        top_k_modulate: int = 5,
        min_score: float = 0.2,
        blocksize: int = 50,
    ) -> None:
        """
        Args:
            num_matched: Kept for interface compatibility (not used for edge count).
            num_msts: Number of MSTs to build (k). More MSTs = more redundancy.
                      3 is a good default for 1dSfM scenes; use 5 for very large/ambiguous scenes.
            lambda_modulation: Blending weight in [0,1] between similarity and hop distance.
                               0 = pure similarity, 1 = pure connectivity. 0.3 balances well
                               for 1dSfM where scenes have moderate ambiguity.
            min_rank_threshold: Minimum similarity to allow an edge to be modulation-boosted.
                                0.5 is relaxed enough for 1dSfM's wide-baseline pairs while
                                still filtering junk matches.
            top_k_modulate: Only modulate the top-k candidates per image after each MST.
                            Prevents low-quality edges from getting undeserved boosts.
            min_score: Minimum similarity to include an edge as a candidate at all.
                       0.2 is lower than the paper's 0.9 because 1dSfM scenes have many
                       wide-baseline pairs with moderate similarity.
            blocksize: Block size for similarity matrix computation (inherited).
        """
        super().__init__(num_matched=num_matched, min_score=min_score, blocksize=blocksize)
        self._num_msts = num_msts
        self._lambda = lambda_modulation
        self._min_rank_threshold = min_rank_threshold
        self._top_k_modulate = top_k_modulate
        # Diagnostics: per-MST edge counts, populated during pair selection.
        self._mst_edge_counts: List[int] = []

    def __repr__(self) -> str:
        return (
            f"KMstRetriever(num_msts={self._num_msts}, lambda={self._lambda}, "
            f"min_rank={self._min_rank_threshold}, top_k_mod={self._top_k_modulate}, "
            f"min_score={self._min_score})"
        )

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Compute image pairs via multi-MST selection with distance modulation."""
        if global_descriptors is None:
            raise ValueError("Global descriptors need to be provided")

        sim = self.compute_similarity_matrix(global_descriptors)
        self._latest_similarity_matrix = sim.detach().cpu().clone()

        pairs = self._select_pairs_multi_mst(sim.cpu().numpy())
        logger.info("KMstRetriever selected %d pairs from %d MSTs.", len(pairs), self._num_msts)
        return pairs

    def _select_pairs_multi_mst(self, sim: np.ndarray) -> VisibilityGraph:
        """Build k MSTs with connectivity-aware score modulation.

        Follows Algorithm 1 from Wei et al.:
        1. Build candidate edge list from similarity matrix.
        2. First MST uses raw similarity weights.
        3. Subsequent MSTs modulate scores using hop distances on the current
           union graph, boosting edges that bridge weakly connected regions.
        4. Return union of all MST edges.

        Args:
            sim: (N, N) similarity matrix (upper triangular populated).

        Returns:
            Sorted list of (i, j) pairs with i < j.
        """
        num_nodes = sim.shape[0]
        self._mst_edge_counts = []

        if num_nodes <= 1:
            return []

        # Symmetrize the similarity matrix (it's upper-triangular from compute_similarity_matrix).
        sim_full = np.maximum(sim, sim.T)

        # Build candidate edge list: all (i, j) with i < j and similarity >= min_score.
        edges: List[Tuple[int, int]] = []
        sims: List[float] = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                s = sim_full[i, j]
                if s >= self._min_score:
                    edges.append((i, j))
                    sims.append(s)

        if len(edges) == 0:
            logger.warning("No candidate edges above min_score=%.2f", self._min_score)
            return []

        sim_arr = np.array(sims, dtype=np.float64)
        weights = 1.0 - sim_arr  # Lower weight = higher similarity = preferred by MST.

        # Adjacency list for the union graph (built incrementally).
        adj: Dict[int, Set[int]] = {}
        selected: Set[Tuple[int, int]] = set()
        all_mst_edges: List[Tuple[int, int]] = []

        for m in range(self._num_msts):
            if m == 0:
                # First MST: use raw weights (no modulation).
                current_weights = weights.copy()
            else:
                # Modulate scores using hop distances on the current union graph.
                current_weights = self._modulate_weights(
                    edges, sim_arr, adj, num_nodes
                )

            mst = _build_mst_kruskal(edges, current_weights, num_nodes, masked=selected)
            self._mst_edge_counts.append(len(mst))

            # Update union graph adjacency and mask.
            for i, j in mst:
                selected.add((i, j))
                adj.setdefault(i, set()).add(j)
                adj.setdefault(j, set()).add(i)
                all_mst_edges.append((i, j))

            logger.info("MST %d/%d: %d edges (total so far: %d)", m + 1, self._num_msts, len(mst), len(all_mst_edges))

            if len(mst) == 0:
                logger.info("No more edges available for MST construction, stopping early.")
                break

        # Deduplicate and sort.
        unique_pairs = sorted(set(all_mst_edges))
        return unique_pairs

    def _modulate_weights(
        self,
        edges: List[Tuple[int, int]],
        sim_arr: np.ndarray,
        adj: Dict[int, Set[int]],
        num_nodes: int,
    ) -> np.ndarray:
        """Compute distance-modulated edge weights for the next MST iteration.

        Implements Eq. 6 from Wei et al.:
            s_ij = (1 - lambda) * r_ij + lambda * d_bar(i,j)

        where d_bar is the normalized hop distance in the current union graph.
        Only the top-k candidates per image (by raw similarity) are eligible for
        modulation; all others keep their raw weights. Edges below
        min_rank_threshold are penalized.

        Args:
            edges: Candidate edge list.
            sim_arr: Raw similarity scores for each edge.
            adj: Current union graph adjacency list.
            num_nodes: Total number of nodes.

        Returns:
            Weight array (1 - modulated_score) for Kruskal's.
        """
        hop_dist = _compute_hop_distances(adj, num_nodes)

        # Graph diameter: max finite hop distance.
        finite_mask = hop_dist >= 0
        if finite_mask.any():
            diameter = int(hop_dist[finite_mask].max())
        else:
            diameter = 1  # Degenerate: no edges yet.
        if diameter == 0:
            diameter = 1

        # Identify top-k candidates per image by raw similarity.
        # Build per-image sorted candidate lists.
        per_image_top: Dict[int, Set[int]] = {}
        image_candidates: Dict[int, List[Tuple[float, int]]] = {}
        for idx, (i, j) in enumerate(edges):
            image_candidates.setdefault(i, []).append((sim_arr[idx], idx))
            image_candidates.setdefault(j, []).append((sim_arr[idx], idx))

        top_edge_indices: Set[int] = set()
        for img, cands in image_candidates.items():
            cands.sort(key=lambda x: -x[0])  # Descending by similarity.
            for rank, (_, eidx) in enumerate(cands):
                if rank >= self._top_k_modulate:
                    break
                top_edge_indices.add(eidx)

        # Compute modulated weights.
        modulated = np.empty(len(edges), dtype=np.float64)
        for idx, (i, j) in enumerate(edges):
            r_ij = sim_arr[idx]

            if idx in top_edge_indices and r_ij >= self._min_rank_threshold:
                # Normalized hop distance (Eq. 5).
                d = hop_dist[i, j]
                if d < 0:
                    d_bar = 1.0  # Disconnected -> max distance.
                else:
                    d_bar = d / diameter

                # Modulated score (Eq. 6).
                s_ij = (1.0 - self._lambda) * r_ij + self._lambda * d_bar
                modulated[idx] = 1.0 - s_ij
            else:
                # Not in top-k or below threshold: use raw weight.
                modulated[idx] = 1.0 - r_ij

        return modulated

    def evaluate(self, num_images: int, visibility_graph: VisibilityGraph) -> GtsfmMetricsGroup:
        """Evaluate retriever output with graph topology metrics.

        Extends the base retriever metrics with graph-level quality indicators
        that show how well the multi-MST selection strategy connects the image set.
        """
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
            # Report diameter of the largest connected component.
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
            f.write(f"num_msts: {self._num_msts}\n")
            f.write(f"lambda_modulation: {self._lambda}\n")
            f.write(f"min_rank_threshold: {self._min_rank_threshold}\n")
            f.write(f"top_k_modulate: {self._top_k_modulate}\n")
            f.write(f"min_score: {self._min_score}\n")
            f.write(f"total_pairs: {len(pairs)}\n")
            for m, count in enumerate(self._mst_edge_counts):
                f.write(f"mst_{m + 1}_edges: {count}\n")

        logger.info("Saved KMst stats to %s", stats_path)
