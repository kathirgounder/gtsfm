"""Retriever implementation which uses a global image descriptor to suggest potential image pairs.
Note: Similarity computation based off of Paul-Edouard Sarlin's HLOC:
Reference: https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/pairs_from_retrieval.py
https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf
Authors: John Lambert
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()
MAX_NUM_IMAGES = 10000


class _UnionFind:
    """Disjoint-set with path compression + union by rank, for Kruskal's MST."""
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def build_mst_augment_edges(
    W: np.ndarray, mst_min_score: float = 0.20, num_msts: int = 2
) -> set:
    """Build k minimum spanning forests over edges with similarity >= mst_min_score,
    returning the union of edges. Use as a connectivity-augmenting overlay on kNN
    pairs — guarantees the chosen subgraph spans all nodes (modulo the floor).

    Edge weight = 1 - similarity, so MST picks highest-similarity bridges.
    Each subsequent MST excludes prior-tree edges → k structurally-distinct trees.

    Input:
      W: (N, N) similarity matrix (asymmetric or symmetric; symmetrized internally).
      mst_min_score: weak floor for MST candidate edges.
      num_msts: number of trees (k=2 for redundancy; k=1 for minimal sparsity).

    Returns:
      Set of (i, j) edges (i < j), unionized across the k MSTs.
    """
    W = np.asarray(W, dtype=np.float64)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, -np.inf)
    N = W.shape[0]

    candidates = []
    sims = []
    for i in range(N):
        for j in range(i + 1, N):
            s = W[i, j]
            if not np.isinf(s) and s >= mst_min_score:
                candidates.append((i, j))
                sims.append(s)
    if not candidates:
        return set()
    weights = 1.0 - np.array(sims, dtype=np.float64)
    order = np.argsort(weights, kind="stable")

    selected = set()
    for _ in range(num_msts):
        uf = _UnionFind(N)
        this_tree = []
        for idx in order:
            e = candidates[idx]
            if e in selected:
                continue
            if uf.union(e[0], e[1]):
                this_tree.append(e)
                if len(this_tree) == N - 1:
                    break
        if not this_tree:
            break
        for e in this_tree:
            selected.add(e)
    return selected


def diffuse_view_graph(W: np.ndarray, beta: float = 0.4, num_hops: int = 3) -> np.ndarray:
    """Reinforce a view-graph similarity matrix using Katz-style polynomial diffusion.

    Boosts the score of edges (i, j) that participate in many high-confidence
    triangles / 3-hop paths, and damps edges with no such support. Lets the
    downstream top-K + min_score filter pick a cleaner subgraph.

    Input:
      W:  (N, N) numpy array of pairwise similarity scores, e.g. cosine on L2-
          normalized MegaLoc descriptors. Can be asymmetric, upper-triangular,
          or symmetric — we symmetrize internally. Diagonal is ignored.
      beta: decay factor for longer paths (0.2-0.6 recommended).
            Lower = trust direct similarities more (less smoothing).
            Higher = trust multi-hop paths more (more smoothing).
      num_hops: how many path lengths to sum (3 = direct + 2-hop + 3-hop).

    Output:
      W_reinforced: (N, N) symmetric matrix of reinforced similarities,
                    rescaled so its max equals max(W). Drop-in replacement for
                    the input — existing min_score thresholds remain meaningful.
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]

    # 1. Symmetrize and zero diagonal.
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)

    # 2. Row-normalize to a row-stochastic transition matrix P.
    #    Keeps matrix powers bounded; gives the polynomial a random-walk
    #    interpretation: P^k[i, j] = probability of reaching j from i in k hops.
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = W / row_sums

    # 3. Decaying polynomial: W_acc = sum_{k=1..num_hops} beta^(k-1) * P^k
    W_acc = np.zeros((N, N), dtype=np.float64)
    P_power = np.eye(N)
    decay = 1.0
    for _ in range(num_hops):
        P_power = P_power @ P
        W_acc += decay * P_power
        decay *= beta

    # 4. Symmetrize the accumulator (P^k is asymmetric).
    W_acc = 0.5 * (W_acc + W_acc.T)

    # 5. Rescale to match the input's range so existing thresholds still apply.
    np.fill_diagonal(W_acc, 0.0)
    if W_acc.max() > 0 and W.max() > 0:
        W_acc = W_acc * (W.max() / W_acc.max())

    return W_acc


@dataclass
class SubBlockSimilarityResult:
    i_start: int
    i_end: int
    j_start: int
    j_end: int
    sub_block: torch.Tensor


class SimilarityRetriever(RetrieverBase):
    def __init__(
        self,
        num_matched: int,
        min_score: float = 0.1,
        blocksize: int = 50,
        use_diffusion_preprocessing: bool = False,
        diffusion_beta: float = 0.4,
        diffusion_num_hops: int = 3,
        use_mst_augmentation: bool = False,
        mst_min_score: float = 0.20,
        num_msts: int = 2,
    ) -> None:
        """
        Args:
            num_matched: Top-K matches per query for kNN pair selection.
            min_score: Minimum allowed similarity score for kNN pairs.
            blocksize: Block size for similarity matrix sub-block computation.
            use_diffusion_preprocessing: Apply Katz polynomial diffusion to W before
                top-K + min_score selection. Reinforces edges with triangle support.
            diffusion_beta / diffusion_num_hops: see diffuse_view_graph().
            use_mst_augmentation: Add union of k minimum spanning trees on top of the
                kNN pair set. Guarantees graph connectivity via highest-similarity
                bridges, even if some bridge edges score below `min_score`. Independent
                of diffusion — both can be combined.
            mst_min_score: Weak floor for MST candidate edges.
            num_msts: Number of trees (k=2 typical).
        """
        self._num_matched = num_matched
        self._blocksize = blocksize
        self._min_score = min_score
        self._use_diffusion_preprocessing = use_diffusion_preprocessing
        self._diffusion_beta = diffusion_beta
        self._diffusion_num_hops = diffusion_num_hops
        self._use_mst_augmentation = use_mst_augmentation
        self._mst_min_score = mst_min_score
        self._num_msts = num_msts
        self._latest_similarity_matrix: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return f"""
        SimilarityRetriever:
            Num. frames matched: {self._num_matched}
            Block size: {self._blocksize}
            Minimum score: {self._min_score}
            Diffusion: enabled={self._use_diffusion_preprocessing} beta={self._diffusion_beta} hops={self._diffusion_num_hops}
        """

    def set_num_matched(self, n) -> None:
        """Set the number of matched frames for similarity matching."""
        self._num_matched = n

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Compute potential image pairs.

        Args:
            global_descriptors: the global descriptors for the retriever, if needed.
            image_fnames: file names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            List of (i1,i2) image pairs.
        """
        if global_descriptors is None:
            raise ValueError("Global descriptors need to be provided")
        sim = self.compute_similarity_matrix(global_descriptors)
        # Stash a CPU copy so downstream code can decide how to persist diagnostics.
        self._latest_similarity_matrix = sim.detach().cpu().clone()
        return self.compute_pairs_from_similarity_matrix(
            sim=sim, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )

    def compute_similarity_matrix(self, global_descriptors: List[np.ndarray]) -> torch.Tensor:
        """Compute a similarity matrix between all pairs of images.
        We use block matching, to avoid excessive memory usage.
        We cannot fit more than 50x50 sized block into memory, on a 16 GB RAM machine.
        A similar blocked exhaustive matching implementation can be found in COLMAP:
        https://github.com/colmap/colmap/blob/dev/src/feature/matching.cc#L899

        Args:
            global_descriptors: global descriptors, one per image.

        Returns:
            Similarity matrix as tensor of shape (num_images, num_images).
        """
        num_images = len(global_descriptors)
        if num_images > MAX_NUM_IMAGES:
            raise RuntimeError("Cannot construct similarity matrix of this size.")

        sub_block_results: List[SubBlockSimilarityResult] = []
        num_blocks = math.ceil(num_images / self._blocksize)

        # TODO(Ayush, John): do we still need to do sub_block computations?
        for block_i in range(num_blocks):
            for block_j in range(block_i, num_blocks):
                sub_block_results.append(
                    self._compute_similarity_sub_block(
                        global_descriptors=global_descriptors, block_i=block_i, block_j=block_j
                    )
                )

        sim = self._aggregate_sub_blocks(sub_block_results=sub_block_results, num_images=num_images)
        return sim

    def _compute_similarity_sub_block(self, global_descriptors: List[np.ndarray], block_i: int, block_j: int):
        """Compute a sub-block of an global descriptor based similarity matrix.
        Args:
            global_descriptors: global descriptors, one per image.
            block_i: row index of sub-block.
            block_j: column index of sub-block.
        Returns:
            sub-block similarity result.
        """
        num_images = len(global_descriptors)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_blocks = math.ceil(num_images / self._blocksize)
        # only compute the upper triangular portion of the similarity matrix.
        logger.info("Computing matching block (%d/%d,%d/%d)", block_i, num_blocks - 1, block_j, num_blocks - 1)
        i_start = block_i * self._blocksize
        i_end = (block_i + 1) * self._blocksize
        i_end = min(i_end, num_images)
        j_start = block_j * self._blocksize
        j_end = (block_j + 1) * self._blocksize
        j_end = min(j_end, num_images)
        # Form (K,D) for K images.
        block_i_query_descriptors = torch.from_numpy(np.array(global_descriptors[i_start:i_end]))
        block_j_query_descriptors = torch.from_numpy(np.array(global_descriptors[j_start:j_end]))
        # Einsum equivalent to (img_descriptors @ img_descriptors.T)
        sim_block = torch.einsum(
            "id,jd->ij", block_i_query_descriptors.to(device), block_j_query_descriptors.to(device)
        )
        return SubBlockSimilarityResult(i_start=i_start, i_end=i_end, j_start=j_start, j_end=j_end, sub_block=sim_block)

    def _aggregate_sub_blocks(self, sub_block_results: List[SubBlockSimilarityResult], num_images: int) -> torch.Tensor:
        """Aggregate results from many independently computed sub-blocks of the similarity matrix into a single matrix.

        Args:
            sub_block_results: Metadata and results of similarity matrix sub-block computation.
            num_images: Number of images to compare for matching.

        Returns:
            sim: Tensor of shape (num_images, num_images) representing similarity matrix.
        """
        sim = torch.zeros((num_images, num_images))
        for sr in sub_block_results:
            sim[sr.i_start : sr.i_end, sr.j_start : sr.j_end] = sr.sub_block  # noqa E203
        return sim

    def compute_pairs_from_similarity_matrix(
        self, sim: torch.Tensor, image_fnames: List[str], plots_output_dir: Optional[Path] = None
    ) -> VisibilityGraph:
        """

        Args:
            sim: Tensor of shape (num_images, num_images) representing similarity matrix.
            image_fnames: the names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            pair_indices: (i1,i2) image pairs.
        """
        num_images = len(image_fnames)
        # Avoid self-matching and disallow lower triangular portion
        is_invalid_mat = ~np.triu(np.ones((num_images, num_images), dtype=bool))
        np.fill_diagonal(a=is_invalid_mat, val=True)

        # Optional diffusion preprocessing: reinforce edges with multi-hop path
        # support before applying top-K + min_score. Raw `sim` is preserved for
        # diagnostic export via save_diagnostics().
        if self._use_diffusion_preprocessing:
            sim_np = sim.detach().cpu().numpy()
            sim_reinforced = diffuse_view_graph(
                sim_np, beta=self._diffusion_beta, num_hops=self._diffusion_num_hops
            )
            if plots_output_dir is not None:
                os.makedirs(plots_output_dir, exist_ok=True)
                np.savetxt(
                    fname=str(plots_output_dir / "similarity_matrix_reinforced.txt"),
                    X=sim_reinforced, fmt="%.4f", delimiter=",",
                )
            logger.info(
                "Diffusion preprocessing applied: beta=%.2f, hops=%d. raw max=%.3f, "
                "reinforced max=%.3f, mean lift=%.3f",
                self._diffusion_beta, self._diffusion_num_hops,
                float(sim_np.max()), float(sim_reinforced.max()),
                float((sim_reinforced - np.maximum(sim_np, sim_np.T)).mean()),
            )
            sim_for_selection = torch.from_numpy(sim_reinforced).to(sim.device).type_as(sim)
        else:
            sim_for_selection = sim

        pairs = pairs_from_score_matrix(
            sim_for_selection, invalid=is_invalid_mat,
            num_select=self._num_matched, min_score=self._min_score,
        )
        n_knn = len(pairs)

        # Optional MST augmentation: union the kNN pair set with edges from k MSTs
        # over the raw similarity matrix (with mst_min_score floor). MSTs guarantee
        # global connectivity via highest-similarity bridges, even if those bridge
        # edges scored below `min_score` and were rejected by kNN.
        if self._use_mst_augmentation:
            mst_edges = build_mst_augment_edges(
                sim.detach().cpu().numpy(),
                mst_min_score=self._mst_min_score,
                num_msts=self._num_msts,
            )
            knn_set = set((min(i, j), max(i, j)) for (i, j) in pairs)
            mst_only = mst_edges - knn_set
            pairs = sorted(knn_set | mst_edges)
            logger.info(
                "MST augmentation: kNN=%d pairs + MST adds %d new pairs (%d total). "
                "mst_min_score=%.2f, num_msts=%d",
                n_knn, len(mst_only), len(pairs),
                self._mst_min_score, self._num_msts,
            )
        else:
            logger.info("Found %d pairs from the Similarity Retriever.", n_knn)
        return pairs

    def save_diagnostics(
        self, image_fnames: List[str], pairs: VisibilityGraph, plots_output_dir: Optional[Path]
    ) -> None:
        """Persist diagnostic outputs for the latest similarity matrix (if available)."""

        if plots_output_dir is None:
            return

        if self._latest_similarity_matrix is None:
            logger.warning("No cached similarity matrix available to save.")
            return

        os.makedirs(plots_output_dir, exist_ok=True)

        sim_cpu = self._latest_similarity_matrix
        heatmap_path = plots_output_dir / "similarity_matrix.jpg"
        matrix_path = plots_output_dir / "similarity_matrix.txt"
        named_pairs_path = plots_output_dir / "similarity_named_pairs.txt"

        sim_np = np.triu(sim_cpu.numpy())
        plt.imshow(sim_np)
        plt.title("Image Similarity Matrix")
        plt.savefig(str(heatmap_path), dpi=500)
        plt.close("all")
        logger.info("Saved similarity heatmap to %s", heatmap_path)

        np.savetxt(
            fname=str(matrix_path),
            X=sim_cpu.numpy(),
            fmt="%.2f",
            delimiter=",",
        )
        logger.info("Saved similarity matrix values to %s", matrix_path)

        named_pairs = [(image_fnames[i], image_fnames[j]) for i, j in pairs]
        with open(named_pairs_path, "w") as fid:
            for (name_i, name_j), (i, j) in zip(named_pairs, pairs):
                fid.write("%.4f %s %s\n" % (sim_cpu[i, j].item(), name_i, name_j))
        logger.info("Saved similarity pair scores to %s", named_pairs_path)

        self._latest_similarity_matrix = None


def pairs_from_score_matrix(
    scores: torch.Tensor, invalid: np.ndarray, num_select: int, min_score: Optional[float] = None
) -> VisibilityGraph:
    """Identify image pairs from a score matrix.

    Note: Similarity computation here is based off of Paul-Edouard Sarlin's HLOC:
    Reference: https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/pairs_from_retrieval.py

    Args:
        scores: (K1,K2) for matching K1 images against K2 images.
        invalid: (K1,K2) boolean array indicating invalid match pairs (e.g. self-matches).
        num_select: Number of matches to select, per query.
        min_score: Minimum allowed similarity score.

    Returns:
        pairs: Tuples representing pairs (i1,i2) of images.
    """
    N = scores.shape[0]
    # if there are only N images to choose from, selecting more than N is not allowed
    num_select = min(num_select, N)
    assert scores.shape == invalid.shape
    invalid_tensor = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        # logical OR.
        invalid_tensor |= scores < min_score
    scores.masked_fill_(invalid_tensor, float("-inf"))
    topk = torch.topk(scores, k=num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()
    pairs = []
    for i_raw, j_raw in zip(*np.where(valid)):
        j = int(indices[i_raw, j_raw])
        pairs.append((int(i_raw), j))
    return pairs
