"""Hybrid retriever: COLMAP-verified pairs intersected with MegaLoc top-K similarity.

Returns the image pairs that are BOTH geometrically verified by COLMAP (so verified correspondences
exist in the database) AND selected by MegaLoc similarity (each image's top-K most-similar neighbors
above ``min_score``). This keeps COLMAP's fast, robust verified correspondences while sparsifying its
dense vocab-tree view graph (e.g. ~140k pairs on St Peter's) down to the tuned MegaLoc selection, so the
downstream Metis cluster tree stays manageable. Correspondences are still read from the COLMAP db (via
``ColmapCorrespondenceGenerator``) for the surviving pairs.

Authors: Kathirvel Gounder
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.colmap_db_retriever import ColmapDBRetriever
from gtsfm.retriever.retriever_base import RetrieverBase
from gtsfm.retriever.similarity_retriever import SimilarityRetriever

logger = logger_utils.get_logger()


class ColmapDBMegaLocRetriever(RetrieverBase):
    """Intersect COLMAP-verified pairs with MegaLoc top-K similarity to sparsify the view graph."""

    def __init__(self, database_path: str, num_matched: int, min_score: float = 0.1) -> None:
        """
        Args:
            database_path: path to the COLMAP database.db (features + verified two-view geometries).
            num_matched: number of top MegaLoc matches to keep per image (the similarity top-K).
            min_score: minimum MegaLoc similarity score to accept a match.
        """
        self._colmap_retriever = ColmapDBRetriever(database_path)
        self._similarity_retriever = SimilarityRetriever(num_matched=num_matched, min_score=min_score)

    def __repr__(self) -> str:
        return (
            "ColmapDBMegaLocRetriever:\n"
            f"    {self._colmap_retriever}\n"
            f"    {self._similarity_retriever}"
        )

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Return COLMAP-verified pairs that are also in MegaLoc's per-image top-K.

        Args:
            global_descriptors: MegaLoc descriptors, one per image (required for the similarity filter).
            image_fnames: file names of the images.
            plots_output_dir: directory to save plots to. If None, plots are not saved.

        Returns:
            Sparsified visibility graph: sorted (i, j) pairs with i < j.
        """
        colmap_pairs = self._colmap_retriever.get_image_pairs(
            global_descriptors=None, image_fnames=image_fnames, plots_output_dir=plots_output_dir
        )

        if global_descriptors is None:
            logger.warning(
                "🗄️🔎 ColmapDBMegaLocRetriever: no global descriptors provided; returning all %d COLMAP "
                "pairs unfiltered (set image_pairs_generator.global_descriptor to enable the MegaLoc filter).",
                len(colmap_pairs),
            )
            return colmap_pairs

        megaloc_pairs = self._similarity_retriever.get_image_pairs(
            global_descriptors=global_descriptors,
            image_fnames=image_fnames,
            plots_output_dir=plots_output_dir,
        )

        # Strict intersection: keep pairs that are both COLMAP-verified and in MegaLoc's top-K. Both
        # retrievers return (i, j) with i < j over the same gtsfm image-index space, so set-and is exact.
        pairs: VisibilityGraph = sorted(set(colmap_pairs) & set(megaloc_pairs))
        logger.info(
            "🗄️🔎 ColmapDBMegaLocRetriever: %d COLMAP-verified ∩ %d MegaLoc top-K = %d pairs (from %d images).",
            len(colmap_pairs),
            len(megaloc_pairs),
            len(pairs),
            len(image_fnames),
        )
        return pairs
