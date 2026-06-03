"""Retriever that pulls image pairs directly from a COLMAP shared database.

Bypass for the MegaLoc + similarity-matrix retrieval pipeline when the COLMAP
frontend has already done all the work (feature extraction + exhaustive matching
+ geometric verification). The DB's `two_view_geometries` table with `config != 0`
IS the verified-pair set — no point recomputing descriptors and similarity to
re-derive a less-complete subset.

Wins vs `Similarity` retriever on a shared-DB run:
  • Skip MegaLoc descriptor compute (~2 min on a 500-image dataset)
  • Skip similarity-matrix construction (~1 min)
  • Skip Dask graph construction overhead for ~10× more candidate edges than
    actually have DB matches (~2-3 min on 500-image dataset)

Total savings: ~5-6 min per run. More importantly, the visibility graph emitted
matches GLOMAP's exact verified-pair input, so back-end comparisons are
apples-to-apples.

Authors: Kathir Gounder
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()

_COLMAP_MAX_IMAGE_ID = 2**31 - 1


class ColmapDbPairsRetriever(RetrieverBase):
    """Read verified image pairs directly from a COLMAP `two_view_geometries` table."""

    def __init__(self, database_path: str) -> None:
        """Initialize with a path to the COLMAP shared database."""
        super().__init__()
        self._database_path = database_path

    def __repr__(self) -> str:
        return f"ColmapDbPairsRetriever(database_path={self._database_path})"

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Return all GTSFM-indexed pairs with `config != 0` in the COLMAP DB.

        Resolves COLMAP's image_ids → file_names → GTSFM image indices via the
        loader-supplied `image_fnames` list. Pairs whose images aren't present in
        the loader (e.g. because the user subsampled) are silently skipped.
        """
        del global_descriptors  # not needed; pairs come straight from DB
        del plots_output_dir

        name_to_gtsfm_idx = {Path(fn).name: i for i, fn in enumerate(image_fnames)}
        conn = sqlite3.connect(self._database_path)
        try:
            colmap_id_to_name = {
                row[0]: row[1] for row in conn.execute("SELECT image_id, name FROM images")
            }
            verified_rows = conn.execute(
                "SELECT pair_id FROM two_view_geometries WHERE config != 0"
            ).fetchall()
        finally:
            conn.close()

        pairs: VisibilityGraph = []
        n_skipped_missing = 0
        for (pair_id,) in verified_rows:
            colmap_id1 = pair_id // _COLMAP_MAX_IMAGE_ID
            colmap_id2 = pair_id % _COLMAP_MAX_IMAGE_ID
            n1 = colmap_id_to_name.get(colmap_id1)
            n2 = colmap_id_to_name.get(colmap_id2)
            if n1 is None or n2 is None:
                n_skipped_missing += 1
                continue
            i = name_to_gtsfm_idx.get(Path(n1).name)
            j = name_to_gtsfm_idx.get(Path(n2).name)
            if i is None or j is None or i == j:
                n_skipped_missing += 1
                continue
            pairs.append((min(i, j), max(i, j)))

        logger.info(
            "ColmapDbPairsRetriever: %d verified pairs from %s "
            "(%d images present in loader, %d DB pairs skipped due to missing-in-loader)",
            len(pairs), self._database_path, len(name_to_gtsfm_idx), n_skipped_missing,
        )
        return pairs
