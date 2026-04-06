# Short-name exports for retriever classes.
#
# Usage (Hydra/Python):
#   _target_: gtsfm.retriever.Exhaustive
#   _target_: gtsfm.retriever.JointSimilaritySequential
#   _target_: gtsfm.retriever.Sequential
#   _target_: gtsfm.retriever.Similarity

from .exhaustive_retriever import ExhaustiveRetriever
from .joint_similarity_sequential_retriever import JointSimilaritySequentialRetriever
from .kmst_retriever import KMstRetriever
from .sequential_retriever import SequentialRetriever
from .similarity_retriever import SimilarityRetriever

Exhaustive = ExhaustiveRetriever
JointSimilaritySequential = JointSimilaritySequentialRetriever
KMst = KMstRetriever
Sequential = SequentialRetriever
Similarity = SimilarityRetriever

__all__ = [
    "Exhaustive",
    "JointSimilaritySequential",
    "KMst",
    "Sequential",
    "Similarity",
]
