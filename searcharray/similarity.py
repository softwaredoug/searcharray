"""Similarity functions given term stats."""
from typing import Protocol
import numpy as np


class Similarity(Protocol):
    """Similarity function protocol."""

    def __call__(self, term_freqs: np.ndarray, doc_freqs: np.ndarray, doc_lens: np.ndarray,
                 avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate similarity scores."""
        ...


def bm25_similarity(k1: float = 1.2, b: float = 0.75) -> Similarity:
    def bm25(term_freqs: np.ndarray, doc_freqs: np.ndarray,
             doc_lens: np.ndarray,
             avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate BM25 scores."""
        # Sum doc freqs
        sum_dfs = np.sum(doc_freqs, axis=0)
        # Calculate idf
        idf = np.log(1 + (num_docs - sum_dfs + 0.5) / (sum_dfs + 0.5))
        # Calculate tf
        tf = term_freqs / (term_freqs + k1 * (1 - b + b * doc_lens / avg_doc_lens))
        return idf * tf
    return bm25


default_bm25 = bm25_similarity()
