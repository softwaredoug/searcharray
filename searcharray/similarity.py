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
    """BM25 similarity function, as in Lucene 9."""
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


def bm25_legacy_similarity(k1: float = 1.2, b: float = 0.75) -> Similarity:
    """BM25 similarity prior to LUCENE-8563 with k1 + 1 in numerator."""
    # (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength))
    def bm25(term_freqs: np.ndarray, doc_freqs: np.ndarray,
             doc_lens: np.ndarray,
             avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate BM25 scores."""
        # Sum doc freqs
        sum_dfs = np.sum(doc_freqs, axis=0)
        # Calculate idf
        idf = np.log(1 + (num_docs - sum_dfs + 0.5) / (sum_dfs + 0.5))
        # Calculate tf
        tf = (term_freqs * (k1 + 1)) / (term_freqs + k1 * (1 - b + b * doc_lens / avg_doc_lens))
        return idf * tf
    return bm25


def classic_similarity() -> Similarity:
    """Classic Lucene TF-IDF similarity function."""
    def classic(term_freqs: np.ndarray, doc_freqs: np.ndarray,
                doc_lens: np.ndarray,
                avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate classic TF-IDF scores."""
        # Sum doc freqs
        sum_dfs = np.sum(doc_freqs, axis=0)
        # Calculate idf as log((docCount+1)/(docFreq+1)) + 1
        idf = np.log((num_docs + 1) / (sum_dfs + 1)) + 1
        length_norm = 1.0 / np.sqrt(doc_lens)
        # Calculate tf
        tf = np.sqrt(term_freqs)
        return idf * tf * length_norm
    return classic


default_bm25 = bm25_similarity()
