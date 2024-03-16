"""Similarity functions given term stats."""
from typing import Protocol, Dict, Any
import numpy as np


class ScoringContext:
    """Scoring context for similarity functions."

    May be reused across multiple calls to similarity functions over
    the lifetime of the index, allowing caching of any non-query-specific
    information.

    """

    def __init__(self, term_freqs: np.ndarray, doc_freqs: np.ndarray,
                 doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int):
        """Initialize."""
        self.term_freqs = term_freqs
        self.doc_freqs = doc_freqs
        self.doc_lens = doc_lens
        self.avg_doc_lens = avg_doc_lens
        self.num_docs = num_docs
        self.working: Dict[str, Any] = {}


class Similarity(Protocol):
    """Similarity function protocol."""

    def __call__(self, context: ScoringContext) -> np.ndarray:
        """Calculate similarity scores."""
        ...


def compute_idf(num_docs, sum_dfs):
    """Calculate idf."""
    return np.log(1 + (num_docs - sum_dfs + 0.5) / (sum_dfs + 0.5))


def compute_tfs(term_freqs: np.ndarray, doc_lens, avg_doc_lens, k1, b):
    adj_doc_lens = doc_lens / avg_doc_lens
    adj_doc_lens *= b
    adj_doc_lens += 1 - b
    adj_doc_lens *= k1
    # Divide tf in place for perf, but this means
    # we can't use the same term_freqs for different k1, b
    term_freqs /= (term_freqs + adj_doc_lens)
    return term_freqs


def bm25_similarity(k1: float = 1.2, b: float = 0.75) -> Similarity:
    """BM25 similarity function, as in Lucene 9."""
    def bm25(context: ScoringContext) -> np.ndarray:
        """Calculate BM25 scores."""
        # Sum doc freqs
        sum_dfs = np.sum(context.doc_freqs, axis=0)
        # Calculate idf
        idf = compute_idf(context.num_docs, sum_dfs)
        # Calculate tf
        # tf = term_freqs / (term_freqs + k1 * (1 - b + b * doc_lens / avg_doc_lens))
        tf = compute_tfs(context.term_freqs,
                         context.doc_lens,
                         context.avg_doc_lens, k1, b)
        return idf * tf
    return bm25


def bm25_legacy_similarity(k1: float = 1.2, b: float = 0.75) -> Similarity:
    """BM25 similarity prior to LUCENE-8563 with k1 + 1 in numerator."""
    # (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength))
    def bm25(context: ScoringContext) -> np.ndarray:
        """Calculate BM25 scores."""
        # Sum doc freqs
        sum_dfs = np.sum(context.doc_freqs, axis=0)
        # Calculate idf
        idf = np.log(1 + (context.num_docs - sum_dfs + 0.5) / (sum_dfs + 0.5))
        # Calculate tf
        tf = (context.term_freqs * (k1 + 1)) / (context.term_freqs + k1 * (1 - b + b * context.doc_lens / context.avg_doc_lens))
        return idf * tf
    return bm25


def classic_similarity() -> Similarity:
    """Classic Lucene TF-IDF similarity function."""
    def classic(context: ScoringContext) -> np.ndarray:
        """Calculate classic TF-IDF scores."""
        # Sum doc freqs
        sum_dfs = np.sum(context.doc_freqs, axis=0)
        # Calculate idf as log((docCount+1)/(docFreq+1)) + 1
        idf = np.log((context.num_docs + 1) / (sum_dfs + 1)) + 1
        length_norm = 1.0 / np.sqrt(context.doc_lens)
        # Calculate tf
        tf = np.sqrt(context.term_freqs)
        return idf * tf * length_norm
    return classic


default_bm25 = bm25_similarity()
