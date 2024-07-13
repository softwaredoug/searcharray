"""Similarity functions given term stats."""
from typing import Protocol, Dict, Any
import numpy as np


class ScoringContext:
    """Scoring context for similarity functions."

    May be reused across multiple calls to similarity functions over
    the lifetime of the index, allowing caching of any non-query-specific
    information.

    """

    def __init__(self, doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int):
        """Initialize."""
        self.doc_lens = doc_lens
        self.avg_doc_lens = avg_doc_lens
        self.num_docs = num_docs
        self.working: Dict[str, Any] = {}

    def same_as(self, other: "ScoringContext") -> bool:
        """Check if the context is the same as another."""
        if other is None:
            return False
        return (self.doc_lens is other.doc_lens
                and self.avg_doc_lens == other.avg_doc_lens
                and self.num_docs == other.num_docs)


class Similarity(Protocol):
    """Similarity function protocol."""

    def __call__(self, term_freqs: np.ndarray, doc_freqs: np.ndarray,
                 doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate similarity scores."""
        ...


def compute_idf(num_docs, dfs):
    """Calculate idf."""
    return np.sum(np.log(1 + (num_docs - dfs + 0.5) / (dfs + 0.5)))


def compute_adj_doc_lens(doc_lens, avg_doc_lens, k1, b):
    if avg_doc_lens == 0:
        adj_doc_lens = np.zeros_like(doc_lens, dtype=np.float32)
    else:
        adj_doc_lens = doc_lens / avg_doc_lens
    adj_doc_lens *= b
    adj_doc_lens += 1 - b
    adj_doc_lens *= k1
    # Divide tf in place for perf, but this means
    # we can't use the same term_freqs for different k1, b
    return adj_doc_lens


def bm25_similarity(k1: float = 1.2, b: float = 0.75) -> Similarity:
    """BM25 similarity function, as in Lucene 9."""
    context = None
    def bm25(term_freqs: np.ndarray, doc_freqs: np.ndarray,
             doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate BM25 scores."""
        # Sum doc freqs
        # Calculate idf
        nonlocal context
        new_context = ScoringContext(doc_lens, avg_doc_lens, num_docs)
        if context is None or not context.same_as(new_context):
            context = new_context

        idf = compute_idf(context.num_docs, doc_freqs)
        try:
            adj_doc_lens = context.working["adj_doc_lens"]
            term_freqs /= (term_freqs + adj_doc_lens)
            term_freqs *= idf
            return term_freqs
        except (KeyError, ValueError):
            try:
                adj_doc_lens = compute_adj_doc_lens(context.doc_lens, context.avg_doc_lens, k1, b)
                context.working["adj_doc_lens"] = adj_doc_lens
                term_freqs /= (term_freqs + adj_doc_lens)
                term_freqs *= idf
                return term_freqs
            except ValueError:
                adj_doc_lens = compute_adj_doc_lens(doc_lens, avg_doc_lens, k1, b)
                term_freqs /= (term_freqs + adj_doc_lens)
                term_freqs *= idf
                return term_freqs
    return bm25


def bm25_legacy_similarity(k1: float = 1.2, b: float = 0.75) -> Similarity:
    """BM25 similarity prior to LUCENE-8563 with k1 + 1 in numerator."""
    # (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength))
    def bm25(term_freqs: np.ndarray, doc_freqs: np.ndarray,
             doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
        """Calculate BM25 scores."""
        # Calculate idf
        idf = compute_idf(num_docs, doc_freqs)
        # Calculate tf
        tf = (term_freqs * (k1 + 1)) / (term_freqs + k1 * (1 - b + b * doc_lens / avg_doc_lens))
        return idf * tf
    return bm25


def classic_similarity() -> Similarity:
    """Classic Lucene TF-IDF similarity function."""
    def classic(term_freqs: np.ndarray, doc_freqs: np.ndarray,
                doc_lens: np.ndarray, avg_doc_lens: int, num_docs: int) -> np.ndarray:
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
