import numpy as np

from searcharray.similarity import bm25_similarity
from test_utils import w_scenarios


def arr(x):
    if isinstance(x, int) or isinstance(x, float):
        return np.array([x], dtype=np.float32)
    if isinstance(x, list):
        return np.array(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)


lucene_bm25_scenarios = {
    "tf_2": {
        "term_freqs": 2,
        "doc_freqs": 14,
        "doc_lens": 4,
        "avg_doc_len": 2.7322686,
        "num_docs": 8516,
        "expected": 3.52482
    },
    "tf_1": {
        "term_freqs": 1,
        "doc_freqs": 5,
        "doc_lens": 35,
        "avg_doc_len": 50.580456,
        "num_docs": 8514,
        "expected": 3.8199246
    },
    "rambo_tmdb": {
        "term_freqs": 2,  # freq, occurrences of term within document
        "doc_freqs": 7,   # n, number of documents containing term
        "doc_lens": 44,   # "dl, length of field (approximate)
        "avg_doc_len": 50.580456,  # avgdl, average length of field
        "num_docs": 8514,    # N, total number of documents with field
        "expected": 4.5636616
    },
    "the_tmdb": {
        "term_freqs": 25,  # freq, occurrences of term within document
        "doc_freqs": 7823,   # n, number of documents containing term
        "doc_lens": 152,   # "dl, length of field (approximate)
        "avg_doc_len": 119.18542,  # avgdl, average length of field
        "num_docs": 8516,    # N, total number of documents with field
        "expected": 0.08028283
    }
}


@w_scenarios(lucene_bm25_scenarios)
def test_bm25_similarity_matches_lucene(term_freqs, doc_freqs, doc_lens, avg_doc_len, num_docs, expected):
    default_bm25 = bm25_similarity(k1=1.2, b=0.75)
    bm25 = default_bm25(arr(term_freqs),
                        arr(doc_freqs),
                        arr(doc_lens),
                        avg_doc_len,
                        num_docs)
    assert np.isclose(bm25, expected).all()
