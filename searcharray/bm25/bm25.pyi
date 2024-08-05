import numpy as np
from numpy.typing import NDArray


def bm25_score(term_freqs: NDArray[np.float32],
               doc_lens: NDArray[np.float32],
               avg_doc_lens: float,
               idf: float,
               b: float,
               k1: float):
    ...
