"""Phrase search by dumb numpy position differences subtraction, bigram by bigram."""
import numpy as np
from typing import List
import logging

from time import perf_counter

logger = logging.getLogger(__name__)

# When running in pytest
import sys  # noqa
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def vstack_with_mask(posns: List[np.ndarray], phrase_freqs: np.ndarray, width: int = 0, pad: int = -100):
    """Vertically stack arrays, only accepting those that fit within width.

    Parameters
    ----------
    posns: list of np.ndarray term positions for a given term across multiple docs
    width: int, max number of positions we accept
    pad: int, value to pad with when no position present
    phrase_freqs: np.ndarray, phrase freqs for each doc.
                 -1 indicates "not yet processed" 0 means "defiiitely not a match"

    Returns
    -------
    vstacked: np.ndarray, vertically stacked arrays of those <= width
    mask: np.ndarray, boolean mask of which arrays were accepted (so you can recompute them a different way)
    """
    # 0.19
    to_concat = []
    empty = np.zeros(width, dtype=np.int32) + pad
    for idx, array in enumerate(posns):
        arr_len = array.shape[0]
        if arr_len != width:
            if phrase_freqs[idx] == -1:
                phrase_freqs[idx] = -2  # Mark as skip this round
            to_concat.append(empty)
        else:
            to_concat.append(array)
    return np.vstack(to_concat)


def stack_term_posns(term_posns: List[List[np.ndarray]], phrase_freqs: np.ndarray, width: int = 10):
    # Pad for easy difference computation
    start = perf_counter()
    keep_term_posns = []
    for term_posn in term_posns:
        logger.info(f"Stacking {len(term_posn)} docs")

        this_term_posns = vstack_with_mask(term_posn, phrase_freqs, width=width)
        keep_term_posns.append(this_term_posns)
    logger.info(f"Stacking term posns took {perf_counter() - start:.4f} seconds")
    return keep_term_posns


def _compute_phrase_freqs(term_posns, phrase_freqs: np.ndarray, slop=1):
    start = perf_counter()

    to_compute = phrase_freqs == -1

    if not np.any(to_compute):
        return phrase_freqs

    # Only examine masked
    term_posns = [term_posn[to_compute] for term_posn in term_posns]

    prior_term = term_posns[0]
    logger.info(f"Preamble took {perf_counter() - start:.4f} seconds")
    for term in term_posns[1:]:
        is_same_term = (term.shape == prior_term.shape) and np.all(term[0] == prior_term[0])

        # Compute positional differences
        #
        # Each row of posn_diffs is a term posn diff matrix
        # Where columns are prior_term posns, rows are term posns
        # This shows every possible term diff
        #
        # Example:
        #   prior_term = array([[0, 4],[0, 4])
        #         term = array([[1, 2, 3],[1, 2, 3]])
        #
        #
        #   posn_diffs =
        #
        #     array([[ term[0] - prior_term[0], term[0] - prior_term[1] ],
        #            [ term[1] - prior_term[0], ...
        #            [ term[2] - prior_term[0], ...
        #
        #    or in our example
        #
        #     array([[ 1, -3],
        #            [ 2, -2],
        #            [ 3, -1]])
        #
        #  We care about all locations where posn == slop (or perhaps <= slop)
        #  that is term is slop away from prior_term. Usually slop == 1 (ie 1 posn away)
        #  for normal phrase matching
        #
        posn_diffs = term[:, :, np.newaxis] - prior_term[:, np.newaxis, :]
        logger.info(f"posn_diffs took {perf_counter() - start:.4f} seconds")

        # For > 2 terms, we need to connect a third term by making prior_term = term
        # and repeating
        #
        # BUT
        # we only want those parts of term that are adjacent to prior_term
        # before continuing, so we don't accidentally get a partial phrase
        # so we need to make sure to
        # Pad out any rows in 'term' where posn diff != slop
        # so they're not considered on subsequent iterations
        term_mask = np.any(posn_diffs == 1, axis=2)
        term[~term_mask] = -100
        logger.info(f"term mask took {perf_counter() - start:.4f} seconds")

        # Count how many times the row term is 1 away from the col term
        per_doc_diffs = np.sum(posn_diffs == slop, axis=1, dtype=np.int8)
        logger.info(f"count slops took {perf_counter() - start:.4f} seconds")

        # Doc-wise sum to get a 'term freq' for the prior_term - term bigram
        bigram_freqs = np.sum(per_doc_diffs == slop, axis=1)
        if is_same_term:
            satisfies_slop = per_doc_diffs == slop
            consecutive_ones = satisfies_slop[:, 1:] & satisfies_slop[:, :-1]
            consecutive_ones = np.sum(consecutive_ones, axis=1)
            # ceiling divide?
            # Really these show up as
            # 1 1 1 0 1
            # we need to treat the 2nd consecutive 1 as 'not a match'
            # and also update 'term' to not include it
            bigram_freqs -= -np.floor_divide(consecutive_ones, -2)
            logger.info(f"same terms took {perf_counter() - start:.4f} seconds")

        # Last loop, bigram_freqs is the full phrase term freq

        # Update mask to eliminate any non-matches
        phrase_freqs[to_compute] = bigram_freqs

        # Should only keep positions of 'prior term' that are adjacent to the
        # one prior to it...
        prior_term = term
        logger.info(f"Loop took {perf_counter() - start:.4f} seconds")

    return phrase_freqs


def compute_phrase_freqs(term_posns, phrase_freqs, slop=1, width=10):
    """Compute phrase freq using matrix-diff method for docs up to width posns. Skip others.

    Parameters
    ----------
    term_posns: list of np.ndarray term positions for a given term across multiple docs
    phrase_freqs: np.ndarray, phrase freqs for each doc present in term_posns

    Returns
    -------
    phrase_freqs: np.ndarray, phrase freqs for each doc present in mask

    See Also
    --------
    Colab notebook: https://colab.research.google.com/drive/1NRxeO8Ya8jSlFP5YwZaGh1-43kDH4OXG?authuser=1#scrollTo=5JZV8svpauYB
    """
    logger.info(f"Computing phrase freqs for width:{width} {len(term_posns)} terms")
    start = perf_counter()
    if len(term_posns[0]) != len(phrase_freqs):
        raise ValueError("term_posns and phrase_freqs must be same length")
    stacked = stack_term_posns(term_posns, phrase_freqs, width=width)
    phrase_freqs = _compute_phrase_freqs(stacked, phrase_freqs, slop=slop)
    phrase_freqs[phrase_freqs == -2] = -1
    logger.info(f"compute_phrase_freqs took {perf_counter() - start:.4f} seconds")
    return phrase_freqs
