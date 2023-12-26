import numpy as np
from typing import List


def _self_adjs(prior_posns, next_posns):
    """Given two arrays of positions, return the self adjacencies.

    Prior is a subset of next, find all the locations in next in prior,
    then get all the adjacent positions, returning a numpy array.
    """
    in1d = np.in1d(next_posns, prior_posns)
    if len(in1d) == 0:
        return np.array([])

    # Where there's an intersection
    start_indices = np.argwhere(in1d).flatten()
    # The adjacent to the intersections
    adj_indices = np.argwhere(in1d).flatten() + 1
    adj_indices = adj_indices[adj_indices < len(next_posns)]

    arr = np.union1d(next_posns[start_indices], next_posns[adj_indices])
    arr.sort()
    return arr


def scan_merge_ins(term_posns: List[List[np.ndarray]],
                   phrase_freqs: np.ndarray, slop=1) -> np.ndarray:
    """Merge bigram, by bigram, using np.searchsorted to find if insert posns match slop.

    Description:
    ------------
    See https://colab.research.google.com/drive/1EeqHYuCiqyptd-awS67Re78pqVdTfH4A

    Parameters:
    -----------
    term_posns: List[List[np.ndarray]] - for each term, a list of positions
    phrase_freqs: np.ndarray - the frequency of each phrase, the output buffer
    slop: int - allowed distance between terms
    """
    # Any identical terms, default to shitty algo for now
    # if len(tokens) != len(set(tokens)):
    #     return self.phrase_freq_shitty(tokens, slop=slop)

    # Iterate each phrase with its next term
    prior_posns: List[np.ndarray] = term_posns[0]
    for term_cnt, curr_posns in enumerate(term_posns[1:]):
        assert len(prior_posns) == len(curr_posns)
        bigram_freqs = np.zeros(len(curr_posns))
        cont_posns: List[np.ndarray] = []
        for idx in range(len(curr_posns)):

            # Find insert position of every next term in prior term's positions
            # Intuition:
            # https://colab.research.google.com/drive/1EeqHYuCiqyptd-awS67Re78pqVdTfH4A
            if len(prior_posns[idx]) == 0:
                bigram_freqs[idx] = 0
                cont_posns.append(np.asarray([]))
                continue
            priors_in_self = _self_adjs(prior_posns[idx], curr_posns[idx])
            takeaway = 0
            satisfies_slop = None
            cont_indices = None
            # Different term
            if len(priors_in_self) == 0:
                ins_posns = np.searchsorted(prior_posns[idx], curr_posns[idx], side='right')
                prior_adjacents = prior_posns[idx][ins_posns - 1]
                adjacents = curr_posns[idx] - prior_adjacents
                satisfies_slop = (adjacents <= slop) & ~(ins_posns == 0)
                cont_indices = np.argwhere(satisfies_slop).flatten()
            # Overlapping term
            else:
                adjacents = np.diff(priors_in_self)
                satisfies_slop = adjacents <= slop
                consecutive_slops = satisfies_slop[1:] & satisfies_slop[:-1]
                sum_consecutive = np.sum(consecutive_slops)
                takeaway = -np.floor_divide(sum_consecutive, -2)  # ceiling divide
                cont_indices = np.argwhere(satisfies_slop).flatten() + 1

            bigram_freqs[idx] = np.sum(satisfies_slop) - takeaway
            cont_posn = curr_posns[idx][cont_indices]
            cont_posns.append(cont_posn)
        phrase_freqs = bigram_freqs
        prior_posns = cont_posns
    return phrase_freqs
