import numpy as np
from typing import List


def advance_to(next_posns, posn, next_start, next_idx=0):
    while next_idx < next_start and not next_posns[next_idx] > posn:
        next_idx += 1
    return next_idx


def _scan_merge(prior_posns: np.ndarray,
                prior_starts: np.ndarray,
                next_posns: np.ndarray,
                next_starts: np.ndarray,
                slop=1):
    """Merge two term position lists together into a single list of bigrams.

    This is intentionally written to be naive, C-like for later porting to C

    See notebook:
    https://colab.research.google.com/drive/10zjUYHGtwMfJMPXz-BHwHe_v6j4MUZzm?authuser=1#scrollTo=W6HuiFGaYCiX

    """
    next_idx = 0
    cont_nexts = []
    cont_next = []
    bigram_freq = 0
    bigram_freqs = []
    last_prior = -2
    next_start = next_starts[0]
    prior_start = prior_starts[0]
    start_idx = 0
    prior_idx = 0
    while prior_idx < len(prior_posns):
        # Scan next until just past p
        prior = prior_posns[prior_idx]
        next_idx = advance_to(next_posns, posn=prior,
                              next_start=next_start,
                              next_idx=next_idx)

        # Reset to head of next location,
        # Re-advance next
        if next_idx >= next_start or prior_idx >= prior_start:
            next_idx = next_start
            prior_idx = prior_start
            last_prior = -2

            start_idx += 1
            if start_idx >= len(next_starts):
                break
            next_start = next_starts[start_idx]
            prior_start = prior_starts[start_idx]

            # Save and reset output
            cont_nexts.append(np.array(cont_next))
            bigram_freqs.append(bigram_freq)

            # print("Resetting with")
            # print(cont_nexts)
            # print(bigram_freqs)

            cont_next = []
            bigram_freq = 0

        prior = prior_posns[prior_idx]
        next_idx = advance_to(next_posns, posn=prior,
                              next_start=next_start, next_idx=next_idx)

        next_posn = next_posns[next_idx]
        # Check if within slop
        # And is not double counting 0->1->2 (only happens if prior/next identical)
        dist = next_posn - prior
        if dist <= slop and prior != (last_prior + 1):
            cont_next.append(next_posn)
            bigram_freq += 1
            last_prior = prior

        prior_idx += 1

    # Save last output
    cont_nexts.append(np.array(cont_next))
    bigram_freqs.append(bigram_freq)
    return bigram_freqs, cont_nexts


def scan_merge_all(prior_posns: List, next_posns: List, slop=1):
    prior_starts = np.cumsum([lst.shape[0] for lst in prior_posns])
    next_starts = np.cumsum([lst.shape[0] for lst in next_posns])
    prior_posns = np.concatenate(prior_posns)
    next_posns = np.concatenate(next_posns)

    result = _scan_merge(
        prior_posns, prior_starts,
        next_posns, next_starts,
        slop=slop
    )
    return result