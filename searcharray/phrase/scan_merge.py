import numpy as np
from typing import List


def advance_to_scan(arr, target, next_start, idx):
    while idx < next_start and not arr[idx] > target:
        idx += 1
    return idx


def advance_after_binsearch(arr, target, next_start, idx):
    """Scan arr from idx until just after target. Return next_start if exhausted.

    Will manually create a skip index if next_start - idx is large

    arr from idx -> next_start should be sorted

    """
    beg = idx
    end = next_start - 1
    space = end - beg
    if space < 1000:
        advance_to_scan(arr, target, next_start, idx)

    # print(f"Binsearching {target} in arr[{beg}:{end}] = {arr[beg:end]}")
    if arr[idx] > target:
        # print("Already past, returning idx")
        return idx

    mid = None
    while True:
        space = end - beg
        if space <= 1000:
            return advance_to_scan(arr, target, next_start, beg)

        mid = beg + (space // 2)
        if arr[mid] == target:
            # print(f"Found exact match, returning {mid + 1}")
            return mid + 1
        elif arr[mid] > target and arr[mid - 1] <= target:
            # print(f"Found one past, returning {mid}")
            return mid
        elif arr[mid] < target:
            # print(f"Moving beg to {mid}")
            # print(f"       end is {end}")
            # print(f"arr[{mid}] is {arr[mid]}")
            beg = mid
        else:  # elif arr[mid] > target:
            # print(f"       beg is {beg}")
            # print(f"Moving end to {mid}")
            # print(f"arr[{mid}] is {arr[mid]}")
            end = mid

    # print(f"Exhausted binsearch, returning {next_start}")
    return next_start


def scan_merge(prior_posns: np.ndarray,
               prior_starts: np.ndarray,
               next_posns: np.ndarray,
               next_starts: np.ndarray,
               scan_algo=advance_to_scan,
               slop=1):
    """Merge two term position lists together into a single list of bigrams.

    Each position list is a flattened representation of multiple docs, ie

    prior_posns: [0,2,1,4,5]
    prior_starts: [0,2]

    Points to the first position of each doc in the prior_posns list for this term

    (same for next_posns)

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
        # Scan next until just past prior
        prior = prior_posns[prior_idx]
        next_idx = scan_algo(next_posns, target=prior,
                             next_start=next_start,
                             idx=next_idx)

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

            cont_next = []
            bigram_freq = 0

        prior = prior_posns[prior_idx]
        next_idx = scan_algo(next_posns, target=prior,
                             next_start=next_start, idx=next_idx)

        if next_idx >= next_start:
            continue

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


def scan_merge_inplace(prior_posns: np.ndarray,
                       prior_starts: np.ndarray,
                       next_posns: np.ndarray,
                       next_starts: np.ndarray,
                       scan_algo=advance_to_scan,
                       slop=1):
    """Merge two term position lists together into a single list of bigrams.

    Same as scan_merge, but doesn't need a dynamic array output, instead
    overwrites prior* buffers
    """
    next_idx = 0
    bigram_freq = 0
    bigram_freqs = []  # this could also be preallocated array output
    last_prior = -2
    output_idx = 0
    next_start = next_starts[0]
    prior_start = prior_starts[0]
    start_idx = 0
    prior_idx = 0

    while prior_idx < prior_starts[-1]:
        # Scan next until just past p
        prior = prior_posns[prior_idx]
        next_idx = scan_algo(next_posns,
                             target=prior,
                             next_start=next_start,
                             idx=next_idx)

        # Reset to head of next location,
        # Re-advance next
        if next_idx >= next_start or prior_idx >= prior_start:
            next_idx = next_start
            prior_idx = prior_start
            last_prior = -2

            prior_starts[start_idx] = output_idx

            start_idx += 1
            if start_idx >= len(next_starts):
                break
            next_start = next_starts[start_idx]
            prior_start = prior_starts[start_idx]

            # Save and reset output
            bigram_freqs.append(bigram_freq)

            bigram_freq = 0

        prior = prior_posns[prior_idx]
        next_idx = scan_algo(next_posns,
                             target=prior,
                             next_start=next_start,
                             idx=next_idx)
        if next_idx >= next_start:
            continue

        next_posn = next_posns[next_idx]
        # Check if within slop
        # And is not double counting 0->1->2 (only happens if prior/next identical)
        dist = next_posn - prior
        if dist <= slop and prior != (last_prior + 1):
            prior_posns[output_idx] = next_posn
            output_idx += 1
            bigram_freq += 1
            last_prior = prior

        prior_idx += 1

    # Save last output
    bigram_freqs.append(bigram_freq)
    if start_idx < len(next_starts):
        prior_starts[start_idx] = output_idx

    return bigram_freqs, prior_posns, prior_starts


def scan_merge_bigram(prior_posns: List, next_posns: List, slop=1):
    prior_starts = np.cumsum([lst.shape[0] for lst in prior_posns])
    next_starts = np.cumsum([lst.shape[0] for lst in next_posns])
    prior_posns = np.concatenate(prior_posns)
    next_posns = np.concatenate(next_posns)

    result = scan_merge(
        prior_posns, prior_starts,
        next_posns, next_starts,
        slop=slop
    )
    return result
