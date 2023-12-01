"""Phrase search as wide span detection."""
import numpy as np


def advance_after(arr, target, idx, next_start):
    """Scan arr from idx until just after target. Return next_start if exhausted.

    arr from idx -> next_start should be sorted

    """
    while idx < next_start and not arr[idx] > target:
        idx += 1
    return idx


def next_wide_span(posns, starts, start_idx, idxs):
    """Given N terms find possible acceptable phrases for a slop.

    Parameters:
    -----------
    posns - N Numpy arrays, holding all matching document term positions
    starts - each term N start boundaries, for all D docs
    start_idx - where we are in the starts (aka doc positon)
    idxs - where we are in each term

    A wide span begins with the first term in a phrase, and finds the
    first location after that where the last term occurs. Acceptable
    wide spans are

    Query - "foo bar baz"
          posn   1   2   3   4
    t1         foo foo
    t2                 bar
    t3                     baz

          posn   1   2   3 ...   N -1  N
    t1         foo foo
    t2                 bar       bar
    t3                                baz

    To confirm these wide spans are acceptable phrase matches, we have
    to walk backwards from the end and see if phrases occur in order within span

    After consuming this wide span, the next phrase candidate can be had by advancing
    T1 to N+1th position.
    """
    #  30100000   20.651    0.000   33.100    0.000 wide_spans.py:16(next_wide_span)
    # in-place
    #  30100000   18.470    0.000   29.952    0.000 wide_spans.py:16(next_wide_span)
    #  30100000   18.078    0.000   29.297    0.000 wide_spans.py:16(next_wide_span)

    span_idx = idxs[0]
    target = posns[0][span_idx]

    term_idx = 1
    for term in posns[1:]:
        next_start = starts[term_idx][start_idx]
        span_idx = advance_after(posns[term_idx],
                                 target=target,
                                 idx=idxs[term_idx],
                                 next_start=next_start)
        if span_idx == next_start:
            return [start[start_idx] for start in starts]

        target = posns[term_idx][span_idx]
        idxs[term_idx] = span_idx
        term_idx += 1

    return idxs


def get_back_span(posns, starts, start_idx, idxs):
    """Given N terms find minimal back span.

    Parameters:
    ----------
    posns - N Numpy arrays, holding all matching document term positions
    starts - each term N start boundaries, for all D docs
    start_idx - where we are in the starts (aka doc positon)
    idxs - where we are in each term

    Returns:
    --------
    Shortest span backwards, plus positional diff

    A back span for 'foo bar baz' of a wide span, starts with end position
    then goes backwards

    Query - "foo bar baz"
          posn   1   2   3   4
    t1          foo* foo
    t2                  bar*
    t3                       baz*

          posn   1   2   3 ...   N -1  N
    t1         foo*  foo <- scan here
    t2                  bar*      bar <- scan here
    t3                                baz

    Walking backwards at idxs[-1] we  scan idxs[-2] to find
    the first occurence (going backwards)
    """
    last_posn = posns[-1][idxs[-1]]
    back_span_idxs = [idxs[-1]]

    term_idx = len(posns) - 2
    for posn in posns[::-1][1:]:
        curr_posn = posn[idxs[term_idx]]
        while curr_posn < last_posn:
            idxs[term_idx] += 1
            if idxs[term_idx] >= starts[term_idx][start_idx]:
                break
            curr_posn = posn[idxs[term_idx]]
        back_span_idxs.append(idxs[term_idx] - 1)
        last_posn = posn[back_span_idxs[-1]]
        term_idx -= 1
    back_span_idxs = back_span_idxs[::-1]
    return back_span_idxs, (posns[-1][back_span_idxs[-1]] - posns[0][back_span_idxs[0]])


def collect_span(posns, starts, start_idx, idxs, acceptable):
    # Advance 0th beyond
    beg_posn = posns[0][idxs[0]]
    last_end_posn = posns[-1][idxs[-1]]
    if (last_end_posn - beg_posn) <= acceptable:
        return 1
    else:
        back_span, posn_diff = get_back_span(posns, starts, start_idx, idxs)
        if posn_diff <= acceptable:
            return 1
    return 0


def all_wide_spans_of_slop(posns, starts, slop=1):
    """Collect all wide spans of multiple docs for all posn terms.

    Parameters:
    -----------
    posns - list of single concatenated array of all doc posns for each term
    starts - doc start posn boundaries per term
    slop - allowed (in order) moves of term

    Returns:
    --------
    phrase frequency of terms in posns satisfying given slop

    """
    start_idx = 0
    span_idxs = np.zeros(len(starts), dtype=np.uint32)
    phrases_per_doc = np.zeros(len(starts[0]), dtype=np.uint32)
    # Acceptable posn difference (<=)
    #       2-gram   3-gram
    # slop
    #
    #    1  1        2
    #    2  2        3
    acceptable = len(posns) + (slop - 2)
    first_term_start = starts[0]
    num_starts = len(starts[0])
    while True:
        if start_idx >= num_starts:
            break
        elif span_idxs[0] == first_term_start[start_idx]:
            span_idxs = [start[start_idx] for start in starts]
            start_idx += 1
        else:
            span_idxs = next_wide_span(posns, starts, start_idx, span_idxs)
            if span_idxs[0] == first_term_start[start_idx]:
                # New doc / next start
                continue

            last_end_posn = posns[-1][span_idxs[-1]]
            phrases_per_doc[start_idx] += collect_span(posns, starts, start_idx, span_idxs, acceptable)

            span_idxs[0] = advance_after(posns[0],
                                         target=last_end_posn,
                                         idx=span_idxs[0],
                                         next_start=first_term_start[start_idx])

        if span_idxs[0] == first_term_start[-1]:
            break

    return phrases_per_doc
