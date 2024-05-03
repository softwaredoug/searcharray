# These are modified versions of sortednp:
#   https://gitlab.sauerburger.com/frank/sortednp
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
cimport numpy as np
import numpy as np
from enum import Enum

cimport searcharray.roaringish.snp_ops
from searcharray.roaringish.snp_ops cimport _galloping_search, DTYPE_t


cdef DTYPE_t ALL_BITS = 0xFFFFFFFFFFFFFFFF


cdef extern from "stddef.h":
    # Assuming size_t is available via stddef.h for the example's simplicity
    # and portability, though it's not directly used here.
    int __builtin_popcountll(unsigned long long x)

# Popcount reduce key-value pair
# for words 0xKKKKKKKK...KKKKVVVV...VVVV
# Returning two parallel arrays:
#   - keys   - the keys
#   - values - the popcount of the values with those keys
cdef _popcount64_reduce(DTYPE_t[:] arr,
                        DTYPE_t key_shift,
                        DTYPE_t value_mask):
    cdef float[:] popcounts = np.zeros(arr.shape[0], dtype=np.float32)
    cdef DTYPE_t[:] keys = np.empty(arr.shape[0], dtype=np.uint64)
    # cdef int i = 0
    cdef float* popcounts_ptr = &popcounts[0]
    cdef DTYPE_t* keys_ptr = &keys[0]
    cdef DTYPE_t* arr_ptr = &arr[0]
    cdef DTYPE_t last_key = 0xFFFFFFFFFFFFFFFF

    for _ in range(arr.shape[0]):
        popcounts_ptr[0] += __builtin_popcountll(arr_ptr[0] & value_mask)
        if arr_ptr[0] >> key_shift != last_key:
            last_key = arr_ptr[0] >> key_shift
            keys_ptr[0] = last_key
            popcounts_ptr += 1
            keys_ptr += 1
        arr_ptr += 1
    return keys, popcounts, keys_ptr - &keys[0]


def popcount64_reduce(arr, key_shift, value_mask):
    cdef DTYPE_t[:] arr_view = arr
    keys, popcounts, results_idx = _popcount64_reduce(arr_view, key_shift, value_mask)
    return np.array(keys[:results_idx]), np.array(popcounts[:results_idx])


cdef _payload_slice(DTYPE_t[:] arr,
                    DTYPE_t payload_msb_mask,
                    DTYPE_t min_payload,
                    DTYPE_t max_payload):
    cdef DTYPE_t[:] sliced = np.zeros(arr.shape[0], dtype=np.uint64)
    cdef DTYPE_t* sliced_ptr = &sliced[0]
    cdef DTYPE_t* arr_ptr = &arr[0]

    while arr_ptr < &arr[arr.shape[0]]:
        if (arr_ptr[0] & payload_msb_mask) >= min_payload and (arr_ptr[0] & payload_msb_mask) <= max_payload:
            sliced_ptr[0] = arr_ptr[0]
            sliced_ptr += 1
        arr_ptr += 1

    return sliced, sliced_ptr - &sliced[0]


def payload_slice(arr, payload_msb_mask,
                  min_payload=0, max_payload=0xFFFFFFFFFFFFFFFF):
    cdef DTYPE_t[:] arr_view = arr
    sliced, sliced_len = _payload_slice(arr_view, payload_msb_mask,
                                        min_payload, max_payload)
    return np.array(sliced[:sliced_len])


cdef _as_dense_array(DTYPE_t[:] indices,  # Its likely these indices are sorted, if that helps
                     float[:] values,
                     DTYPE_t size):

    cdef DTYPE_t* indices_ptr = &indices[0]
    cdef float* values_ptr = &values[0]
    cdef float[:] arr = np.zeros(size, dtype=np.float32)

    while indices_ptr < &indices[indices.shape[0]]:
        arr[indices_ptr[0]] = values_ptr[0]
        indices_ptr += 1
        values_ptr += 1

    return arr


def as_dense(indices, values, size):
    cdef DTYPE_t[:] indices_view = indices
    cdef float[:] values_view = values
    if len(indices) != len(values):
        raise ValueError("indices and values must have the same length")
    return np.array(_as_dense_array(indices_view, values_view, size))


cdef DTYPE_t ceiling_div(DTYPE_t dividend, DTYPE_t divisor):
    return (dividend + divisor - 1) // divisor


cdef _count_same_term(DTYPE_t target, DTYPE_t other,
                      DTYPE_t payload_mask, DTYPE_t header_mask):
    cdef DTYPE_t overlap = (target & (other >> 1)) & payload_mask
    cdef DTYPE_t num_overlaps = __builtin_popcountll(overlap)
    cdef DTYPE_t consecs = __builtin_popcountll(overlap & overlap << 1)
    cdef DTYPE_t num_consecs = ceiling_div(consecs, 2)
    cdef DTYPE_t bigram_freq = num_overlaps - num_consecs
    print( " >") 
    print( " > CHECK SAME TERM")
    print(f" >     target: {(target & payload_mask):0x}")
    print(f" >      other: {(other & payload_mask):0x}")
    target = ((other << 1) &  other) | (target & header_mask)
    print(f" > new target: {(target & payload_mask):0x}")
    print(f" > num_overlaps: {num_overlaps}")
    print(f" > phrase_fre: {bigram_freq}")
    print( " >") 
    return bigram_freq, target


cdef _count_diff_term(DTYPE_t target, DTYPE_t other, DTYPE_t other_adj,
                      DTYPE_t cont_mask, DTYPE_t payload_msb,
                      DTYPE_t payload_mask, DTYPE_t header_mask):
    cdef DTYPE_t bigram_freq = 0
    cdef bint same_adj_header = False
    cdef DTYPE_t adj_bit_set = 0
    cdef bint same_header = False

    same_adj_header = ((target + (header_mask & -header_mask)) & header_mask) == (other_adj & header_mask)
    if same_adj_header:
        print("SAME ADJ HEADER")
        if cont_mask == 0 and (target & payload_msb == payload_msb):
            cont_mask = 1
        adj_bit_set = other_adj & cont_mask
        print(f"ADJ BIT SET: {adj_bit_set:0x}")
        if adj_bit_set == 0:
            cont_mask = 0
    # Either the intersect has adjacent bits, or the adjacent bits are in the other array
    same_header = (target & header_mask) == (other & header_mask)
    inner_bigrams = ((target & payload_mask) & 
                     ((other & payload_mask) >> 1))
    if same_header and inner_bigrams > 0 and adj_bit_set > 0:
        print("INNER + ADJ")
        target = inner_bigrams << 1 | (target & header_mask)
        bigram_freq = __builtin_popcountll(inner_bigrams) + 1
    elif same_header and inner_bigrams > 0:
        print("INNER")
        target = inner_bigrams << 1 | (target & header_mask)
        bigram_freq = __builtin_popcountll(inner_bigrams)
    elif same_adj_header and adj_bit_set > 0:
        print("ADJ")
        bigram_freq = 1
    else:
        print("EXIT!")
        bigram_freq = 0
        target = 0
    return bigram_freq, target, cont_mask


cdef _phrase_search(list encoded_posns,
                    DTYPE_t header_mask, DTYPE_t key_mask, 
                    DTYPE_t header_bits, DTYPE_t key_bits,
                    double[:] phrase_freqs_out):  # Output
    """Given encoded positions, count occurences of phrases.

    hdr | payload
    encoded_posns[0] - first term

    123 14  000010   123 15 000000
    123 14  000001   123 15 001000
    123 14  000000   123 15 100000
                ^^          ^
                phrase (adjacent bits, even across payloads)

    1. Intersect the headers, starting with the shortest list

                         123 14        123 15        124 12      124 13       <-- shortest
                123 13   123 14        123 15        124 12
       123 12   123 13   123 14        123 15        124 12               124 14   <-- longest

                         ______        ______        ______              <-- intersected w/ shortest
                ------                                                   <-- intersected - 1
                                                                  ------ <-- intersected + 1

    2. Collapse to intersected, keeping doc id (aka key) and payload with MSB or LSB set

       arr1 (payloads)   (bits + adj LHS/RHS) 
       arr2 (keys)       123           123           124
    """
    # TODO - reintroduce optimization starting with shortest, for now lets 
    # get this to work going left to right
    cdef DTYPE_t[:] shortest_arr = encoded_posns[0]
    cdef DTYPE_t[:] curr_arr = encoded_posns[0]
    cdef DTYPE_t shortest_arr_idx = 0

    cdef DTYPE_t curr_arr_idx = 0
    cdef DTYPE_t payload_mask = ALL_BITS & ~header_mask
    cdef DTYPE_t payload_num_bits = 64 - header_bits
    # for arr in encoded_posns:
    #     if shortest_arr is None or arr.shape[0] < shortest_arr.shape[0]:
    #         shortest_arr = arr
    #         shortest_arr_idx = curr_arr_idx
    #    curr_arr_idx += 1
    cdef np.intp_t i_other = 0
    cdef np.intp_t i_other_adj = 0
    cdef np.intp_t i_shortest = 0
    cdef DTYPE_t i_arr = 0
    cdef DTYPE_t value_shortest = 0 
    cdef DTYPE_t target = 0
    cdef DTYPE_t target_adj = 0
    cdef DTYPE_t write_key = 0
    cdef DTYPE_t inner_bigrams = 0
    cdef DTYPE_t cont_mask = 0
    cdef DTYPE_t dir_adj_bit_set = 0
    cdef DTYPE_t payload_msb = 1 << (payload_num_bits - 1)
    cdef double bigram_freq = 0.0
    cdef bint same_term = False
    cdef bint same_header = False

    # For each item in the shortest array,
    # Find any bigram match before / after
    while i_shortest < shortest_arr.shape[0]:
        # Find phrases left -> right
        curr_arr_idx = shortest_arr_idx + 1
        target = shortest_arr[i_shortest]
        cont_mask = 0
        bigram_freq = 0.0
        print("****************")
        print("SEARCHING BIGRAMS...")
        print(f"New target: { (target & header_mask):0x} {(target & payload_mask):0x}")
        print(f"i_shortest: {i_shortest}")
        while curr_arr_idx < len(encoded_posns):
            # The target is the current term's intersecting bigram
            # Each new term can produce up to two new targets, one for the starting
            # 64 bit word, where we look for bit intersections. 
            # The other is for the adjacent bit, where we continue to scan down for adjacent 
            # bits, and that will either produce one phrase or be terminated
            curr_arr = encoded_posns[curr_arr_idx]

            # Really each array should have its own index for these
            # same_term = (encoded_posns[curr_arr_idx] is encoded_posns[curr_arr_idx - 1])
            i_other = 0
            i_other_adj = 0
            # Direct intersected
            _galloping_search(curr_arr,
                              target,
                              header_mask,
                              &i_other,
                              encoded_posns[curr_arr_idx].shape[0])
            same_header = (i_other < encoded_posns[curr_arr_idx].shape[0]) and (target & header_mask) == (curr_arr[i_other] & header_mask)
            same_term = same_header and (
                __builtin_popcountll(target & payload_mask) > 0
            ) and (
                target == curr_arr[i_other]
            )
            if not same_term:
                _galloping_search(curr_arr,
                                  target + (header_mask & -header_mask),  # +1 on header
                                  header_mask,
                                  &i_other_adj,
                                  curr_arr.shape[0])
                print( " >")
                print( " > CHECK DIFF TERM")
                print(f" >     target:{(target & header_mask):0x} {(target & payload_mask):0x}")
                print(f" >      other:{(curr_arr[i_other] & header_mask):0x} {(curr_arr[i_other] & payload_mask):0x}")
                print(f" >  other_adj:{(curr_arr[i_other_adj] & header_mask):0x} {(curr_arr[i_other_adj] & payload_mask):0x}")
                print(f" > same_header: {same_header}")

                bigram_freq, target, cont_mask = _count_diff_term(target,
                                                                  curr_arr[i_other], curr_arr[i_other_adj],
                                                                  cont_mask, payload_msb,
                                                                  payload_mask, header_mask)
                print(f" > new target: {(target & header_mask):0x} {(target & payload_mask):0x}")
                print(f" >      other: {(curr_arr[i_other] & payload_mask):0x}")
                print(f" > other_adj: {(curr_arr[i_other_adj] & payload_mask):0x}")
                print(f" > phrase_fre: {bigram_freq}")
                print( " >")
            if target == 0 and cont_mask == 0:
                break
            curr_arr_idx += 1
            cont_mask <<= 1

        write_key = target >> (64 - key_bits)
        phrase_freqs_out[write_key] += bigram_freq
        print(f"Collecting {write_key}:{phrase_freqs_out[write_key]}")
        i_shortest += 1

        # Go right -> left
        # Phrase freq for this is min(right, left)


def phrase_search(encoded_posns, header_mask, key_mask,
                  header_bits, key_bits,
                  phrase_freqs):
    if len(encoded_posns) > 64 - header_bits:
        raise ValueError(f"Phrase too long - max phrase len: {64 - header_bits} ")
    cdef double[:] phrase_freqs_view = phrase_freqs
    _phrase_search(encoded_posns, header_mask, key_mask, 
                   header_bits, key_bits,
                   phrase_freqs_view)
