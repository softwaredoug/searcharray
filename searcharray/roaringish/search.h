#ifndef SEARCH_H
#define SEARCH_H

#include <stdint.h>

void exp_search(uint64_t *array,
	  					  uint64_t target,
								const uint64_t mask,
								uint64_t* idx_out,
								const uint64_t size) {
		uint64_t value = array[idx_out[0]] & mask;
		target &= mask;

		if (target <= value) {
			return;
		}

		int delta = 1;
		uint64_t end = size - 1;
		int i_prev = *idx_out;

		// Exponential search
		while (value < target) {
			i_prev = *idx_out;
			*idx_out += delta;
			if (size <= *idx_out) {
				*idx_out = end;
				value = array[*idx_out] & mask;
				break;
			}
			value = array[*idx_out] & mask;
			delta *= 2;
		}

		int i_right = *idx_out + 1;
		// binary search
		while (i_prev + 1 < i_right) {
			*idx_out = (i_right + i_prev) / 2;
			value = array[*idx_out] & mask;
			if (target <= value) {
				i_right = *idx_out;
			} else {
				i_prev = *idx_out;
			}
		}

		*idx_out = i_right;
}

#endif
