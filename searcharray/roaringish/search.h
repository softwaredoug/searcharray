#ifndef SEARCH_H
#define SEARCH_H

#include <stdint.h>

void exp_search(uint64_t *array,
	  					  uint64_t target,
								const uint64_t mask,
								uint64_t* idx_out,
								const uint64_t size) {
		uint64_t value = array[idx_out[0]] & mask;
		uint64_t posns[2] = {0, 0};
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

		// int i_right = *idx_out + 1;
		posns[0] = i_prev;
		posns[1] = *idx_out + 1;
		while (posns[0] + 1 < posns[1]) {
			*idx_out = (posns[0] + posns[1]) / 2;
			value = array[*idx_out] & mask;
			posns[target <= value] = *idx_out;
		}

		*idx_out = posns[1];
}

#endif
