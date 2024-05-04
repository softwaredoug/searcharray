#ifndef SEARCH_H
#define SEARCH_H

#include <stdint.h>
#include <stdio.h>

void exp_search(uint64_t *array,
	  					  uint64_t target,
								const uint64_t mask,
								uint64_t* idx_out,
								const uint64_t size) {
		uint64_t value = array[*idx_out] & mask;
		uint64_t posns[2] = {*idx_out, *idx_out};
		target &= mask;

		if (target <= value) {
			return;
		}

		int delta = 1;

		// Exponential search
		while (value < target) {
			posns[0] = posns[1];
			posns[1] += delta;
			if (size <= posns[1]) {
				posns[1] = size - 1;
				value = array[posns[1]] & mask;
				break;
			}
			value = array[posns[1]] & mask;
			delta *= 2;
		}

		posns[1] = *idx_out + 1;
		// branchless binary search
		while (posns[0] + 1 < posns[1]) {
			*idx_out = (posns[0] + posns[1]) / 2;
			value = array[*idx_out] & mask;
			posns[value < target] = *idx_out;
		}

		*idx_out = posns[1];
}

#endif
