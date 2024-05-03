#ifndef SEARCH_H
#define SEARCH_H

#include <stdint.h>
#include <stdio.h>
#include <mach/mach_time.h>

// TEMP!
uint64_t bin_time = 0;
uint64_t exp_time = 0;
uint64_t total_time = 0;


void exp_search(uint64_t *array,
	  					  uint64_t target,
								const uint64_t mask,
								uint64_t* idx_out,
								const uint64_t size) {
		uint64_t start_time = mach_absolute_time();
		uint64_t *curr = &array[idx_out[0]];
		uint64_t *prev;   //= &array[idx_out[0]];
		uint64_t *right;  // = &array[idx_out[0]];
		int delta = 1;

		target &= mask;

		/*if (target <= (*curr & mask)) {
			return;
		}*/

		// unrolled exp search
		prev = curr;
		/*while ((curr + 64 < array + size) &&
				((*curr & mask) <= target) && (curr += 1) &&
		    ((*curr & mask) <= target) && (curr += 2) &&
		    ((*curr & mask) <= target) && (curr += 4) &&
		    ((*curr & mask) <= target) && (curr += 8) &&
		    ((*curr & mask) <= target) && (curr += 16) &&
		    ((*curr & mask) <= target) && (curr += 32)) {
				prev = curr;
		}*/
		while ((*curr & mask) < target) {
				prev = curr;
				curr += delta;
				// if exceeded size, break
				if ( curr >= array + size) {
					break;
				}
				delta *= 2;
		}
		exp_time += mach_absolute_time() - start_time;
		start_time = mach_absolute_time();

		right = curr + 1;
		// binary search
		while (prev + 1 < right) {
			curr = (right - prev) / 2 + prev;
			if (target <= (*curr & mask)) {
				right = curr;
			} else {
				prev = curr;
			}
		}
		bin_time += mach_absolute_time() - start_time;
		*idx_out = right - array;
		printf("***************\n");
		printf("  EXP Time taken: %llu (%lf)\n", exp_time, (double)exp_time / (double)(bin_time + exp_time));
		printf("  BIN Time taken: %llu (%lf)\n", bin_time, (double)bin_time / (double)(bin_time + exp_time));
		total_time += mach_absolute_time() - start_time;
}

#endif
