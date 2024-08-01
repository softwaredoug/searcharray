#ifndef SCATTER_ASSIGN_H
#define SCATTER_ASSIGN_H

#include <stdio.h>

void scatter_naive(float *array,
				           const unsigned long long *indices,
								   const float *values,
									 int n) {

		const unsigned long long *end = indices + n;
    const unsigned long long *unroll_end = indices + (n / 10) * 10;
		while (indices < unroll_end) {
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
				array[*indices++] = *values++;
		}
		while (indices < end) {
				array[*indices++] = *values++;
		}

}

#endif // SCATTER_ASSIGN_H
