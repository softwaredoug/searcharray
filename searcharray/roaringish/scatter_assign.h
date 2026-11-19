#ifndef SCATTER_ASSIGN_H
#define SCATTER_ASSIGN_H

#include <stdint.h>

void scatter_naive(float *array,
                   const uint64_t *indices,
                   const float *values,
                   int n) {
    const uint64_t *end = indices + n;
    const uint64_t *unroll_end = indices + (n / 10) * 10;
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
