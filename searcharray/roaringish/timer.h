#ifndef TIMER_H
#define TIMER_H


#include <stdio.h>
#include "mach/mach_time.h"


uint64_t timestamp() {
    return mach_absolute_time();
}


void print_elapsed(uint64_t elapsed, const char *msg) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double nanos = elapsed * ((double)(info.numer) / (double)(info.denom));
    if (nanos > 1000000000) {
        printf("%s Elapsed time: %lf s\n", msg, nanos / 1000000000.0);
    } else if (nanos > 1000000) {
        printf("%s Elapsed time: %lf ms\n", msg, nanos / 1000000.0);
    } else {
        printf("%s Elapsed time: %lf ns\n", msg, nanos);
    }
}


#endif // TIMER_H
