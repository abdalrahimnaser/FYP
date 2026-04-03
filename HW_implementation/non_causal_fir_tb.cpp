#include "non_causal_fir.h"
#include <stdio.h>


int main() {
    float sig_out[NO_SYMBOLS];
    non_causal_fir_top(x, taps_I, taps_Q, bias, sig_out);
    for(int i=0; i<NO_SYMBOLS; i++) {
        printf("i: %d, sig_out: %f\n", i, sig_out[i]);
    }
    return 0;
}