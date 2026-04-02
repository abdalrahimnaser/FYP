#include "non_causal_fir.h"
#include <stdio.h>

void fir_non_causal(const float input[], const float taps[], float output[]){

	float shift_reg[NO_TAPS]={0};
    float tmp;
	#pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0 // allow parallel access to shift_reg (synthesie into registers not BRAM) - needed when combined with UNROLL

    for(int i=HALF_TAPS; i<NO_TAPS; i++){
        #pragma HLS pipeline II=1 // OR UNROLL
        shift_reg[i] = input[i-HALF_TAPS];
    }

	for(int j = 0; j < NO_SYMBOLS; j ++ ) {
		#pragma HLS pipeline II=1

		float acc = 0;
		for (int i = 0; i < NO_TAPS-1; i++) {
			// TODO: implement UNROLL logic here, just add the relevant pragma
			// refer to https://github.com/Xilinx/xup_high_level_synthesis_design_flow/blob/main/source/fir/notebook/fir_part2.ipynb
			// though that needs to be applied to both shift reg and taps. as taps needs to be accessed in parallel as well
			acc += shift_reg[i] * taps[i];
			shift_reg[i] = shift_reg[i + 1];
		}

		acc += tmp * taps[NO_TAPS-1];
		tmp = (j<(NO_SYMBOLS-HALF_TAPS-1)) ? input[j+1+HALF_TAPS] : 0;
		shift_reg[NO_TAPS-1] = tmp;
		output[j] = acc;
	}
}

void fir_non_causal_cmplx(const float input_I[], const float input_Q[], const float taps_I[], const float taps_Q[], float output[]){

	float shift_reg_I[NO_TAPS]={0};
	float shift_reg_Q[NO_TAPS]={0};
    float tmp_I, tmp_Q;
	#pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0 // allow parallel access to shift_reg (synthesie into registers not BRAM) - needed when combined with UNROLL

    for(int i=HALF_TAPS; i<NO_TAPS; i++){
        #pragma HLS pipeline II=1 // OR UNROLL
		shift_reg_I[i] = input_I[i-HALF_TAPS];
		shift_reg_Q[i] = input_Q[i-HALF_TAPS];
    }

	for(int j = 0; j < NO_SYMBOLS; j ++ ) {
		#pragma HLS pipeline II=1

		float acc = 0;
		for (int i = 0; i < NO_TAPS-1; i++) {
			// TODO: implement UNROLL logic here, just add the relevant pragma
			// refer to https://github.com/Xilinx/xup_high_level_synthesis_design_flow/blob/main/source/fir/notebook/fir_part2.ipynb
			// though that needs to be applied to both shift reg and taps. as taps needs to be accessed in parallel as well
			acc += shift_reg_I[i] * taps_I[i] + shift_reg_Q[i] * taps_Q[i];
			shift_reg_I[i] = shift_reg_I[i + 1];
			shift_reg_Q[i] = shift_reg_Q[i + 1];
		}

		acc += tmp_I * taps_I[NO_TAPS-1] + tmp_Q * taps_Q[NO_TAPS-1];
		tmp_I = (j<(NO_SYMBOLS-HALF_TAPS-1)) ? input_I[j+1+HALF_TAPS] : 0;
		tmp_Q = (j<(NO_SYMBOLS-HALF_TAPS-1)) ? input_Q[j+1+HALF_TAPS] : 0;
		shift_reg_I[NO_TAPS-1] = tmp_I;
		shift_reg_Q[NO_TAPS-1] = tmp_Q;
		output[j] = acc;
	}
}



void non_causal_fir_top(const float sig_in[], const float taps_I[], const float taps_Q[], float sig_out[]) {
    #pragma HLS INTERFACE m_axi port=sig_in bundle=gmem0 offset=slave depth=NO_SYMBOLS*2
    #pragma HLS INTERFACE m_axi port=sig_out bundle=gmem1 offset=slave depth=NO_SYMBOLS
    #pragma HLS INTERFACE s_axilite port=sig_in bundle=control // sig?
    #pragma HLS INTERFACE s_axilite port=sig_out bundle=control // sig?
    #pragma HLS INTERFACE s_axilite port=return bundle=control


	fir_non_causal_cmplx(sig_in, sig_in+NO_SYMBOLS, taps_I, taps_Q, sig_out);
}
