#include "MLP.h"
#include <hls_math.h>

template<int K, int N, int M, int D, bool activation_en>
void FullyConnectedLayer(const float A[], const float B[], const float bias[], float C[]){
	  for (int n = 0; n < N / D; ++n) {

	   float acc[D][M];
	   #pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

	    for (int k = 0; k < K; ++k) {

	      float a_buffer[D];
	      for (int nd = 0; nd < D; ++nd) {
	        #pragma HLS PIPELINE II=1
	        a_buffer[nd] = A[n * D * K + nd * K + k];
	      }

	      for (int m = 0; m < M; ++m) {
	        #pragma HLS PIPELINE II=1
	        const auto b_val = B[k * M + m];
	        for (int nd = 0; nd < D; ++nd) {
	          #pragma HLS UNROLL
	          const auto prev = (k > 0) ? acc[nd][m] : static_cast<float>(0);
	          acc[nd][m] = prev + a_buffer[nd] * b_val;
	        }
	      }
	    }

	    for (int nd = 0; nd < D; ++nd) {
	      for (int m = 0; m < M; ++m) {
	        #pragma HLS LOOP_FLATTEN
	        #pragma HLS PIPELINE II=1
	        C[n * D * M + nd * M + m] = (activation_en==true)? sin(acc[nd][m] + bias[n * D * M + nd * M + m]):(acc[nd][m] + bias[n * D * M + nd * M + m]) ;
	      }
	    }
	  }
}


template<int no_taps, int no_symbols, int half_taps>
void fir_cmplx(const float input_I[], const float input_Q[], const float taps_I[], const float taps_Q[], const float bias, float output[]){

	float shift_reg_I[no_taps]={0};
	float shift_reg_Q[no_taps]={0};
    float tmp_I, tmp_Q;
	// add the below pragma for both shift_reg_I, shift_reg_Q , taps_I , taps_Q IN ADDITION TO UNROLL below 
	#pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0 // allow parallel access to shift_reg (synthesie into registers not BRAM) - needed when combined with UNROLL

    for(int i=half_taps; i<no_taps; i++){
        #pragma HLS pipeline II=1 // OR UNROLL
		shift_reg_I[i] = input_I[i-half_taps];
		shift_reg_Q[i] = input_Q[i-half_taps];
    }

	for(int j = 0; j < no_symbols; j ++ ) {
		#pragma HLS pipeline II=1

		float acc = 0;
		for (int i = 0; i < no_taps-1; i++) {
			// TODO: implement UNROLL logic here, just add the relevant pragma
			// refer to https://github.com/Xilinx/xup_high_level_synthesis_design_flow/blob/main/source/fir/notebook/fir_part2.ipynb
			// though that needs to be applied to both shift reg and taps. as taps needs to be accessed in parallel as well
			acc += shift_reg_I[i] * taps_I[i] + shift_reg_Q[i] * taps_Q[i];
			shift_reg_I[i] = shift_reg_I[i + 1];
			shift_reg_Q[i] = shift_reg_Q[i + 1];
		}

		acc += tmp_I * taps_I[no_taps-1] + tmp_Q * taps_Q[no_taps-1];
		tmp_I = (j<(no_symbols-half_taps-1)) ? input_I[j+1+half_taps] : 0;
		tmp_Q = (j<(no_symbols-half_taps-1)) ? input_Q[j+1+half_taps] : 0;
		shift_reg_I[no_taps-1] = tmp_I;
		shift_reg_Q[no_taps-1] = tmp_Q;
		output[j] = acc + bias;
	}
}


void MultilayerPerceptron(const float sig_in[], float sig_out[]) {
#pragma HLS INTERFACE m_axi port=sig_in bundle=gmem0 offset=slave depth=NO_SYMBOLS*2
#pragma HLS INTERFACE m_axi port=sig_out bundle=gmem1 offset=slave depth=NO_SYMBOLS*2
#pragma HLS INTERFACE s_axilite port=sig_in bundle=control // sig?
#pragma HLS INTERFACE s_axilite port=sig_out bundle=control // sig?
#pragma HLS INTERFACE s_axilite port=return bundle=control

	float input_1[2];
	float input_2[NEURONS_1];
	float input_3[NEURONS_2];
	float output_3[2];

	float fir1_out[NO_SYMBOLS];
	float fir2_out[NO_SYMBOLS];

	fir_cmplx<NO_TAPS, NO_SYMBOLS, HALF_TAPS>(sig_in, sig_in+NO_SYMBOLS, taps1_I, taps1_Q, bias1, fir1_out);
	fir_cmplx<NO_TAPS, NO_SYMBOLS, HALF_TAPS>(sig_in, sig_in+NO_SYMBOLS, taps2_I, taps2_Q, bias2, fir2_out);

	#ifndef __SYNTHESIS__
	// for(int j = 0; j<5000; j++){printf("%.9g ", (double)filtered_I[j]);}
	#endif

	for (int i=0; i<NO_SYMBOLS; i++){
		#pragma HLS PIPELINE II=1

		input_1[0]=fir1_out[i];input_1[1]=fir2_out[i];
		// K N M D
		FullyConnectedLayer<2, 1, NEURONS_1, 1, true>(input_1, weights_1, biases_1, input_2);
		FullyConnectedLayer<NEURONS_1, 1, NEURONS_2, 1, true>(input_2, weights_2, biases_2, input_3);
		FullyConnectedLayer<NEURONS_2, 1, NEURONS_3, 1, false>(input_3, weights_3, biases_3, output_3);

		sig_out[i] = fir1_out[i] + output_3[0];
		sig_out[i+NO_SYMBOLS] = fir2_out[i] + output_3[1];

}

}
