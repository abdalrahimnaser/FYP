#include "MLP.h"
#include <hls_math.h>

template<int K, int N, int M, int D>
void FullyConnectedLayer(const float A[], const float B[], const float bias[], const bool activation_en, float C[]){
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


void fir(const float input[], const float taps[], float output[]){

	static float shift_reg[NO_TAPS];
	#pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0

	for(int j = 0; j < NO_SYMBOLS; j ++ ) {
		#pragma HLS pipeline II=1

		float acc = 0;
		for (int i = NO_TAPS - 1; i > 0; i--) {
			shift_reg[i] = shift_reg[i - 1];
			acc += shift_reg[i] * taps[i];
		}

		acc += input[j] * taps[0];
		shift_reg[0] = input[j];
		output[j] = acc;
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

	float filtered_I[NO_SYMBOLS];
	float filtered_Q[NO_SYMBOLS];

	fir(sig_in, taps_I, filtered_I);
	fir(sig_in+NO_SYMBOLS, taps_Q, filtered_Q);

	#ifndef __SYNTHESIS__
	for(int j = 0; j<5000; j++){printf("%.9g ", (double)filtered_I[j]);}
	#endif

	for (int i=0; i<NO_SYMBOLS; i++){
		#pragma HLS PIPELINE II=1

		input_1[0]=filtered_I[i];input_1[1]=filtered_Q[i];
		// K N M D
		FullyConnectedLayer<2, 1, NEURONS_1, 1>(input_1, weights_1, biases_1, true, input_2);
		FullyConnectedLayer<NEURONS_1, 1, NEURONS_2, 1>(input_2, weights_2,biases_1, true, input_3);
		FullyConnectedLayer<NEURONS_2, 1, NEURONS_3, 1>(input_3, weights_3, biases_3, false, output_3);

		sig_out[i] = filtered_I[i] + output_3[0];
		sig_out[i+NO_SYMBOLS] = filtered_Q[i] + output_3[1];

}

}
