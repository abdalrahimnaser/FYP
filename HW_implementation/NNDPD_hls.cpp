#include "NNDPD_hls.h"
#include <hls_math.h>

typedef ap_fixed<32,12> cordic_angle_t;
typedef ap_fixed<32,12> cordic_val_t;

static data_t cordic_sin(data_t theta_rad)
{
#pragma HLS INLINE

    // Precomputed arctan(2^-i) table — 16 iterations (from PDF)
    static const cordic_angle_t cordic_phase[16] = {
        (cordic_angle_t)0.7853981634,  // arctan(2^0)  = 45.000°
        (cordic_angle_t)0.4636476090,  // arctan(2^-1) = 26.565°
        (cordic_angle_t)0.2449786631,  // arctan(2^-2) = 13.928°
        (cordic_angle_t)0.1243549945,  // arctan(2^-3) =  7.125°
        (cordic_angle_t)0.0624188100,  // arctan(2^-4) =  3.576°
        (cordic_angle_t)0.0312398334,  // arctan(2^-5) =  1.790°
        (cordic_angle_t)0.0156237286,  // arctan(2^-6) =  0.895°
        (cordic_angle_t)0.0078123411,  // arctan(2^-7) =  0.448°
        (cordic_angle_t)0.0039062301,  // arctan(2^-8)
        (cordic_angle_t)0.0019531225,  // arctan(2^-9)
        (cordic_angle_t)0.0009765622,  // arctan(2^-10)
        (cordic_angle_t)0.0004882812,  // arctan(2^-11)
        (cordic_angle_t)0.0002441406,  // arctan(2^-12)
        (cordic_angle_t)0.0001220703,  // arctan(2^-13)
        (cordic_angle_t)0.0000610352,  // arctan(2^-14)
        (cordic_angle_t)0.0000305176   // arctan(2^-15)
    };
    #pragma HLS ARRAY_PARTITION variable=cordic_phase complete dim=1

    // CORDIC gain compensation factor K = prod(cos(arctan(2^-i)))
    static const cordic_val_t CORDIC_SCALE = (cordic_val_t)0.6072529350;
    static const cordic_angle_t PI         = (cordic_angle_t)3.14159265;
    static const cordic_angle_t PI_HALF    = (cordic_angle_t)1.57079633;

    // Cast input to internal CORDIC type
    cordic_angle_t theta = (cordic_angle_t)theta_rad;

    // Normalize to [-pi/2, pi/2] for CORDIC convergence (from PDF)
    bool negate_result = false;
    if (theta > PI_HALF) {
        theta = PI - theta;
        negate_result = true;
    } else if (theta < -PI_HALF) {
        theta = -PI - theta;
        negate_result = true;
    }

    // CORDIC initialisation
    cordic_val_t   current_cos   = CORDIC_SCALE;
    cordic_val_t   current_sin   = (cordic_val_t)0;
    cordic_angle_t current_theta = (cordic_angle_t)0;

    // Iterative micro-rotations (pipelined, II=1 — matches PDF)
    CORDIC_LOOP: for (int j = 0; j < 16; j++) {
    #pragma HLS PIPELINE II=1
    //#pragma HLS UNROLL factor=2          // optional: trades LUTs for latency
        int direction = (theta > current_theta) ? 1 : -1;
        cordic_val_t   cos_temp = current_cos;
        cordic_val_t   sin_temp = current_sin;
        current_cos   = cos_temp - (cordic_val_t)(direction * (sin_temp >> j));
        current_sin   = sin_temp + (cordic_val_t)(direction * (cos_temp >> j));
        current_theta = current_theta + (cordic_angle_t)(direction * cordic_phase[j]);
    }

    if (negate_result)
        current_sin = -current_sin;

    return static_cast<data_t>(current_sin);
}

template<int K, int N, int M, int D, bool activation_en>
void FullyConnectedLayer(const data_t A[], const data_t B[], const data_t bias[], data_t C[]){
      for (int n = 0; n < N / D; ++n) {

      
       data_t acc[D][M];
       data_t tmp;
       #pragma HLS ARRAY_PARTITION variable=acc dim=1 complete


        for (int k = 0; k < K; ++k) {


          data_t a_buffer[D];
          for (int nd = 0; nd < D; ++nd) {
            #pragma HLS PIPELINE II=1
            a_buffer[nd] = A[n * D * K + nd * K + k];
          }


          for (int m = 0; m < M; ++m) {
            #pragma HLS PIPELINE II=1
            const data_t b_val = B[k * M + m];
            for (int nd = 0; nd < D; ++nd) {
              #pragma HLS UNROLL
              const data_t prev = (k > 0) ? acc[nd][m] : static_cast<data_t>(0);
              acc[nd][m] = prev + static_cast<data_t>(a_buffer[nd]) * static_cast<data_t>(b_val);
            }
          }
        }


        for (int nd = 0; nd < D; ++nd) {
          for (int m = 0; m < M; ++m) {
            #pragma HLS LOOP_FLATTEN // should i remove this? what happens to lut cnt and latency.
            #pragma HLS PIPELINE II=1
            tmp = static_cast<data_t>(acc[nd][m] + static_cast<data_t>(bias[n * D * M + nd * M + m]));
            C[n * D * M + nd * M + m] =(activation_en==true)? static_cast<data_t>(cordic_sin(tmp)) : tmp;
          }
        }
      }
}



template<int no_taps, int no_symbols, int half_taps>
void fir_cmplx(const data_t input_I[], const data_t input_Q[], const data_t taps_I[], const data_t taps_Q[], const data_t bias, data_t output[]){


    data_t shift_reg_I[no_taps]={0};
    data_t shift_reg_Q[no_taps]={0};
    data_t tmp_I, tmp_Q;
    #pragma HLS ARRAY_PARTITION variable=shift_reg_I complete dim=0 // allow parallel access to shift_reg (synthesie into registers not BRAM) - needed when combined with UNROLL
    #pragma HLS ARRAY_PARTITION variable=shift_reg_Q complete dim=0 
    #pragma HLS ARRAY_PARTITION variable=taps_I complete dim=0 
    #pragma HLS ARRAY_PARTITION variable=taps_Q complete dim=0 


    for(int i=half_taps; i<no_taps; i++){
        #pragma HLS pipeline II=1 // OR UNROLL
        shift_reg_I[i] = input_I[i-half_taps];
        shift_reg_Q[i] = input_Q[i-half_taps];
    }


    for(int j = 0; j < no_symbols; j ++ ) {
        #pragma HLS pipeline II=1


        data_t acc = 0;
        for (int i = 0; i < no_taps-1; i++) {
            #pragma HLS UNROLL
            // refer to [https://github.com/Xilinx/xup_high_level_synthesis_design_flow/blob/main/source/fir/notebook/fir_part2.ipynb]
            acc += static_cast<data_t>(shift_reg_I[i]) * static_cast<data_t>(taps_I[i]) +
                   static_cast<data_t>(shift_reg_Q[i]) * static_cast<data_t>(taps_Q[i]);
            shift_reg_I[i] = shift_reg_I[i + 1];
            shift_reg_Q[i] = shift_reg_Q[i + 1];
        }


        acc += static_cast<data_t>(tmp_I) * static_cast<data_t>(taps_I[no_taps-1]) +
               static_cast<data_t>(tmp_Q) * static_cast<data_t>(taps_Q[no_taps-1]);
        tmp_I = (j<(no_symbols-half_taps-1)) ? input_I[j+1+half_taps] : static_cast<data_t>(0);
        tmp_Q = (j<(no_symbols-half_taps-1)) ? input_Q[j+1+half_taps] : static_cast<data_t>(0);
        shift_reg_I[no_taps-1] = tmp_I;
        shift_reg_Q[no_taps-1] = tmp_Q;
        output[j] = static_cast<data_t>(acc + static_cast<data_t>(bias));
    }
}



void MultilayerPerceptron(const data_t sigI_in[], const data_t sigQ_in[], data_t sigI_out[], data_t sigQ_out[]) {
#pragma HLS INTERFACE m_axi port=sigI_in bundle=gmem0 offset=slave depth=NO_SYMBOLS
#pragma HLS INTERFACE m_axi port=sigQ_in bundle=gmem1 offset=slave depth=NO_SYMBOLS
#pragma HLS INTERFACE m_axi port=sigI_out bundle=gmem0 offset=slave depth=NO_SYMBOLS
#pragma HLS INTERFACE m_axi port=sigQ_out bundle=gmem1 offset=slave depth=NO_SYMBOLS
#pragma HLS INTERFACE s_axilite port=sigI_in bundle=control // sig?
#pragma HLS INTERFACE s_axilite port=sigQ_in bundle=control // sig?
#pragma HLS INTERFACE s_axilite port=sigI_out bundle=control // sig?
#pragma HLS INTERFACE s_axilite port=sigQ_out bundle=control // sig?
#pragma HLS INTERFACE s_axilite port=return bundle=control


    data_t input_1[2];
    data_t input_2[NEURONS_1];
    data_t input_3[NEURONS_2];
    data_t output_3[2];


    data_t inI[NO_SYMBOLS];
    data_t inQ[NO_SYMBOLS];
    data_t fir1_out[NO_SYMBOLS];
    data_t fir2_out[NO_SYMBOLS];
    //data_t output[NO_SYMBOLS*2];

	#pragma HLS BIND_STORAGE variable=inI type=RAM_S2P // ram_s2p means two reads on one port, and two writes on the other port ... i only need two reads on one and one write on other
    #pragma HLS BIND_STORAGE variable=inQ type=RAM_S2P
	//#pragma HLS DATAFLOW

    for (int i=0; i<NO_SYMBOLS; i++) {
    #pragma HLS PIPELINE II=1
        inI[i] = sigI_in[i];
		inQ[i] = sigQ_in[i];
    }




    fir_cmplx<NO_TAPS, NO_SYMBOLS, HALF_TAPS>(inI, inQ, taps1_I, taps1_Q, bias1, fir1_out);
    fir_cmplx<NO_TAPS, NO_SYMBOLS, HALF_TAPS>(inI, inQ, taps2_I, taps2_Q, bias2, fir2_out);


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


        sigI_out[i] = static_cast<data_t>(static_cast<data_t>(fir1_out[i]) + static_cast<data_t>(output_3[0]));
        sigQ_out[i] = static_cast<data_t>(static_cast<data_t>(fir2_out[i]) + static_cast<data_t>(output_3[1]));


    }


    // for (int i=0; i<NO_SYMBOLS*2; i++) {
    //  #pragma HLS PIPELINE II=1
    //  sig_out[i] = output[i];
    // }
}
