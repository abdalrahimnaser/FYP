#include <math.h>
#include <stdio.h>
#include "tb_NNDPD_hls.h"
#include "NNDPD_hls.h"

// Q(6.26): same scale as Python export (2**26)
static constexpr double Q626_SCALE = 67108864.0;

static void PrintPythonArrayFloat(const char *name, const float *arr, int n) {
  printf("%s = [", name);
  for (int i = 0; i < n; ++i) {
    if (i) printf(",\n ");
    printf("%.9g", (double)arr[i]);
  }
  printf("]\n");
}

int main() {

  static float sigI_out[NO_SYMBOLS];
  static float sigQ_out[NO_SYMBOLS];
  float mse = 0;
  float val = 0;

  NNDPD(x, x+NO_SYMBOLS, sigI_out, sigQ_out);
   //PrintPythonArrayFloat("sig_out", sig_out, NO_SYMBOLS * 2);
   for(int i = 0; i < NO_SYMBOLS; i++){
    val = ((x[i]) - (sigI_out[i]));
    mse+= val * val;

    printf("    %d: %f\n", i, static_cast<float>(sigI_out[i]));
   }

    for(int i = 0; i < NO_SYMBOLS; i++){
     val = ((x[i+NO_SYMBOLS]) - (sigQ_out[i]));
     mse+= val * val;

     printf("    %d: %f\n", i, static_cast<float>(sigI_out[i]));
    }
  mse = mse / (NO_SYMBOLS*2); // should be close to the val in utils.ipynb, 0.9152914517872547-ish
  printf("mse: %f\n", mse);

  return 0;
}
