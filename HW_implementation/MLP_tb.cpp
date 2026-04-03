#include <math.h>
#include <stdio.h>
#include "MLP_tb.h"
#include "MLP.h"

static void PrintPythonArrayFloat(const char *name, const float *arr, int n) {
  printf("%s = [", name);
  for (int i = 0; i < n; ++i) {
    if (i) printf(",\n ");
    printf("%.9g", (double)arr[i]);
  }
  printf("]\n");
}

int main() {

  static float sig_out[NO_SYMBOLS * 2];



  MultilayerPerceptron(x, sig_out);

  //PrintPythonArrayFloat("sig_out", sig_out, NO_SYMBOLS * 2);
  for(int i = 0; i < NO_SYMBOLS * 2; i++){
    printf("    %d: %.9g\n", i, (double)sig_out[i]);
  }


  return 0;
}
