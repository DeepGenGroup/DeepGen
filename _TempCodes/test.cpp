#include <iostream>
#include <cmath>

#define M 128
#define N 128
#define K 128

void elementWise(float *A, float *B) {
  for (int i=0; i<M; i++) {
    for (int j=0; j<K; j++) {
      float ldA = A[i*K+j];        // load 
      float result = sqrt(ldA);
      B[i*K+j] = result;           // store
    }
  }
}

void matmul(float *A, float *B, float *C) {
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      float sum = 0;
      for (int k=0; k<K; k++) {
        float ldA = A[i*K+k];     // load
        float ldB = B[k*N+j];     // load
        float result = ldA * ldB;
        sum = sum + result;
      }
      C[i*N+j] = sum;             // store
    }
  }
}

void elemWise_matmul(float *A, float *B, float *C) {
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      float sum = 0;
      for (int k=0; k<K; k++) {
        float ldA = A[i*K+k];     // load
        float resultA = sqrt(ldA);
        float ldB = B[k*N+j];     // load
        float result = resultA * ldB;
        sum = sum + result;
      }
      C[i*N+j] = sum;             // store
    }
  }
}

int main() {
  float *A, *B, *C;
  init(A, B, C);
  elementWise(A, A);
  matmul(A, B, C);
}