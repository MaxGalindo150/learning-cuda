#include <cstdlib>
#include <ctime>
#include <cuda/cmath>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <stdio.h>

__global__ void vecAdd(float *A, float *B, float *C, int vectorLength) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < vectorLength) {
    C[i] = A[i] + B[i];
  }
}

void initArray(float *A, int length) {
  srand(std::time({}));
  for (int i = 0; i < length; ++i) {
    A[i] = rand() / static_cast<float>(RAND_MAX);
  }
}

void serialVecAdd(float *A, float *B, float *C, int length) {
  for (int i = 0; i < length; ++i) {
    C[i] = A[i] + B[i];
  }
}

bool vectorApproximatelyEqual(float *A, float *B, int length,
                              float epsilon = 0.00001) {
  for (int i = 0; i < length; i++) {
    if (fabs(A[i] - B[i]) > epsilon) {
      printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
      return false;
    }
  }
  return true;
}

void unifiedMemExample(int vectorLength) {
  // Pointers to memory vectors
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *comparisonResult =
      static_cast<float *>(malloc(vectorLength * sizeof(float)));

  // Use unified memory to allocate buffers
  cudaMallocManaged(&A, vectorLength * sizeof(float));
  cudaMallocManaged(&B, vectorLength * sizeof(float));
  cudaMallocManaged(&C, vectorLength * sizeof(float));

  // Initialize vectors on the host
  initArray(A, vectorLength);
  initArray(B, vectorLength);

  // Launch the kernel. Unified memory will make sure A, B, and C are
  // accesible to the GPU
  int threads = 256;
  int blocks = cuda::ceil_div(vectorLength, threads);
  vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
  // Wait for the kernel to complete execution
  cudaDeviceSynchronize();

  // Perform computation serially on CPU for comparison
  serialVecAdd(A, B, comparisonResult, vectorLength);

  // Confirm that CPU and GPU got the same answer
  if (vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
    printf("Unified Memory: CPU and GPU answers match!!!\n");
  } else {
    printf("Unified Memory: Error - CPU and GPU answers do not match\n");
  }

  // Clean up
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(comparisonResult);
}

int main(int argc, char **argv) {
  int vectorLength = 1024;
  if (argc >= 2) {
    vectorLength = atoi(argv[1]);
  }
  unifiedMemExample(vectorLength);
  return 0;
}
