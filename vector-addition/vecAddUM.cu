#include "../common/cuda_check.cuh"
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

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // Use unified memory to allocate buffers
  CUDA_CHECK(cudaMallocManaged(&A, vectorLength * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&B, vectorLength * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&C, vectorLength * sizeof(float)));

  // Initialize vectors on the host
  initArray(A, vectorLength);
  initArray(B, vectorLength);

  // Launch the kernel. Unified memory will make sure A, B, and C are
  // accesible to the GPU
  int threads = 256;
  int blocks = cuda::ceil_div(vectorLength, threads);
  CUDA_CHECK(cudaEventRecord(start, /*stream=*/0));
  vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
  CUDA_LAUNCH_CHECK();

  CUDA_CHECK(cudaEventRecord(stop, 0));

  CUDA_CHECK(cudaEventSynchronize(stop));

  float kernelMs = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));
  printf("Kernel time: %.3f ms\n", kernelMs);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Perform computation serially on CPU for comparison
  serialVecAdd(A, B, comparisonResult, vectorLength);

  // Confirm that CPU and GPU got the same answer
  if (vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
    printf("Unified Memory: CPU and GPU answers match!!!\n");
  } else {
    printf("Unified Memory: Error - CPU and GPU answers do not match\n");
  }

  // Clean up
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
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
