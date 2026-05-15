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

void explicitMem(int vectorLength) {
  // Pointer for host memory
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *comparisonResult =
      static_cast<float *>(malloc(vectorLength * sizeof(float)));

  // Pointers for device memory
  float *devA = nullptr;
  float *devB = nullptr;
  float *devC = nullptr;

  // Allocate Host Memory using cudaMallocHost API. This is best practice
  // when buffers will be used for copies between CPU and GPU memory
  CUDA_CHECK(cudaMallocHost(&A, vectorLength * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&B, vectorLength * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&C, vectorLength * sizeof(float)));

  // Initialize vectors on the host
  initArray(A, vectorLength);
  initArray(B, vectorLength);

  // start-allocate-and-copy
  // Allocate memory on the GPU
  CUDA_CHECK(cudaMalloc(&devA, vectorLength * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&devB, vectorLength * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&devC, vectorLength * sizeof(float)));

  cudaEvent_t h2dStart, h2dStop, kStart, kStop, d2hStart, d2hStop;
  CUDA_CHECK(cudaEventCreate(&h2dStart));
  CUDA_CHECK(cudaEventCreate(&h2dStop));
  CUDA_CHECK(cudaEventCreate(&kStart));
  CUDA_CHECK(cudaEventCreate(&kStop));
  CUDA_CHECK(cudaEventCreate(&d2hStart));
  CUDA_CHECK(cudaEventCreate(&d2hStop));

  CUDA_CHECK(cudaEventRecord(h2dStart));
  CUDA_CHECK(
      cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyDefault));
  CUDA_CHECK(
      cudaMemcpy(devB, B, vectorLength * sizeof(float), cudaMemcpyDefault));
  CUDA_CHECK(cudaEventRecord(h2dStop));

  CUDA_CHECK(cudaMemset(devC, 0, vectorLength * sizeof(float)));
  // end-allocate-and-copy

  int threads = 256;
  int blocks = cuda::ceil_div(vectorLength, threads);
  CUDA_CHECK(cudaEventRecord(kStart));
  vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
  CUDA_CHECK(cudaEventRecord(kStop));
  CUDA_LAUNCH_CHECK();

  CUDA_CHECK(cudaEventRecord(d2hStart));
  CUDA_CHECK(
      cudaMemcpy(C, devC, vectorLength * sizeof(float), cudaMemcpyDefault));
  CUDA_CHECK(cudaEventRecord(d2hStop));

  CUDA_CHECK(cudaEventSynchronize(d2hStop));

  float h2dMs = 0.f, kMs = 0.f, d2hMs = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&h2dMs, h2dStart, h2dStop));
  CUDA_CHECK(cudaEventElapsedTime(&kMs, kStart, kStop));
  CUDA_CHECK(cudaEventElapsedTime(&d2hMs, d2hStart, d2hStop));

  const double h2dBytes = 2.0 * vectorLength * sizeof(float);
  const double d2hBytes = 1.0 * vectorLength * sizeof(float);
  const double h2dGB = h2dBytes / (h2dMs / 1000.0) / 1e9;
  const double d2hGB = d2hBytes / (d2hMs / 1000.0) / 1e9;

  printf("\n--- Timings (N = %d) ---\n", vectorLength);
  printf("H2D:    %8.3f ms  (%6.2f GB/s)\n", h2dMs, h2dGB);
  printf("Kernel: %8.3f ms\n", kMs);
  printf("D2H:    %8.3f ms  (%6.2f GB/s)\n", d2hMs, d2hGB);
  printf("Total:  %8.3f ms\n\n", h2dMs + kMs + d2hMs);

  // Perform computation serially on CPU for comparison
  serialVecAdd(A, B, comparisonResult, vectorLength);

  // Confirm that CPU and GPU got the same answer
  if (vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
    printf("Explict Memory: CPU and GPU answers match\n");
  } else {
    printf("Explicit Memory: Error - CPU and GPU anwers to not match\n");
  }

  // Clean up
  CUDA_CHECK(cudaEventDestroy(h2dStart));
  CUDA_CHECK(cudaEventDestroy(h2dStop));
  CUDA_CHECK(cudaEventDestroy(kStart));
  CUDA_CHECK(cudaEventDestroy(kStop));
  CUDA_CHECK(cudaEventDestroy(d2hStart));
  CUDA_CHECK(cudaEventDestroy(d2hStop));
  CUDA_CHECK(cudaFree(devA));
  CUDA_CHECK(cudaFree(devB));
  CUDA_CHECK(cudaFree(devC));
  CUDA_CHECK(cudaFreeHost(A));
  CUDA_CHECK(cudaFreeHost(B));
  CUDA_CHECK(cudaFreeHost(C));
  free(comparisonResult);
}
// explict-memory-end

int main(int argc, char **argv) {
  int vectorLength = 1024;
  if (argc >= 2) {
    vectorLength = atoi(argv[1]);
  }
  explicitMem(vectorLength);
  return 0;
}
