__global__ void vecAdd(float *A, float *B, float *C, int vectorLength) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < vectorLength) {
    C[i] = A[i] + B[i];
  }
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
}
