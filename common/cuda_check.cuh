#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "[CUDA] %s:%d error: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(_err));                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_LAUNCH_CHECK()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
  } while (0)
