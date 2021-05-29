#pragma once

#include <cstdio>

#ifdef __CUDACC__
#  define CUDA_HOSTDEV __host__ __device__
#  define CUDA_HOST __host__
#  define CUDA_DEV __device__
#  define CUDA_GLOBAL __global__
#  define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#  define CUDA_HOSTDEV
#  define CUDA_HOST
#  define CUDA_DEV
#  define CUDA_GLOBAL
#  define gpuErrchk(ans) ans
#endif