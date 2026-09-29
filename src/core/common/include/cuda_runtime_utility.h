#pragma once

#include <cstdio>
#include <cstdlib>

#ifdef Q_USE_CUDA
#include <cuda_runtime.h>
#define HD __host__ __device__
#endif

#ifdef Q_USE_HIP
#include <hip/hip_runtime.h>
#define HD __host__ __device__
#endif

#include "../../cuda/include/gpu_interface.h"

inline void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::printf(">>> FATAL: CUDA call failed with error code %d: %s\n", status, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

inline void check_cudaMalloc(void** dev_ptr, size_t size) {
    check_cuda(cudaMalloc(dev_ptr, size));
}
