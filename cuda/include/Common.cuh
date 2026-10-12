#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                          \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// Shared across brute-force and Barnes-Hut so both integrate identically
// and their outputs stay directly comparable in the correctness check.
constexpr float kG = 1.0f;
constexpr float kSoftening2 = 1e-2f * 1e-2f;
