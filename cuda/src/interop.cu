#include "Interop.cuh"

#include "Common.cuh"

namespace {
constexpr int kBlockSize = 256;

__global__ void copyPositionsToVBOKernel(BodiesSoA bodies, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bodies.n) return;
    out[3 * i + 0] = bodies.px[i];
    out[3 * i + 1] = bodies.py[i];
    out[3 * i + 2] = bodies.pz[i];
}
}  // namespace

void copyPositionsToVBO(const BodiesSoA& d_bodies, float* vboPositions) {
    int n = d_bodies.n;
    int grid = (n + kBlockSize - 1) / kBlockSize;
    copyPositionsToVBOKernel<<<grid, kBlockSize>>>(d_bodies, vboPositions);
    CUDA_CHECK(cudaGetLastError());
}
