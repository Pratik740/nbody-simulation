#include "BruteForce.cuh"

#include "Common.cuh"

constexpr int kBlockSize = 256;
// One tile of bodies cached in shared memory per outer loop iteration, so
// each body's attributes are read once from global memory and reused by
// every thread in the block instead of every thread re-fetching them.
constexpr int kTileSize = kBlockSize;

__global__ void computeAccelerationsBruteForceKernel(BodiesSoA bodies, float* __restrict__ ax,
                                                       float* __restrict__ ay, float* __restrict__ az) {
    __shared__ float sx[kTileSize];
    __shared__ float sy[kTileSize];
    __shared__ float sz[kTileSize];
    __shared__ float sm[kTileSize];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (i < bodies.n) {
        px = bodies.px[i];
        py = bodies.py[i];
        pz = bodies.pz[i];
    }

    float accx = 0.0f, accy = 0.0f, accz = 0.0f;

    for (int tileStart = 0; tileStart < bodies.n; tileStart += kTileSize) {
        int j = tileStart + threadIdx.x;
        if (j < bodies.n) {
            sx[threadIdx.x] = bodies.px[j];
            sy[threadIdx.x] = bodies.py[j];
            sz[threadIdx.x] = bodies.pz[j];
            sm[threadIdx.x] = bodies.mass[j];
        } else {
            sm[threadIdx.x] = 0.0f;  // padding body contributes zero force
        }
        __syncthreads();

        int tileLen = min(kTileSize, bodies.n - tileStart);
        for (int k = 0; k < tileLen; ++k) {
            float dx = sx[k] - px;
            float dy = sy[k] - py;
            float dz = sz[k] - pz;
            float distSq = dx * dx + dy * dy + dz * dz + kSoftening2;
            float invDist = rsqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;
            float f = kG * sm[k] * invDist3;
            accx += f * dx;
            accy += f * dy;
            accz += f * dz;
        }
        __syncthreads();
    }

    if (i < bodies.n) {
        ax[i] = accx;
        ay[i] = accy;
        az[i] = accz;
    }
}

__global__ void updatePositionsKernel(BodiesSoA bodies, const float* __restrict__ ax,
                                       const float* __restrict__ ay, const float* __restrict__ az, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bodies.n) return;
    float halfDt2 = 0.5f * dt * dt;
    bodies.px[i] += bodies.vx[i] * dt + ax[i] * halfDt2;
    bodies.py[i] += bodies.vy[i] * dt + ay[i] * halfDt2;
    bodies.pz[i] += bodies.vz[i] * dt + az[i] * halfDt2;
}

__global__ void updateVelocitiesKernel(BodiesSoA bodies, const float* __restrict__ axOld,
                                        const float* __restrict__ ayOld, const float* __restrict__ azOld,
                                        const float* __restrict__ axNew, const float* __restrict__ ayNew,
                                        const float* __restrict__ azNew, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bodies.n) return;
    float halfDt = 0.5f * dt;
    bodies.vx[i] += (axOld[i] + axNew[i]) * halfDt;
    bodies.vy[i] += (ayOld[i] + ayNew[i]) * halfDt;
    bodies.vz[i] += (azOld[i] + azNew[i]) * halfDt;
}

static int gridFor(int n) { return (n + kBlockSize - 1) / kBlockSize; }

void computeAccelerationsBruteForce(const BodiesSoA& d_bodies, float* d_ax, float* d_ay, float* d_az) {
    computeAccelerationsBruteForceKernel<<<gridFor(d_bodies.n), kBlockSize>>>(d_bodies, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());
}

void runBruteForceSimulation(BodiesSoA& d_bodies, int steps, float dt, StepCallback perStepCallback,
                              void* user) {
    int n = d_bodies.n;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *axA, *ayA, *azA, *axB, *ayB, *azB;
    CUDA_CHECK(cudaMalloc(&axA, bytes)); CUDA_CHECK(cudaMalloc(&ayA, bytes)); CUDA_CHECK(cudaMalloc(&azA, bytes));
    CUDA_CHECK(cudaMalloc(&axB, bytes)); CUDA_CHECK(cudaMalloc(&ayB, bytes)); CUDA_CHECK(cudaMalloc(&azB, bytes));

    float *axOld = axA, *ayOld = ayA, *azOld = azA;
    float *axNew = axB, *ayNew = ayB, *azNew = azB;

    computeAccelerationsBruteForce(d_bodies, axOld, ayOld, azOld);

    int grid = gridFor(n);
    for (int step = 0; step < steps; ++step) {
        updatePositionsKernel<<<grid, kBlockSize>>>(d_bodies, axOld, ayOld, azOld, dt);
        CUDA_CHECK(cudaGetLastError());

        computeAccelerationsBruteForce(d_bodies, axNew, ayNew, azNew);

        updateVelocitiesKernel<<<grid, kBlockSize>>>(d_bodies, axOld, ayOld, azOld, axNew, ayNew, azNew, dt);
        CUDA_CHECK(cudaGetLastError());

        if (perStepCallback) perStepCallback(d_bodies, step, user);

        float *tx = axOld, *ty = ayOld, *tz = azOld;
        axOld = axNew; ayOld = ayNew; azOld = azNew;
        axNew = tx; ayNew = ty; azNew = tz;
    }

    cudaFree(axA); cudaFree(ayA); cudaFree(azA);
    cudaFree(axB); cudaFree(ayB); cudaFree(azB);
}
