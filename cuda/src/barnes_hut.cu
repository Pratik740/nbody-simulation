#include "BarnesHut.cuh"

#include <algorithm>
#include <vector>

#include "BruteForce.cuh"  // reuses updatePositionsKernel/updateVelocitiesKernel via the integration loop below
#include "Common.cuh"

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxTraversalStack = 256;
// Bound on how deep a chain of forced single-child cells can go when two
// bodies keep landing in the same octant (near-coincident positions). 64
// halvings of any realistic coordinate range exhausts float precision, so
// this only ever triggers on literal duplicate positions.
constexpr int kMaxChainDepth = 64;

// Octree cells, indexed separately from bodies: node index < n means "body
// i", node index >= n means "cell (node index - n)". Kept as its own
// struct (rather than folded into BodiesSoA) because only cells carry a
// bounding box / accumulated center of mass -- bodies are exact point
// sources and are read directly from BodiesSoA instead.
struct CellArrays {
    int maxCells = 0;
    float* mass = nullptr;     // accumulated via atomicAdd during build
    float* comWX = nullptr;    // mass-weighted position sums (divide by mass to read center of mass)
    float* comWY = nullptr;
    float* comWZ = nullptr;
    float* centerX = nullptr;  // geometric bounding-box center, fixed at allocation time
    float* centerY = nullptr;
    float* centerZ = nullptr;
    float* halfWidth = nullptr;
    int* child = nullptr;      // 8 slots per cell; -1 empty, -2 locked, else a node index

    static CellArrays alloc(int maxCells) {
        CellArrays c;
        c.maxCells = maxCells;
        CUDA_CHECK(cudaMalloc(&c.mass, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.comWX, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.comWY, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.comWZ, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.centerX, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.centerY, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.centerZ, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.halfWidth, sizeof(float) * maxCells));
        CUDA_CHECK(cudaMalloc(&c.child, sizeof(int) * maxCells * 8));
        return c;
    }

    void free() {
        cudaFree(mass); cudaFree(comWX); cudaFree(comWY); cudaFree(comWZ);
        cudaFree(centerX); cudaFree(centerY); cudaFree(centerZ); cudaFree(halfWidth);
        cudaFree(child);
    }
};

__device__ __forceinline__ int octantOf(float px, float py, float pz, float cx, float cy, float cz) {
    int o = 0;
    if (px >= cx) o |= 1;
    if (py >= cy) o |= 2;
    if (pz >= cz) o |= 4;
    return o;
}

__device__ __forceinline__ void childCenter(float cx, float cy, float cz, float halfWidth, int oct,
                                             float& ox, float& oy, float& oz) {
    float q = halfWidth * 0.5f;
    ox = cx + ((oct & 1) ? q : -q);
    oy = cy + ((oct & 2) ? q : -q);
    oz = cz + ((oct & 4) ? q : -q);
}

__device__ __forceinline__ void addContribution(CellArrays cells, int nodeIdx, int n, float mass, float px,
                                                 float py, float pz) {
    int c = nodeIdx - n;
    atomicAdd(&cells.mass[c], mass);
    atomicAdd(&cells.comWX[c], mass * px);
    atomicAdd(&cells.comWY[c], mass * py);
    atomicAdd(&cells.comWZ[c], mass * pz);
}

__device__ __forceinline__ int allocateCell(int* cellCounter, int n) {
    return n + atomicAdd(cellCounter, 1);
}

__global__ void resetCellsKernel(CellArrays cells) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cells.maxCells) return;
    cells.mass[c] = 0.0f;
    cells.comWX[c] = 0.0f;
    cells.comWY[c] = 0.0f;
    cells.comWZ[c] = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) cells.child[c * 8 + k] = -1;
}

// One thread per body. Walks down from the root, accumulating this body's
// contribution into every cell it passes through (so cell mass/center-of-
// mass end up correct with no separate bottom-up summarization pass), and
// claims an empty slot via atomicCAS. On colliding with another body
// already occupying a slot, builds a *private* chain of new cells (nothing
// else can reach them yet) to separate the two bodies, then publishes the
// whole chain to the parent slot in one release-fenced store -- so no other
// thread ever observes a half-built subtree.
__global__ void buildTreeKernel(BodiesSoA bodies, CellArrays cells, int* cellCounter, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float bx = bodies.px[i], by = bodies.py[i], bz = bodies.pz[i], bm = bodies.mass[i];

    int current = n;  // root node index (cell slot 0)
    float cx = cells.centerX[0], cy = cells.centerY[0], cz = cells.centerZ[0], hw = cells.halfWidth[0];
    addContribution(cells, current, n, bm, bx, by, bz);

    for (;;) {
        int j = octantOf(bx, by, bz, cx, cy, cz);
        int c = current - n;
        int* slotPtr = &cells.child[c * 8 + j];
        // volatile: the "ch == -2" spin below has no atomic op in it, so
        // without forcing a real reload here the compiler can treat this
        // load as loop-invariant and cache it, leaving a spinning thread
        // stuck rereading a stale value and never seeing another thread's
        // fenced publish -- a real, reproducible hang at higher body counts
        // (the 3-way-collision case is rare at n=3, common at n=500+).
        int ch = *(volatile int*)slotPtr;

        if (ch >= n) {
            current = ch;
            float ncx, ncy, ncz;
            childCenter(cx, cy, cz, hw, j, ncx, ncy, ncz);
            cx = ncx; cy = ncy; cz = ncz; hw *= 0.5f;
            addContribution(cells, current, n, bm, bx, by, bz);
            continue;
        }
        if (ch == -1) {
            if (atomicCAS(slotPtr, -1, i) == -1) return;  // placed as a leaf
            continue;                                      // lost the race, re-examine the slot
        }
        if (ch == -2) continue;  // another thread is subdividing this slot; spin

        // ch is an existing body occupying this slot: lock it and subdivide.
        if (atomicCAS(slotPtr, ch, -2) != ch) continue;

        float occx = bodies.px[ch], occy = bodies.py[ch], occz = bodies.pz[ch], ocm = bodies.mass[ch];

        float chainCx, chainCy, chainCz;
        childCenter(cx, cy, cz, hw, j, chainCx, chainCy, chainCz);
        float chainHw = hw * 0.5f;

        int firstCell = -1, prevCell = -1, prevOct = -1;
        for (int depth = 0; depth < kMaxChainDepth; ++depth) {
            int newCell = allocateCell(cellCounter, n);
            int cc = newCell - n;
            cells.centerX[cc] = chainCx; cells.centerY[cc] = chainCy; cells.centerZ[cc] = chainCz;
            cells.halfWidth[cc] = chainHw;
            addContribution(cells, newCell, n, ocm, occx, occy, occz);
            addContribution(cells, newCell, n, bm, bx, by, bz);

            if (firstCell == -1) firstCell = newCell;
            if (prevCell != -1) cells.child[(prevCell - n) * 8 + prevOct] = newCell;  // private link, no race

            int octCh = octantOf(occx, occy, occz, chainCx, chainCy, chainCz);
            int octI = octantOf(bx, by, bz, chainCx, chainCy, chainCz);
            if (octCh != octI || depth == kMaxChainDepth - 1) {
                cells.child[cc * 8 + octCh] = ch;
                cells.child[cc * 8 + octI] = i;  // on a forced depth bailout this overwrites octCh's slot;
                                                   // acceptable only for literally-duplicate coordinates
                break;
            }

            prevCell = newCell; prevOct = octCh;
            float nx, ny, nz;
            childCenter(chainCx, chainCy, chainCz, chainHw, octCh, nx, ny, nz);
            chainCx = nx; chainCy = ny; chainCz = nz; chainHw *= 0.5f;
        }

        __threadfence();     // publish all of the above before anyone can see firstCell
        *slotPtr = firstCell;
        return;
    }
}

__global__ void computeForcesBarnesHutKernel(BodiesSoA bodies, CellArrays cells, int n, float theta2,
                                              float* __restrict__ ax, float* __restrict__ ay,
                                              float* __restrict__ az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = bodies.px[i], py = bodies.py[i], pz = bodies.pz[i];
    float accx = 0.0f, accy = 0.0f, accz = 0.0f;

    int stack[kMaxTraversalStack];
    int sp = 0;
    stack[sp++] = n;  // root

    while (sp > 0) {
        int node = stack[--sp];
        if (node < n) {
            if (node == i) continue;
            float dx = bodies.px[node] - px, dy = bodies.py[node] - py, dz = bodies.pz[node] - pz;
            float distSq = dx * dx + dy * dy + dz * dz + kSoftening2;
            float invDist = rsqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;
            float f = kG * bodies.mass[node] * invDist3;
            accx += f * dx; accy += f * dy; accz += f * dz;
        } else {
            int c = node - n;
            float cmass = cells.mass[c];
            if (cmass <= 0.0f) continue;
            float comx = cells.comWX[c] / cmass, comy = cells.comWY[c] / cmass, comz = cells.comWZ[c] / cmass;
            float dx = comx - px, dy = comy - py, dz = comz - pz;
            float distSq = dx * dx + dy * dy + dz * dz;
            float size = 2.0f * cells.halfWidth[c];

            if (size * size < theta2 * distSq) {
                float distSqSoft = distSq + kSoftening2;
                float invDist = rsqrtf(distSqSoft);
                float invDist3 = invDist * invDist * invDist;
                float f = kG * cmass * invDist3;
                accx += f * dx; accy += f * dy; accz += f * dz;
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    int ch = cells.child[c * 8 + k];
                    if (ch != -1 && ch != -2 && sp < kMaxTraversalStack) stack[sp++] = ch;
                }
            }
        }
    }

    ax[i] = accx; ay[i] = accy; az[i] = accz;
}

constexpr int kBBBlockSize = 256;
constexpr int kBBMaxBlocks = 128;

__global__ void boundingBoxPartialKernel(BodiesSoA bodies, float* blockMinX, float* blockMinY, float* blockMinZ,
                                          float* blockMaxX, float* blockMaxY, float* blockMaxZ) {
    __shared__ float sMinX[kBBBlockSize], sMinY[kBBBlockSize], sMinZ[kBBBlockSize];
    __shared__ float sMaxX[kBBBlockSize], sMaxY[kBBBlockSize], sMaxZ[kBBBlockSize];

    int tid = threadIdx.x;
    float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;

    for (int i = blockIdx.x * blockDim.x + tid; i < bodies.n; i += blockDim.x * gridDim.x) {
        minX = fminf(minX, bodies.px[i]); maxX = fmaxf(maxX, bodies.px[i]);
        minY = fminf(minY, bodies.py[i]); maxY = fmaxf(maxY, bodies.py[i]);
        minZ = fminf(minZ, bodies.pz[i]); maxZ = fmaxf(maxZ, bodies.pz[i]);
    }
    sMinX[tid] = minX; sMinY[tid] = minY; sMinZ[tid] = minZ;
    sMaxX[tid] = maxX; sMaxY[tid] = maxY; sMaxZ[tid] = maxZ;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sMinX[tid] = fminf(sMinX[tid], sMinX[tid + stride]);
            sMinY[tid] = fminf(sMinY[tid], sMinY[tid + stride]);
            sMinZ[tid] = fminf(sMinZ[tid], sMinZ[tid + stride]);
            sMaxX[tid] = fmaxf(sMaxX[tid], sMaxX[tid + stride]);
            sMaxY[tid] = fmaxf(sMaxY[tid], sMaxY[tid + stride]);
            sMaxZ[tid] = fmaxf(sMaxZ[tid], sMaxZ[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMinX[blockIdx.x] = sMinX[0]; blockMinY[blockIdx.x] = sMinY[0]; blockMinZ[blockIdx.x] = sMinZ[0];
        blockMaxX[blockIdx.x] = sMaxX[0]; blockMaxY[blockIdx.x] = sMaxY[0]; blockMaxZ[blockIdx.x] = sMaxZ[0];
    }
}

struct BoundingBox { float cx, cy, cz, halfWidth; };

BoundingBox computeBoundingBox(const BodiesSoA& d_bodies) {
    int numBlocks = std::min(kBBMaxBlocks, (d_bodies.n + kBBBlockSize - 1) / kBBBlockSize);
    numBlocks = std::max(numBlocks, 1);

    float *dMinX, *dMinY, *dMinZ, *dMaxX, *dMaxY, *dMaxZ;
    CUDA_CHECK(cudaMalloc(&dMinX, sizeof(float) * numBlocks));
    CUDA_CHECK(cudaMalloc(&dMinY, sizeof(float) * numBlocks));
    CUDA_CHECK(cudaMalloc(&dMinZ, sizeof(float) * numBlocks));
    CUDA_CHECK(cudaMalloc(&dMaxX, sizeof(float) * numBlocks));
    CUDA_CHECK(cudaMalloc(&dMaxY, sizeof(float) * numBlocks));
    CUDA_CHECK(cudaMalloc(&dMaxZ, sizeof(float) * numBlocks));

    boundingBoxPartialKernel<<<numBlocks, kBBBlockSize>>>(d_bodies, dMinX, dMinY, dMinZ, dMaxX, dMaxY, dMaxZ);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> minX(numBlocks), minY(numBlocks), minZ(numBlocks);
    std::vector<float> maxX(numBlocks), maxY(numBlocks), maxZ(numBlocks);
    CUDA_CHECK(cudaMemcpy(minX.data(), dMinX, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(minY.data(), dMinY, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(minZ.data(), dMinZ, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(maxX.data(), dMaxX, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(maxY.data(), dMaxY, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(maxZ.data(), dMaxZ, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
    cudaFree(dMinX); cudaFree(dMinY); cudaFree(dMinZ); cudaFree(dMaxX); cudaFree(dMaxY); cudaFree(dMaxZ);

    float gMinX = minX[0], gMinY = minY[0], gMinZ = minZ[0];
    float gMaxX = maxX[0], gMaxY = maxY[0], gMaxZ = maxZ[0];
    for (int b = 1; b < numBlocks; ++b) {
        gMinX = std::min(gMinX, minX[b]); gMaxX = std::max(gMaxX, maxX[b]);
        gMinY = std::min(gMinY, minY[b]); gMaxY = std::max(gMaxY, maxY[b]);
        gMinZ = std::min(gMinZ, minZ[b]); gMaxZ = std::max(gMaxZ, maxZ[b]);
    }

    BoundingBox box;
    box.cx = 0.5f * (gMinX + gMaxX);
    box.cy = 0.5f * (gMinY + gMaxY);
    box.cz = 0.5f * (gMinZ + gMaxZ);
    float half = std::max({gMaxX - gMinX, gMaxY - gMinY, gMaxZ - gMinZ}) * 0.5f;
    box.halfWidth = std::max(half, 1.0f) * 1.001f;  // pad so no body sits exactly on the boundary
    return box;
}

int gridFor(int n, int block) { return (n + block - 1) / block; }

void buildAndComputeForces(BodiesSoA& d_bodies, CellArrays& cells, int* d_cellCounter, float theta2, float* ax,
                            float* ay, float* az) {
    int n = d_bodies.n;

    resetCellsKernel<<<gridFor(cells.maxCells, kBlockSize), kBlockSize>>>(cells);
    CUDA_CHECK(cudaGetLastError());

    BoundingBox box = computeBoundingBox(d_bodies);
    CUDA_CHECK(cudaMemcpy(&cells.centerX[0], &box.cx, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&cells.centerY[0], &box.cy, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&cells.centerZ[0], &box.cz, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&cells.halfWidth[0], &box.halfWidth, sizeof(float), cudaMemcpyHostToDevice));

    int one = 1;
    CUDA_CHECK(cudaMemcpy(d_cellCounter, &one, sizeof(int), cudaMemcpyHostToDevice));  // slot 0 = root, reserved

    buildTreeKernel<<<gridFor(n, kBlockSize), kBlockSize>>>(d_bodies, cells, d_cellCounter, n);
    CUDA_CHECK(cudaGetLastError());

    computeForcesBarnesHutKernel<<<gridFor(n, kBlockSize), kBlockSize>>>(d_bodies, cells, n, theta2, ax, ay, az);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void computeAccelerationsBarnesHut(const BodiesSoA& d_bodies, float* d_ax, float* d_ay, float* d_az, float theta) {
    int n = d_bodies.n;
    int maxCells = std::max(4 * n + 1024, 1024);
    CellArrays cells = CellArrays::alloc(maxCells);
    int* d_cellCounter;
    CUDA_CHECK(cudaMalloc(&d_cellCounter, sizeof(int)));

    BodiesSoA bodiesCopy = d_bodies;  // struct of pointers; mutate nothing, just reuse the launcher signature
    buildAndComputeForces(bodiesCopy, cells, d_cellCounter, theta * theta, d_ax, d_ay, d_az);

    cells.free();
    cudaFree(d_cellCounter);
}

void runBarnesHutSimulation(BodiesSoA& d_bodies, int steps, float dt, float theta, StepCallback perStepCallback,
                             void* user) {
    int n = d_bodies.n;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float theta2 = theta * theta;

    int maxCells = std::max(4 * n + 1024, 1024);
    CellArrays cells = CellArrays::alloc(maxCells);
    int* d_cellCounter;
    CUDA_CHECK(cudaMalloc(&d_cellCounter, sizeof(int)));

    float *axA, *ayA, *azA, *axB, *ayB, *azB;
    CUDA_CHECK(cudaMalloc(&axA, bytes)); CUDA_CHECK(cudaMalloc(&ayA, bytes)); CUDA_CHECK(cudaMalloc(&azA, bytes));
    CUDA_CHECK(cudaMalloc(&axB, bytes)); CUDA_CHECK(cudaMalloc(&ayB, bytes)); CUDA_CHECK(cudaMalloc(&azB, bytes));

    float *axOld = axA, *ayOld = ayA, *azOld = azA;
    float *axNew = axB, *ayNew = ayB, *azNew = azB;

    buildAndComputeForces(d_bodies, cells, d_cellCounter, theta2, axOld, ayOld, azOld);

    int grid = gridFor(n, kBlockSize);
    for (int step = 0; step < steps; ++step) {
        updatePositionsKernel<<<grid, kBlockSize>>>(d_bodies, axOld, ayOld, azOld, dt);
        CUDA_CHECK(cudaGetLastError());

        buildAndComputeForces(d_bodies, cells, d_cellCounter, theta2, axNew, ayNew, azNew);

        updateVelocitiesKernel<<<grid, kBlockSize>>>(d_bodies, axOld, ayOld, azOld, axNew, ayNew, azNew, dt);
        CUDA_CHECK(cudaGetLastError());

        if (perStepCallback) perStepCallback(d_bodies, step, user);

        float *tx = axOld, *ty = ayOld, *tz = azOld;
        axOld = axNew; ayOld = ayNew; azOld = azNew;
        axNew = tx; ayNew = ty; azNew = tz;
    }

    cells.free();
    cudaFree(d_cellCounter);
    cudaFree(axA); cudaFree(ayA); cudaFree(azA);
    cudaFree(axB); cudaFree(ayB); cudaFree(azB);
}
