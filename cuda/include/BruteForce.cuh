#pragma once

#include "BodiesSoA.hpp"

// Device-resident bodies in/out. Advances `steps` velocity-Verlet steps of
// size `dt` using the tiled O(n^2) kernel. If `perStepCallback` is non-null,
// it's invoked after every step with the current device bodies (used by the
// renderer to push updated positions into the mapped VBO without an extra
// device->host round trip).
using StepCallback = void (*)(const BodiesSoA& d_bodies, int step, void* user);

void runBruteForceSimulation(BodiesSoA& d_bodies, int steps, float dt,
                              StepCallback perStepCallback = nullptr, void* user = nullptr);

// Single acceleration evaluation, exposed for the brute-force/Barnes-Hut
// cross-check in main.cu.
void computeAccelerationsBruteForce(const BodiesSoA& d_bodies, float* d_ax, float* d_ay, float* d_az);

// Integration kernels, shared with barnes_hut.cu so both force methods step
// the same velocity-Verlet scheme. Declared (not defined) here; the
// definitions live in bruteforce.cu.
__global__ void updatePositionsKernel(BodiesSoA bodies, const float* ax, const float* ay, const float* az,
                                       float dt);
__global__ void updateVelocitiesKernel(BodiesSoA bodies, const float* axOld, const float* ayOld,
                                        const float* azOld, const float* axNew, const float* ayNew,
                                        const float* azNew, float dt);
