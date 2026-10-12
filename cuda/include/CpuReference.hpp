#pragma once

#include "BodiesSoA.hpp"

// Single-threaded host O(n^2) velocity-Verlet integrator, numerically
// matching the GPU kernels (same constants, same float precision) so it can
// serve as the correctness oracle for both the brute-force and Barnes-Hut
// GPU paths.
void runCpuReference(BodiesSoA& h_bodies, int steps, float dt);
