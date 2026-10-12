#pragma once

#include "BodiesSoA.hpp"

// Builds an octree over `d_bodies` each step and traverses it per body with
// the Barnes-Hut multipole-acceptance criterion (theta), giving O(n log n)
// force evaluation instead of brute force's O(n^2). Mirrors
// runBruteForceSimulation's velocity-Verlet integration so results are
// directly comparable.
void runBarnesHutSimulation(BodiesSoA& d_bodies, int steps, float dt, float theta);

// Single acceleration evaluation (build + traverse once), exposed for the
// brute-force cross-check in main.cu.
void computeAccelerationsBarnesHut(const BodiesSoA& d_bodies, float* d_ax, float* d_ay, float* d_az, float theta);
