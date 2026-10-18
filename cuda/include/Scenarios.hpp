#pragma once

#include "BodiesSoA.hpp"

// Star + two-body demo matching src/main.cpp's CPU scenario. Requires
// h.n == 3 -- see scenarios.cpp for why padding extra slots is a bad idea.
void initOrbitScenario(BodiesSoA& h);

// Uniform-random cube of bodies, for benchmarking/rendering at any n.
void initRandomScenario(BodiesSoA& h, unsigned seed);
