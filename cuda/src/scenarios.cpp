#include "Scenarios.hpp"

#include <cmath>
#include <random>

#include "Common.cuh"

void initOrbitScenario(BodiesSoA& h) {
    h.px[0] = 0; h.py[0] = 0; h.pz[0] = 0;
    h.vx[0] = 0; h.vy[0] = 0; h.vz[0] = 0;
    h.mass[0] = 1000.0f;

    float r = 10.0f;
    float orbitalSpeed = std::sqrt(kG * 1000.0f / r);
    h.px[1] = r; h.py[1] = 0; h.pz[1] = 0;
    h.vx[1] = 0; h.vy[1] = orbitalSpeed; h.vz[1] = 0;
    h.mass[1] = 1.0f;

    h.px[2] = 0; h.py[2] = 20; h.pz[2] = 5;
    h.vx[2] = -3; h.vy[2] = 0; h.vz[2] = 0;
    h.mass[2] = 0.5f;
}

void initRandomScenario(BodiesSoA& h, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(-50.0f, 50.0f);
    std::uniform_real_distribution<float> vel(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mass(0.5f, 5.0f);
    for (int i = 0; i < h.n; ++i) {
        h.px[i] = pos(rng); h.py[i] = pos(rng); h.pz[i] = pos(rng);
        h.vx[i] = vel(rng); h.vy[i] = vel(rng); h.vz[i] = vel(rng);
        h.mass[i] = mass(rng);
    }
}
