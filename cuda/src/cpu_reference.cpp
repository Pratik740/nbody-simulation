#include "CpuReference.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "Common.cuh"

static void computeAccelerations(const BodiesSoA& b, std::vector<float>& ax, std::vector<float>& ay,
                                  std::vector<float>& az) {
    int n = b.n;
    std::fill(ax.begin(), ax.end(), 0.0f);
    std::fill(ay.begin(), ay.end(), 0.0f);
    std::fill(az.begin(), az.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float dx = b.px[j] - b.px[i];
            float dy = b.py[j] - b.py[i];
            float dz = b.pz[j] - b.pz[i];
            float distSq = dx * dx + dy * dy + dz * dz + kSoftening2;
            float dist = std::sqrt(distSq);
            float invDist3 = 1.0f / (distSq * dist);

            float fi = kG * b.mass[j] * invDist3;
            float fj = kG * b.mass[i] * invDist3;
            ax[i] += fi * dx; ay[i] += fi * dy; az[i] += fi * dz;
            ax[j] -= fj * dx; ay[j] -= fj * dy; az[j] -= fj * dz;
        }
    }
}

void runCpuReference(BodiesSoA& b, int steps, float dt) {
    int n = b.n;
    std::vector<float> axOld(n), ayOld(n), azOld(n);
    std::vector<float> axNew(n), ayNew(n), azNew(n);

    computeAccelerations(b, axOld, ayOld, azOld);

    for (int step = 0; step < steps; ++step) {
        float halfDt2 = 0.5f * dt * dt;
        for (int i = 0; i < n; ++i) {
            b.px[i] += b.vx[i] * dt + axOld[i] * halfDt2;
            b.py[i] += b.vy[i] * dt + ayOld[i] * halfDt2;
            b.pz[i] += b.vz[i] * dt + azOld[i] * halfDt2;
        }

        computeAccelerations(b, axNew, ayNew, azNew);

        float halfDt = 0.5f * dt;
        for (int i = 0; i < n; ++i) {
            b.vx[i] += (axOld[i] + axNew[i]) * halfDt;
            b.vy[i] += (ayOld[i] + ayNew[i]) * halfDt;
            b.vz[i] += (azOld[i] + azNew[i]) * halfDt;
        }

        std::swap(axOld, axNew);
        std::swap(ayOld, ayNew);
        std::swap(azOld, azNew);
    }
}
