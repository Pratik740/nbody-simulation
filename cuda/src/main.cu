#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "BarnesHut.cuh"
#include "BodiesSoA.hpp"
#include "BruteForce.cuh"
#include "CpuReference.hpp"

namespace {

void initOrbitScenario(BodiesSoA& h) {
    // Same star + two-body scenario as the CPU demo (src/main.cpp), so GPU
    // and CPU runs can be eyeballed against each other at a glance.
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

struct Args {
    std::string mode = "bruteforce";  // bruteforce | barnes_hut | compare
    int n = 3;
    int steps = 500;
    float dt = 0.01f;
    float theta = 0.5f;
    std::string scenario = "orbit";  // orbit | random
    unsigned seed = 42;
    bool printFinal = false;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val = [&](const char* flag) -> const char* {
            size_t len = std::strlen(flag);
            if (arg.rfind(flag, 0) == 0) return arg.c_str() + len;
            return nullptr;
        };
        if (auto v = val("--mode=")) a.mode = v;
        else if (auto v = val("--n=")) a.n = std::atoi(v);
        else if (auto v = val("--steps=")) a.steps = std::atoi(v);
        else if (auto v = val("--dt=")) a.dt = std::atof(v);
        else if (auto v = val("--theta=")) a.theta = std::atof(v);
        else if (auto v = val("--scenario=")) a.scenario = v;
        else if (auto v = val("--seed=")) a.seed = static_cast<unsigned>(std::atoi(v));
        else if (arg == "--print-final") a.printFinal = true;
    }
    return a;
}

BodiesSoA cloneHost(const BodiesSoA& src) {
    BodiesSoA dst = BodiesSoA::allocHost(src.n);
    size_t bytes = sizeof(float) * src.n;
    std::memcpy(dst.px, src.px, bytes); std::memcpy(dst.py, src.py, bytes); std::memcpy(dst.pz, src.pz, bytes);
    std::memcpy(dst.vx, src.vx, bytes); std::memcpy(dst.vy, src.vy, bytes); std::memcpy(dst.vz, src.vz, bytes);
    std::memcpy(dst.mass, src.mass, bytes);
    return dst;
}

double maxPositionDelta(const BodiesSoA& a, const BodiesSoA& b) {
    double maxDelta = 0.0;
    for (int i = 0; i < a.n; ++i) {
        double dx = a.px[i] - b.px[i];
        double dy = a.py[i] - b.py[i];
        double dz = a.pz[i] - b.pz[i];
        maxDelta = std::max(maxDelta, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return maxDelta;
}

void printState(const BodiesSoA& h, const char* label) {
    int shown = std::min(h.n, 5);
    for (int i = 0; i < shown; ++i) {
        std::printf("%s p%d pos=(%.6f, %.6f, %.6f) vel=(%.6f, %.6f, %.6f)\n", label, i, h.px[i], h.py[i], h.pz[i],
                    h.vx[i], h.vy[i], h.vz[i]);
    }
}

// Runs one GPU method on a fresh device copy of h_bodies, timing it, and
// returns the final state on the host.
BodiesSoA runGpu(const BodiesSoA& h_bodies, const Args& args, bool bruteForce, double* outMs) {
    BodiesSoA d_bodies = BodiesSoA::allocDevice(args.n);
    d_bodies.copyToDevice(h_bodies);

    auto t0 = std::chrono::high_resolution_clock::now();
    if (bruteForce) {
        runBruteForceSimulation(d_bodies, args.steps, args.dt);
    } else {
        runBarnesHutSimulation(d_bodies, args.steps, args.dt, args.theta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    *outMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    BodiesSoA result = BodiesSoA::allocHost(args.n);
    result.copyToHost(d_bodies);
    d_bodies.freeDevice();
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    BodiesSoA h_bodies = BodiesSoA::allocHost(args.n);
    if (args.scenario == "orbit") {
        // Exactly 3 bodies, matching src/main.cpp's demo. Padding extra slots
        // with mass-0 bodies at a shared (0,0,0) would force every pair of
        // them through the tree's duplicate-position fallback path for no
        // reason, so random is the scenario to use for n > 3.
        if (args.n != 3) { std::fprintf(stderr, "orbit scenario requires n=3 (use --scenario=random for other n)\n"); return 1; }
        initOrbitScenario(h_bodies);
    } else {
        initRandomScenario(h_bodies, args.seed);
    }

    if (args.mode == "bruteforce" || args.mode == "barnes_hut") {
        bool bruteForce = (args.mode == "bruteforce");
        double ms;
        BodiesSoA result = runGpu(h_bodies, args, bruteForce, &ms);
        std::printf("[GPU %s] n=%d steps=%d theta=%.2f time=%.3f ms (%.3f ms/step)\n", args.mode.c_str(), args.n,
                    args.steps, args.theta, ms, ms / args.steps);
        if (args.printFinal) printState(result, "GPU");
        result.freeHost();
    } else if (args.mode == "compare") {
        double bfMs, bhMs;
        BodiesSoA bfResult = runGpu(h_bodies, args, /*bruteForce=*/true, &bfMs);
        BodiesSoA bhResult = runGpu(h_bodies, args, /*bruteForce=*/false, &bhMs);

        BodiesSoA cpuResult = cloneHost(h_bodies);
        auto c0 = std::chrono::high_resolution_clock::now();
        runCpuReference(cpuResult, args.steps, args.dt);
        auto c1 = std::chrono::high_resolution_clock::now();
        double cpuMs = std::chrono::duration<double, std::milli>(c1 - c0).count();

        std::printf("[CPU reference]   n=%d steps=%d time=%.3f ms\n", args.n, args.steps, cpuMs);
        std::printf("[GPU bruteforce]  n=%d steps=%d time=%.3f ms (%.3f ms/step)\n", args.n, args.steps, bfMs,
                    bfMs / args.steps);
        std::printf("[GPU barnes_hut]  n=%d steps=%d theta=%.2f time=%.3f ms (%.3f ms/step)\n", args.n, args.steps,
                    args.theta, bhMs, bhMs / args.steps);

        double bfVsCpu = maxPositionDelta(bfResult, cpuResult);
        double bhVsCpu = maxPositionDelta(bhResult, cpuResult);
        double bhVsBf = maxPositionDelta(bhResult, bfResult);
        std::printf("max |bruteforce_gpu - cpu|  = %.6e (expect ~1e-3, float rounding only)\n", bfVsCpu);
        std::printf("max |barnes_hut_gpu - cpu|   = %.6e (expect small but larger than above -- theta approximation)\n",
                    bhVsCpu);
        std::printf("max |barnes_hut_gpu - bruteforce_gpu| = %.6e (isolates the theta approximation error)\n", bhVsBf);

        if (args.printFinal) {
            printState(cpuResult, "CPU");
            printState(bfResult, "BF ");
            printState(bhResult, "BH ");
        }

        bfResult.freeHost(); bhResult.freeHost(); cpuResult.freeHost();
    } else {
        std::fprintf(stderr, "unknown mode: %s (expected bruteforce|barnes_hut|compare)\n", args.mode.c_str());
        h_bodies.freeHost();
        return 1;
    }

    h_bodies.freeHost();
    return 0;
}
