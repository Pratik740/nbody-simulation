#!/usr/bin/env bash
# Builds both CUDA binaries. Run from anywhere; paths are relative to this
# script's location, so this also works after git clone/pull into a fresh
# Colab runtime.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build

echo "Building nbody_cuda (brute-force / Barnes-Hut benchmark + correctness CLI)..."
nvcc -O3 -std=c++17 -arch=sm_75 \
    -Iinclude \
    src/main.cu src/bruteforce.cu src/barnes_hut.cu src/cpu_reference.cpp src/scenarios.cpp \
    -o build/nbody_cuda

echo "Building nbody_render (CUDA-OpenGL interop renderer)..."
nvcc -O3 -std=c++17 -arch=sm_75 \
    -Iinclude \
    src/render_main.cu src/bruteforce.cu src/barnes_hut.cu src/scenarios.cpp src/interop.cu \
    src/egl_context.cpp src/gl_compat.cpp src/ppm.cpp \
    -lEGL -lGL \
    -o build/nbody_render

echo "Built build/nbody_cuda and build/nbody_render"
