#!/usr/bin/env bash
# Builds the CUDA n-body binary. Run from anywhere; paths are relative to
# this script's location, so this also works after `git clone`/`git pull`
# into a fresh Colab runtime.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_75 \
    -Iinclude \
    src/main.cu src/bruteforce.cu src/barnes_hut.cu src/cpu_reference.cpp \
    -o build/nbody_cuda

echo "Built cuda/build/nbody_cuda"
