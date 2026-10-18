#!/usr/bin/env bash
# Builds both CUDA binaries. Run from anywhere; paths are relative to this
# script's location, so this also works after git clone/pull into a fresh
# Colab runtime.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build

# -arch=native (CUDA 11.8+) targets whatever GPU is actually in this box --
# T4 (Turing, sm_75), L40S (Ada, sm_89), or anything else -- instead of a
# hardcoded sm_75 that would just JIT-compile (correct, but unoptimized)
# on newer hardware. Falls back to sm_75 for older nvcc that predates
# -arch=native.
ARCH=native
if ! nvcc -arch=native --version >/dev/null 2>&1; then
    echo "nvcc doesn't support -arch=native; falling back to sm_75 (Turing/T4)" >&2
    ARCH=sm_75
fi

echo "Building nbody_cuda (brute-force / Barnes-Hut benchmark + correctness CLI)..."
nvcc -O3 -std=c++17 -arch=$ARCH \
    -Iinclude \
    src/main.cu src/bruteforce.cu src/barnes_hut.cu src/cpu_reference.cpp src/scenarios.cpp \
    -o build/nbody_cuda

echo "Building nbody_render (CUDA-OpenGL interop renderer)..."
# Link GLVND's libOpenGL.so, not legacy Mesa libGL.so: this renderer loads
# GL >=3.2 entry points via eglGetProcAddress (the GLVND model). On boxes
# with the NVIDIA driver installed (e.g. Colab's /usr/lib64-nvidia), NVIDIA
# ships its own complete, self-contained copies of libOpenGL.so/libGL.so
# that don't depend on Mesa's libglapi.so at all. Mesa's copies at the
# standard multiarch path do need it, and chasing that dependency down is
# more fragile than just preferring NVIDIA's own libraries when present --
# point the linker there first via -L so it picks those instead.
NVIDIA_LIBDIR_FLAG=""
if [ -d /usr/lib64-nvidia ]; then
    NVIDIA_LIBDIR_FLAG="-L/usr/lib64-nvidia"
fi
nvcc -O3 -std=c++17 -arch=$ARCH \
    -Iinclude \
    src/render_main.cu src/bruteforce.cu src/barnes_hut.cu src/scenarios.cpp src/interop.cu \
    src/egl_context.cpp src/gl_compat.cpp src/ppm.cpp \
    $NVIDIA_LIBDIR_FLAG -lEGL -lOpenGL \
    -o build/nbody_render

echo "Built build/nbody_cuda and build/nbody_render"
