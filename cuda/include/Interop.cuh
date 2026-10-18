#pragma once

#include "BodiesSoA.hpp"

// Writes bodies.{px,py,pz} into `vboPositions`, an interleaved (x,y,z)
// float3-per-body device buffer. `vboPositions` is expected to be the
// pointer CUDA got back from cudaGraphicsResourceGetMappedPointer() on an
// OpenGL VBO -- so this kernel is a device-to-device copy from the
// simulation's SoA buffers straight into GPU memory OpenGL will read
// directly, with no host round trip at all (the "zero-copy" part of the
// interop pipeline).
void copyPositionsToVBO(const BodiesSoA& d_bodies, float* vboPositions);
