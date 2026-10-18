#pragma once

#include <cstdint>
#include <string>

// Dumps an RGBA8 buffer (as returned by glReadPixels) to a binary PPM (P6).
// glReadPixels' origin is bottom-left, PPM's is top-left, so this flips
// rows while writing rather than requiring the caller to.
void writePPM(const std::string& path, int width, int height, const uint8_t* rgba);
