#include "Ppm.hpp"

#include <cstdio>
#include <vector>

void writePPM(const std::string& path, int width, int height, const uint8_t* rgba) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "writePPM: failed to open %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "P6\n%d %d\n255\n", width, height);

    std::vector<uint8_t> row(static_cast<size_t>(width) * 3);
    for (int y = height - 1; y >= 0; --y) {  // flip: glReadPixels is bottom-up, PPM is top-down
        const uint8_t* src = rgba + static_cast<size_t>(y) * width * 4;
        for (int x = 0; x < width; ++x) {
            row[x * 3 + 0] = src[x * 4 + 0];
            row[x * 3 + 1] = src[x * 4 + 1];
            row[x * 3 + 2] = src[x * 4 + 2];
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
}
