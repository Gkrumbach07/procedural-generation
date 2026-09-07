// Minimal PNG decoder for tiles (8/16-bit gray, gray+alpha, RGB, RGBA; no
// interlace).  Godot's built-in Image loader strips 16-bit PNGs to 8 bits,
// which would destroy the height precision, so the tile loader decodes
// height.png / water.png itself.  Inflate is delegated to Godot's zlib via
// PackedByteArray::decompress (COMPRESSION_DEFLATE == zlib stream).
#pragma once
#include <cstdint>
#include <vector>

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

namespace globe {

struct PngImage {
    int width = 0, height = 0, channels = 0, bit_depth = 0;
    std::vector<uint16_t> data16;  // used when bit_depth == 16 (row-major, channels interleaved)
    std::vector<uint8_t> data8;    // used when bit_depth == 8
    bool ok = false;
    const char *error = "";
};

inline uint32_t be32(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

inline int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

inline PngImage decode_png(const godot::PackedByteArray &bytes) {
    PngImage out;
    const int64_t n = bytes.size();
    const uint8_t *p = bytes.ptr();
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (n < 8 || memcmp(p, sig, 8) != 0) {
        out.error = "not a PNG";
        return out;
    }
    int64_t pos = 8;
    godot::PackedByteArray idat;
    int color_type = -1;
    while (pos + 8 <= n) {
        uint32_t len = be32(p + pos);
        const uint8_t *type = p + pos + 4;
        const uint8_t *data = p + pos + 8;
        if (pos + 12 + int64_t(len) > n) {
            out.error = "truncated chunk";
            return out;
        }
        if (memcmp(type, "IHDR", 4) == 0) {
            out.width = int(be32(data));
            out.height = int(be32(data + 4));
            out.bit_depth = data[8];
            color_type = data[9];
            if (data[12] != 0) {
                out.error = "interlaced PNG unsupported";
                return out;
            }
        } else if (memcmp(type, "IDAT", 4) == 0) {
            int64_t old = idat.size();
            idat.resize(old + len);
            memcpy(idat.ptrw() + old, data, len);
        } else if (memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + int64_t(len);
    }
    switch (color_type) {
        case 0: out.channels = 1; break;
        case 2: out.channels = 3; break;
        case 4: out.channels = 2; break;
        case 6: out.channels = 4; break;
        default: out.error = "unsupported colour type"; return out;
    }
    if (out.bit_depth != 8 && out.bit_depth != 16) {
        out.error = "unsupported bit depth";
        return out;
    }
    const int bps = out.bit_depth / 8;
    const int bpp = bps * out.channels;                      // bytes per pixel
    const int64_t stride = int64_t(out.width) * bpp;          // bytes per row (unfiltered)
    const int64_t raw_size = int64_t(out.height) * (stride + 1);
    godot::PackedByteArray raw = idat.decompress(raw_size, godot::FileAccess::COMPRESSION_DEFLATE);
    if (raw.size() != raw_size) {
        out.error = "inflate failed";
        return out;
    }
    std::vector<uint8_t> cur(stride), prev(stride, 0);
    const uint8_t *r = raw.ptr();
    if (bps == 2) out.data16.resize(size_t(out.width) * out.height * out.channels);
    else out.data8.resize(size_t(out.width) * out.height * out.channels);
    for (int y = 0; y < out.height; ++y) {
        const uint8_t filter = r[y * (stride + 1)];
        const uint8_t *src = r + y * (stride + 1) + 1;
        for (int64_t x = 0; x < stride; ++x) {
            int a = x >= bpp ? cur[x - bpp] : 0;
            int b = prev[x];
            int c = x >= bpp ? prev[x - bpp] : 0;
            int v = src[x];
            switch (filter) {
                case 0: break;
                case 1: v += a; break;
                case 2: v += b; break;
                case 3: v += (a + b) / 2; break;
                case 4: v += paeth(a, b, c); break;
                default: out.error = "bad filter"; return out;
            }
            cur[x] = uint8_t(v);
        }
        if (bps == 2) {
            uint16_t *dst = out.data16.data() + size_t(y) * out.width * out.channels;
            for (int64_t i = 0; i < stride / 2; ++i) dst[i] = uint16_t((cur[2 * i] << 8) | cur[2 * i + 1]);
        } else {
            memcpy(out.data8.data() + size_t(y) * stride, cur.data(), stride);
        }
        std::swap(cur, prev);
    }
    out.ok = true;
    return out;
}

}  // namespace globe
