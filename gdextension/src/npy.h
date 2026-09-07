// Tiny reader for NumPy .npy files (v1/v2 headers, C order, little endian).
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace globe {

struct NpyArray {
    std::string dtype;  // e.g. "<i4", "<f4", "|u1"
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
    bool ok = false;
    int64_t count() const {
        int64_t c = 1;
        for (int64_t s : shape) c *= s;
        return c;
    }
};

inline NpyArray read_npy(const godot::String &path) {
    NpyArray out;
    godot::PackedByteArray bytes = godot::FileAccess::get_file_as_bytes(path);
    const int64_t n = bytes.size();
    const uint8_t *p = bytes.ptr();
    if (n < 10 || p[0] != 0x93 || memcmp(p + 1, "NUMPY", 5) != 0) return out;
    const int major = p[6];
    int64_t header_len, off;
    if (major == 1) {
        header_len = p[8] | (p[9] << 8);
        off = 10;
    } else {
        header_len = p[8] | (p[9] << 8) | (p[10] << 16) | (int64_t(p[11]) << 24);
        off = 12;
    }
    std::string header(reinterpret_cast<const char *>(p + off), size_t(header_len));
    auto find_val = [&](const char *key) -> std::string {
        size_t k = header.find(key);
        if (k == std::string::npos) return "";
        size_t colon = header.find(':', k);
        size_t start = header.find_first_not_of(" ", colon + 1);
        size_t end;
        if (header[start] == '\'' || header[start] == '"') {
            end = header.find(header[start], start + 1);
            return header.substr(start + 1, end - start - 1);
        }
        if (header[start] == '(') {
            end = header.find(')', start);
            return header.substr(start + 1, end - start - 1);
        }
        end = header.find_first_of(",}", start);
        return header.substr(start, end - start);
    };
    out.dtype = find_val("'descr'");
    if (find_val("'fortran_order'").find("True") != std::string::npos) return out;
    std::string shape = find_val("'shape'");
    size_t i = 0;
    while (i < shape.size()) {
        while (i < shape.size() && (shape[i] == ' ' || shape[i] == ',')) ++i;
        if (i >= shape.size()) break;
        size_t j = i;
        while (j < shape.size() && isdigit(shape[j])) ++j;
        if (j > i) out.shape.push_back(std::stoll(shape.substr(i, j - i)));
        i = j + 1;
    }
    const int64_t data_off = off + header_len;
    out.data.assign(p + data_off, p + n);
    out.ok = true;
    return out;
}

}  // namespace globe
