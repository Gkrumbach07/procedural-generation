#include "tile_loader.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "png16.h"

using namespace godot;

namespace globe {

void TileLoader::_bind_methods() {
    ClassDB::bind_static_method("TileLoader", D_METHOD("tile_dir", "world_dir", "lod", "face", "x", "y"), &TileLoader::tile_dir);
    ClassDB::bind_static_method("TileLoader", D_METHOD("tile_exists", "world_dir", "lod", "face", "x", "y"), &TileLoader::tile_exists);
    ClassDB::bind_static_method("TileLoader", D_METHOD("load", "world_dir", "lod", "face", "x", "y"), &TileLoader::load);
    ClassDB::bind_static_method("TileLoader", D_METHOD("decode_png16", "bytes", "lo", "hi", "zero_is_none"), &TileLoader::decode_png16);
    ClassDB::bind_static_method("TileLoader", D_METHOD("decode_png8", "bytes"), &TileLoader::decode_png8);
    ClassDB::bind_static_method("TileLoader", D_METHOD("height_to_image", "height", "size"), &TileLoader::height_to_image);
    ClassDB::bind_static_method("TileLoader", D_METHOD("build_mesh_arrays", "size", "skirt"), &TileLoader::build_mesh_arrays);
    ClassDB::bind_static_method("TileLoader", D_METHOD("read_json", "path"), &TileLoader::read_json);
}

String TileLoader::tile_dir(const String &world_dir, int lod, int face, int x, int y) {
    return world_dir + String("/tiles/L") + String::num_int64(lod) + String("/f") + String::num_int64(face) + String("/") + String::num_int64(x) + String("_") + String::num_int64(y);
}

bool TileLoader::tile_exists(const String &world_dir, int lod, int face, int x, int y) {
    return FileAccess::file_exists(tile_dir(world_dir, lod, face, x, y) + String("/meta.json"));
}

Dictionary TileLoader::read_json(const String &path) {
    Dictionary d;
    if (!FileAccess::file_exists(path)) return d;
    String text = FileAccess::get_file_as_string(path);
    Variant v = JSON::parse_string(text);
    if (v.get_type() == Variant::DICTIONARY) return v;
    return d;
}

PackedFloat32Array TileLoader::decode_png16(const PackedByteArray &bytes, double lo, double hi, bool zero_is_none) {
    PackedFloat32Array out;
    PngImage img = decode_png(bytes);
    if (!img.ok || img.bit_depth != 16 || img.channels != 1) {
        UtilityFunctions::push_error(String("decode_png16: ") + (img.ok ? "expected 16-bit gray" : img.error));
        return out;
    }
    out.resize(int64_t(img.width) * img.height);
    float *dst = out.ptrw();
    const float flo = float(lo), fspan = float(hi - lo);
    for (size_t i = 0; i < img.data16.size(); ++i) {
        uint16_t v = img.data16[i];
        if (zero_is_none) {
            dst[i] = v == 0 ? 0.0f : (float(v - 1) / 65534.0f) * fspan + flo;
        } else {
            dst[i] = (float(v) / 65535.0f) * fspan + flo;
        }
    }
    return out;
}

Ref<Image> TileLoader::decode_png8(const PackedByteArray &bytes) {
    PngImage img = decode_png(bytes);
    if (!img.ok || img.bit_depth != 8) {
        UtilityFunctions::push_error(String("decode_png8: ") + (img.ok ? "expected 8-bit" : img.error));
        return Ref<Image>();
    }
    Image::Format fmt = img.channels == 4 ? Image::FORMAT_RGBA8 : img.channels == 3 ? Image::FORMAT_RGB8 : img.channels == 2 ? Image::FORMAT_LA8 : Image::FORMAT_L8;
    PackedByteArray data;
    data.resize(int64_t(img.data8.size()));
    memcpy(data.ptrw(), img.data8.data(), img.data8.size());
    return Image::create_from_data(img.width, img.height, false, fmt, data);
}

Dictionary TileLoader::load(const String &world_dir, int lod, int face, int x, int y) {
    Dictionary out;
    out["ok"] = false;
    String dir = tile_dir(world_dir, lod, face, x, y);
    Dictionary meta = read_json(dir + String("/meta.json"));
    if (meta.is_empty()) {
        out["error"] = String("missing meta.json in ") + dir;
        return out;
    }
    const double hmin = meta.get("height_min", 0.0), hmax = meta.get("height_max", 1.0);
    const double wmin = meta.get("water_min", 0.0), wmax = meta.get("water_max", 1.0);
    PackedFloat32Array height = decode_png16(FileAccess::get_file_as_bytes(dir + String("/height.png")), hmin, hmax, false);
    if (height.size() == 0) {
        out["error"] = String("bad height.png in ") + dir;
        return out;
    }
    PackedFloat32Array water = decode_png16(FileAccess::get_file_as_bytes(dir + String("/water.png")), wmin, wmax, true);
    Ref<Image> layers = decode_png8(FileAccess::get_file_as_bytes(dir + String("/layers.png")));
    Ref<Image> flow = decode_png8(FileAccess::get_file_as_bytes(dir + String("/flow.png")));
    const int size = int(meta.get("size", int(std::lround(std::sqrt(double(height.size()))))));
    out["ok"] = true;
    out["lod"] = lod;
    out["face"] = face;
    out["x"] = x;
    out["y"] = y;
    out["size"] = size;
    out["height"] = height;
    out["water"] = water;
    out["layers"] = layers;
    out["flow"] = flow;
    out["meta"] = meta;
    out["height_min"] = hmin;
    out["height_max"] = hmax;
    return out;
}

Ref<Image> TileLoader::height_to_image(const PackedFloat32Array &height, int size) {
    PackedByteArray data;
    data.resize(int64_t(size) * size * 4);
    memcpy(data.ptrw(), height.ptr(), size_t(size) * size * 4);
    return Image::create_from_data(size, size, false, Image::FORMAT_RF, data);
}

// Vertices at (i, 0, j) for i, j in 0..size-1, UV = (i, j)/(size-1).  With a
// skirt, an extra ring of vertices is added around the grid with UV clamped
// to the edge and the y set to -1 (the shader drops skirt vertices by a
// fixed amount).  Index layout: two triangles per cell, counter-clockwise
// seen from +y.
Array TileLoader::build_mesh_arrays(int size, bool skirt) {
    PackedVector3Array verts;
    PackedVector2Array uvs;
    PackedInt32Array idx;
    const int n = size;
    const int ext = skirt ? n + 2 : n;
    verts.resize(int64_t(ext) * ext);
    uvs.resize(int64_t(ext) * ext);
    for (int a = 0; a < ext; ++a) {
        for (int b = 0; b < ext; ++b) {
            int i = skirt ? a - 1 : a, j = skirt ? b - 1 : b;
            const bool is_skirt = skirt && (i < 0 || j < 0 || i >= n || j >= n);
            int ci = std::min(std::max(i, 0), n - 1), cj = std::min(std::max(j, 0), n - 1);
            verts[a * ext + b] = Vector3(real_t(ci), is_skirt ? real_t(-1) : real_t(0), real_t(cj));
            uvs[a * ext + b] = Vector2(real_t(ci) / real_t(n - 1), real_t(cj) / real_t(n - 1));
        }
    }
    idx.resize(int64_t(ext - 1) * (ext - 1) * 6);
    int k = 0;
    for (int a = 0; a < ext - 1; ++a) {
        for (int b = 0; b < ext - 1; ++b) {
            int v00 = a * ext + b, v10 = (a + 1) * ext + b, v01 = a * ext + b + 1, v11 = (a + 1) * ext + b + 1;
            idx[k++] = v00; idx[k++] = v01; idx[k++] = v10;
            idx[k++] = v10; idx[k++] = v01; idx[k++] = v11;
        }
    }
    Array arrays;
    arrays.resize(Mesh::ARRAY_MAX);
    arrays[Mesh::ARRAY_VERTEX] = verts;
    arrays[Mesh::ARRAY_TEX_UV] = uvs;
    arrays[Mesh::ARRAY_INDEX] = idx;
    return arrays;
}

}  // namespace globe
