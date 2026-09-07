#pragma once
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace globe {

// Decodes baked tiles (PLAN 3 / 12.1).  Synchronous; GDScript wraps it in
// WorkerThreadPool tasks.  Returned Dictionary:
//   size (int, T+1), height (PackedFloat32Array, [j*size + i], metres),
//   water (PackedFloat32Array, water surface metres, 0 = none),
//   layers (Image RGBA8), flow (Image RGB8), meta (Dictionary), ok (bool),
//   height_min/height_max (float)
class TileLoader : public godot::RefCounted {
    GDCLASS(TileLoader, godot::RefCounted)

protected:
    static void _bind_methods();

public:
    static godot::String tile_dir(const godot::String &world_dir, int lod, int face, int x, int y);
    static bool tile_exists(const godot::String &world_dir, int lod, int face, int x, int y);
    static godot::Dictionary load(const godot::String &world_dir, int lod, int face, int x, int y);
    // 16-bit PNG -> floats via (lo, hi); zero_is_none maps value 0 to 0.0 and 1..65535 to [lo, hi]
    static godot::PackedFloat32Array decode_png16(const godot::PackedByteArray &bytes, double lo, double hi, bool zero_is_none);
    static godot::Ref<godot::Image> decode_png8(const godot::PackedByteArray &bytes);
    // PackedFloat32Array (row-major [j*size+i]) -> Image FORMAT_RF for a height texture
    static godot::Ref<godot::Image> height_to_image(const godot::PackedFloat32Array &height, int size);
    // Flat (size x size) grid mesh arrays for ArrayMesh with an optional skirt ring
    static godot::Array build_mesh_arrays(int size, bool skirt);
    // Read manifest.json / tiles/index.json
    static godot::Dictionary read_json(const godot::String &path);
};

}  // namespace globe
