#pragma once
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include "cubesphere_math.h"

namespace globe {

// Static cube-sphere helpers exposed to GDScript (PLAN 12.1).
class CubeSphere : public godot::Object {
    GDCLASS(CubeSphere, godot::Object)

protected:
    static void _bind_methods();

public:
    static godot::Vector3 to_sphere(int face, double u, double v);
    // [face, u, v]
    static godot::Array from_sphere(const godot::Vector3 &p);
    // (x, y, lod) of the tile containing (face, u, v)
    static godot::Vector3i tile_of(int face, double u, double v, int lod, int n_fine, int T);
    // Jacobian columns as Basis: x = dp/du, y = dp/dv, z = p (unit sphere)
    static godot::Basis jacobian(int face, double u, double v);
    // Orthonormal tangent frame at anchor: Basis(x = e1, y = anchor, z = e2)
    static godot::Basis tangent_frame(const godot::Vector3 &anchor);
    // Gnomonic local position of a sphere point Q at height h (PLAN 12.2)
    static godot::Vector3 gnomonic_local(const godot::Vector3 &q, double h, const godot::Basis &frame, double r_planet);
    static double planet_radius(int N, double cell_size_m);
    // Triangle soup (2*(size-1)^2 triangles) of a tile's vertices projected
    // into `frame`, for ConcavePolygonShape3D.set_faces (PLAN 12.3).  `height`
    // is the tile's size*size heights, [j * size + i]; empty on bad input.
    static godot::PackedVector3Array build_collision_faces(int face, int lod, int tile_x, int tile_y, int T, int n_fine,
                                                           double r_planet, const godot::Basis &frame,
                                                           const godot::PackedFloat32Array &height);
};

}  // namespace globe
