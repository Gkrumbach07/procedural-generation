// Cube-sphere mapping — C++ twin of bake/globe/cubesphere.py.
// Header-only, no Godot dependency, so it can be unit-tested standalone
// (gdextension/tests/test_cubesphere.cpp) and used from the GDExtension.
//
// The arithmetic mirrors the *scalar* numba functions (to_sphere, face_of,
// from_sphere, project_to_face) operation for operation, so results are
// bit-identical on bake/tests/data/cubesphere_vectors.bin when compiled
// without FP contraction (-ffp-contract=off; see gdextension/tests/Makefile)
// against a libm whose tan/atan agree with the generating glibc.  The same
// flag is required for the GDExtension build of this header if bit-parity
// with the bake is relied upon at runtime (the runtime itself only needs
// PLAN 2.2's 1e-6).  `test_cubesphere --tolerant` relaxes the check to 1e-12
// for other libms.
#pragma once
#include <cmath>
#include <cstdint>

namespace globe {

struct Vec3 {
    double x, y, z;
};

// Faces 0..5 = +X, -X, +Y, -Y, +Z, -Z; BASES[f] = {right, up, normal}
// (OpenGL cubemap convention; identical to Python BASES).
static const double BASES[6][3][3] = {
    {{0.0, 0.0, -1.0}, {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}},   // 0 +X
    {{0.0, 0.0, 1.0}, {0.0, -1.0, 0.0}, {-1.0, 0.0, 0.0}},   // 1 -X
    {{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}},     // 2 +Y
    {{1.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, -1.0, 0.0}},   // 3 -Y
    {{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 1.0}},    // 4 +Z
    {{-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}},  // 5 -Z
};

static const double HALF_PI = 1.5707963267948966;
static const double TWO_OVER_PI = 0.6366197723675814;

inline double planet_radius(int N, double cell_size_m) {
    return N * cell_size_m * 4.0 / (2.0 * 3.141592653589793);
}

// Face-local (u, v) -> unit vector.  Valid slightly outside [0, 1).
inline Vec3 to_sphere(int face, double u, double v) {
    const double s = std::tan((u - 0.5) * HALF_PI);
    const double t = std::tan((v - 0.5) * HALF_PI);
    const double *r = BASES[face][0];
    const double *up = BASES[face][1];
    const double *n = BASES[face][2];
    const double x = n[0] + s * r[0] + t * up[0];
    const double y = n[1] + s * r[1] + t * up[1];
    const double z = n[2] + s * r[2] + t * up[2];
    const double inv = 1.0 / std::sqrt(x * x + y * y + z * z);
    return {x * inv, y * inv, z * inv};
}

// argmax |component| with sign; ties resolve X > Y > Z (same as Python).
inline int face_of(double x, double y, double z) {
    const double ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
    if (ax >= ay && ax >= az) return x >= 0.0 ? 0 : 1;
    if (ay >= az) return y >= 0.0 ? 2 : 3;
    return z >= 0.0 ? 4 : 5;
}

// Project onto a given face's parametrisation (may fall outside [0, 1)).
inline void project_to_face(int face, double x, double y, double z, double &u, double &v) {
    const double *r = BASES[face][0];
    const double *up = BASES[face][1];
    const double *n = BASES[face][2];
    const double d = x * n[0] + y * n[1] + z * n[2];
    const double s = (x * r[0] + y * r[1] + z * r[2]) / d;
    const double t = (x * up[0] + y * up[1] + z * up[2]) / d;
    u = std::atan(s) * TWO_OVER_PI + 0.5;
    v = std::atan(t) * TWO_OVER_PI + 0.5;
}

// Unit vector -> (face, u, v), u, v in the closed interval [0, 1]: a point
// exactly on a cube edge/corner goes to the higher-priority face (X > Y > Z)
// with u or v exactly 1.0, so callers computing a cell index must clamp,
// min(floor(u*N), N-1) (tile_of below does).
inline int from_sphere(double x, double y, double z, double &u, double &v) {
    const int face = face_of(x, y, z);
    project_to_face(face, x, y, z, u, v);
    return face;
}

// Tile containing (face, u, v) at a given LOD: tile edge = T fine cells at
// LOD 0, doubling per LOD.  Returns (x, y) tile indices.
inline void tile_of(double u, double v, int N_fine, int T, int lod, int &tx, int &ty) {
    int tiles = N_fine / T;
    for (int l = 0; l < lod && tiles > 1; ++l) tiles /= 2;
    if (tiles < 1) tiles = 1;
    int ix = static_cast<int>(std::floor(u * tiles));
    int iy = static_cast<int>(std::floor(v * tiles));
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (ix >= tiles) ix = tiles - 1;
    if (iy >= tiles) iy = tiles - 1;
    tx = ix;
    ty = iy;
}

// Gnomonic projection of Q onto the tangent plane at anchor A with frame
// (e1, e2, A): returns local (x, z) in metres (PLAN 12.2).  Q and A unit.
inline void gnomonic_local(const Vec3 &Q, const Vec3 &A, const Vec3 &e1, const Vec3 &e2, double R_planet,
                           double &lx, double &lz) {
    const double d = Q.x * A.x + Q.y * A.y + Q.z * A.z;
    const double dx = Q.x / d, dy = Q.y / d, dz = Q.z / d;
    lx = (dx * e1.x + dy * e1.y + dz * e1.z) * R_planet;
    lz = (dx * e2.x + dy * e2.y + dz * e2.z) * R_planet;
}

// Orthonormal tangent frame at A: e1 ⊥ A (built from the world axis least
// aligned with A), e2 = A × e1.
inline void tangent_frame(const Vec3 &A, Vec3 &e1, Vec3 &e2) {
    Vec3 ax = std::fabs(A.x) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
    const double d = ax.x * A.x + ax.y * A.y + ax.z * A.z;
    e1 = {ax.x - A.x * d, ax.y - A.y * d, ax.z - A.z * d};
    const double inv = 1.0 / std::sqrt(e1.x * e1.x + e1.y * e1.y + e1.z * e1.z);
    e1 = {e1.x * inv, e1.y * inv, e1.z * inv};
    e2 = {A.y * e1.z - A.z * e1.y, A.z * e1.x - A.x * e1.z, A.x * e1.y - A.y * e1.x};
}

}  // namespace globe
