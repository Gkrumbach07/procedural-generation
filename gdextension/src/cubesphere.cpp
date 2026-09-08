#include "cubesphere.h"

#include <vector>

using namespace godot;

namespace globe {

void CubeSphere::_bind_methods() {
    ClassDB::bind_static_method("CubeSphere", D_METHOD("to_sphere", "face", "u", "v"), &CubeSphere::to_sphere);
    ClassDB::bind_static_method("CubeSphere", D_METHOD("from_sphere", "p"), &CubeSphere::from_sphere);
    ClassDB::bind_static_method("CubeSphere", D_METHOD("tile_of", "face", "u", "v", "lod", "n_fine", "T"), &CubeSphere::tile_of);
    ClassDB::bind_static_method("CubeSphere", D_METHOD("jacobian", "face", "u", "v"), &CubeSphere::jacobian);
    ClassDB::bind_static_method("CubeSphere", D_METHOD("tangent_frame", "anchor"), &CubeSphere::tangent_frame);
    ClassDB::bind_static_method("CubeSphere", D_METHOD("gnomonic_local", "q", "h", "frame", "r_planet"), &CubeSphere::gnomonic_local);
    ClassDB::bind_static_method("CubeSphere", D_METHOD("planet_radius", "N", "cell_size_m"), &CubeSphere::planet_radius);
    ClassDB::bind_static_method(
        "CubeSphere",
        D_METHOD("build_collision_faces", "face", "lod", "tile_x", "tile_y", "T", "n_fine", "r_planet", "frame", "height"),
        &CubeSphere::build_collision_faces);
}

Vector3 CubeSphere::to_sphere(int face, double u, double v) {
    Vec3 p = globe::to_sphere(face, u, v);
    return Vector3(real_t(p.x), real_t(p.y), real_t(p.z));
}

Array CubeSphere::from_sphere(const Vector3 &p) {
    double u, v;
    int face = globe::from_sphere(double(p.x), double(p.y), double(p.z), u, v);
    Array a;
    a.push_back(face);
    a.push_back(u);
    a.push_back(v);
    return a;
}

Vector3i CubeSphere::tile_of(int face, double u, double v, int lod, int n_fine, int T) {
    (void)face;
    int tx, ty;
    globe::tile_of(u, v, n_fine, T, lod, tx, ty);
    return Vector3i(tx, ty, lod);
}

Basis CubeSphere::jacobian(int face, double u, double v) {
    const double a = (u - 0.5) * HALF_PI, b = (v - 0.5) * HALF_PI;
    const double s = std::tan(a), t = std::tan(b);
    const double dsdu = HALF_PI * (1.0 + s * s), dtdv = HALF_PI * (1.0 + t * t);
    const double *r = BASES[face][0];
    const double *up = BASES[face][1];
    const double *n = BASES[face][2];
    const double qx = n[0] + s * r[0] + t * up[0], qy = n[1] + s * r[1] + t * up[1], qz = n[2] + s * r[2] + t * up[2];
    const double ql = std::sqrt(qx * qx + qy * qy + qz * qz);
    const double px = qx / ql, py = qy / ql, pz = qz / ql;
    double dux = r[0] * dsdu, duy = r[1] * dsdu, duz = r[2] * dsdu;
    double pd = px * dux + py * duy + pz * duz;
    Vector3 ju(real_t((dux - px * pd) / ql), real_t((duy - py * pd) / ql), real_t((duz - pz * pd) / ql));
    double dvx = up[0] * dtdv, dvy = up[1] * dtdv, dvz = up[2] * dtdv;
    pd = px * dvx + py * dvy + pz * dvz;
    Vector3 jv(real_t((dvx - px * pd) / ql), real_t((dvy - py * pd) / ql), real_t((dvz - pz * pd) / ql));
    Basis B;
    B.set_column(0, ju);
    B.set_column(1, jv);
    B.set_column(2, Vector3(real_t(px), real_t(py), real_t(pz)));
    return B;
}

Basis CubeSphere::tangent_frame(const Vector3 &anchor) {
    Vec3 A{double(anchor.x), double(anchor.y), double(anchor.z)};
    const double l = std::sqrt(A.x * A.x + A.y * A.y + A.z * A.z);
    A = {A.x / l, A.y / l, A.z / l};
    Vec3 e1, e2;
    globe::tangent_frame(A, e1, e2);
    Basis B;
    B.set_column(0, Vector3(real_t(e1.x), real_t(e1.y), real_t(e1.z)));
    B.set_column(1, Vector3(real_t(A.x), real_t(A.y), real_t(A.z)));
    B.set_column(2, Vector3(real_t(e2.x), real_t(e2.y), real_t(e2.z)));
    return B;
}

Vector3 CubeSphere::gnomonic_local(const Vector3 &q, double h, const Basis &frame, double r_planet) {
    Vector3 e1 = frame.get_column(0), A = frame.get_column(1), e2 = frame.get_column(2);
    Vec3 Q{double(q.x), double(q.y), double(q.z)};
    Vec3 Av{double(A.x), double(A.y), double(A.z)};
    Vec3 E1{double(e1.x), double(e1.y), double(e1.z)};
    Vec3 E2{double(e2.x), double(e2.y), double(e2.z)};
    double lx, lz;
    globe::gnomonic_local(Q, Av, E1, E2, r_planet, lx, lz);
    return Vector3(real_t(lx), real_t(h), real_t(lz));
}

double CubeSphere::planet_radius(int N, double cell_size_m) {
    return globe::planet_radius(N, cell_size_m);
}

PackedVector3Array CubeSphere::build_collision_faces(int face, int lod, int tile_x, int tile_y, int T, int n_fine,
                                                     double r_planet, const Basis &frame,
                                                     const PackedFloat32Array &height) {
    PackedVector3Array out;
    const int64_t n = height.size();
    const int size = int(std::llround(std::sqrt(double(n))));
    if (face < 0 || face > 5 || lod < 0 || T < 1 || n_fine < 1 || size < 2 || int64_t(size) * size != n) return out;
    const Vector3 c0 = frame.get_column(0), c1 = frame.get_column(1), c2 = frame.get_column(2);
    const Vec3 e1{double(c0.x), double(c0.y), double(c0.z)};
    const Vec3 A{double(c1.x), double(c1.y), double(c1.z)};
    const Vec3 e2{double(c2.x), double(c2.y), double(c2.z)};
    // one tan per row and per column instead of two per vertex
    std::vector<double> su(size), tv(size);
    for (int k = 0; k < size; ++k) {
        su[k] = std::tan((globe::tile_uv(tile_x, k, T, n_fine, lod) - 0.5) * HALF_PI);
        tv[k] = std::tan((globe::tile_uv(tile_y, k, T, n_fine, lod) - 0.5) * HALF_PI);
    }
    const float *h = height.ptr();
    std::vector<Vector3> pts(size_t(size) * size);
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            const Vec3 q = globe::to_sphere_st(face, su[i], tv[j]);
            double lx, lz;
            globe::gnomonic_local_clamped(q, A, e1, e2, r_planet, globe::GNOMONIC_MIN_D, lx, lz);
            pts[size_t(j) * size + i] = Vector3(real_t(lx), real_t(h[j * size + i]), real_t(lz));
        }
    }
    out.resize(int64_t(size - 1) * (size - 1) * 6);
    Vector3 *w = out.ptrw();
    int64_t k = 0;
    for (int j = 0; j < size - 1; ++j) {
        for (int i = 0; i < size - 1; ++i) {
            const Vector3 &v00 = pts[size_t(j) * size + i];
            const Vector3 &v10 = pts[size_t(j) * size + i + 1];
            const Vector3 &v01 = pts[size_t(j + 1) * size + i];
            const Vector3 &v11 = pts[size_t(j + 1) * size + i + 1];
            w[k] = v00; w[k + 1] = v01; w[k + 2] = v10;
            w[k + 3] = v10; w[k + 4] = v01; w[k + 5] = v11;
            k += 6;
        }
    }
    return out;
}

}  // namespace globe
