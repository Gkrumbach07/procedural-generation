#include "cubesphere.h"

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

}  // namespace globe
