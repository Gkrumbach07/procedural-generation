// Standalone test: the C++ cube-sphere math matches the Python reference on
// bake/tests/data/cubesphere_vectors.bin.  Build and run with
//   make -C gdextension/tests   (or: g++ -std=c++17 -O2 -I../src test_cubesphere.cpp -o test_cubesphere && ./test_cubesphere)
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#include "cubesphere_math.h"

using namespace globe;

static int failures = 0;
#define CHECK(cond, ...)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            ++failures;                                              \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);         \
            std::printf(__VA_ARGS__);                                \
            std::printf("\n");                                       \
        }                                                            \
    } while (0)

#pragma pack(push, 1)
struct Rec {
    int32_t face;
    double u, v, x, y, z;
};
#pragma pack(pop)

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../../bake/tests/data/cubesphere_vectors.bin";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("cannot open %s\n", path);
        return 2;
    }
    char magic[4];
    in.read(magic, 4);
    CHECK(std::memcmp(magic, "CSV1", 4) == 0, "bad magic");
    int32_t K = 0;
    in.read(reinterpret_cast<char *>(&K), 4);
    std::vector<Rec> recs(K);
    in.read(reinterpret_cast<char *>(recs.data()), sizeof(Rec) * K);
    CHECK(static_cast<size_t>(in.gcount()) == sizeof(Rec) * K, "short read");

    double worst = 0.0, worst_rt = 0.0;
    for (const Rec &r : recs) {
        Vec3 p = to_sphere(r.face, r.u, r.v);
        double e = std::max({std::fabs(p.x - r.x), std::fabs(p.y - r.y), std::fabs(p.z - r.z)});
        worst = std::max(worst, e);
        double u2, v2;
        int f2 = from_sphere(p.x, p.y, p.z, u2, v2);
        Vec3 q = to_sphere(f2, u2, v2);
        double e2 = std::max({std::fabs(p.x - q.x), std::fabs(p.y - q.y), std::fabs(p.z - q.z)});
        worst_rt = std::max(worst_rt, e2);
        CHECK(u2 >= 0.0 && u2 <= 1.0 && v2 >= 0.0 && v2 <= 1.0, "uv out of range");
    }
    CHECK(worst < 1e-12, "to_sphere mismatch vs python: %g", worst);
    CHECK(worst_rt < 1e-12, "round trip error: %g", worst_rt);
    std::printf("vectors: %d records, max |dp| vs python = %.3g, round trip = %.3g\n", K, worst, worst_rt);

    // random round trips incl. near-edge points
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> G(0.0, 1.0);
    double worst_r = 0.0;
    for (int i = 0; i < 200000; ++i) {
        double x = G(rng), y = G(rng), z = G(rng);
        double n = std::sqrt(x * x + y * y + z * z);
        x /= n; y /= n; z /= n;
        double u, v;
        int f = from_sphere(x, y, z, u, v);
        Vec3 p = to_sphere(f, u, v);
        worst_r = std::max(worst_r, std::max({std::fabs(p.x - x), std::fabs(p.y - y), std::fabs(p.z - z)}));
    }
    CHECK(worst_r < 1e-12, "random round trip error %g", worst_r);

    // tile_of sanity
    int tx, ty;
    tile_of(0.999, 0.0, 4096, 256, 0, tx, ty);
    CHECK(tx == 15 && ty == 0, "tile_of lod0: %d %d", tx, ty);
    tile_of(0.999, 0.0, 4096, 256, 4, tx, ty);
    CHECK(tx == 0 && ty == 0, "tile_of lod4: %d %d", tx, ty);

    // gnomonic: the anchor maps to the origin, a nearby point to ~its arc distance
    Vec3 A = to_sphere(4, 0.5, 0.5), e1, e2;
    tangent_frame(A, e1, e2);
    double lx, lz;
    gnomonic_local(A, A, e1, e2, 1000.0, lx, lz);
    CHECK(std::fabs(lx) < 1e-9 && std::fabs(lz) < 1e-9, "anchor not at origin");
    Vec3 Q = to_sphere(4, 0.5 + 0.01, 0.5);
    gnomonic_local(Q, A, e1, e2, 1000.0, lx, lz);
    double dist = std::sqrt(lx * lx + lz * lz);
    double arc = std::acos(Q.x * A.x + Q.y * A.y + Q.z * A.z) * 1000.0;
    CHECK(std::fabs(dist - arc) / arc < 1e-3, "gnomonic distance %g vs arc %g", dist, arc);

    if (failures == 0) std::printf("OK\n");
    return failures == 0 ? 0 : 1;
}
