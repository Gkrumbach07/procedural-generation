// Standalone test: the C++ cube-sphere math matches the Python reference on
// bake/tests/data/cubesphere_vectors.bin (format CSV2, see
// bake/tests/gen_vectors.py).  Build and run with
//   make -C gdextension/tests test
// (or: g++ -std=c++17 -O2 -ffp-contract=off -I../src test_cubesphere.cpp -o test_cubesphere && ./test_cubesphere)
// Default mode requires bit equality with the scalar numba reference (glibc
// libm, no FMA contraction); pass --tolerant to relax to 1e-12 for other libms.
#include <algorithm>
#include <cstdint>
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
    int32_t face2;
    double u2, v2;
};
#pragma pack(pop)
static_assert(sizeof(Rec) == 64, "Rec must match gen_vectors.py '<idddddidd'");

int main(int argc, char **argv) {
    const char *path = "../../bake/tests/data/cubesphere_vectors.bin";
    bool tolerant = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tolerant") == 0)
            tolerant = true;
        else
            path = argv[i];
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("cannot open %s\n", path);
        return 2;
    }
    char magic[4];
    in.read(magic, 4);
    CHECK(std::memcmp(magic, "CSV2", 4) == 0, "bad magic (expected CSV2; regenerate with bake/tests/gen_vectors.py)");
    int32_t K = 0;
    in.read(reinterpret_cast<char *>(&K), 4);
    std::vector<Rec> recs(K);
    in.read(reinterpret_cast<char *>(recs.data()), sizeof(Rec) * K);
    CHECK(static_cast<size_t>(in.gcount()) == sizeof(Rec) * K, "short read");

    // to_sphere and from_sphere against the Python scalar reference: bit-exact
    // by default (--tolerant: 1e-12), checked independently (from_sphere is
    // evaluated on the file's x, y, z, not the recomputed point)
    double worst = 0.0, worst_rt = 0.0, worst_uv = 0.0;
    int n_bits_p = 0, n_bits_uv = 0;
    for (const Rec &r : recs) {
        Vec3 p = to_sphere(r.face, r.u, r.v);
        double e = std::max({std::fabs(p.x - r.x), std::fabs(p.y - r.y), std::fabs(p.z - r.z)});
        worst = std::max(worst, e);
        if (!(p.x == r.x && p.y == r.y && p.z == r.z)) ++n_bits_p;
        double u2, v2;
        int f2 = from_sphere(r.x, r.y, r.z, u2, v2);
        CHECK(f2 == r.face2, "from_sphere face %d != %d", f2, r.face2);
        worst_uv = std::max(worst_uv, std::max(std::fabs(u2 - r.u2), std::fabs(v2 - r.v2)));
        if (!(f2 == r.face2 && u2 == r.u2 && v2 == r.v2)) ++n_bits_uv;
        CHECK(u2 >= 0.0 && u2 <= 1.0 && v2 >= 0.0 && v2 <= 1.0, "uv out of range");
        // round trip closure on the recomputed point
        double u3, v3;
        int f3 = from_sphere(p.x, p.y, p.z, u3, v3);
        Vec3 q = to_sphere(f3, u3, v3);
        double e2 = std::max({std::fabs(p.x - q.x), std::fabs(p.y - q.y), std::fabs(p.z - q.z)});
        worst_rt = std::max(worst_rt, e2);
    }
    if (tolerant) {
        CHECK(worst < 1e-12, "to_sphere mismatch vs python: %g", worst);
        CHECK(worst_uv < 1e-12, "from_sphere mismatch vs python: %g", worst_uv);
    } else {
        CHECK(n_bits_p == 0, "to_sphere not bit-identical to python on %d records (max |dp| = %g); build with -ffp-contract=off or run --tolerant", n_bits_p, worst);
        CHECK(n_bits_uv == 0, "from_sphere not bit-identical to python on %d records (max |duv| = %g); build with -ffp-contract=off or run --tolerant", n_bits_uv, worst_uv);
    }
    CHECK(worst_rt < 1e-12, "round trip error: %g", worst_rt);
    std::printf("vectors: %d records, max |dp| vs python = %.3g (%d records differ), max |duv| = %.3g (%d differ), round trip = %.3g%s\n",
                K, worst, n_bits_p, worst_uv, n_bits_uv, worst_rt, tolerant ? " [tolerant]" : "");

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
