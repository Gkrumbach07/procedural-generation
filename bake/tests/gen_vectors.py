"""Generate tests/data/cubesphere_vectors.bin (shared with gdextension/tests).

Format (little endian): magic b"CSV2", int32 K, then K records packed as
``'<idddddidd'`` (64 bytes):
    int32 face, float64 u, v, x, y, z, int32 face2, float64 u2, v2
where (x, y, z) = to_sphere(face, u, v) and (face2, u2, v2) =
from_sphere(x, y, z), both computed with the *scalar* numba functions
that every kernel uses (the vectorised ``to_sphere_v`` differs by 1 ulp
on some records).  (u, v) are drawn in [0, 1) with extra samples within
1e-4 of edges and corners.

The checked-in file was generated on x86-64 Linux with glibc's libm
(``tan``/``atan``); PLAN section 5's bit-identical Python/C++ test relies
on that libm and on compiling the C++ side without FP contraction
(``-ffp-contract=off``).
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globe.cubesphere import from_sphere, to_sphere  # noqa: E402

MAGIC = b"CSV2"
RECORD = "<idddddidd"


def main(path: Path, K: int = 5000, seed: int = 12345):
    rng = np.random.default_rng(seed)
    face = rng.integers(0, 6, K)
    u = rng.random(K)
    v = rng.random(K)
    n_edge = K // 5
    idx = rng.choice(K, n_edge, replace=False)
    u[idx] = np.where(rng.random(n_edge) < 0.5, rng.random(n_edge) * 1e-4, 1 - rng.random(n_edge) * 1e-4)
    idx2 = rng.choice(K, n_edge, replace=False)
    v[idx2] = np.where(rng.random(n_edge) < 0.5, rng.random(n_edge) * 1e-4, 1 - rng.random(n_edge) * 1e-4)
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<i", K))
        for k in range(K):
            f = int(face[k])
            x, y, z = to_sphere(f, float(u[k]), float(v[k]))
            f2, u2, v2 = from_sphere(x, y, z)
            fh.write(struct.pack(RECORD, f, float(u[k]), float(v[k]), x, y, z, int(f2), u2, v2))
    print(f"wrote {K} records to {path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data" / "cubesphere_vectors.bin"
    main(out)
