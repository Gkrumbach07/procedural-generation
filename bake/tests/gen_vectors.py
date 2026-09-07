"""Generate tests/data/cubesphere_vectors.bin (shared with gdextension/tests).

Format (little endian): magic b"CSV1", int32 K, then K records of
    int32 face, float64 u, float64 v, float64 x, float64 y, float64 z
where (x, y, z) = to_sphere(face, u, v).  (u, v) are drawn in [0, 1) with
extra samples within 1e-4 of edges and corners.
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globe.cubesphere import to_sphere_v  # noqa: E402


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
    p = to_sphere_v(face, u, v)
    with open(path, "wb") as fh:
        fh.write(b"CSV1")
        fh.write(struct.pack("<i", K))
        for k in range(K):
            fh.write(struct.pack("<iddddd", int(face[k]), u[k], v[k], p[k, 0], p[k, 1], p[k, 2]))
    print(f"wrote {K} records to {path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data" / "cubesphere_vectors.bin"
    main(out)
