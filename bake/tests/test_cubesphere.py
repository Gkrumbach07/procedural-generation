"""Phase 1 tests (PLAN.md section 5)."""
import struct
from pathlib import Path

import numpy as np
import pytest

from globe import cubesphere as cs
from globe.cubesphere import Grid, from_sphere, from_sphere_v, jacobian, to_sphere, to_sphere_v, transfer_velocity
from globe.field import FaceField

DATA = Path(__file__).resolve().parent / "data" / "cubesphere_vectors.bin"


def _random_unit(rng, n):
    p = rng.normal(size=(n, 3))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_bases_orthonormal():
    for f in range(6):
        r, u, n = cs.BASES[f]
        assert abs(np.dot(r, u)) < 1e-12 and abs(np.dot(r, n)) < 1e-12 and abs(np.dot(u, n)) < 1e-12
        assert np.allclose([np.linalg.norm(r), np.linalg.norm(u), np.linalg.norm(n)], 1.0)
        # face centre maps to the normal
        assert np.allclose(to_sphere_v([f], [0.5], [0.5])[0], n)


def test_roundtrip_random_1M():
    rng = np.random.default_rng(1)
    p = _random_unit(rng, 1_000_000)
    f, u, v = from_sphere_v(p)
    assert u.min() >= 0.0 and u.max() <= 1.0 and v.min() >= 0.0 and v.max() <= 1.0
    q = to_sphere_v(f, u, v)
    assert np.abs(p - q).max() < 1e-6
    # and uv -> sphere -> uv
    f2, u2, v2 = from_sphere_v(q)
    assert np.array_equal(f, f2)
    assert np.abs(u - u2).max() < 1e-9 and np.abs(v - v2).max() < 1e-9


def test_roundtrip_edges_and_corners():
    rng = np.random.default_rng(2)
    K = 200_000
    face = rng.integers(0, 6, K)
    u = rng.random(K)
    v = rng.random(K)
    eps = rng.random(K) * 1e-4
    u[: K // 2] = np.where(rng.random(K // 2) < 0.5, eps[: K // 2], 1 - eps[: K // 2])
    v[K // 4 :] = np.where(rng.random(K - K // 4) < 0.5, eps[K // 4 :], 1 - eps[K // 4 :])
    p = to_sphere_v(face, u, v)
    f2, u2, v2 = from_sphere_v(p)
    q = to_sphere_v(f2, u2, v2)
    assert np.abs(p - q).max() < 1e-6
    # the exact corners
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                pc = np.array([x, y, z]) / np.sqrt(3)
                f, uu, vv = from_sphere(*pc)
                assert np.allclose(to_sphere(f, uu, vv), pc, atol=1e-12)
                assert min(uu, 1 - uu) < 1e-12 and min(vv, 1 - vv) < 1e-12


def test_scalar_and_vector_versions_agree():
    rng = np.random.default_rng(3)
    p = _random_unit(rng, 5000)
    f, u, v = from_sphere_v(p)
    for k in range(0, 5000, 50):
        fs, us, vs = from_sphere(p[k, 0], p[k, 1], p[k, 2])
        assert fs == f[k] and abs(us - u[k]) < 1e-15 and abs(vs - v[k]) < 1e-15
        assert np.allclose(to_sphere(fs, us, vs), to_sphere_v([fs], [us], [vs])[0], atol=1e-15)


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(4)
    for _ in range(200):
        f = int(rng.integers(0, 6))
        u, v = rng.random(2) * 1.2 - 0.1  # include a little halo
        h = 1e-6
        J = np.array(jacobian(f, u, v)).reshape(2, 3)
        fu = (np.array(to_sphere(f, u + h, v)) - np.array(to_sphere(f, u - h, v))) / (2 * h)
        fv = (np.array(to_sphere(f, u, v + h)) - np.array(to_sphere(f, u, v - h))) / (2 * h)
        assert np.abs(J[0] - fu).max() < 1e-6
        assert np.abs(J[1] - fv).max() < 1e-6
        p = np.array(to_sphere(f, u, v))
        assert abs(np.dot(J[0], p)) < 1e-9 and abs(np.dot(J[1], p)) < 1e-9


def test_transfer_preserves_3d_velocity():
    """The 3-D tangent velocity is continuous across a face transition."""
    rng = np.random.default_rng(5)
    n_cross = 0
    for _ in range(2000):
        f = int(rng.integers(0, 6))
        # start just outside a random edge
        u, v = rng.random(2)
        side = rng.integers(0, 4)
        d = rng.random() * 0.02 + 1e-6
        if side == 0:
            u = -d
        elif side == 1:
            u = 1 + d
        elif side == 2:
            v = -d
        else:
            v = 1 + d
        a, b = rng.normal(size=2)
        J = np.array(jacobian(f, u, v)).reshape(2, 3)
        v3 = a * J[0] + b * J[1]
        f2, u2, v2, a2, b2 = transfer_velocity(f, u, v, a, b)
        assert f2 != f
        n_cross += 1
        assert 0 <= u2 <= 1 and 0 <= v2 <= 1
        assert np.allclose(to_sphere(f, u, v), to_sphere(f2, u2, v2), atol=1e-12)
        J2 = np.array(jacobian(f2, u2, v2)).reshape(2, 3)
        w3 = a2 * J2[0] + b2 * J2[1]
        assert np.abs(v3 - w3).max() < 1e-9 * max(1.0, np.abs(v3).max())
    assert n_cross == 2000


def test_particle_straight_line_across_edge():
    """A particle moving in a straight line in face A's (extended)
    parametrisation, and the same particle after transfer moving in a
    straight line in face B, stay within one cell of each other for a
    halo's worth of cells beyond the edge (PLAN 2.6)."""
    N = 256
    H = 4
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(300):
        f = int(rng.integers(0, 6))
        u0 = 1.0 - rng.random() * 0.05
        v0 = 0.1 + 0.8 * rng.random()
        ang = rng.uniform(-1.2, 1.2)
        a, b = np.cos(ang) / N, np.sin(ang) / N  # one cell per unit time along u
        # advance to the edge in face f
        t_edge = (1.0 - u0) / a
        u, v = u0 + a * t_edge + 1e-9, v0 + b * t_edge
        f2, u2, v2, a2, b2 = transfer_velocity(f, u, v, a, b)
        assert f2 != f
        for t in np.linspace(0.5, H, 8):
            p_ref = np.array(to_sphere(f, u + a * t, v + b * t))  # extrapolated straight line in A
            p_new = np.array(to_sphere(f2, u2 + a2 * t, v2 + b2 * t))
            dist_cells = np.arccos(np.clip(np.dot(p_ref, p_new), -1, 1)) / (np.pi / 2 / N)
            worst = max(worst, dist_cells)
    assert worst < 0.5, worst


def test_cell_area_sums_to_sphere():
    for N in (16, 64):
        g = Grid(N, 4, 50.0)
        total = g.interior_cell_area.astype(np.float64).sum()
        assert abs(total / (4 * np.pi * g.R_planet**2) - 1.0) < 0.2 / N**2
        ratio = g.interior_cell_area.max() / g.interior_cell_area.min()
        assert 1.2 < ratio < 1.5  # EAC: ~1.4 at N->inf
    guu, guv, gvv = cs.metric_uv(4, 0.5, 0.5)
    assert abs(guu * (2 / np.pi) ** 2 - 1.0) < 1e-12 and abs(guv) < 1e-12


def _analytic_halo_error(N, fn):
    g = Grid(N, 4, 50.0)
    f = FaceField.from_function(g, fn, dtype=np.float64, name="t")
    ref = f.data.copy()
    f.data[:, g.halo_mask] = np.nan
    f.exchange_halos()
    err = np.abs(f.data - ref)
    assert not np.isnan(f.data).any()
    return float(err[:, g.edge_halo_mask].max()), float(err[:, g.corner_mask].max())


def test_halo_exchange_analytic():
    fn = lambda p: p[..., 0] * p[..., 1] + p[..., 2] ** 2
    e64, c64 = _analytic_halo_error(64, fn)
    e128, c128 = _analytic_halo_error(128, fn)
    # bilinear interpolation error: O(h^2) with h ~ 1/N
    assert e64 < 3e-4 and c64 < 1e-3
    assert e128 < e64 / 3 and c128 < c64 / 3


def test_halo_exchange_linear_exact():
    # bilinear and plane fits reproduce (near-)linear functions on the sphere
    g = Grid(32, 4, 50.0)
    f = FaceField.from_function(g, lambda p: 0.3 * p[..., 0] - 0.7 * p[..., 1] + 0.2 * p[..., 2], dtype=np.float64, name="lin")
    ref = f.data.copy()
    f.data[:, g.halo_mask] = 0
    f.exchange_halos()
    assert np.abs(f.data - ref)[:, g.halo_mask].max() < 2e-3


def test_laplacian_eigenfunction():
    g = Grid(64, 4, 50.0)
    R = g.R_planet
    for name, fn, l in (
        ("Y20", lambda p: 3 * p[..., 2] ** 2 - 1, 2),
        ("Y11", lambda p: p[..., 0], 1),
        ("Y33", lambda p: p[..., 0] ** 3 - 3 * p[..., 0] * p[..., 1] ** 2, 3),
    ):
        f = FaceField.from_function(g, fn, dtype=np.float64, name=name)
        f.exchange_halos()  # use the exchanged halo, as real fields do
        L = f.laplacian().interior
        mask = np.abs(f.interior) > 0.3 * np.abs(f.interior).max()
        ratio = L[mask] / f.interior[mask]
        expect = -l * (l + 1) / R**2
        assert np.abs(ratio / expect - 1).max() < 0.02, (name, ratio.min(), ratio.max(), expect)
        # no seam artefacts: laplacian residual near edges comparable to interior
        res = np.abs(L - expect * f.interior)
        H = g.H
        edge = res[:, [0, -1], :].max()
        inner = res[:, 8:-8, 8:-8].max()
        assert edge < 5 * inner + 1e-12


def test_gradient_of_z():
    g = Grid(64, 4, 50.0)
    f = FaceField.from_function(g, lambda p: p[..., 2], dtype=np.float64, name="z")
    f.exchange_halos()
    gr = f.gradient()
    n = gr.vec_norm().interior
    z = g.interior_centers[..., 2]
    expect = np.sqrt(1 - z**2) / g.R_planet
    assert np.abs(n - expect).max() / expect.max() < 1e-3
    # vector halo exchange re-expresses the analytic gradient correctly
    gr2 = gr.copy()
    gr2.data[:, g.halo_mask] = np.nan
    gr2.exchange_halos()
    inner = g.halo_mask.copy()
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    e = np.abs(gr2.data - gr.data)[:, inner]
    assert np.nanmax(e) / np.abs(gr.data[:, inner]).max() < 2e-3


def test_owner_table():
    g = Grid(32, 4, 50.0)
    H, N, NE = g.H, g.N, g.NE
    own = g.owner
    ident = np.arange(6 * NE * NE).reshape(6, NE, NE)
    assert np.array_equal(own[:, H : H + N, H : H + N], ident[:, H : H + N, H : H + N])
    of, oi, oj = g.unflat_index(own)
    assert (oi[:, g.halo_mask] >= H).all() and (oi[:, g.halo_mask] < H + N).all()
    assert (of[:, g.halo_mask] != np.arange(6)[:, None]).all()
    # owner of a halo cell is the geometrically nearest interior cell (edge blocks)
    ei, ej = np.nonzero(g.edge_halo_mask)
    for f in range(6):
        p = g.centers[f, ei, ej]
        q = g.centers.reshape(-1, 3)[own[f, ei, ej]]
        d = np.arccos(np.clip(np.sum(p * q, 1), -1, 1)) / (np.pi / 2 / N)
        assert d.max() < 0.75


def test_vector_file_matches():
    """The checked-in vector file (shared with the C++ test) matches."""
    with open(DATA, "rb") as fh:
        assert fh.read(4) == b"CSV1"
        (K,) = struct.unpack("<i", fh.read(4))
        rec = np.frombuffer(fh.read(), dtype=np.dtype([("face", "<i4"), ("u", "<f8"), ("v", "<f8"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8")]))
    assert rec.shape[0] == K
    p = to_sphere_v(rec["face"], rec["u"], rec["v"])
    assert np.abs(p - np.stack([rec["x"], rec["y"], rec["z"]], 1)).max() < 1e-12
    f, u, v = from_sphere_v(p)
    # from_sphere may pick the other face exactly on an edge; positions must agree
    q = to_sphere_v(f, u, v)
    assert np.abs(p - q).max() < 1e-12
