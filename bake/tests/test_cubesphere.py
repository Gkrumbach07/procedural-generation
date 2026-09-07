"""Phase 1 tests (PLAN.md section 5)."""
import struct
from pathlib import Path

import numpy as np
import pytest

from globe import cubesphere as cs
from globe.cubesphere import BASES, Grid, from_sphere, from_sphere_v, jacobian, to_sphere, to_sphere_v, transfer_velocity
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


def test_exact_edge_points():
    """Points exactly on a cube edge/corner: from_sphere returns the closed
    interval [0, 1] (u or v == 1.0 exactly on the higher-priority face) and
    every consumer clamps with min(floor(u*N), N-1)."""
    N = 16
    g = Grid(N, 4, 50.0)
    fld = FaceField.from_function(g, lambda p: p[..., 2], dtype=np.float64, name="z")
    pts = []
    for f in range(6):
        r, up, n = BASES[f]
        for s in (-1.0, 1.0):
            for t in (0.0, 0.3, 0.5):
                for a, b in ((s, t), (t, s)):
                    p = n + a * r + b * up
                    pts.append(p / np.linalg.norm(p))
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                pts.append(np.array([x, y, z]) / np.sqrt(3))
    pts = np.array(pts)
    n_one = 0
    fv, uv, vv = from_sphere_v(pts)
    assert uv.min() >= 0.0 and uv.max() <= 1.0 and vv.min() >= 0.0 and vv.max() <= 1.0
    for k, p in enumerate(pts):
        f2, u2, v2 = from_sphere(*p)
        assert 0.0 <= u2 <= 1.0 and 0.0 <= v2 <= 1.0
        assert f2 == fv[k] and u2 == uv[k] and v2 == vv[k]
        assert np.abs(np.array(to_sphere(f2, u2, v2)) - p).max() < 1e-12
        if u2 == 1.0 or v2 == 1.0:
            n_one += 1
            # documented behaviour: the naive index is out of range, the clamped one is not
            assert max(int(np.floor(u2 * N)), int(np.floor(v2 * N))) == N
            assert min(int(np.floor(u2 * N)), N - 1) <= N - 1 and min(int(np.floor(v2 * N)), N - 1) <= N - 1
        # consumers accept the closed interval without an index error
        assert np.isfinite(fld.sample_bilinear(f2, u2, v2)) and np.isfinite(fld.sample_nearest(f2, u2, v2))
        f3, u3, v3, a3, b3 = transfer_velocity(f2, u2, v2, 0.01, -0.02)
        assert 0.0 <= u3 <= 1.0 and 0.0 <= v3 <= 1.0
    assert n_one > 0, "no point returned u or v == 1.0 exactly; the documented edge behaviour changed"


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


def test_particle_crossing_traces_great_circle():
    """PLAN 2.6 / 5: a particle stepping with constant parameter velocity,
    transferred with ``transfer_velocity`` after every step, crosses a face
    edge and stays on the analytic great circle through its start point
    and initial tangent velocity.  Constant-(a, b) motion is only *locally*
    geodesic in EAC (parameter lines are great circles only when
    axis-aligned), so the bound is a fraction of a cell over 2H steps."""
    N = 256
    H = 4
    cell = np.pi / 2 / N
    rng = np.random.default_rng(6)
    worst = 0.0
    n_trials = 0
    for _ in range(400):
        f = int(rng.integers(0, 6))
        side = int(rng.integers(0, 4))  # u=0, u=1, v=0, v=1
        d = rng.uniform(0.5, H - 1.5) / N  # distance from the edge
        along = 0.05 + 0.9 * rng.random()
        ang = rng.uniform(-1.2, 1.2)  # heading relative to the outward edge normal
        if side == 0:
            u, v, a, b = d, along, -np.cos(ang), np.sin(ang)
        elif side == 1:
            u, v, a, b = 1.0 - d, along, np.cos(ang), np.sin(ang)
        elif side == 2:
            u, v, a, b = along, d, np.sin(ang), -np.cos(ang)
        else:
            u, v, a, b = along, 1.0 - d, np.sin(ang), np.cos(ang)
        # scale (a, b) so the physical speed is one cell of arc per step
        J = np.array(jacobian(f, u, v)).reshape(2, 3)
        v3 = a * J[0] + b * J[1]
        scale = cell / np.linalg.norm(v3)
        a, b, v3 = a * scale, b * scale, v3 * scale
        p0 = np.array(to_sphere(f, u, v))
        sp = np.linalg.norm(v3)
        dirn = v3 / sp
        crossed = False
        for k in range(1, 2 * H + 1):
            u, v = u + a, v + b
            f2, u, v, a, b = transfer_velocity(f, u, v, a, b)
            crossed |= f2 != f
            f = f2
            p = np.array(to_sphere(f, u, v))
            p_gc = p0 * np.cos(sp * k) + dirn * np.sin(sp * k)
            dev = np.arccos(np.clip(np.dot(p, p_gc), -1, 1)) / cell
            worst = max(worst, dev)
        assert crossed, (f, side, ang)
        n_trials += 1
    assert n_trials == 400
    assert worst < 0.25, worst


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


def test_halo_exchange_linear_is_monotone():
    """``w_lin`` is a convex combination everywhere (edge blocks: bilinear;
    corner blocks: inverse-distance over the 4 nearest cells), so a field in
    [0, 1] stays in [0, 1] after ``exchange_halos(linear=True)``."""
    for N in (32, 128):
        g = Grid(N, 4, 50.0)
        hm = g.halo
        assert (hm.w_lin >= 0).all() and np.abs(hm.w_lin.sum(1) - 1).max() < 1e-5
        rng = np.random.default_rng(0)
        f = FaceField(g, np.zeros((6, g.NE, g.NE), np.float64), name="mask")
        f.interior[...] = rng.integers(0, 2, f.interior.shape)
        f.exchange_halos(linear=True)
        assert f.data[:, g.halo_mask].min() >= 0 and f.data[:, g.halo_mask].max() <= 1
        assert f.data[:, g.corner_mask].min() >= 0 and f.data[:, g.corner_mask].max() <= 1
    # ... and the corner weights are still O(h^2) accurate, not nearest-only
    fn = lambda p: p[..., 0] * p[..., 1] + p[..., 2] ** 2
    for N, bound in ((32, 2e-3), (64, 5e-4)):
        g = Grid(N, 4, 50.0)
        f = FaceField.from_function(g, fn, dtype=np.float64, name="t")
        ref = f.data.copy()
        f.data[:, g.halo_mask] = np.nan
        f.exchange_halos(linear=True)
        assert np.abs(f.data - ref)[:, g.corner_mask].max() < bound, N


def test_halo_exchange_linear_exact():
    # bilinear and plane fits reproduce (near-)linear functions on the sphere
    g = Grid(32, 4, 50.0)
    f = FaceField.from_function(g, lambda p: 0.3 * p[..., 0] - 0.7 * p[..., 1] + 0.2 * p[..., 2], dtype=np.float64, name="lin")
    ref = f.data.copy()
    f.data[:, g.halo_mask] = 0
    f.exchange_halos()
    assert np.abs(f.data - ref)[:, g.halo_mask].max() < 2e-3


def _rotation_field(g, omega):
    """Contravariant cell components of w = omega x p (a smooth tangent
    field on the sphere), derived independently of HaloMap.rot."""
    w3 = np.cross(np.asarray(omega, float), g.centers)
    U, V = g._uv_grid
    out = np.empty((6, g.NE, g.NE, 2), np.float64)
    for f in range(6):
        ju, jv = cs.jacobian_v(np.full(U.shape, f), U, V)
        guu, guv, gvv = (ju * ju).sum(-1), (ju * jv).sum(-1), (jv * jv).sum(-1)
        ru, rv = (ju * w3[f]).sum(-1), (jv * w3[f]).sum(-1)
        det = guu * gvv - guv * guv
        out[f, ..., 0] = (gvv * ru - guv * rv) / det * g.N  # per-u -> per-cell
        out[f, ..., 1] = (guu * rv - guv * ru) / det * g.N
    return out


def test_vector_halo_exchange_analytic():
    """Vector halo exchange reproduces a smooth tangent field's cell
    components in the *destination* face basis (checks HaloMap.rot for
    both edge and corner blocks)."""
    for N in (32, 64):
        g = Grid(N, 4, 50.0)
        for omega in ((0, 0, 1), (1, 0, 0), (0.3, -0.5, 0.8)):
            ref = _rotation_field(g, omega)
            f = FaceField(g, ref.copy(), is_vector=True, name="w")
            f.data[:, g.halo_mask] = np.nan
            f.exchange_halos()
            assert not np.isnan(f.data).any()
            err = np.abs(f.data - ref).max(-1) / np.abs(ref).max()
            e_edge = err[:, g.edge_halo_mask].max()
            e_corner = err[:, g.corner_mask].max()
            assert e_edge < 1e-5, (N, omega, e_edge)
            assert e_corner < 1e-4, (N, omega, e_corner)


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
        # no seam artefacts: laplacian residual on every face edge (both the
        # i- and j-edges; 4 of the 12 cube edges are only v-edges) comparable
        # to the interior
        res = np.abs(L - expect * f.interior)
        inner = res[:, 8:-8, 8:-8].max()
        edge = max(res[:, [0, -1], :].max(), res[:, :, [0, -1]].max())
        assert edge < 5 * inner + 1e-12, (name, edge / inner)


def test_gradient_of_z():
    g = Grid(64, 4, 50.0)
    f = FaceField.from_function(g, lambda p: p[..., 2], dtype=np.float64, name="z")
    f.exchange_halos()
    gr = f.gradient()
    n = gr.vec_norm().interior
    z = g.interior_centers[..., 2]
    expect = np.sqrt(1 - z**2) / g.R_planet
    assert np.abs(n - expect).max() / expect.max() < 1e-3, np.abs(n - expect).max() / expect.max()
    # vector halo exchange re-expresses the analytic gradient correctly
    gr2 = gr.copy()
    gr2.data[:, g.halo_mask] = np.nan
    gr2.exchange_halos()
    inner = g.halo_mask.copy()
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    e = np.abs(gr2.data - gr.data)[:, inner]
    rel = np.nanmax(e) / np.abs(gr.data[:, inner]).max()
    assert rel < 2e-3, rel


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


REC = np.dtype([("face", "<i4"), ("u", "<f8"), ("v", "<f8"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("face2", "<i4"), ("u2", "<f8"), ("v2", "<f8")])


def test_vector_file_matches():
    """PLAN section 5: the checked-in vector file (shared with the C++ test)
    is reproduced *bit-identically* by the scalar numba to_sphere/from_sphere
    (the implementations every kernel uses; the C++ header mirrors them).
    The vectorised NumPy versions may differ by an ulp."""
    assert REC.itemsize == 64
    with open(DATA, "rb") as fh:
        assert fh.read(4) == b"CSV2"
        (K,) = struct.unpack("<i", fh.read(4))
        rec = np.frombuffer(fh.read(), dtype=REC)
    assert rec.shape[0] == K
    ref = np.stack([rec["x"], rec["y"], rec["z"]], 1)
    # scalar path: bit-exact, forward and inverse checked independently
    ps = np.array([to_sphere(int(f), float(u), float(v)) for f, u, v in zip(rec["face"], rec["u"], rec["v"])])
    assert np.array_equal(ps, ref)
    for r in rec:
        f2, u2, v2 = from_sphere(float(r["x"]), float(r["y"]), float(r["z"]))
        assert f2 == r["face2"] and u2 == r["u2"] and v2 == r["v2"], r
    # vectorised path: within 2 ulp
    eps = np.finfo(np.float64).eps
    assert np.abs(to_sphere_v(rec["face"], rec["u"], rec["v"]) - ref).max() <= 2 * eps
    f, u, v = from_sphere_v(ref)
    assert np.array_equal(f, rec["face2"])
    assert np.abs(u - rec["u2"]).max() <= 2 * eps and np.abs(v - rec["v2"]).max() <= 2 * eps
