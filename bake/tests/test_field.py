import numpy as np

from globe.cubesphere import Grid, face_of_v
from globe.field import FaceField, bilinear_ext


def test_from_interior_save_load_roundtrip(tmp_path):
    g = Grid(16, 4, 50.0)
    rng = np.random.default_rng(0)
    arr = rng.random((6, 16, 16)).astype(np.float32)
    f = FaceField.from_interior(g, arr, name="x")
    assert np.array_equal(f.interior, arr)
    f.save(tmp_path)
    f2 = FaceField.load(tmp_path, "x", g)
    assert np.array_equal(f2.data, f.data)
    vec = rng.random((6, 16, 16, 2)).astype(np.float32)
    v = FaceField.from_interior(g, vec, is_vector=True, name="wind")
    v.save(tmp_path)
    v2 = FaceField.load(tmp_path, "wind", g)
    assert v2.is_vector and np.array_equal(v2.data, v.data)


def test_integer_exchange_uses_nearest():
    g = Grid(16, 4, 50.0)
    fid = FaceField.zeros(g, 1, np.int32, name="fid")
    fid.data[...] = np.arange(6)[:, None, None]
    fid.exchange_halos()
    fo = face_of_v(g.centers)
    # edge blocks are gathered from the single neighbouring face that owns the
    # halo cell's centre; corner blocks use a KD-tree nearest across the three
    # faces meeting at a cube corner, so only the edge blocks are checked exactly
    assert np.array_equal(fid.data[:, g.edge_halo_mask], fo[:, g.edge_halo_mask])
    assert (fid.data[:, g.halo_mask] != np.arange(6)[:, None]).all()
    assert fid.data.dtype == np.int32


def test_sample_bilinear_matches_analytic():
    g = Grid(64, 4, 50.0)
    f = FaceField.from_function(g, lambda p: p[..., 0] + 0.5 * p[..., 1] * p[..., 2], dtype=np.float64, name="t")
    rng = np.random.default_rng(1)
    p = rng.normal(size=(20000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    s = f.sample_sphere(p)
    ref = p[:, 0] + 0.5 * p[:, 1] * p[:, 2]
    assert np.abs(s - ref).max() < 2e-3
    # kernel-side sampler agrees with the vectorised one
    fi, fj = g.uv_cell(0.371, 0.882)
    assert abs(bilinear_ext(f.data[2], float(fi), float(fj)) - f.sample_bilinear(2, 0.371, 0.882)) < 1e-12


def test_directional_derivative_consistency():
    g = Grid(32, 4, 50.0)
    f = FaceField.from_function(g, lambda p: p[..., 2], dtype=np.float64, name="z")
    f.exchange_halos()
    w = FaceField.zeros(g, 2, np.float32, is_vector=True, name="w")
    w.data[..., 0] = 0.7
    w.data[..., 1] = -0.2
    dd = f.directional_derivative(w).interior
    vd = f.gradient().vec_dot(w)[:, g.H : g.H + g.N, g.H : g.H + g.N]
    assert np.abs(dd - vd).max() < 1e-6 * np.abs(dd).max()


def test_sample_window_across_face_edge():
    from globe.cubesphere import to_sphere_v

    g = Grid(32, 4, 50.0)
    fn = lambda p: p[..., 0] + 0.5 * p[..., 1] * p[..., 2] + p[..., 2] ** 2
    f = FaceField.from_function(g, fn, dtype=np.float64, name="t")
    f.exchange_halos()
    # window on face 4 extending 12 coarse cells past the +u edge and 6 past -v
    R = 2
    i0, i1, j0, j1 = 20, 44, -6, 10
    for order in (1, 3):
        w = f.sample_window(4, i0, i1, j0, j1, R=R, order=order)
        assert w.shape == ((i1 - i0) * R, (j1 - j0) * R)
        ui = (np.arange(i0 * R, i1 * R) + 0.5) / (32 * R)
        vj = (np.arange(j0 * R, j1 * R) + 0.5) / (32 * R)
        U, V = np.meshgrid(ui, vj, indexing="ij")
        ref = fn(to_sphere_v(np.full(U.shape, 4), U, V))
        err = np.abs(w - ref).max()
        assert err < (3e-3 if order == 1 else 3e-4), (order, err)
    # vector field: gradient of z sampled past the edge must equal the analytic
    # gradient expressed in face 4's components
    z = FaceField.from_function(g, lambda p: p[..., 2], dtype=np.float64, name="z")
    z.exchange_halos()
    gz = z.gradient()
    w = gz.sample_window(4, 24, 35, 8, 24, R=1)  # 3 cells past the +u edge (inside the halo)
    ref = gz.data[4, g.H + 24 : g.H + 35, g.H + 8 : g.H + 24]  # face-4 components incl. its halo
    assert w.shape == ref.shape
    assert np.abs(w - ref).max() < 3e-3 * np.abs(ref).max()
    # integer field: nearest
    ids = FaceField.zeros(g, 1, np.int32, name="id")
    ids.data[...] = np.arange(6)[:, None, None]
    ids.exchange_halos()
    wi = ids.sample_window(4, 28, 36, 0, 4, R=1)
    assert wi.dtype == np.int32 and set(np.unique(wi)) <= {4, 0}
