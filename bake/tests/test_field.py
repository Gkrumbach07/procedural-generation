import numpy as np

from globe.cubesphere import Grid
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
