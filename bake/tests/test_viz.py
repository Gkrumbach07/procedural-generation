import numpy as np

from globe.cubesphere import Grid
from globe.field import FaceField
from globe.viz import quicklook as ql
from globe.viz.unfold import SLOTS, unfold_field_interior


def test_net_edges_are_continuous():
    g = Grid(32, 4)
    n = 32
    net = unfold_field_interior(g.interior_centers)

    def block(f):
        r, c = SLOTS[f]
        return net[r * n : (r + 1) * n, c * n : (c + 1) * n]

    cell = np.pi / 2 / n
    for a, b, side in [(4, 0, "right"), (4, 1, "left"), (4, 2, "up"), (4, 3, "down"), (0, 5, "right")]:
        A, B = block(a), block(b)
        if side == "right":
            d = np.linalg.norm(A[:, -1] - B[:, 0], axis=-1)
        elif side == "left":
            d = np.linalg.norm(A[:, 0] - B[:, -1], axis=-1)
        elif side == "up":
            d = np.linalg.norm(A[0, :] - B[-1, :], axis=-1)
        else:
            d = np.linalg.norm(A[-1, :] - B[0, :], axis=-1)
        assert d.max() < 1.05 * cell


def seam_discontinuity(field: FaceField) -> float:
    """Max |gradient jump| across face edges relative to the interior
    gradient scale (PLAN 15 seam test helper).  Vector fields are compared
    via their physical norm (cell components are basis-dependent)."""
    if field.is_vector:
        field = field.vec_norm()
    f = field.copy()
    f.exchange_halos()
    d = f.data.astype(np.float64)
    H, N = f.H, f.N
    gi = d[:, 2:, :] - d[:, :-2, :]
    gj = d[:, :, 2:] - d[:, :, :-2]
    inner = max(np.abs(gi[:, H : H + N - 2, H : H + N]).mean(), np.abs(gj[:, H : H + N, H : H + N - 2]).mean(), 1e-12)
    # centred differences straddling the four edges of every face
    edges = [
        np.abs(gi[:, H - 1, H : H + N]),
        np.abs(gi[:, H + N - 1, H : H + N]),
        np.abs(gj[:, H : H + N, H - 1]),
        np.abs(gj[:, H : H + N, H + N - 1]),
    ]
    return float(max(e.mean() for e in edges) / inner)


def test_seam_metric_on_smooth_function():
    g = Grid(64, 4)
    f = FaceField.from_function(g, lambda p: np.sin(3 * p[..., 0]) * p[..., 1] + p[..., 2] ** 2, dtype=np.float32, name="s")
    assert seam_discontinuity(f) < 3.0


def test_seam_metric_detects_seams():
    """Negative tests: the PLAN 15 metric rises above 3 on real seams."""
    from globe.field import rotation_field
    from globe.stubs import fbm_noise

    g = Grid(64, 4)
    fn = lambda p: np.sin(3 * p[..., 0]) * p[..., 1] + p[..., 2] ** 2 + 0.3 * p[..., 0] * p[..., 2] ** 3
    f = FaceField.from_function(g, fn, dtype=np.float32, name="s")
    assert seam_discontinuity(f) < 1.5
    s = f.copy()
    s.data += (np.arange(6)[:, None, None] * 0.05 * np.ptp(f.interior)).astype(np.float32)
    assert seam_discontinuity(s) > 3.0  # per-face offset
    m = f.copy()
    m.data[0] = m.data[0].T.copy()
    assert seam_discontinuity(m) > 3.0  # one face mis-oriented
    n = f.copy()
    for k in range(6):
        n.data[k] = fbm_noise(g, np.random.default_rng(k), 5)[k]
    assert seam_discontinuity(n) > 3.0  # per-face noise, forgot the sphere
    c = FaceField.zeros(g, 2, np.float32, is_vector=True, name="c")
    c.data[..., 0] = 1.0
    assert seam_discontinuity(c) > 3.0  # constant cell components are not a continuous tangent field
    assert seam_discontinuity(rotation_field(g, (0.3, -0.5, 0.8), "w")) < 3.0  # a rigid rotation is


def test_quicklook_writes(tmp_path):
    g = Grid(32, 4)
    h = FaceField.from_function(g, lambda p: 1000 * p[..., 2] * p[..., 0], dtype=np.float32, name="h")
    q = FaceField.from_function(g, lambda p: np.exp(4 * p[..., 1]), dtype=np.float32, name="q")
    out = ql.quicklook_height(tmp_path / "h.png", h, 0.0, g.cell_size_m, discharge=q, water=h)
    assert out.exists()
    img = ql.render_labels(FaceField.full(g, 3, dtype=np.int32))
    assert img.shape == (6, 32, 32, 3)
    ql.save_image(tmp_path / "v.png", ql.render_vector(h.gradient(), base=ql.render_scalar(h)))
