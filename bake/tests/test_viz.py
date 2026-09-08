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
    """Max |gradient jump| across face edges relative to the gradient scale
    *next to that edge* (PLAN 15 seam test helper).  Vector fields are
    compared via their physical norm (cell components are basis-dependent).

    Each edge line (centred differences straddling the edge) is divided by
    the mean of the two parallel lines just inside the face, not by the
    whole-face mean: a seam is a *jump* confined to the edge line, while a
    field that is simply concentrated near an edge (a trunk river along a
    face boundary, a coastal band) ramps up over several lines and is not a
    seam.  Normalising against the face mean flags such ramps — e.g. the
    tiny preset's ``momentum`` (edge 3.18 x the face mean, but 2.91 / 2.43
    on the two lines inside it) and the small preset's ``water_surface``
    (3.03) — while the local form scores them 1.2 and 1.1 and still scores
    every real seam below (per-face offset, mis-oriented face, per-face
    noise, constant cell components) above 4.9."""
    if field.is_vector:
        field = field.vec_norm()
    f = field.copy()
    f.exchange_halos()
    d = f.data.astype(np.float64)
    H, N = f.H, f.N
    gi = np.abs(d[:, 2:, :] - d[:, :-2, :])
    gj = np.abs(d[:, :, 2:] - d[:, :, :-2])
    # [edge line, first line inside, second line inside] for the four edges
    edges = [
        [gi[:, H - 1 + k, H : H + N] for k in range(3)],
        [gi[:, H + N - 1 - k, H : H + N] for k in range(3)],
        [gj[:, H : H + N, H - 1 + k] for k in range(3)],
        [gj[:, H : H + N, H + N - 1 - k] for k in range(3)],
    ]
    return float(max(e[0].mean() / max(0.5 * (e[1].mean() + e[2].mean()), 1e-12) for e in edges))


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


def test_seam_metric_ignores_features_that_hug_an_edge():
    """False-positive test: a field that is smooth on the sphere but has a
    narrow, steep ridge lying *along* a cube edge (|x| = |z|) is continuous
    there — as a trunk river or a coastal band along a face boundary is.
    Normalised against the face-wide gradient mean such a field scores 8.7;
    against the lines next to the edge it scores 1.4."""
    g = Grid(64, 4)
    f = FaceField.from_function(g, lambda p: np.exp(-((p[..., 0] ** 2 - p[..., 2] ** 2) ** 2) / 2e-3), dtype=np.float32, name="ridge")
    assert f.interior.max() > 0.9 and float(np.median(f.interior)) < 0.1  # it really is a narrow ridge
    assert seam_discontinuity(f) < 3.0
    # and it is still caught once a jump is added on top of the ridge
    s = f.copy()
    s.data += (np.arange(6)[:, None, None] * 0.2).astype(np.float32)
    assert seam_discontinuity(s) > 3.0


def test_quicklook_writes(tmp_path):
    g = Grid(32, 4)
    h = FaceField.from_function(g, lambda p: 1000 * p[..., 2] * p[..., 0], dtype=np.float32, name="h")
    q = FaceField.from_function(g, lambda p: np.exp(4 * p[..., 1]), dtype=np.float32, name="q")
    out = ql.quicklook_height(tmp_path / "h.png", h, 0.0, g.cell_size_m, discharge=q, water=h)
    assert out.exists()
    img = ql.render_labels(FaceField.full(g, 3, dtype=np.int32))
    assert img.shape == (6, 32, 32, 3)
    ql.save_image(tmp_path / "v.png", ql.render_vector(h.gradient(), base=ql.render_scalar(h)))


def test_terrain_scale_pins_the_palette_across_frames():
    """`render_height` derives its palette and vertical exaggeration from
    each array, so an unchanged region re-tints when the rest of the world
    changes — which makes an animation show motion that never happened.
    `terrain_scale` freezes all three so a series shares one scale."""
    g = Grid(32, 4)
    a = FaceField.from_function(g, lambda p: 800 * p[..., 2], dtype=np.float32, name="a")
    b = a.copy()
    b.data[0] += 4000.0  # one face grows a plateau; the other five are untouched
    keep = (slice(1, 6),)
    auto_a, auto_b = ql.render_height(a)[keep], ql.render_height(b)[keep]
    assert not np.array_equal(auto_a, auto_b)  # the untouched faces re-tint
    sc = ql.terrain_scale(a)
    fix_a, fix_b = ql.render_height(a, **sc)[keep], ql.render_height(b, **sc)[keep]
    assert np.array_equal(fix_a, fix_b)  # pinned: untouched faces render identically
    assert set(sc) == {"z_factor", "hmax", "hmin"}
