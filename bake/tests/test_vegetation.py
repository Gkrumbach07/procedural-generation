"""Vegetation grown with the zoom levels' erosion (globe.erosion.vegetation)."""
import numpy as np

from globe.config import WorldParams
from globe.erosion import vegetation as veg
from globe.erosion.maps import step

from test_erosion import make_window


def _cap(p, n=64, cell_m=5.0, **kw):
    """Capacity of a flat, warm, forested, gently sloping patch with ``kw``
    overriding its fields."""
    f = {"forest": np.full((n, n), 0.85, np.float32), "temp_c": np.full((n, n), 12.0, np.float32),
         "surface_m": np.add.outer(np.arange(n) * 0.2 * cell_m, np.zeros(n)), "pool_m": np.zeros((n, n)),
         "area_km2": np.full((n, n), 0.05), "active": np.ones((n, n), bool)}
    f.update(kw)
    return veg.capacity(p, f["forest"], f["temp_c"], f["surface_m"], f["pool_m"], f["area_km2"], cell_m, f["active"])


def test_capacity_keeps_trees_off_channels_pools_cliffs_and_the_tree_line():
    p = veg.VegParams()
    n = 64
    base = _cap(p)
    assert base.min() > 0.4, base.min()
    # a stream: its channel fills a 5 m cell, only a sliver of a 76 m one
    stream = np.full((n, n), 0.05)
    stream[:, 30:33] = 20.0
    assert _cap(p, area_km2=stream)[5:-5, 30:33].max() == 0.0
    assert _cap(p, cell_m=76.0, area_km2=stream)[5:-5, 31].min() > 0.2
    # standing water
    pool = np.zeros((n, n))
    pool[20:40, 20:40] = 2.0
    assert _cap(p, pool_m=pool)[22:38, 22:38].max() == 0.0
    # past the tree line
    assert _cap(p, temp_c=np.full((n, n), -8.0, np.float32)).max() == 0.0
    # a cliff: 60 degrees
    steep = np.add.outer(np.arange(n) * 5.0 * np.tan(np.radians(60.0)), np.zeros(n))
    assert _cap(p, surface_m=steep)[2:-2].max() == 0.0
    # dry ridges carry less than the ground along the streams
    dry = _cap(p, area_km2=np.full((n, n), 0.0005)).mean()
    wet = _cap(p, area_km2=np.full((n, n), 0.02)).mean()
    assert dry < 0.9 * wet, (dry, wet)


def test_cover_spreads_to_capacity_and_dies_where_there_is_none():
    p = veg.VegParams()
    rng = np.random.default_rng(0)
    n = 96
    cap = np.full((n, n), 0.8, np.float32)
    cap[:, 60:] = 0.0                       # a lake on the right
    cover = np.zeros((n, n), np.float32)
    cover[:, 60:] = 0.9                     # full of drowned trees at the start
    for _ in range(200):
        cover = veg.grow(p, cover, cap, rng)
    left = cover[:, :56]
    assert 0.55 < left.mean() <= 0.8, left.mean()
    assert cover[:, 60:].max() < 1e-3
    assert (cover >= 0.0).all() and (cover <= 1.0).all()


def _banded_roots_run(hold: float | None, N: int = 96, iters: int = 60, period: int = 16):
    p = WorldParams.small_world(0)
    st = make_window(N, "tilt", p, iters=iters)
    H = p.world.halo
    jj = np.arange(st.height.shape[2])
    rooted = np.sin(2 * np.pi * (jj - H) / period) > 0
    if hold is not None:
        st.roots = np.ascontiguousarray(np.broadcast_to((hold * rooted)[None, None, :], st.height.shape).astype(np.float32))
    for i in range(iters):
        step(st, p, (i,))
    surf = st.surface()[st.interior][0]
    r = np.broadcast_to(rooted[H:H + N][None, :], surf.shape)
    return float(surf[r].mean()), float(surf[~r].mean())


def test_roots_hold_the_ground():
    """McDonald's ``depositionRate * (1 - treedensity)``: bands under roots wear
    down less than the bare bands beside them, and no roots at all is the
    erosion without vegetation."""
    none = _banded_roots_run(None)
    assert _banded_roots_run(0.0) == none
    rooted_0, bare_0 = none                  # the bands differ a little on their own
    rooted, bare = _banded_roots_run(0.9)
    assert rooted - bare > 0.1 + (rooted_0 - bare_0), (rooted, bare, rooted_0, bare_0)
