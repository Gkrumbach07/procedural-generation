"""Zoom bake tests (globe/zoom, docs/zoom-windows.md).

World: test_refine's tiny preset with stub tectonics / climate / erosion and
the real hydro + watershed stages (the zoom reads the planet's ``flow_dir``
/ ``flow_acc`` for its inflow).  Checks: tile cores cover the product; the
flood-tree flux of a plane is its upstream rain; inflow never spawns on the
array border; a two-level zoom with 2 x 2 tiles per level writes every
product cell, stays finite, keeps water on or above the ground, holds each
level to its parent at the parent's cell, takes inflow from the planet and
from the level above, is deterministic, and lists itself for the globe
viewer with its pages.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from globe.zoom import bake as zb
from globe.zoom import index as zindex

from test_refine import _params, _world_to_watersheds


LEVELS = (zb.ZoomLevel(4, 12, 4, tile=24, margin=4, hold_every=2), zb.ZoomLevel(8, 6, 3, tile=24, margin=4, hold_every=0, hold_scale=1.0))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    params = _params()
    root = tmp_path_factory.mktemp("zoom") / "tiny"
    store = _world_to_watersheds(root, params)
    (root / "viewer").mkdir()
    return {"store": store, "params": params, "root": root}


def _land_spot(world):
    """A coarse cell well inside a face with land around it."""
    from globe.io.world_store import WorldStore

    st = WorldStore(world["root"])
    grid = world["params"].coarse_grid()
    h = st.load_field("height", grid).interior
    N = grid.N
    best = None
    for f in range(6):
        for i in range(10, N - 10):
            for j in range(10, N - 10):
                land = float((h[f, i - 4:i + 4, j - 4:j + 4] > 0).mean())
                if best is None or land > best[0]:
                    best = (land, f, i, j)
    return best[1:]


def test_tile_cores_cover_the_product():
    for n, tile in ((48, 24), (50, 24), (1024, 1024), (768, 512), (10, 64)):
        starts, c = zb.tile_starts(5, n, tile)
        cov = np.zeros(n + 10, int)
        for a in starts:
            cov[a:a + c] += 1
        assert c <= max(tile, 1) or len(starts) == 1
        assert cov[5:5 + n].min() >= 1 and cov[:5].max() == 0 and cov[5 + n:].max() == 0, (n, tile, starts, c)


def test_flux_is_the_rain_upstream():
    """``drainage``: every cell's flux is its own weight plus its donors', so
    what reaches the drains is all the rain, and a cell on a plane carries
    at least the column above it that cannot leave sideways."""
    n = 20
    i = np.arange(n, dtype=np.float64)
    surface = np.repeat(i[:, None], n, 1) + 0.001 * np.arange(n)[None, :]   # rises along i
    ocean = np.zeros((n, n), bool)
    w = np.random.default_rng(1).random((n, n))
    recv, flux = zb.drainage(surface, ocean, w)
    r = recv.reshape(n, n)
    assert np.all(r[0, :] == -1) and np.all(r[:, 0] == -1)
    into_drain = np.zeros(n * n, bool)
    ok = recv >= 0
    into_drain[ok] = recv[ok] < 0          # never: a receiver index is a cell
    rec = recv[ok]
    drains = np.flatnonzero(recv < 0)
    feeds = np.isin(rec, drains)
    assert flux.ravel()[np.flatnonzero(ok)[feeds]].sum() == pytest.approx(w[1:-1, 1:-1].sum(), rel=1e-9)
    assert np.all(flux >= w - 1e-12)


def test_inflow_spawns_off_the_border():
    s = np.random.default_rng(0).random((16, 16))
    w = zb.spawn_at(np.array([0.0, 15.0, 7.2]), np.array([3.0, 15.0, 7.7]), np.array([1.0, 2.0, 3.0]), s, 2)
    assert w.sum() == pytest.approx(6.0)
    assert w[0, :].sum() == 0 and w[-1, :].sum() == 0 and w[:, 0].sum() == 0 and w[:, -1].sum() == 0


@pytest.fixture(scope="module")
def zoom(world, tmp_path_factory):
    spot = _land_spot(world)
    out = world["root"] / "zoom" / "t"
    zb.run_zoom(world["root"], spot, LEVELS, name="t", resume=False)
    return {"spot": spot, "out": out, "levels": [zb.load_level(out, lv.R) for lv in LEVELS]}


def test_zoom_levels_write_every_product_cell_and_stay_physical(world, zoom):
    for lv, res in zip(LEVELS, zoom["levels"]):
        a = res.arrays
        sl = res.geo.product()
        land = ~a["ocean"][sl]
        assert a["done"][sl][land].all(), lv
        for k in ("height", "sediment", "discharge", "water_surface", "flux"):
            assert np.isfinite(a[k]).all(), (lv, k)
        surf = a["height"] + a["sediment"]
        assert np.all(a["water_surface"][sl][land] >= surf[sl][land] - 1e-3)
        assert np.all(a["sediment"] >= -1e-4)
        assert len(res.stats["tiles"]) == 4, res.stats["tiles"]


def test_zoom_level_is_held_to_its_parent(world, zoom):
    """The drift correction: a few parent cells out (blocks of 4 parent
    cells) the refined surface keeps the parent's -- the level's
    ``held_p90_m`` [before, after the correction] falls, and after it the
    block means are small against the change the level made.  (The
    Gaussian holds from about two parent cells up; an alternation at the
    parent's own cell is detail it leaves, by design.)"""
    for k, (lv, res) in enumerate(zip(LEVELS, zoom["levels"])):
        a = res.arrays
        f = lv.R // (LEVELS[k - 1].R if k else 1)
        before, after = res.stats["held_p90_m"]
        spread = res.stats["change_std_m"]
        assert after <= before + 1e-6 and after < 0.3 * spread, (lv, before, after, spread)
        cells = a["done"] & ~a["ocean"]
        assert zb.held_p90(a["height"] + a["sediment"] - a["plain"], cells, 4 * f, res.geo.product()) == pytest.approx(after, abs=0.01)


def test_zoom_takes_inflow_from_the_planet_and_the_level_above(zoom):
    for res in zoom["levels"]:
        assert res.stats["inflow_total"] > 0.0, res.stats
        assert any(t.get("inflow", 0.0) > 0.0 for t in res.stats["tiles"]), res.stats["tiles"]


def test_zoom_is_deterministic(world, zoom, tmp_path):
    out2 = tmp_path / "again"
    zb.run_zoom(world["root"], zoom["spot"], LEVELS[:1], out=out2, name="again", resume=False)
    a = zb.load_level(out2, LEVELS[0].R).arrays
    b = zoom["levels"][0].arrays
    for k in ("height", "sediment", "discharge"):
        assert np.array_equal(a[k], b[k]), k


def test_zoom_pages_and_viewer_list(world, zoom):
    out = zoom["out"]
    for lv in LEVELS:
        assert (out / f"L{lv.R}.html").stat().st_size > 1000
    html = (out / "view.html").read_text()
    assert "../../viewer/index.html#lat=" in html
    zooms = zindex.write(world["root"])
    assert [z["name"] for z in zooms] == ["t"]
    z = zooms[0]
    assert z["href"] == "../zoom/t/view.html" and len(z["levels"]) == 2
    for lv in z["levels"]:
        assert len(lv["corners"]) == 4 and all(-90 <= c[0] <= 90 and -180 <= c[1] <= 180 for c in lv["corners"])
    js = (world["root"] / "viewer" / "zooms.js").read_text()
    assert js.startswith("GLOBE_VIEWER.setZooms(") and json.loads(js[len("GLOBE_VIEWER.setZooms("):-3])[0]["name"] == "t"
    info = json.loads((out / "zoom.json").read_text())
    assert info["spot"] == list(zoom["spot"]) and -90 <= info["lat"] <= 90
