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
from scipy import ndimage

from globe.zoom import bake as zb
from globe.zoom import index as zindex

from test_refine import _params, _world_to_watersheds


LEVELS = (zb.ZoomLevel(4, 12, 4, tile=24, margin=4, hold_every=2, hold_scale=1.0), zb.ZoomLevel(8, 6, 3, tile=24, margin=4, hold_every=0, hold_scale=1.0))


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


def test_tile_passes_never_overlap_and_cover_every_tile():
    for n, tile, m in ((4096, 1024, 64), (768, 256, 64), (48, 24, 4), (100, 30, 20)):
        starts, c = zb.tile_starts(0, n, tile)
        windows = zb.tile_windows(starts, c, m)
        passes = zb.tile_passes(windows, c, m)
        assert sorted(w for ps in passes for w in ps) == sorted(windows)
        for ps in passes:
            for k, w1 in enumerate(ps):
                for w2 in ps[k + 1:]:
                    apart = (w1[2] + c + m <= w2[2] - m or w2[2] + c + m <= w1[2] - m or
                             w1[3] + c + m <= w2[3] - m or w2[3] + c + m <= w1[3] - m)
                    assert apart, (n, tile, m, w1, w2)


def test_inflow_spawns_off_the_border():
    s = np.random.default_rng(0).random((16, 16))
    w = zb.spawn_at(np.array([0.0, 15.0, 7.2]), np.array([3.0, 15.0, 7.7]), np.array([1.0, 2.0, 3.0]), s, 2)
    assert w.sum() == pytest.approx(6.0)
    assert w[0, :].sum() == 0 and w[-1, :].sum() == 0 and w[:, 0].sum() == 0 and w[:, -1].sum() == 0


@pytest.fixture(scope="module")
def zoom(world, tmp_path_factory):
    spot = _land_spot(world)
    out = world["root"] / "zoom" / "t"
    zb.run_zoom(world["root"], spot, LEVELS, name="t", resume=False, planet=False)
    return {"spot": spot, "out": out, "levels": [zb.load_level(out, lv.R) for lv in LEVELS]}


def test_zoom_bake_reports_its_progress(zoom, tmp_path, monkeypatch):
    """globe/zoom/progress.py: a finished bake leaves its plan with every level
    done (fraction 1); on a synthetic plan a running level counts the tiles'
    iterations over its expected work, the first level cut from nothing
    counts nothing, and a tile writes only when the environment names a
    directory."""
    from globe.zoom import progress

    rep = progress.read(zoom["out"])
    assert rep is not None and rep["fraction"] == 1.0 and rep["level"] is None
    assert [e["state"] for e in json.loads((zoom["out"] / "progress" / "plan.json").read_text())["levels"]] == ["done"] * len(LEVELS)

    out = tmp_path / "z"
    d = progress.write_plan(out, [{"R": 8, "work": 0.0, "tiles": 1, "state": "pending"},
                                  {"R": 32, "work": 100.0, "tiles": 2, "state": "pending"},
                                  {"R": 128, "work": 300.0, "tiles": 1, "state": "pending"}])
    progress.set_level(out, 8, "done")
    progress.set_level(out, 32, "running")
    monkeypatch.delenv(progress.ENV, raising=False)
    progress.tile_tick(32, 0, 0, 49, 100)
    assert not list(d.glob("L32_*.json"))
    monkeypatch.setenv(progress.ENV, str(d))
    progress.tile_tick(32, 0, 0, 49, 100)                  # tile 1 half way, tile 2 not started
    rep = progress.read(out)
    assert rep["level"] == 32 and rep["level_index"] == 1 and rep["levels"] == [8, 32, 128]
    assert rep["fraction"] == pytest.approx((1.0 + 100.0 * 0.25) / 401.0, abs=1e-3)
    assert "iteration 50/100" in rep["detail"] and "1/2 tiles" in rep["detail"]


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


def test_zoom_is_deterministic_and_the_same_in_parallel(world, zoom, tmp_path):
    """A rerun matches byte for byte, and so does one with the tiles of a
    pass in worker processes (each tile seeds its own particles)."""
    out2 = tmp_path / "again"
    zb.run_zoom(world["root"], zoom["spot"], LEVELS[:1], out=out2, name="again", resume=False, workers=2, planet=False)
    a = zb.load_level(out2, LEVELS[0].R).arrays
    b = zoom["levels"][0].arrays
    for k in ("height", "sediment", "discharge"):
        assert np.array_equal(a[k], b[k]), k


def test_detail_noise_is_drained_before_anything_erodes():
    """``refine.zoom.drain_noise``: the noise's closed depressions are filled to
    their spill, nothing else moves, and what is left drains to the sea or the
    array's border."""
    from globe.hydro.priority_flood import priority_flood_flat
    from globe.refine.zoom import drain_noise

    n = 64
    i = np.arange(n, dtype=np.float64)[:, None]
    surface = 300.0 - 4.0 * np.repeat(i, n, 1)              # falls towards i = n, sea beyond
    ocean = surface < 0.0
    rng = np.random.default_rng(4)
    noise = 30.0 * rng.normal(size=(n, n))
    noisy = surface + ndimage.gaussian_filter(noise, 1.2)
    add = drain_noise(noisy, ocean)
    filled = noisy + add
    assert (add >= 0.0).all() and add.max() > 1.0
    drain = ocean.copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    fr = priority_flood_flat(filled.astype(np.float32), drain, None)
    assert float((fr.filled.reshape(filled.shape) - filled).max()) < 1e-3   # no closed depression left
    assert float(drain_noise(surface, ocean).max()) == 0.0                  # a surface that already drains is untouched


def test_levels_below_the_coarse_grid_see_hardness_low_passed_and_capped(world):
    """``refine.hardness_smooth_cells`` / ``hardness_max``: the hardness every
    level below the coarse grid reads is tectonics' field low-passed without
    seams and capped -- a 2-cell zebra of strata loses most of its contrast --
    and the field itself when both are off; the planet's shared inputs are
    rewritten when the knobs change."""
    from globe.io.world_store import WorldStore
    from globe.refine import basin_job as bj
    from globe.zoom import planet as zp

    params = world["params"]
    grid = params.coarse_grid()
    h = WorldStore(world["root"]).load_field("hardness", grid)
    assert bj.refined_hardness(h, params) is h
    from globe.field import FaceField

    data = np.empty(np.shape(h.data), np.float64)
    data[:] = np.where(np.arange(data.shape[1]) % 2 == 0, 0.3, 0.95)[None, :, None]      # strata a cell wide
    zebra = FaceField(grid, data, name="hardness")
    zebra.exchange_halos()
    soft = params.with_overrides(refine={"hardness_smooth_cells": 1.5, "hardness_max": 0.85})
    out = bj.refined_hardness(zebra, soft)
    H = grid.H
    core = lambda f: np.asarray(f.data)[:, H:-H, H:-H]
    assert core(out).max() <= 0.85 + 1e-6
    assert np.ptp(core(out)[:, 8:-8, 8:-8]) < 0.1 * np.ptp(core(zebra)[:, 8:-8, 8:-8])
    assert zp._input_stamp(world["root"], params) != zp._input_stamp(world["root"], soft)


def test_pool_threads_follow_the_running_tiles_particles(monkeypatch):
    """A worker takes the share of the cores its tile's particles are of the
    running tiles' (one thread at least), and a tile's demand counts only
    while it runs."""
    import os

    import numba

    cpus = os.cpu_count() or 1
    before = numba.get_num_threads()
    demand = zb.pool_demand()
    monkeypatch.setattr(zb, "_DEMAND", demand)
    try:
        with zb._Demand(3.0), zb._Demand(1.0):
            assert demand[0] == pytest.approx(4.0)
            zb._share_threads(3.0)
            assert numba.get_num_threads() == max(1, min(round(cpus * 0.75), numba.config.NUMBA_NUM_THREADS))
            zb._share_threads(1e-6)
            assert numba.get_num_threads() == 1
        assert demand[0] == 0.0
        zb._share_threads(1.0)
        assert numba.get_num_threads() == min(cpus, numba.config.NUMBA_NUM_THREADS)
    finally:
        numba.set_num_threads(before)


def test_zoom_pages_and_viewer_list(world, zoom):
    out = zoom["out"]
    for lv in LEVELS:
        assert (out / f"L{lv.R}.html").stat().st_size > 1000
    # view.html opens the globe viewer's 3-D view over the zoom (its satellite ground, trees
    # and water), and links the self-contained meshes
    html = (out / "view.html").read_text()
    assert "../../viewer/index.html#v=tilt&lat=" in html and "layer=satellite" in html
    assert all(f'href="L{lv.R}.html"' in html for lv in LEVELS)
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


# --------------------------------------------------------------------------
# levels placed in fine cells (game-scale levels: Geometry.size / pad)
# --------------------------------------------------------------------------
#: a level of 1.5 coarse cells at R = 32 with a 12-cell guard, below LEVELS
FINE = zb.ZoomLevel(32, 2, 4, tile=24, margin=4, hold_every=2, hold_scale=1.0, size=48)


def test_fine_cell_geometry_and_level_specs():
    """A level with ``size`` is placed in fine cells: its product is ``size``
    fine cells centred on the spot's corner, its guard ``fine_pad(margin)``
    fine cells, and every work-array index is one face fine cell; a
    coarse-cell level's geometry, saved fields and resume key are what they
    were before fine cells existed; ``--levels`` specs parse both."""
    N = 1024
    lv = zb.ZoomLevel(2048, 1, 100, size=1024)
    g = zb.place(5, 212, 902, lv, N)
    pad = zb.fine_pad(lv.margin)
    assert g.fine and (g.size, g.pad, g.p0, g.n, g.NE) == (1024, pad, pad, 1024, 1024 + 2 * pad)
    assert g.product_origin == (212 * 2048 - 512, 902 * 2048 - 512)
    assert (g.ci0, g.cj0, g.cells, g.guard) == (211, 901, 1, 0)
    assert g.fine_origin == (g.fi0 - pad, g.fj0 - pad) and g.corner == ((g.fi0 - pad) / 2048, (g.fj0 - pad) / 2048)
    assert g.win.fine_ext() == (g.fi0 - pad, g.fi0 - pad + g.NE, g.fj0 - pad, g.fj0 - pad + g.NE) and g.win.NE == g.NE
    with pytest.raises(ValueError):
        g.origin
    assert zb.Geometry(**g.to_dict()) == g
    # the tiles' windows fit the work array (prepare_tile's bounds), and cover the product
    starts, c = zb.tile_starts(g.p0, g.n, 768)
    for a in starts:
        assert a - lv.margin >= 1 and a + c + lv.margin <= g.NE - 1
    assert starts[0] == g.p0 and starts[-1] + c == g.p0 + g.n
    # shifted onto the face near an edge, never within a coarse cell of it
    e = zb.place(0, 0, N - 1, lv, N)
    assert e.fine_origin[0] >= lv.R and e.fine_origin[1] + e.NE <= (N - 1) * lv.R

    # coarse-cell levels: unchanged
    for lvc in zb.DEFAULT_LEVELS:
        gc = zb.place(5, 212, 902, lvc, N)
        guard = max(1, -(-(lvc.margin + 2) // lvc.R))
        assert not gc.fine and gc.to_dict() == {"face": 5, "ci0": 212 - lvc.cells // 2, "cj0": 902 - lvc.cells // 2, "cells": lvc.cells, "guard": guard, "R": lvc.R}
        assert (gc.NE, gc.p0, gc.n) == ((lvc.cells + 2 * guard + 2) * lvc.R, (guard + 1) * lvc.R, lvc.cells * lvc.R)
        assert gc.corner == gc.origin and gc.fine_origin == (gc.origin[0] * lvc.R, gc.origin[1] * lvc.R)
        assert "size" not in zb.level_key(lvc)
    assert zb.level_key(lv)["size"] == 1024

    assert zb.parse_level("128:8:150") == zb.ZoomLevel(128, 8, 150)
    assert zb.parse_level("2048:0.5:100") == zb.ZoomLevel(2048, 1, 100, size=1024)
    assert zb.parse_level("512:1024f:200:512:32") == zb.ZoomLevel(512, 2, 200, 512, 32, size=1024)
    with pytest.raises(ValueError):
        zb.parse_level("512:0.3:10")
    assert zb.parse_level("2048:2048f:60:2048:64:inflow_cap=2") == zb.GAME_LEVELS[1]
    assert "inflow_cap" not in zb.level_key(zb.DEFAULT_LEVELS[0]) and zb.level_key(zb.GAME_LEVELS[1])["inflow_cap"] == 2.0

    # the game levels below the default ones: each product inside its parent's (face fine cells), and
    # every level's texture within the viewer's limit
    from globe.viz import zoomtex

    chain = [zb.place(5, 212, 902, lvc, N) for lvc in zb.DEFAULT_LEVELS + zb.GAME_LEVELS]
    for gp, gc in zip(chain, chain[1:]):
        k = gc.R // gp.R
        a, b = gp.product_origin
        assert a * k <= gc.product_origin[0] and gc.product_origin[0] + gc.n <= (a + gp.n) * k
        assert b * k <= gc.product_origin[1] and gc.product_origin[1] + gc.n <= (b + gp.n) * k
        zb._child_coords(gc, gp)                                  # inside the parent's work array
    for gc in chain:
        x0, side = zoomtex.crop_window(gc.NE, gc.p0, gc.n)
        assert 3 * side <= zoomtex.MAX_TEX and side == gc.n + 2 * zoomtex.CROP


def test_child_coordinates_follow_the_face_across_fine_and_coarse_levels():
    """``_child_coords``: a child's array cell and the parent coordinate it
    samples are the same point of the face, whether either is placed in
    coarse or in fine cells."""
    N = 256
    pairs = [
        (zb.ZoomLevel(32, 8, 1), zb.ZoomLevel(128, 2, 1)),                   # coarse -> coarse
        (zb.ZoomLevel(128, 2, 1), zb.ZoomLevel(512, 1, 1, size=384)),        # coarse -> fine, off the coarse grid
        (zb.ZoomLevel(512, 1, 1, size=384), zb.ZoomLevel(2048, 1, 1, size=1000)),   # fine -> fine
    ]
    for lp, lc in pairs:
        gp, gc = zb.place(3, 100, 57, lp, N), zb.place(3, 100, 57, lc, N)
        I, J = zb._child_coords(gc, gp)
        fo_c, fo_p = gc.fine_origin, gp.fine_origin
        for k in (0, gc.p0, gc.NE // 2, gc.NE - 1):
            u = (fo_c[0] + k + 0.5) / (N * gc.R)             # the child cell's centre, as a face coordinate
            v = (fo_c[1] + k + 0.5) / (N * gc.R)
            assert I[k, 0] == pytest.approx(u * N * gp.R - fo_p[0] - 0.5, abs=1e-9)
            assert J[0, k] == pytest.approx(v * N * gp.R - fo_p[1] - 0.5, abs=1e-9)
        # the child's product lies in the parent's
        pr = gc.product()[0]
        assert I[pr.start, 0] >= gp.p0 - 0.5 and I[pr.stop - 1, 0] <= gp.p0 + gp.n - 0.5


def test_inflow_cap_lets_the_rain_spawn_on_the_slopes():
    """``ZoomLevel.inflow_cap``: a tile whose inflow is 100x its rain spawns
    nearly every particle at the crossing and leaves its slopes unworn; capped
    at 2x the rain, the inflow keeps its crossing but the rain's particles
    reach the slopes (many more cells change away from the inflow's path),
    the particle budget is the same, and the stats keep the uncapped inflow."""
    from globe.config import WorldParams
    from globe.refine.zoom import zoom_params

    params = zoom_params(WorldParams.tiny_world(), 8)
    n = 42
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    height = (2.0 * i + 0.3 * np.sin(j * 0.9) + 0.2 * np.cos(i * 1.3 + j * 0.4)).astype(np.float64)   # falls towards i = 0
    inwin = np.zeros((n, n), bool)
    inwin[1:-1, 1:-1] = True
    precip = np.full((n, n), 0.01)
    src = np.zeros((n, n))
    src[n - 3, 21] = 100.0 * float(precip[inwin].sum())            # a river entering at the top edge
    metric = np.zeros((n, n, 3), np.float32)
    metric[..., 0] = metric[..., 2] = 1.0
    arrays = {"height": height, "sediment": np.zeros((n, n)), "discharge": np.zeros((n, n)), "momentum": np.zeros((n, n, 2)),
              "hardness": np.full((n, n), 0.5), "precip": precip, "evap": np.ones((n, n)), "metric": metric, "metric_inv": metric.copy(), "plain": height.copy()}
    job = {"a": 1, "b": 1, "c": n - 2, "sl": None, "land": inwin, "active": inwin, "inwin": inwin, "ocean": np.zeros((n, n), bool), "src": src, "arrays": arrays, "f": 1}
    out = {}
    for cap in (0.0, 2.0):
        lv = zb.ZoomLevel(8, 1, 2, tile=n - 2, margin=0, hold_every=0, inflow_cap=cap)
        out[cap] = zb.erode_tile(params, lv, 8, job, (1, 2, 3))
    rain = out[0.0]["stats"]["rain"]
    assert out[0.0]["stats"]["inflow"] == pytest.approx(100.0 * rain) and "inflow_uncapped" not in out[0.0]["stats"]
    assert out[2.0]["stats"]["inflow"] == pytest.approx(2.0 * rain) and out[2.0]["stats"]["inflow_uncapped"] == pytest.approx(100.0 * rain)
    assert out[2.0]["stats"]["particles_per_cell"] == out[0.0]["stats"]["particles_per_cell"]
    off_path = inwin & (np.abs(j - 21) > 6)
    moved = {cap: int((np.abs(o["height"] - height) > 1e-6)[off_path].sum()) for cap, o in out.items()}
    assert moved[2.0] > 3 * max(moved[0.0], 1), moved


def test_a_time_lapse_frame_keeps_the_flood_trees_streams():
    """A snapshot's ``discharge`` is the flood tree of the frame's own surface
    (:func:`zoom.bake.tile_water`), the field the finished level draws its
    rivers from -- not the particles' own discharge, which is an average over
    the iteration's passes and swelled and faded with them, so a time lapse of
    it pulsed instead of growing."""
    from globe.config import WorldParams
    from globe.refine.zoom import zoom_params

    params = zoom_params(WorldParams.tiny_world(), 8)
    n = 42
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    height = (2.0 * i + 0.3 * np.sin(j * 0.9) + 0.2 * np.cos(i * 1.3 + j * 0.4)).astype(np.float64)
    inwin = np.zeros((n, n), bool)
    inwin[1:-1, 1:-1] = True
    precip = np.full((n, n), 0.01)
    src = np.zeros((n, n))
    src[n - 3, 21] = 10.0 * float(precip[inwin].sum())
    metric = np.zeros((n, n, 3), np.float32)
    metric[..., 0] = metric[..., 2] = 1.0
    arrays = {"height": height, "sediment": np.zeros((n, n)), "discharge": np.zeros((n, n)), "momentum": np.zeros((n, n, 2)),
              "hardness": np.full((n, n), 0.5), "precip": precip, "evap": np.ones((n, n)), "metric": metric,
              "metric_inv": metric.copy(), "plain": height.copy()}
    ocean = np.zeros((n, n), bool)
    job = {"a": 1, "b": 1, "c": n - 2, "sl": None, "land": inwin, "active": inwin, "inwin": inwin, "ocean": ocean,
           "src": src, "arrays": arrays, "f": 1, "snapshots": 3}
    lv = zb.ZoomLevel(8, 1, 4, tile=n - 2, margin=0, hold_every=0)
    out = zb.erode_tile(params, lv, 8, job, (1, 2, 3))

    frames = out["frames"]
    assert [f["it"] for f in frames] == [1, 3, 4] and all(f["factor"] == 1 for f in frames)   # the first, then evenly to the last
    weight = np.where(inwin, precip + src, 0.0).astype(np.float32)          # the tile's own rain and its inflow
    for f in frames:
        acc, _pool = zb.tile_water(f["surface"], ocean | ~inwin, weight)
        assert f["discharge"] == pytest.approx(acc, rel=1e-5), f["it"]
    # the inflow river runs to the outlet in every frame: the pulse was a river that grew and
    # then faded out, while the flood tree carries what enters the tile all the way through it
    for f in frames:
        assert float(np.asarray(f["discharge"])[inwin].max()) > float(src.sum())


@pytest.fixture(scope="module")
def fine_zoom(world, zoom, tmp_path_factory):
    """``LEVELS`` and ``FINE`` below them, in a world-like directory of its own
    (so the zoom list of the other tests stays as it was): the coarse levels
    resume from the ``zoom`` fixture's files."""
    import shutil

    base = tmp_path_factory.mktemp("fine") / "w"
    out = base / "zoom" / "tf"
    out.mkdir(parents=True)
    for lv in LEVELS:
        for ext in ("npz", "json"):
            shutil.copy(zoom["out"] / f"L{lv.R}.{ext}", out / f"L{lv.R}.{ext}")
    logs = []
    zb.run_zoom(world["root"], zoom["spot"], LEVELS + (FINE,), out=out, name="tf", resume=True, planet=False, log=logs.append)
    shutil.copy(world["root"] / "manifest.json", base / "manifest.json")
    (base / "viewer").mkdir()
    return {"out": out, "base": base, "logs": logs, "levels": [zb.load_level(out, lv.R) for lv in LEVELS + (FINE,)]}


def test_fine_cell_level_chains_writes_its_product_and_is_held(world, zoom, fine_zoom):
    assert sum("resumed" in m for m in fine_zoom["logs"]) == len(LEVELS)
    parent, res = fine_zoom["levels"][-2], fine_zoom["levels"][-1]
    g = res.geo
    assert g.fine and (g.n, g.p0, g.NE) == (48, zb.fine_pad(FINE.margin), 48 + 2 * zb.fine_pad(FINE.margin))
    json_geo = json.loads((fine_zoom["out"] / "L32.json").read_text())["geometry"]
    assert zb.Geometry(**json_geo) == g
    a = res.arrays
    assert a["height"].shape == (g.NE, g.NE)
    sl = g.product()
    land = ~a["ocean"][sl]
    assert land.any() and a["done"][sl][land].all()
    outside = np.ones((g.NE, g.NE), bool)
    outside[g.p0 - FINE.margin:g.p0 + g.n + FINE.margin, g.p0 - FINE.margin:g.p0 + g.n + FINE.margin] = False
    assert not a["done"][outside].any()                       # nothing written beyond the tiles' windows
    for k in ("height", "sediment", "discharge", "water_surface", "flux"):
        assert np.isfinite(a[k]).all(), k
    surf = a["height"] + a["sediment"]
    assert np.all(a["water_surface"][sl][land] >= surf[sl][land] - 1e-3)
    assert len(res.stats["tiles"]) == 4 and res.stats["inflow_total"] > 0.0
    before, after = res.stats["held_p90_m"]
    assert after <= before + 1e-6
    # the level starts from its parent: the plain is the parent's surface at the same points of the face
    from scipy.ndimage import map_coordinates

    I, J = zb._child_coords(g, parent.geo)
    plain = map_coordinates(parent.surface().astype(np.float64), [I, J], order=3, mode="nearest")
    assert np.allclose(a["plain"], plain, atol=1e-3)
    # the level's product is inside its parent's, in face fine cells
    pf0, pn = parent.geo.product_origin[0] * (g.R // parent.geo.R), parent.geo.n * (g.R // parent.geo.R)
    assert pf0 <= g.product_origin[0] and g.product_origin[0] + g.n <= pf0 + pn


def test_fine_cell_level_texture_maps_its_core_onto_the_face(fine_zoom):
    """``index.write`` writes the fine level's texture cropped to its core and
    the guard it has (under ``CROP``); the record's ``ci0 R - p0`` is the face
    fine cell of pixel 0 -- the viewer's offset -- with ``ci0`` fractional,
    and the pixels are the work array's cells from ``x0``; the corners
    outline the product in fine cells."""
    from globe.viz import zoomtex

    zooms = zindex.write(fine_zoom["base"], log=None)
    z = zooms[0]
    assert [lv["R"] for lv in z["levels"]] == [lv.R for lv in LEVELS + (FINE,)]
    for lv, res in zip(z["levels"], fine_zoom["levels"]):
        tex, g = lv["tex"], res.geo
        m = min(zoomtex.CROP, g.p0)
        assert (tex["NE"], tex["p0"], tex["n"], tex["x0"], tex["array_NE"]) == (g.n + 2 * m, m, g.n, g.p0 - m, g.NE)
        assert tex["ci0"] * tex["R"] - tex["p0"] == g.fine_origin[0] + tex["x0"]
        assert tex["cj0"] * tex["R"] - tex["p0"] == g.fine_origin[1] + tex["x0"]
        assert (tex["fi0"], tex["fj0"]) == g.product_origin
        img = _decode_tex(fine_zoom["base"] / "viewer" / tex["file"], "tf", g.R)
        assert img.shape == (3 * tex["NE"], tex["NE"], 4)
        if tex["biome"]:                                   # the level's own biomes, where the world has a climate
            from globe.derive.biomes import N_BIOMES
            assert np.asarray(res.arrays["biome"]).max() < N_BIOMES
            bio = img[2 * tex["NE"]:, :, 0]
            assert np.array_equal(bio, np.asarray(res.arrays["biome"])[tex["x0"]:tex["x0"] + tex["NE"], tex["x0"]:tex["x0"] + tex["NE"]].T)
        surf = res.surface()
        step = (tex["h1"] - tex["h0"]) / 65535.0
        for x, y in ((tex["p0"], tex["p0"]), (tex["p0"] + g.n - 1, tex["p0"] + 3), (0, tex["NE"] - 1)):
            code = int(img[y, x, 0]) * 256 + int(img[y, x, 1])
            assert abs(tex["h0"] + code * step - float(surf[tex["x0"] + x, tex["x0"] + y])) <= step * 1.01
    g = fine_zoom["levels"][-1].geo
    assert isinstance(z["levels"][-1]["tex"]["ci0"], float) and z["levels"][-1]["tex"]["ci0"] == g.fi0 / g.R
    from globe.cubesphere import to_sphere_v

    N = world_N = json.loads((fine_zoom["base"] / "manifest.json").read_text())["N_c"]
    p = to_sphere_v(np.array([g.face]), np.array([g.fi0 / g.R / N]), np.array([g.fj0 / g.R / N]))[0]
    assert z["levels"][-1]["corners"][0] == pytest.approx([np.degrees(np.arcsin(p[2])), np.degrees(np.arctan2(p[1], p[0]))], abs=1e-4)
    assert world_N == N


# --------------------------------------------------------------------------
# the planet at a zoom level's resolution (globe/zoom/planet.py)
# --------------------------------------------------------------------------
def test_hashed_noise_agrees_where_windows_overlap():
    from globe.zoom import planet as zp

    a = zp.hashed_ridged(7, 2, -5, 10, 40, 30, 16.0)
    b = zp.hashed_ridged(7, 2, 15, 22, 40, 30, 16.0)
    assert np.allclose(a[20:, 12:], b[:20, :18])
    assert np.abs(a).max() <= 1.0 + 1e-6 and a.std() > 0.05
    c = zp.hashed_ridged(8, 2, -5, 10, 40, 30, 16.0)
    assert not np.allclose(a, c)


PLANET = None


def _planet_level():
    from globe.zoom import planet as zp

    return zp.PlanetLevel(R=4, iterations=3, tile=32, margin=8, hold_every=2, seam_cells=4)


@pytest.fixture(scope="module")
def planet(world):
    from globe.zoom import planet as zp

    out = zp.run_planet(world["root"], _planet_level(), workers=2)
    return {"out": out, "level": _planet_level()}


def test_planet_level_outputs_are_complete_and_physical(world, planet):
    from globe.zoom import planet as zp

    out, lv = planet["out"], planet["level"]
    info = json.loads((out / "planet.json").read_text())
    N = world["params"].coarse_grid().N
    assert info["active_cells"] > 0 and (out / info["quicklook"]).exists()
    for f in range(6):
        a = {k: np.load(zp.out_path(out, lv.R, f, k)) for k in zp.OUT_FIELDS}
        for k, v in a.items():
            assert v.shape == (N * lv.R, N * lv.R) and np.isfinite(v).all(), (f, k)
        assert (a["water_surface"] >= a["height"] + a["sediment"] - 1e-3).all()
        assert (a["sediment"] >= -1e-4).all() and (a["discharge"] >= 0).all()
    assert sum(fc["seam_cells"] for fc in info["faces"]) > 0


def test_planet_level_keeps_a_time_lapse_of_its_erosion(world, planet):
    """Every tile keeps its core every few iterations
    (``PlanetLevel.snapshots``) and :mod:`globe.zoom.planet_frames` stitches
    frame k of all of them into six faces: the frames share one grid, grow
    towards the finished level, and hold it where no tile ran (the sea)."""
    from globe.zoom import planet as zp
    from globe.zoom import planet_frames as pfr

    out, lv = planet["out"], planet["level"]
    info = json.loads((out / "planet.json").read_text())
    assert info.get("frames") == pfr.FRAMES_NAME.format(R=lv.R)
    lapse = pfr.load(out, lv.R)
    N = world["params"].coarse_grid().N
    n = N * lv.R
    res = n // int(lapse["factor"][0])
    T = len(lapse["iterations"])
    assert 1 < T <= lv.snapshots + 1
    assert list(lapse["iterations"]) == sorted(lapse["iterations"]) and int(lapse["iterations"][-1]) == lv.iterations
    for k in ("surface", "discharge"):
        assert lapse[k].shape == (T, 6, res, res), k
        assert np.isfinite(np.asarray(lapse[k], np.float32)).all(), k
    # the last frame is the level as the tiles left it: the same land, cell for cell where a
    # tile ran (the finish blends seams and the coast after them, so allow a little)
    tiles = pfr.tile_frames(out)
    assert tiles, "no per-tile frames were kept"
    face, ta0, tb0, _ = tiles[0]
    surf = np.load(zp.out_path(out, lv.R, face, "height")) + np.load(zp.out_path(out, lv.R, face, "sediment"))
    k = n // res
    red = surf.reshape(res, k, res, k).mean(axis=(1, 3))
    last = np.asarray(lapse["surface"][-1, face], np.float32)
    land = red > 0.0
    assert land.any()
    assert np.abs(last - red)[land].mean() < max(50.0, 0.05 * float(np.abs(red[land]).mean()))


def test_the_planet_time_lapse_joins_the_viewers_timeline(world, planet):
    """The level's frames are timeline entries of their own (stage
    ``planet``), after the coarse erosion's and before the final state, each
    labelled with the level's cell and its iteration."""
    from globe.viz import viewer as vw
    from globe.zoom import planet_frames as pfr

    out, lv = planet["out"], planet["level"]
    frames, _final = vw.collect_frames(world["root"], None, log=lambda m: None, planet=str(out))
    pf = [f for f in frames if f.stage == "planet"]
    assert len(pf) == len(pfr.load(out, lv.R)["iterations"])
    # last of the erosion frames, before the final state (this world keeps no coarse frames)
    assert frames[-1].stage == "final" and [f.stage for f in frames[-len(pf) - 1:-1]] == ["planet"] * len(pf)
    cell = world["params"].world.cell_size_m / lv.R
    assert all(f"{cell:.0f} m" in f.label or f"{cell / 1000.0:.1f} km" in f.label for f in pf)
    assert [f.key for f in pf] == sorted(f.key for f in pf) and pf[-1].key == lv.iterations
    assert pf[0].res == pf[-1].res and "discharge" in pf[0].ch


def test_the_planet_lapse_survives_a_viewer_export(world, planet, tmp_path):
    """The whole way out: ``export_viewer --planet`` writes the level's frames
    as timeline entries with textures of their own, so the page plays the 1.2
    km erosion after the coarse one."""
    from globe.viz import viewer as vw

    out = vw.export_viewer(world["root"], out=tmp_path / "v", planet=str(planet["out"]), log=lambda m: None)
    meta = json.loads((out.parent / "data" / "meta.js").read_text()[len("GLOBE_VIEWER.setMeta("):-3])
    pf = [f for f in meta["frames"] if f["stage"] == "planet"]
    assert pf, [f["stage"] for f in meta["frames"]]
    assert [f["stage"] for f in meta["frames"][-len(pf) - 1:]] == ["planet"] * len(pf) + ["final"]
    for f in pf:
        assert (out.parent / f["file"]).exists() and (out.parent / f["file"]).stat().st_size > 0
        assert f["res"] == pf[0]["res"] and "discharge" in f["layers"] and f["h1"] > f["h0"]
        assert "iteration" in f["label"]


def test_planet_workers_map_the_inputs_the_world_has(world, planet):
    """The inputs the bake wrote once are what a worker would load itself,
    mapped rather than copied; a changed world file makes a worker load
    instead of trusting them."""
    import os

    from globe.refine import basin_job as bj
    from globe.refine.zoom import zoom_params
    from globe.zoom import planet as zp

    lp = zoom_params(world["params"], planet["level"].R)
    grid, fields, derived, fd, fa = zp.planet_inputs(world["root"], lp, planet["out"])
    assert isinstance(fields["height"].data, np.memmap) and isinstance(fd, np.memmap)
    g2, f2, d2 = bj.coarse_inputs(world["root"], lp)
    fd2, fa2 = zp.planet_flow(world["root"])
    for a, b in ((fields, f2), (derived, d2)):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].data.dtype == b[k].data.dtype and np.array_equal(a[k].data, b[k].data) and a[k].is_vector == b[k].is_vector, k
    assert np.array_equal(fd, fd2) and np.array_equal(fa, fa2)
    src = world["root"] / "coarse" / "precip.f3.npy"
    st = src.stat()
    try:
        os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert not isinstance(zp.planet_inputs(world["root"], lp, planet["out"])[1]["height"].data, np.memmap)
    finally:
        os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_planet_level_does_not_depend_on_the_workers(world, planet, tmp_path):
    from globe.zoom import planet as zp

    lv = planet["level"]
    out2 = zp.run_planet(world["root"], lv, out=tmp_path / "serial", faces=[0], workers=1, finish=False)
    g = lv.guard * lv.R
    for k in ("height", "sediment", "discharge"):
        a = np.load(zp.work_path(planet["out"], 0, k))
        b = np.load(zp.work_path(out2, 0, k))
        assert np.array_equal(a, b), k
    # a pass cut off after its tiles wrote (progress.json never recorded it):
    # resuming skips the written tiles instead of eroding them twice
    prog_path = out2 / "progress.json"
    prog = json.loads(prog_path.read_text())
    last = max(prog["passes"], key=lambda k: int(k.split(":")[1]))
    del prog["passes"][last]
    prog_path.write_text(json.dumps(prog))
    zp.run_planet(world["root"], lv, out=out2, faces=[0], workers=1, finish=False)
    rec = json.loads(prog_path.read_text())["passes"][last]
    assert rec.get("resumed_tiles", 0) > 0, rec
    for k in ("height", "sediment", "discharge"):
        assert np.array_equal(np.load(zp.work_path(planet["out"], 0, k)), np.load(zp.work_path(out2, 0, k))), k


def test_zoom_starts_from_the_planet_level(world, zoom, planet, tmp_path):
    """With a finished planet level at the first level's R, that level is the
    planet's rasters over its work array -- not eroded again -- and the next
    level chains from it as from its own."""
    from globe.zoom import planet as zp

    assert zb.planet_dir(world["root"], LEVELS[0].R) == planet["out"]
    out = zb.run_zoom(world["root"], zoom["spot"], LEVELS, out=tmp_path / "from_planet", name="p", resume=False)
    first = zb.load_level(out, LEVELS[0].R)
    assert first.stats["source"] == "planet"
    geo, R = first.geo, LEVELS[0].R
    a0, b0 = geo.origin[0] * R, geo.origin[1] * R
    h = np.load(zp.out_path(planet["out"], R, geo.face, "height"))[a0:a0 + geo.NE, b0:b0 + geo.NE]
    assert np.array_equal(first.arrays["height"], h.astype(np.float32))
    second = zb.load_level(out, LEVELS[1].R)
    assert second.stats.get("source") != "planet" and second.stats["inflow_total"] > 0.0
    sl = second.geo.product()
    assert second.arrays["done"][sl][~second.arrays["ocean"][sl]].all()
    info = json.loads((out / "zoom.json").read_text())
    assert [lv["R"] for lv in info["levels"]] == [lv.R for lv in LEVELS]


def test_planet_level_chained_from_a_coarser_one(world, planet):
    """``PlanetLevel(parent=R)``: every tile starts from the finished coarser
    level -- where it wrote, the child's plain surface is the parent's
    interpolated, not the planet's upsample -- takes the parent's drainage
    across its window edge, and the level finishes like any other.  The
    parent's flood tree carries all the rain it is given to its drains."""
    from globe.io.world_store import WorldStore
    from globe.refine.upsample import Window, upsample_window
    from globe.refine.zoom import zoom_params
    from globe.zoom import planet as zp
    from globe.zoom import planet_chain as pc

    parent = planet["level"]
    lv = zp.PlanetLevel(R=2 * parent.R, iterations=2, tile=32, margin=8, hold_every=2, seam_cells=4, parent=parent.R)
    root = world["root"]
    params = world["params"]
    N = params.coarse_grid().N
    # the parent's drainage
    rp, fp = pc.parent_flow(root, planet["out"], parent, 0)
    recv, flux = np.load(rp), np.load(fp)
    NFp = (N + 2 * parent.guard) * parent.R
    assert recv.shape == (NFp, NFp) and flux.min() >= 0.0 and flux.max() > 0.0
    # a tile's inputs where the parent wrote
    lp = zoom_params(params, lv.R)
    grid, fields, derived, _, _ = zp.planet_inputs(root, lp, planet["out"])
    mc = lv.margin // lv.R
    land_c = np.stack([np.load(root / "coarse" / f"basin_id.f{k}.npy") for k in range(6)]) >= 0
    face, (ta0, tb0, cc) = next((f, t) for f in range(6) for ps in zp.face_tiles(lv, N, land_c, f) for t in ps)
    nc = cc + 2 * mc
    win = Window(face, ta0, ta0 + nc, tb0, tb0 + nc, lv.R)
    up = upsample_window(fields, derived, win, grid)
    t = slice(lv.R - 1, win.NE - (lv.R - 1))
    tr = {k: v[t, t] for k, v in up.items()}
    pc.parent_flow(root, planet["out"], parent, face)
    ch = pc.chained_inputs(planet["out"], parent, lv, N, face, ta0, tb0, tr, grid.cell_size_m, 1, 0.0)
    assert ch["f"] == 2 and np.isfinite(ch["plain"]).all() and (ch["sediment0"] >= 0).all()
    work = np.load(zp.work_path(planet["out"], face, "height")) + np.load(zp.work_path(planet["out"], face, "sediment"))
    gp = parent.guard * parent.R
    i, j = ta0 * lv.R - 1 + nc * lv.R // 2, tb0 * lv.R - 1 + nc * lv.R // 2      # a child cell centred on a parent cell pair
    k = nc * lv.R // 2
    px = (i + 0.5) / lv.R * parent.R - 0.5 + gp
    lo = work[int(np.floor(px)), int(np.floor((j + 0.5) / lv.R * parent.R - 0.5 + gp))]
    hi = work[int(np.ceil(px)), int(np.ceil((j + 0.5) / lv.R * parent.R - 0.5 + gp))]
    assert min(lo, hi) - 50.0 <= ch["plain"][k, k] <= max(lo, hi) + 50.0
    # the whole level
    out = zp.run_planet(root, lv, workers=2)
    info = json.loads((out / "planet.json").read_text())
    assert zp.PlanetLevel.from_dict(info["level"]).parent == parent.R and info["active_cells"] > 0
    tiles = json.loads((out / "progress.json").read_text())["tiles"]
    assert any(t_.get("inflow", 0.0) > 0.0 for t_ in tiles)
    for f in range(6):
        a = {k2: np.load(zp.out_path(out, lv.R, f, k2)) for k2 in zp.OUT_FIELDS}
        assert all(np.isfinite(v).all() for v in a.values())
        assert (a["water_surface"] >= a["height"] + a["sediment"] - 1e-3).all()


def test_planet_finish_in_bounded_memory_matches_the_whole_face_finish(world, planet, tmp_path, monkeypatch):
    """``planet_finish.finish_face`` writes the same outputs as the in-memory
    finish while a face floods whole, and valid ones -- water on or above the
    ground, none on the sea, lakes kept -- when it floods in blocks."""
    import shutil

    from globe.zoom import planet as zp
    from globe.zoom import planet_finish as pf

    lv, src = planet["level"], planet["out"]
    out = tmp_path / "fin"
    shutil.copytree(src, out)
    ref_dir = tmp_path / "ref"
    shutil.copytree(src, ref_dir)
    for f in (0, 3):
        zp.finish_face(world["root"], ref_dir, lv, f)                    # the in-memory finish
        ref = {k: np.load(zp.out_path(ref_dir, lv.R, f, k)) for k in zp.OUT_FIELDS}
        st = pf.finish_face(world["root"], out, lv, f)
        for k in zp.OUT_FIELDS:
            assert np.array_equal(np.load(zp.out_path(out, lv.R, f, k)), ref[k]), (f, k)
        assert not (out / f"L{lv.R}.f{f}.plain_ws.tmp.npy").exists()
    n = world["params"].coarse_grid().N * lv.R
    monkeypatch.setattr(pf, "FLOOD_WHOLE", n // 4)
    monkeypatch.setattr(pf, "FLOOD_BLOCK", n // 4)
    monkeypatch.setattr(pf, "FLOOD_OVERLAP", n // 16)
    lakes = []
    for f in range(6):
        st = pf.finish_face(world["root"], out, lv, f)
        a = {k: np.load(zp.out_path(out, lv.R, f, k)) for k in zp.OUT_FIELDS}
        surf = a["height"] + a["sediment"]
        assert (a["water_surface"] >= surf - 1e-3).all() and np.isfinite(a["water_surface"]).all()
        ref = np.load(zp.out_path(src, lv.R, f, "height"))
        assert np.array_equal(a["height"], ref)
        lakes.append(st["lake_cells"])
    assert sum(lakes) >= 0


def test_planet_flow_is_the_rain_down_the_final_surface(world, planet, tmp_path):
    """``planet_finish.flow_face``: the river map of a finished face is rain
    accumulated down its surface's flood tree -- on a valley tilted to one
    border cell, it never falls down the valley floor and the outlet carries
    the face's rain; a finished level has one for every face."""
    from globe.zoom import planet as zp
    from globe.zoom import planet_finish as pf

    out, lv = planet["out"], planet["level"]
    N = world["params"].coarse_grid().N
    n = N * lv.R
    info = json.loads((out / "planet.json").read_text())
    assert [s["face"] for s in info["flow"]] == list(range(6))
    for f in range(6):
        q = np.load(zp.out_path(out, lv.R, f, "flow"))
        assert q.shape == (n, n) and np.isfinite(q).all() and (q >= 0).all()

    syn = tmp_path / "valley"
    syn.mkdir()
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    c = n // 2
    h = 5000.0 + 5.0 * i + 20.0 * np.abs(j - c)
    rim = (i == 0) | (i == n - 1) | (j == 0) | (j == n - 1)
    h[rim & ~((i == 0) & (j == c))] = 9000.0                       # the border drains; only the valley's mouth is low
    np.save(zp.out_path(syn, lv.R, 0, "height"), h.astype(np.float32))
    np.save(zp.out_path(syn, lv.R, 0, "sediment"), np.zeros((n, n), np.float32))
    st = pf.flow_face(world["root"], syn, lv.R, 0)
    q = np.load(zp.out_path(syn, lv.R, 0, "flow")).astype(np.float64)
    floor = q[1:-1, c]
    assert (np.diff(floor) <= 1e-3 * floor[1:]).all()          # downstream is toward row 0
    rain = np.maximum(np.load(world["root"] / "coarse" / "precip.f0.npy").astype(np.float64), 0.0) / lv.R ** 2
    interior = np.repeat(np.repeat(rain, lv.R, axis=0), lv.R, axis=1)[1:-1, 1:-1].sum()
    assert q[0, c] >= 0.999 * interior and q[0, c] == q.max()      # the mouth takes the face's rain (from three cells)
    assert st["flow_max"] == pytest.approx(q[0, c], rel=1e-6)


def test_planet_inflow_crosses_cube_edges():
    """A window on a face edge takes water from the neighbouring face's
    drainage: a donor beyond the edge is looked up on the face that owns it,
    and when its D8 receiver (in that face's own directions) lies across the
    edge in the window, its whole accumulation enters there -- on the
    window's edge row, and nowhere else."""
    from globe.cubesphere import from_sphere_v, project_to_face_v, to_sphere_v
    from globe.zoom import planet as zp

    N, R, face = 32, 4, 0
    j = 12.0
    p = to_sphere_v(np.array([face]), np.array([-0.5 / N]), np.array([(j + 0.5) / N]))   # ring cell i = -1
    f2, u2, v2 = from_sphere_v(p)
    f2, i2, j2 = int(f2[0]), min(int(u2[0] * N), N - 1), min(int(v2[0] * N), N - 1)
    assert f2 != face
    found = False
    for code in range(8):
        ri, rj = i2 + zp.zb.D8[code, 0], j2 + zp.zb.D8[code, 1]
        pr = to_sphere_v(np.array([f2]), np.array([(ri + 0.5) / N]), np.array([(rj + 0.5) / N]))
        u, v = project_to_face_v(np.array([face]), pr)
        if int(np.floor(u[0] * N)) == 0 and 8 <= int(np.floor(v[0] * N)) < 16:
            found = True
            break
    assert found
    fd = np.full((6, N, N), 255, np.int64)
    fa = np.zeros((6, N, N))
    fd[f2, i2, j2] = code
    fa[f2, i2, j2] = 5.0
    w = zp.planet_inflow(fd, fa, face, 0, 8, 8, 16, R, np.zeros((8 * R + 2, 8 * R + 2)))
    assert w.sum() == pytest.approx(5.0)
    assert w[1:1 + R // 2 + 1].sum() == pytest.approx(5.0)


def test_viewer_final_frame_from_the_planet_level(world, planet):
    """``viewer._planet_final`` reduces the planet level to the frame
    resolution: block-mean surface, block-max rivers (the level's
    accumulated flow, which a finished level has), sea and lakes by derive's
    rules."""
    from globe.viz import viewer
    from globe.zoom import planet as zp

    root, params = world["root"], world["params"]
    grid = params.coarse_grid()
    store = world["store"]
    surf_c = store.load_field("height", grid).interior + store.load_field("sediment", grid).interior
    fd = store.load_field("flow_dir", grid).interior
    lv = planet["level"]
    N = grid.N
    fin = viewer._planet_final(root, json.loads((root / "manifest.json").read_text()), surf_c, fd, planet["out"].relative_to(root), res=N * 2, log=lambda m: None)
    assert fin["surf"].shape == (6, 2 * N, 2 * N)
    k = lv.R // 2
    from globe.viz import detail as dt

    assert dt.river_field(planet["out"], lv.R) == "flow" and fin["river_scale"]["river_full"] > fin["river_scale"]["river_min"]
    q0 = dt.widen_rivers(np.load(zp.out_path(planet["out"], lv.R, 0, "flow")), fin["river_scale"])     # strips give the whole face's widening
    assert np.allclose(fin["discharge"][0], q0.reshape(2 * N, k, 2 * N, k).max(axis=(1, 3)))
    h0 = np.load(zp.out_path(planet["out"], lv.R, 0, "height")) + np.load(zp.out_path(planet["out"], lv.R, 0, "sediment"))
    assert np.allclose(fin["surf"][0], h0.reshape(2 * N, k, 2 * N, k).mean(axis=(1, 3)), atol=1e-2)
    assert (fin["water"] == viewer.WATER_OCEAN).any() and (fin["ws"] >= fin["surf"] - 1e-3).all()


def test_planet_flow_crosses_a_filled_depression_along_its_bed():
    """``planet_finish.flow_face``'s routing (``hydro.bed_routing``): water
    crossing a filled depression runs down its drowned bed and along it to the
    spill, where the plain flood's queue draws a breadth-first ray.  A closed
    basin (rough floor) holds a U-shaped channel from a river's mouth round
    to the spill point, the channel flat, rising or falling toward the spill:
    the fill is the plain flood's, the river's path through the basin is the
    channel (the plain flood's leaves it), the channel's last cell carries the
    river and the basin's rain, every cell's weight reaches a drain once, and
    the tree is acyclic and deterministic."""
    from globe.hydro.bed_routing import priority_flood_bed
    from globe.hydro.priority_flood import priority_flood_flat

    n = 64
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rng = np.random.default_rng(3)
    chan = [(r, 12) for r in range(32, 15, -1)] + [(16, c) for c in range(13, 47)] + [(r, 46) for r in range(17, 33)] + [(32, c) for c in range(47, 51)]
    mouth, spill = chan[0][0] * n + chan[0][1], 32 * n + 51
    on_chan = np.zeros(n * n, bool)
    on_chan[[a * n + b for a, b in chan]] = True
    basin = ((i >= 12) & (i <= 52) & (j >= 10) & (j <= 50)).ravel()
    drain = np.zeros((n, n), bool)
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    w = np.ones(n * n)
    w[mouth] = 1e6                                                   # the river entering the basin

    def path(parent):
        p, c = [], mouth
        while c >= 0:
            p.append(c)
            c = int(parent[c])
        return np.array(p)

    for tilt in (-0.05, 0.0, 0.05):
        h = 200.0 + 0.5 * np.minimum.reduce([i, j, n - 1 - i, n - 1 - j])   # a plateau falling to the border
        h.ravel()[basin] = 50.0 + rng.uniform(0.0, 1.0, int(basin.sum()))
        for k, (a, b) in enumerate(chan):
            h[a, b] = 20.0 + tilt * k
        h[32, 51] = 80.0                                             # the spill: the lake fills to 80 m
        h[32, 52:] = 79.0 - np.arange(n - 52)                        # its outflow to the east border
        h = h.astype(np.float32)
        fr, plain = priority_flood_bed(h, drain), priority_flood_flat(h, drain)
        assert np.array_equal(fr.filled, plain.filled) and fr.filled[20, 20] == 80.0
        live = ~drain.ravel()
        assert (fr.parent[live] >= 0).all() and (fr.parent[~live] == -1).all()
        assert (fr.order[fr.parent[live]] < fr.order[live]).all()
        acc = zb._accumulate(fr.pop_seq, fr.parent, w)
        assert acc[~live].sum() == pytest.approx(w.sum(), rel=1e-12)          # every drop reaches a drain once
        p = path(fr.parent)
        assert spill in p and basin[p].sum() >= 60 and on_chan[p][basin[p]].all(), tilt
        assert acc[32 * n + 50] >= 1e6 + basin.sum()                           # the channel takes the basin's rain
        q = path(plain.parent)
        assert on_chan[q][basin[q]].mean() < 0.2, tilt                         # the plain flood's ray crosses the floor
        again = priority_flood_bed(h, drain)
        assert np.array_equal(again.parent, fr.parent) and np.array_equal(again.pop_seq, fr.pop_seq)


def test_ltd_follows_a_planar_slopes_aspect_and_keeps_the_tree():
    """``hydro.bed_routing.accumulate_ltd`` (D8-LTD on dry ground): on a plane
    falling 30 degrees off the grid's axis a path stays within a cell or two
    of the true flow line, where the lowest-neighbour tree runs a 45-degree
    ray away from it; on rough ground with pits and flats every cell still
    drains to a strictly earlier-popped neighbour, dry receivers are
    strictly lower, and the accumulation is the tree's (rain conserved)."""
    from globe.hydro.bed_routing import accumulate_ltd, priority_flood_bed
    from globe.hydro.tree import accumulate_codes, receiver_codes

    n = 160
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a = np.deg2rad(30.0)
    h = (1000.0 - 0.5 * (np.cos(a) * i + np.sin(a) * j)).astype(np.float32)
    drain = np.zeros((n, n), bool)
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    w = np.ones(n * n)

    def worst_offset(parent):
        c, i0, j0, worst, steps = 5 * n + 5, 5, 5, 0.0, 0
        while parent[c] >= 0:
            c = int(parent[c])
            steps += 1
            worst = max(worst, abs(-np.sin(a) * (c // n - i0) + np.cos(a) * (c % n - j0)))
        return worst, steps

    plain = priority_flood_bed(h, drain)
    lowest, _ = worst_offset(plain.parent)
    fr = priority_flood_bed(h, drain)
    acc = accumulate_ltd(h, fr, w)
    ltd, steps = worst_offset(fr.parent)
    assert steps >= 100 and ltd <= 1.5 and lowest >= 10.0, (ltd, lowest, steps)
    assert acc[drain.ravel()].sum() == pytest.approx(w.sum(), rel=1e-12)

    rng = np.random.default_rng(5)
    rough = ndimage.gaussian_filter(rng.normal(size=(n, n)), 3.0) * 200.0
    rough = np.where(rng.uniform(size=(n, n)) < 0.3, np.round(rough / 5.0) * 5.0, rough).astype(np.float32)   # exact flats
    drain = drain | (rough < np.percentile(rough, 2))
    fr = priority_flood_bed(rough, drain)
    acc = accumulate_ltd(rough, fr, w)
    live = ~drain.ravel()
    par = fr.parent[live].astype(np.int64)
    cells = np.flatnonzero(live)
    assert (par >= 0).all() and (fr.order[par] < fr.order[cells]).all()
    assert (np.abs(par // n - cells // n) <= 1).all() and (np.abs(par % n - cells % n) <= 1).all() and (par != cells).all()
    dry = fr.filled.ravel()[cells] == rough.ravel()[cells]
    assert (fr.filled.ravel()[par[dry]] <= fr.filled.ravel()[cells[dry]]).all()
    assert acc[~live].sum() == pytest.approx(w.sum(), rel=1e-12)
    assert np.allclose(acc, zb._accumulate(fr.pop_seq, fr.parent, w))
    assert np.allclose(accumulate_codes(receiver_codes(fr.parent, n), n, w), acc)


def test_spill_levels_join_tiles_into_one_flood():
    """``hydro.spill_graph``: tiles flooded alone from labelled edge seeds, their
    labels linked across the tile edges and flooded from the sea, give the
    whole window's filled surface exactly -- basins that straddle tiles
    included; and ``hydro.tree.window_exits`` names where a cell's water last
    leaves a sub-window."""
    from globe.hydro.priority_flood import priority_flood_flat
    from globe.hydro.spill_graph import label_flood, spill_levels
    from globe.hydro.tree import window_exits

    rng = np.random.default_rng(1)
    for trial in range(12):
        H, W = (int(x) for x in rng.integers(20, 90, 2))
        s = (ndimage.gaussian_filter(rng.normal(size=(H, W)), rng.uniform(1.0, 5.0)) * 100).astype(np.float32)
        if trial % 3 == 0:
            s = np.round(s / 4.0) * 4.0
        sea = s < np.percentile(s, 5)
        outer = np.zeros((H, W), bool)
        outer[0] = outer[-1] = outer[:, 0] = outer[:, -1] = True
        whole = priority_flood_flat(s, sea | outer)
        ci, cj = H // 2, W // 2
        tid = np.zeros((H, W), int)
        tid[:ci, cj:], tid[ci:, :cj], tid[ci:, cj:] = 1, 2, 3
        label = np.full((H, W), -1, np.int64)
        filled = np.zeros((H, W), np.float32)
        A, B, Wt, nxt = [], [], [], 1
        for t, (a0, a1, b0, b1) in enumerate([(0, ci, 0, cj), (0, ci, cj, W), (ci, H, 0, cj), (ci, H, cj, W)]):
            seed = np.full((a1 - a0, b1 - b0), -1, np.int32)
            rim = np.zeros(seed.shape, bool)
            rim[0] = rim[-1] = rim[:, 0] = rim[:, -1] = True
            seed[rim] = np.arange(nxt, nxt + rim.sum())
            nxt += int(rim.sum())
            seed[(sea | outer)[a0:a1, b0:b1]] = 0
            f_, l_, (ea, eb, ew) = label_flood(s[a0:a1, b0:b1], seed)
            filled[a0:a1, b0:b1], label[a0:a1, b0:b1] = f_, l_
            A.append(ea), B.append(eb), Wt.append(ew)
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            p = np.argwhere(np.ones((H, W), bool))
            q = p + (di, dj)
            ok = (q[:, 0] >= 0) & (q[:, 0] < H) & (q[:, 1] >= 0) & (q[:, 1] < W)
            p, q = p[ok], q[ok]
            cross = tid[p[:, 0], p[:, 1]] != tid[q[:, 0], q[:, 1]]
            p, q = p[cross], q[cross]
            A.append(label[p[:, 0], p[:, 1]]), B.append(label[q[:, 0], q[:, 1]])
            Wt.append(np.maximum(filled[p[:, 0], p[:, 1]], filled[q[:, 0], q[:, 1]]))
        lv = spill_levels(nxt, np.concatenate(A), np.concatenate(B), np.concatenate(Wt), [0])
        assert np.array_equal(np.maximum(filled, lv[label]).astype(np.float32), whole.filled), trial
        ex = window_exits(whole.parent, whole.pop_seq, W, 3, H - 3, 3, W - 3)
        for c in rng.choice(np.flatnonzero(whole.order >= 0), 40):
            path = [int(c)]
            while whole.parent[path[-1]] >= 0:
                path.append(int(whole.parent[path[-1]]))
            ins = [3 <= x // W < H - 3 and 3 <= x % W < W - 3 for x in path]
            want = -1
            if not any(ins):
                want = path[0]
            else:
                last = max(k for k, v in enumerate(ins) if v)
                want = path[last + 1] if last + 1 < len(path) else -1
            assert ex[c] == want, (trial, c)


def _sphere_faces(n, fn):
    from globe.cubesphere import to_sphere_v

    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return [fn(to_sphere_v(np.full(i.shape, f), (i + 0.5) / n, (j + 0.5) / n)).astype(np.float32) for f in range(6)]


def test_route_faces_hands_rivers_over_cube_edges():
    """``planet_finish.route_faces``: a planet falling to a sea around +Y, a
    valley along the great circle through +X and +Y, and a closed basin
    straddling the +X/+Y edge wider than the flood margin.  Every drop of
    rain reaches the sea once (no parcel circles -- the basin's level is
    the planet's, not each face's flood's; the far face's water crosses two
    edges); a river cell on a face's edge that is not passed on inside its
    face is taken up on the neighbouring face within two cells, with at least
    its water; and without the hand-over (a lone face) the sea would miss
    the water that crosses."""
    from globe.zoom import planet_finish as pf

    n, K = 96, 4
    c = np.array([1.0, 1.0, 0.5]) / np.linalg.norm([1.0, 1.0, 0.5])

    def ground(p):
        basin = 900.0 * np.exp(-((np.arccos(np.clip(p @ c, -1, 1)) / 0.12) ** 2))
        return 3000.0 * (1.0 - p[..., 1]) - 600.0 - 400.0 * np.exp(-(p[..., 2] / 0.06) ** 2) - basin + 3.0 * np.sin(40 * p[..., 0]) * np.cos(37 * p[..., 2])

    S = _sphere_faces(n, ground)
    flows = {}
    st = pf.route_faces(n, lambda f: (S[f], S[f] < 0), lambda f: np.ones((n, n)), lambda f, i, j: (S[f][i, j], S[f][i, j] < 0),
                        lambda f, q: flows.__setitem__(f, q.astype(np.float64)), K)
    rain = sum(s["rain"] for s in st)
    assert sum(s["sea"] for s in st) == pytest.approx(rain, rel=1e-9)
    assert sum(s["unresolved"] for s in st) == 0.0 and st[0]["cycled_total"] == 0.0
    assert st[3]["sea"] == 0.0 and st[3]["outflow"] == pytest.approx(st[3]["rain"])
    big = 0.002 * rain
    checked = 0
    for f in range(6):
        q, sea = flows[f], S[f] < 0
        for side in range(4):
            idx = np.arange(1, n - 1)
            a, b, oi, oj = {0: (0 * idx, idx, -1, 0), 1: (0 * idx + n - 1, idx, 1, 0), 2: (idx, 0 * idx, 0, -1), 3: (idx, 0 * idx + n - 1, 0, 1)}[side]
            for x, y in zip(a, b):
                if q[x, y] < big or sea[x, y]:
                    continue
                nb = q[max(x - 1, 0):x + 2, max(y - 1, 0):y + 2]
                if (nb >= q[x, y]).sum() > 1:
                    continue                                       # passed on inside its face
                g, gi, gj = pf._beyond(f, np.array([x + oi]), np.array([y + oj]), n)
                win = flows[int(g[0])][max(gi[0] - 2, 0):gi[0] + 3, max(gj[0] - 2, 0):gj[0] + 3]
                assert win.max() >= q[x, y] * (1 - 1e-9), (f, x, y)
                checked += 1
    assert checked >= 3


# --------------------------------------------------------------------------
# zoom level textures for the globe viewer (globe/viz/zoomtex.py)
# --------------------------------------------------------------------------
def _decode_tex(js_path, name, R, frame=None):
    import base64
    import io
    import re

    from PIL import Image

    text = js_path.read_text()
    head = 'GLOBE_VIEWER.zoomTex("%s", %d, "' % (name, R)
    assert text.startswith(head)
    tail = '", %d);\n' % frame if frame is not None else '");\n'
    assert text.endswith(tail) or re.search(r'", \d+\);\n$', text), text[-20:]
    b64 = text[len(head):text.rindex('"')]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA"))   # no alpha stored when it is all 255


def test_zoom_level_texture_encodes_ground_lakes_ocean_and_rivers(tmp_path):
    """A synthetic level: a slope out of the sea along i, a pit lake on it and
    one channel of large flux down column 40.  ``index.write`` writes its
    texture and sidecar and lists them as the level's ``tex``: the image is
    ``NE`` wide and ``2 NE`` tall, the heights decode back within a code step,
    the sea is transparent and the land opaque, the lake's depth byte is
    above the middle, the channel's river byte above ``river_min_byte`` and
    the dry hillside's 0; a second write leaves the fresh texture alone."""
    root = tmp_path / "w"
    (root / "viewer").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"N_c": 64}))
    zdir = root / "zoom" / "syn"
    zdir.mkdir(parents=True)
    geo = zb.Geometry(face=2, ci0=10, cj0=12, cells=4, guard=1, R=8)
    NE, p0, n = geo.NE, geo.p0, geo.n
    i, j = np.meshgrid(np.arange(NE), np.arange(NE), indexing="ij")
    rng = np.random.default_rng(3)
    height = ((i - 19.5) * 2.0).astype(np.float32)
    sediment = rng.uniform(0.0, 0.4, (NE, NE)).astype(np.float32)
    surf = height + sediment
    ocean = i < 20
    ws = np.maximum(surf, 0.0)
    lake = (i >= 36) & (i <= 38) & (j >= 24) & (j <= 26)
    ws[lake] = surf[lake] + 5.0
    flux = np.ones((NE, NE), np.float32)
    chan = (j == 40) & ~ocean
    flux[chan] = 50.0 * (i[chan] - 19)
    flux[ocean] = 0.0
    arrays = {"height": height, "sediment": sediment, "discharge": flux.copy(), "momentum": np.zeros((NE, NE, 2), np.float32),
              "water_surface": ws.astype(np.float32), "flux": flux, "plain": np.zeros((NE, NE), np.float32), "ocean": ocean,
              "done": np.ones((NE, NE), bool)}
    zb.save_level(zb.LevelResult(geo, arrays, {"geometry": {"face": 2, "ci0": 10, "cj0": 12, "cells": 4, "guard": 1, "R": 8}, "cell_m": 1000.0}), zdir)
    level = {"R": 8, "cell_m": 1000.0, "geometry": {"face": 2, "ci0": 10, "cj0": 12, "cells": 4, "guard": 1, "R": 8}}
    no_npz = {"R": 32, "cell_m": 250.0, "geometry": {"face": 2, "ci0": 11, "cj0": 13, "cells": 2, "guard": 1, "R": 32}}
    (zdir / "zoom.json").write_text(json.dumps({"name": "syn", "spot": [2, 12, 14], "lat": 0.0, "lon": 0.0, "levels": [level, no_npz]}))
    (zdir / "view.html").write_text("<html></html>")

    zooms = zindex.write(root, log=None)
    js = root / "viewer" / "zoomtex" / "syn_L8.js"
    side = root / "viewer" / "zoomtex" / "syn_L8.json"
    assert js.exists() and side.exists()
    lv8, lv32 = zooms[0]["levels"]
    assert "tex" not in lv32 and not (root / "viewer" / "zoomtex" / "syn_L32.js").exists()
    tex = lv8["tex"]
    assert tex == json.loads(side.read_text())
    assert (tex["file"], tex["face"], tex["N"], tex["R"], tex["ci0"], tex["cj0"], tex["cells"], tex["guard"]) == ("zoomtex/syn_L8.js", 2, 64, 8, 10, 12, 4, 1)
    assert (tex["NE"], tex["p0"], tex["n"], tex["lake_range"], tex["cell_m"]) == (NE, p0, n, 200.0, 1000.0)
    listed = json.loads((root / "viewer" / "zooms.js").read_text()[len("GLOBE_VIEWER.setZooms("):-3])
    assert listed[0]["levels"][0]["tex"] == tex and "corners" in listed[0]["levels"][0]

    img = _decode_tex(js, "syn", 8)
    assert img.shape == (3 * NE, NE, 4) and img.dtype == np.uint8
    ground, water = img[:NE], img[NE:2 * NE]                       # (y, x) = (j, i)
    step = (tex["h1"] - tex["h0"]) / 65535.0
    drawn = np.where(lake, ws, surf)                         # a lake's height is its water, not its bed
    for ci, cj in ((5, 5), (19, 30), (20, 30), (p0 + 3, p0 + 7), (NE - 1, NE - 1), (37, 25)):
        code = int(ground[cj, ci, 0]) * 256 + int(ground[cj, ci, 1])
        assert abs(tex["h0"] + code * step - float(drawn[ci, cj])) <= step * 1.01, (ci, cj)
        assert (code < round(-tex["h0"] / step)) == (drawn[ci, cj] < 0)     # sea level on a code
    assert ground[:, :20, 3].max() < 128 and ground[:, 20:, 3].min() >= 128
    assert ground[25, 37, 2] > 127.5 and ground[30, 45, 2] < 127.5
    assert water[40, p0 + n - 1, 0] > tex["river_min_byte"] and water[40, 30, 0] > 0
    assert water[20, 30, 0] == 0 and water[:, :, 3].min() == 255
    # G: the flux itself on the river scale -- the channel unwidened, a cell beside it at the floor
    assert water[40, p0 + n - 1, 1] > tex["river_min_byte"] and water[39, p0 + n - 1, 1] == 0 < water[39, p0 + n - 1, 0]
    # B: sediment on its log byte (all under SED_LO_M here but the odd cell; decodes within a byte)
    from globe.viz import detail as dt
    assert tex["q_hi"] > tex["q_lo"] > 0 and (tex["sed_lo"], tex["sed_hi"], tex["version"]) == (dt.SED_LO_M, dt.SED_HI_M, 5)
    assert tex["biome"] is False and (img[2 * NE:, :, 0] == 0).all()   # a world without a climate: no biomes
    assert tex["veg"] is False                                # no cover grown: A stays 255
    assert (water[:, :, 2] == dt.log_byte(sediment[:NE, :NE], dt.SED_LO_M, dt.SED_HI_M).T).all()
    assert 1 <= tex["river_min_byte"] <= 254 and tex["river_span_byte"] >= 8

    before = (js.stat().st_mtime_ns, side.stat().st_mtime_ns)
    assert zindex.write(root, log=None)[0]["levels"][0]["tex"] == tex
    assert (js.stat().st_mtime_ns, side.stat().st_mtime_ns) == before

    # A: the canopy cover a level grew (version 4)
    from globe.viz import zoomtex as ztex
    arrays["vegetation"] = np.where(ocean, 0.0, (j / (NE - 1.0))).astype(np.float32)
    zb.save_level(zb.LevelResult(geo, arrays, {"geometry": {"face": 2, "ci0": 10, "cj0": 12, "cells": 4, "guard": 1, "R": 8}, "cell_m": 1000.0}), zdir)
    rec = ztex.write_level(root / "viewer", "syn", zdir, 8, 64, force=True)
    assert rec["veg"] is True
    water = _decode_tex(js, "syn", 8)[NE:2 * NE]
    for ci, cj in ((30, 0), (30, NE - 1), (45, 20), (5, 20)):
        assert abs(int(water[cj, ci, 3]) - round(255.0 * float(arrays["vegetation"][ci, cj]))) <= 1, (ci, cj)


def test_time_lapse_frames_are_written_and_share_one_scale(tmp_path):
    """A level's time lapse (``ZoomLevel.snapshots``: its product's surface and
    streams every few iterations, block-averaged): one texture per frame laid
    out as a level's own -- ground on top, streams below -- all on one height
    and one river scale so nothing jumps between them, and a record placing
    them on the face (a frame's cell is the level's times the block factor)."""
    from globe.viz import zoomtex

    zdir = tmp_path / "w" / "zoom" / "syn"
    viewer = tmp_path / "w" / "viewer"
    zdir.mkdir(parents=True)
    geo = zb.Geometry(face=1, ci0=10, cj0=12, cells=4, guard=1, R=8)
    T, S = 3, 64
    i, j = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    surf = np.stack([(i * 2.0 + k * 40.0 + j * 0.5).astype(np.float32) for k in range(T)])
    disch = np.stack([np.full((S, S), 1.0 + k, np.float32) for k in range(T)])
    disch[:, :, 20] = 500.0
    np.savez(zdir / "L8.frames.npz", surface=surf, discharge=disch,
             iterations=np.array([1, 40, 80], np.int32), factor=np.array([2] * T, np.int32))
    (zdir / "L8.json").write_text(json.dumps({"geometry": geo.to_dict(), "cell_m": 1000.0}))

    rec = zoomtex.write_frames(viewer, "syn", zdir, 8, 64, cell_m=1000.0, force=True, log=None)
    assert rec["count"] == T and rec["iterations"] == [1, 40, 80] and rec["cell_m"] == 2000.0
    assert (rec["face"], rec["n"], rec["NE"], rec["p0"]) == (1, S, S, 0)
    fi0, fj0 = geo.product_origin
    assert (rec["res"], rec["oi"], rec["oj"]) == (64 * 8 / 2, fi0 / 2, fj0 / 2)
    step = (rec["h1"] - rec["h0"]) / 65535.0
    for k in range(T):
        img = _decode_tex(viewer / rec["files"][k], "syn", 8)
        assert img.shape == (2 * S, S, 4)
        ground, water = img[:S], img[S:]
        code = ground[:, :, 0].astype(np.int64) * 256 + ground[:, :, 1]
        assert np.abs(rec["h0"] + code * step - surf[k].T).max() <= step * 1.01     # one scale for every frame
        assert (ground[:, :, 3] == 255).all() and (ground[:, :, 2] < 128).all()     # all land, no lake
        assert (water[20, :, 0] > rec["river_min_byte"]).all() and water[10, 10, 0] < rec["river_min_byte"]


def test_restream_frames_takes_a_saved_time_lapse_back_to_the_flood_tree(tmp_path):
    """:func:`zoom.bake.restream_frames` rewrites a saved lapse's streams as the
    flood tree of each frame's own surface, weighted by the rain of the cells a
    frame cell stands for (the level's ``rain_cell`` times its block factor
    squared) -- for the levels baked while a snapshot kept the particles' own
    discharge, which faded with them."""
    zdir = tmp_path / "w" / "zoom" / "syn"
    zdir.mkdir(parents=True)
    T, S, fac, rain = 3, 40, 2, 0.002
    i, j = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    surf = np.stack([(2.0 * i + 0.4 * np.sin(j * 0.7) + 5.0 * k).astype(np.float32) for k in range(T)])
    pulse = np.stack([np.full((S, S), 1.0 / (k + 1), np.float32) for k in range(T)])       # the old field, fading
    np.savez(zdir / "L8.frames.npz", surface=surf, discharge=pulse,
             iterations=np.array([1, 5, 9], np.int32), factor=np.array([fac] * T, np.int32))
    (zdir / "L8.json").write_text(json.dumps({"geometry": zb.Geometry(face=1, ci0=10, cj0=12, cells=4, guard=1, R=8).to_dict(),
                                              "cell_m": 1000.0, "rain_cell": rain}))

    assert zb.restream_frames(zdir, 8) == T
    with np.load(zdir / "L8.frames.npz") as z:
        out, kept = z["discharge"], z["surface"]
    assert kept == pytest.approx(surf) and out.shape == (T, S, S)
    weight = np.full((S, S), rain * fac ** 2, np.float32)
    for k in range(T):
        acc, _pool = zb.tile_water(surf[k], surf[k] < 0.0, weight)
        assert out[k] == pytest.approx(acc, rel=1e-5), k
    # the streams are the land's now: a trunk of hundreds of cells' rain, and as steady from
    # frame to frame as the land is (the surface only rises), where the particles' discharge
    # had halved and halved again
    mx = out.max(axis=(1, 2))
    assert mx.min() > 100.0 * float(weight[0, 0]) and mx.max() <= 1.25 * mx.min()


def test_zoom_texture_is_cropped_to_the_core_and_a_margin(tmp_path):
    """A level with a whole coarse cell of guard at R = 64 and one placed in
    fine cells: the texture keeps the core and ``CROP`` cells round it (as
    many as the array has), every byte the whole-array image has there but
    the height codes (their range is the crop's), and the record's offset
    and core still place each pixel on its face cell.  ``crop_window`` keeps
    ``3 NE`` within ``MAX_TEX`` and refuses a core that cannot fit."""
    from globe.viz import zoomtex

    rng = np.random.default_rng(5)
    for geo in (zb.Geometry(face=2, ci0=10, cj0=12, cells=2, guard=1, R=64),
                zb.Geometry(face=2, ci0=10, cj0=12, cells=1, guard=0, R=64, size=48, fi0=648, fj0=808, pad=40)):
        NE, p0, n = geo.NE, geo.p0, geo.n
        i, j = np.meshgrid(np.arange(NE), np.arange(NE), indexing="ij")
        height = ((i - NE / 3) * 1.5 + 3.0 * np.sin(j / 7.0)).astype(np.float32)
        sediment = rng.uniform(0.0, 0.4, (NE, NE)).astype(np.float32)
        surf = height + sediment
        ocean = surf < 0.0
        ws = np.maximum(surf, 0.0)
        pit = (np.abs(i - (p0 + n // 2)) <= 2) & (np.abs(j - (p0 + 3)) <= 2)
        ws[pit] = surf[pit] + 4.0
        flux = rng.uniform(0.5, 1.5, (NE, NE)).astype(np.float32)
        for col in (p0 - 2, p0 + n // 2, p0 + n + 1):
            flux[:, col] += 40.0 * np.arange(NE)
        flux[ocean] = 0.0
        a = {"height": height, "sediment": sediment, "water_surface": ws.astype(np.float32), "flux": flux, "ocean": ocean}
        full, sc_full = zoomtex.level_image(a, geo.to_dict(), crop=10 ** 6)
        img, sc = zoomtex.level_image(a, geo.to_dict())
        m = min(zoomtex.CROP, p0)
        assert (sc_full["x0"], sc_full["NE"], sc_full["p0"]) == (0, NE, p0)
        assert (sc["x0"], sc["NE"], sc["p0"], sc["n"]) == (p0 - m, n + 2 * m, m, n) and img.shape == (3 * (n + 2 * m), n + 2 * m, 4)
        x0, side = sc["x0"], sc["NE"]
        top, bottom = img[:side], img[side:2 * side]
        ftop, fbottom = full[:NE], full[NE:2 * NE]
        assert np.array_equal(top[:, :, 2:], ftop[x0:x0 + side, x0:x0 + side, 2:])        # lakes, ocean mask
        assert np.array_equal(bottom, fbottom[x0:x0 + side, x0:x0 + side])                # rivers
        assert (bottom[:, :, 0] > sc["river_min_byte"]).any()
        code = top[:, :, 0].astype(np.int64) * 256 + top[:, :, 1]
        step = (sc["h1"] - sc["h0"]) / 65535.0
        drawn = np.where((ws - surf > 0.5) & ~ocean, ws, surf)   # lakes are drawn at their water level, the sea at its bed
        assert np.abs(sc["h0"] + code * step - drawn[x0:x0 + side, x0:x0 + side].T).max() <= step * 1.01
        # the viewer's mapping: face fine cell of pixel x is ci0 R - p0 + x
        fi0 = geo.product_origin[0]
        ci0 = zoomtex._coarse(fi0, geo.R)
        assert ci0 * geo.R - sc["p0"] == geo.fine_origin[0] + x0 and isinstance(ci0, int) == (not geo.fine)
    assert zoomtex.crop_window(10000, 3000, 2600) == (2968, 2664)
    assert zoomtex.crop_window(10000, 3000, 2700) == (2985, 2730)
    assert zoomtex.crop_window(100, 10, 80) == (0, 100)
    with pytest.raises(ValueError):
        zoomtex.crop_window(10000, 3000, 2731)


def test_zoom_textures_for_every_baked_level(world, zoom):
    zooms = {z["name"]: z for z in zindex.write(world["root"], log=None)}
    z = zooms["t"]
    assert [lv["R"] for lv in z["levels"]] == [lv.R for lv in LEVELS]
    for lv, res in zip(z["levels"], zoom["levels"]):
        tex, geo = lv["tex"], res.geo
        assert (tex["face"], tex["R"], tex["ci0"], tex["cj0"], tex["cells"], tex["guard"]) == (geo.face, geo.R, geo.ci0, geo.cj0, geo.cells, geo.guard)
        assert (tex["NE"], tex["p0"], tex["n"]) == (geo.NE, geo.p0, geo.n)
        img = _decode_tex(world["root"] / "viewer" / tex["file"], "t", geo.R)
        assert img.shape == (3 * geo.NE, geo.NE, 4)
        surf = res.surface()
        step = (tex["h1"] - tex["h0"]) / 65535.0
        c = geo.p0 + geo.n // 2
        code = int(img[c, c, 0]) * 256 + int(img[c, c, 1])
        assert abs(tex["h0"] + code * step - float(surf[c, c])) <= step * 1.01
    js = (world["root"] / "viewer" / "zooms.js").read_text()
    assert all("tex" in lv for lv in json.loads(js[len("GLOBE_VIEWER.setZooms("):-3])[0]["levels"])
