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
    resolution: block-mean surface, block-max discharge, sea and lakes by
    derive's rules."""
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
    q0 = np.load(zp.out_path(planet["out"], lv.R, 0, "discharge"))
    assert np.allclose(fin["discharge"][0], q0.reshape(2 * N, k, 2 * N, k).max(axis=(1, 3)))
    h0 = np.load(zp.out_path(planet["out"], lv.R, 0, "height")) + np.load(zp.out_path(planet["out"], lv.R, 0, "sediment"))
    assert np.allclose(fin["surf"][0], h0.reshape(2 * N, k, 2 * N, k).mean(axis=(1, 3)), atol=1e-2)
    assert (fin["water"] == viewer.WATER_OCEAN).any() and (fin["ws"] >= fin["surf"] - 1e-3).all()
