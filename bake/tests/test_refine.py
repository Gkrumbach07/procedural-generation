"""Refine stage tests (PLAN.md sections 10.2 / 15).

World: the tiny preset with stub tectonics / climate / erosion and the
*real* hydro + watershed stages (cheap, and the only source of a
``basins.json`` with outlets, exits and bboxes).  Checks: divide (mask 2)
cells unchanged by a job; a basin rerun is byte-identical in-process and in
a spawned worker; the fine basin id raster is the nearest-upsampled coarse
one; the coarse-cell block means of the refined surface agree with the
upsampled coarse surface (LOD-2 agreement, no crest just inside the
divides) and per-cell changes stay within the detail cap + kernel budget;
coarse lakes survive as fine lakes and lake floors never rise above their
ceiling; the flooded water surface is >= the surface; a window crossing a
face edge upsamples without NaNs or seams; every land cell is written
exactly once; rivers continue across split outlets; the whole stage is
deterministic.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pytest
from scipy import ndimage

from globe.config import WorldParams
from globe.cubesphere import Grid
from globe.erosion import particle as pk
from globe.field import FaceField
from globe.hydro import run as hydro_run
from globe.hydro import watersheds as ws_stage
from globe.io.world_store import WorldStore
from globe.refine import basin_job as bj
from globe.refine import rasterize as rz
from globe.refine import run as refine_run
from globe.refine.upsample import Window, basin_window, detail_noise, ridged_fbm, upsample_window, window_metric
from globe.stubs import stub_climate, stub_erosion, stub_tectonics


def _log(_msg):
    pass


def _params(seed=0):
    p = WorldParams.tiny_world(seed=seed)
    p.refine.workers = 1
    return p


def _world_to_watersheds(path, params):
    store = WorldStore(path, create=True)
    store.init_manifest(params, params.coarse_grid())
    for s, fn in (("tectonics", stub_tectonics), ("climate", stub_climate), ("erosion", stub_erosion)):
        fn(store, params, _log)
        store.mark_stage(s, [], {"stub": True}, 0.0, params=params)
    hydro_run.run(store, params, _log)
    store.mark_stage("hydro", hydro_run.OUTPUTS, {}, 0.0, params=params)
    ws_stage.run(store, params, _log)
    store.mark_stage("watersheds", ws_stage.OUTPUTS, {}, 0.0, params=params)
    return store


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    params = _params()
    root = tmp_path_factory.mktemp("refine") / "tiny"
    store = _world_to_watersheds(root, params)
    t0 = time.time()
    info = refine_run.run(store, params, _log)
    seconds = time.time() - t0
    basins = refine_run.load_basins(store)
    return {"store": store, "params": params, "info": info, "basins": basins, "seconds": seconds}


def _fine(world, name, f):
    return np.asarray(rz.open_fine(world["store"].root, name, f))


def _pick(basins, pred, default=0):
    for b in basins:
        if pred(b):
            return b
    return basins[default]


# --------------------------------------------------------------------------
# one basin job
# --------------------------------------------------------------------------
def test_job_invariants(world):
    """Divide cells unchanged, water surface >= surface, basin id raster,
    every exit block inside the mask and frozen towards the outside (lakes:
    see test_lakes_consistent_with_coarse)."""
    store, params, basins = world["store"], world["params"], world["basins"]
    for b in basins[:3]:
        res = bj.run_basin(store.root, b, params)
        a = res.arrays
        m = a["mask"]
        assert m.shape == (res.win.n, res.win.n)
        frozen = m == pk.MASK_FROZEN
        assert frozen.any()
        # divide cells: exactly the plain upsample (height, sediment, discharge)
        for name, ref in (("height", "height0"), ("sediment", "sediment0"), ("discharge", "discharge0")):
            assert np.array_equal(a[name][frozen], a[ref][frozen]), name
        # no inside cell is 8-adjacent to an outside cell / the border
        inside = m == pk.MASK_ACTIVE
        grown = ndimage.binary_dilation(m == pk.MASK_OUTSIDE, structure=np.ones((3, 3), bool), border_value=True)
        assert not (grown & inside).any()
        # basin id: this id inside the mask, nearest-upsampled elsewhere
        assert (a["basin_id"][m > 0] == b["id"]).all()
        assert not (a["basin_id"][m == 0] == b["id"]).any()
        # every exit block lies inside the mask and touches the outside
        # (its downstream cell is outside the basin) through a frozen cell
        win = res.win
        R = win.R
        assert b["exits"] and b["exits"][0] == b["outlet"]
        for f, i, j in b["exits"]:
            assert f == win.face
            a0, b0 = (i - win.ci0) * R, (j - win.cj0) * R
            blk = m[a0 : a0 + R, b0 : b0 + R]
            assert blk.shape == (R, R) and (blk > 0).all() and (blk == pk.MASK_FROZEN).any()
        ex = bj.exit_cells(b, win, (win.NE, win.NE))
        assert ex.sum() == len(b["exits"]) * R * R
        # flooded water surface >= surface everywhere, finite everywhere
        surf = a["height"] + a["sediment"]
        assert (a["water_surface"] >= surf - 1e-4).all()
        for v in a.values():
            assert np.isfinite(v).all()


def test_frozen_cells_untouched_by_erosion(world):
    """Even with many iterations the frozen ring is bit-identical to the
    upsample (the kernel never writes mask-2 cells)."""
    store, params, basins = world["store"], world["params"], world["basins"]
    p = params.with_overrides(refine={"refine_iterations": 25})
    res = bj.run_basin(store.root, basins[0], p)
    a = res.arrays
    frozen = a["mask"] == pk.MASK_FROZEN
    assert np.array_equal(a["height"][frozen], a["height0"][frozen])
    assert np.array_equal(a["sediment"][frozen], a["sediment0"][frozen])
    assert res.stats["iterations"] == 25 and res.stats["particles"] > 0


def test_basin_rerun_identical_in_process_and_in_worker(world):
    """Determinism: the same basin twice in-process and once in a spawned
    worker (different numba thread count) gives byte-identical arrays."""
    store, params, basins = world["store"], world["params"], world["basins"]
    b = basins[0]
    a1 = bj.run_basin_arrays(store.root, b, params, threads=4)  # 4 threads if the process has them
    a2 = bj.run_basin_arrays(store.root, b, params)  # the window rule (1 thread on a tiny window)
    for k in a1:
        assert np.array_equal(a1[k], a2[k]) and a1[k].dtype == a2[k].dtype, k
    ctx = mp.get_context("spawn")
    with ctx.Pool(1, initializer=bj.pool_init, initargs=(1,)) as pool:
        a3 = pool.apply(bj.run_basin_arrays, (str(store.root), b, params))
    assert bj.job_threads(300) == 1 and bj.job_threads(2000, budget=8) == 8 and bj.job_threads(1000, budget=8) == 2
    for k in a1:
        assert np.array_equal(a1[k], a3[k]), k
    # a different seed changes the result (the rng is really used)
    a4 = bj.run_basin_arrays(store.root, b, _params(seed=1).with_overrides(world={"seed": 1}))
    assert not np.array_equal(a1["height"], a4["height"])


def _dist_to_outside(own: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(own, bool), 1, constant_values=False)
    return ndimage.distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1]


def test_lod2_agreement_and_cell_caps(world):
    """Block mean over R x R (one coarse cell, the LOD-2 vertex scale for R
    = 4) of refined - plain surface over a basin's active non-lake cells is
    ~0 for every whole block (the job removes the coarse-scale drift to a
    5 cm block residual), so there is no systematic step
    just inside the pinned divide either: the mean at chessboard distance
    2..3 from the divide equals the interior mean to < 1 m.  Per cell the
    change stays within the detail cap + twice the kernel's per-iteration
    budget (``iter_erode`` / ``iter_deposit`` + ``thermal_max`` per
    iteration; the drift removal shifts a cell by at most a block mean of
    such changes) — and the refined surface is not the plain one."""
    store, params, basins = world["store"], world["params"], world["basins"]
    R = params.world.R
    ep, rp = params.erosion, params.refine
    fine_cell = params.fine_cell_size_m
    budget_erode = 2 * rp.refine_iterations * fine_cell * (ep.iter_erode + ep.thermal_max)
    budget_dep = 2 * rp.refine_iterations * fine_cell * (ep.iter_deposit + ep.thermal_max)
    grid, fields, derived = bj.coarse_inputs(store.root, params)
    n_full = 0
    for b in basins[:3]:
        res = bj.run_basin(store.root, b, params)
        a = res.arrays
        n = res.win.n
        active = a["mask"] == pk.MASK_ACTIVE
        cells = active & ~a["lake"]
        surf = (a["height"] + a["sediment"]).astype(np.float64)
        plain = (a["height0"] + a["sediment0"]).astype(np.float64)
        d = surf - plain
        assert np.abs(d[active]).max() > 0.1  # something happened

        def blk(x, op):
            return op(x.reshape(n // R, R, n // R, R), axis=(1, 3))

        cnt = blk(cells.astype(np.float64), np.sum)
        dm = blk(np.where(cells, d, 0.0), np.sum) / np.maximum(cnt, 1.0)
        full = cnt == R * R
        n_full += int(full.sum())
        assert (np.abs(dm[full]) <= 0.1).all(), np.abs(dm[full]).max()
        # no crest: cells just inside the frozen ring vs the interior
        dist = _dist_to_outside(a["mask"] > 0)
        near = cells & (dist >= 2) & (dist <= 3)
        inner = cells & (dist >= 4)
        if near.sum() >= 20 and inner.sum() >= 20:
            assert abs(d[near].mean() - d[inner].mean()) < 1.0
        # per-cell caps
        up = upsample_window(fields, derived, res.win, grid)
        H = res.win.H
        cap = rp.detail_amp * up["relief"][H : H + n, H : H + n].astype(np.float64)
        assert (d[cells] <= cap[cells] + budget_dep + 1e-3).all() and (d[cells] >= -cap[cells] - budget_erode - 1e-3).all()
    assert n_full > 10


def test_lakes_consistent_with_coarse(world):
    """Coarse lakes survive refinement: at least 80 % of the coarse lake
    cells (coarse water surface - surface > lake_min_depth) have a fine lake
    cell (fine water surface - surface > lake_min_depth) in their R x R
    block of the raster, and in a job the lake cells' floor never stands
    above the ceiling ``plain fine water surface - lake_min_depth`` nor
    below the plain floor (deposits only)."""
    store, params, basins = world["store"], world["params"], world["basins"]
    R = params.world.R
    grid = params.coarse_grid()
    lake_min = float(params.hydro.lake_min_depth)
    wsc = store.load_field("water_surface", grid).interior
    sc = store.load_field("height", grid).interior + store.load_field("sediment", grid).interior
    bidc = store.load_field("basin_id", grid).interior
    lakec = (wsc - sc > lake_min) & (bidc >= 0)
    assert lakec.sum() >= 10, "the tiny world should have coarse lakes"
    kept = 0
    for f in range(6):
        depth = _fine(world, "water_surface", f) - (_fine(world, "height", f) + _fine(world, "sediment", f))
        blocks = (depth > lake_min).reshape(params.N_c, R, params.N_c, R).any(axis=(1, 3))
        kept += int((blocks & lakec[f]).sum())
    assert kept >= 0.8 * lakec.sum(), (kept, lakec.sum())
    # job-level: lake floors within [plain floor, ceiling]
    by_basin = {}
    for f, i, j in zip(*np.nonzero(lakec)):
        by_basin[int(bidc[f, i, j])] = by_basin.get(int(bidc[f, i, j]), 0) + 1
    bid = max(by_basin, key=by_basin.get)
    b = next(b for b in basins if b["id"] == bid)
    res = bj.run_basin(store.root, b, params)
    a = res.arrays
    lake = a["lake"]
    assert lake.sum() > 0 and res.stats["lakes"] > 0
    surf = a["height"] + a["sediment"]
    plain = a["height0"] + a["sediment0"]
    win = res.win
    NE = win.NE
    up_bid = np.pad(a["basin_id"], win.H, mode="edge")  # only the mask matters for the flood; rebuild ext arrays from the job's own helpers
    mask_ext = np.zeros((NE, NE), np.uint8)
    mask_ext[win.H : win.H + win.n, win.H : win.H + win.n] = a["mask"]
    plain_ext = np.zeros((NE, NE), np.float32)
    plain_ext[win.H : win.H + win.n, win.H : win.H + win.n] = plain
    ocean = np.zeros((NE, NE), bool)
    ocean[win.H : win.H + win.n, win.H : win.H + win.n] = a["basin_id"] < 0
    drain = bj.exit_cells(b, win, (NE, NE)) | ocean
    wsp = bj.local_flood(plain_ext, drain, (mask_ext > 0) | ocean, mask_ext)[win.H : win.H + win.n, win.H : win.H + win.n]
    assert ((wsp - plain)[lake] > lake_min).all()
    assert (surf[lake] <= wsp[lake] - lake_min + 1e-3).all()
    assert (surf[lake] >= plain[lake] - 1e-3).all()
    assert res.stats["lake_discarded_m"] >= 0.0
    del up_bid


def test_settle_lakes_and_block_drift_units():
    """settle_lakes spreads the excess of a lake over the same lake's room
    (sediment first) and discards what does not fit; block_drift's output
    zeroes the block means of a field over the given cells."""
    h = np.array([[10.0, 10.0, 0.0], [0.0, 0.0, 0.0]])  # two lakes: cells (0,0),(0,1) and (1,2)
    s = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    idx = np.array([0, 1, 5])
    lab = np.array([0, 0, 1])
    cap = np.array([12.0, 12.0, 1.0])
    lost = bj.settle_lakes(h.reshape(-1), s.reshape(-1), idx, lab, cap, 2)
    # lake 0: excess 3 at cell 0 (sediment first), room 2 at cell 1 -> 1 discarded; lake 1: excess 1, no room -> discarded
    assert np.allclose(h, [[10.0, 10.0, 0.0], [0.0, 0.0, 0.0]]) and np.allclose(s, [[2.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    assert lost == pytest.approx(2.0)
    rng = np.random.default_rng(3)
    R = 4
    delta = rng.normal(size=(8 * R, 8 * R)) * 5 + np.linspace(-20, 20, 8 * R)[:, None]
    cells = rng.random((8 * R, 8 * R)) > 0.3
    cells[:R, :] = False  # a row of blocks without cells
    F = bj.block_drift(delta, cells, R)
    r = np.where(cells, delta - F, 0.0).reshape(8, R, 8, R).sum(axis=(1, 3)) / np.maximum(cells.reshape(8, R, 8, R).sum(axis=(1, 3)), 1)
    assert np.abs(r).max() < 0.05 and np.isfinite(F).all()


def test_detail_noise_amplitude_and_seed():
    """Noise is bounded by detail_amp x min(slope x cell, relief) x (0.5 +
    0.5 hardness), zero-mean-ish, and depends on the rng stream."""
    win = Window(0, 4, 20, 4, 20, 2)
    NE = win.NE
    slope = np.full((NE, NE), 0.5, np.float32)
    relief = np.full((NE, NE), 40.0, np.float32)
    hard = np.linspace(0, 1, NE, dtype=np.float32)[:, None].repeat(NE, 1)
    p = WorldParams.tiny_world()
    n1 = detail_noise(win, slope, relief, hard, 0.3, 50.0, p.rng("refine", 7))
    n2 = detail_noise(win, slope, relief, hard, 0.3, 50.0, p.rng("refine", 7))
    n3 = detail_noise(win, slope, relief, hard, 0.3, 50.0, p.rng("refine", 8))
    assert np.array_equal(n1, n2) and not np.array_equal(n1, n3)
    amp = 0.3 * np.minimum(slope * 50.0, relief) * (0.5 + 0.5 * hard)
    assert (np.abs(n1) <= amp + 1e-5).all()
    assert np.abs(n1).max() > 0.5 * amp.max()
    assert np.array_equal(detail_noise(win, slope, relief, hard, 0.0, 50.0, p.rng("refine", 7)), np.zeros((NE, NE), np.float32))
    r = ridged_fbm((64, 64), 8.0, np.random.default_rng(1))
    assert abs(float(r.mean())) < 1e-6 and np.abs(r).max() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# windows across a face edge
# --------------------------------------------------------------------------
def _seam_metric_2d(a: np.ndarray, col: int) -> float:
    """PLAN 15 seam metric on a 2-D window: mean |centred difference|
    straddling column ``col`` (the face edge) over the mean interior one."""
    a = np.asarray(a, dtype=np.float64)
    g = np.abs(a[2:, :] - a[:-2, :])  # centred differences along i
    edge = g[col - 1, :].mean()  # straddles rows col-1 | col+1 around the edge at col
    inner = np.concatenate([g[: col - 3].ravel(), g[col + 3 :].ravel()]).mean()
    return float(edge / max(inner, 1e-12))


def test_window_across_face_edge_is_seamless():
    """A window straddling the u = 0 edge of face 0 (12 coarse cells
    beyond it, past the halo) upsamples a smooth function without NaNs and
    with no gradient seam (metric < 3), for scalar, cubic and vector
    fields; the metric of the window is continuous too."""
    g = Grid(32, 4)

    def fn(p):
        return 300.0 * (np.sin(3 * p[..., 0]) * p[..., 1] + p[..., 2] ** 2)

    f = FaceField.from_function(g, fn, dtype=np.float32, name="s")
    R = 2
    win = Window(0, -12, 12, 6, 30, R)  # ci0 < 0: 12 coarse cells on the neighbouring face
    a0, a1, b0, b1 = win.ext_range()
    cub = f.sample_window(0, a0, a1, b0, b1, R, order=3)
    lin = f.sample_window(0, a0, a1, b0, b1, R, order=1)
    col = (0 - a0) * R  # fine row index of the first cell on face 0
    for arr in (cub, lin):
        assert arr.shape == (win.NE, win.NE) and np.isfinite(arr).all()
        assert _seam_metric_2d(arr, col) < 3.0
        # the cubic upsample of a smooth field is close to the function itself
    from globe.cubesphere import to_sphere_v

    ui = (np.arange(a0 * R, a1 * R) + 0.5) / (g.N * R)
    U, V = np.meshgrid(ui, (np.arange(b0 * R, b1 * R) + 0.5) / (g.N * R), indexing="ij")
    exact = fn(to_sphere_v(np.zeros(U.shape, np.int64), U, V))
    assert np.abs(cub - exact).max() < 0.02 * np.ptp(exact)
    # vector field: a rigid rotation is a continuous tangent field
    from globe.field import rotation_field

    w = rotation_field(g, (0.3, -0.5, 0.8), "w")
    vec = w.sample_window(0, a0, a1, b0, b1, R, order=1)
    assert np.isfinite(vec).all()
    metric, inv = window_metric(win, g)
    assert np.isfinite(metric).all() and np.isfinite(inv).all()
    speed = np.sqrt(metric[..., 0] * vec[..., 0] ** 2 + 2 * metric[..., 1] * vec[..., 0] * vec[..., 1] + metric[..., 2] * vec[..., 1] ** 2)
    assert _seam_metric_2d(speed, col) < 3.0
    assert _seam_metric_2d(metric[..., 0], col) < 3.0


def test_edge_basin_job_runs_clean(world):
    """A basin whose bbox touches the face boundary (window past the edge
    and the halo) refines without NaNs; the parts of its window beyond the
    face are not written (rasterize clips to the face)."""
    store, params, basins = world["store"], world["params"], world["basins"]
    N = params.N_c
    b = _pick(basins, lambda b: b["bbox"][0] == 0 or b["bbox"][1] == 0 or b["bbox"][2] == N or b["bbox"][3] == N)
    win = basin_window(b, params.world.R, params.refine.halo_cells)
    assert win.ci0 < 0 or win.cj0 < 0 or win.ci1 > N or win.cj1 > N
    res = bj.run_basin(store.root, b, params)
    for v in res.arrays.values():
        assert np.isfinite(v).all()
    sl = rz.window_face_slices(win, params.N_fine)
    assert sl is not None
    fi, fj, li, lj = sl
    assert fi.start >= 0 and fj.start >= 0 and fi.stop <= params.N_fine and fj.stop <= params.N_fine
    own = res.arrays["basin_id"] == b["id"]
    assert own[li, lj].sum() == own.sum()  # all of the basin's cells lie on the face


# --------------------------------------------------------------------------
# the raster
# --------------------------------------------------------------------------
def test_fine_raster_complete_and_consistent(world):
    """Every fine cell written: basin_id == block-replicated coarse id,
    ocean = plain upsample with water_surface 0, land cells written by their
    basin exactly once (the stage's written-cell count equals the fine
    land area), hardness the upsample, water >= surface."""
    store, params, info = world["store"], world["params"], world["info"]
    R, N = params.world.R, params.N_c
    grid = params.coarse_grid()
    bid_c = store.load_field("basin_id", grid).interior
    land_fine = int((bid_c >= 0).sum()) * R * R
    assert info["cells_written"] == land_fine
    assert rz.fine_exists(store.root, params)
    grid_, fields, derived = bj.coarse_inputs(store.root, params)
    from globe.refine.upsample import upsample_face

    for f in range(6):
        bid = _fine(world, "basin_id", f)
        assert np.array_equal(bid, np.repeat(np.repeat(bid_c[f], R, 0), R, 1))
        h, s, w, q, hd = (_fine(world, n, f) for n in ("height", "sediment", "water_surface", "discharge", "hardness"))
        for a in (h, s, w, q, hd):
            assert np.isfinite(a).all()
        up = upsample_face(fields, derived, f, R, 0, N)
        ocean = bid < 0
        assert np.array_equal(h[ocean], up["height"][ocean]) and np.array_equal(s[ocean], up["sediment"][ocean])
        assert (w[ocean] == 0).all()
        assert np.array_equal(hd, up["hardness"])
        assert (w[~ocean] >= (h + s)[~ocean] - 1e-4).all()
        assert (s >= 0).all() and (q >= 0).all()


def test_feather_and_divides_in_raster(world):
    """Cells on both sides of a divide are the plain upsample (the frozen
    ring), so |surface jump| across divides is the same as in the plain
    upsample; the feather weight ramps 0 -> 1 over feather_cells."""
    store, params = world["store"], world["params"]
    R, N = params.world.R, params.N_c
    grid_, fields, derived = bj.coarse_inputs(store.root, params)
    from globe.refine.upsample import upsample_face

    for f in range(6):
        bid = _fine(world, "basin_id", f)
        surf = _fine(world, "height", f) + _fine(world, "sediment", f)
        up = upsample_face(fields, derived, f, R, 0, N)
        plain = up["height"] + up["sediment"]
        cross = np.zeros(bid.shape, bool)
        cross[1:, :] |= bid[1:, :] != bid[:-1, :]
        cross[:-1, :] |= bid[1:, :] != bid[:-1, :]
        cross[:, 1:] |= bid[:, 1:] != bid[:, :-1]
        cross[:, :-1] |= bid[:, 1:] != bid[:, :-1]
        cross &= bid >= 0
        if cross.any():
            assert np.abs(surf[cross] - plain[cross]).max() < 1e-3
    own = np.zeros((12, 12), bool)
    own[2:10, 3:11] = True
    w = rz.feather_weight(own, 2)
    assert w[2, 3] == 0 and w[3, 4] == 0.5 and w[4, 5] == 1.0 and (w[~own] == 0).all()
    big = np.zeros((24, 24), bool)
    big[2:22, 2:22] = True
    w4 = rz.feather_weight(big, 4)  # smoothstep: 0, 0.156, 0.5, 0.844, 1 at d = 1..5 (rows 2..6, mid column)
    assert np.allclose(w4[2:7, 12], [0.0, 0.15625, 0.5, 0.84375, 1.0])
    assert params.refine.feather_cells >= 2 * R
    w0 = rz.feather_weight(own, 0)
    assert w0[2, 3] == 0 and w0[3, 4] == 1.0
    own[:, :] = True  # touching the array border counts as a divide
    assert rz.feather_weight(own, 2)[0, 5] == 0


def test_rivers_continue_across_split_outlets(world):
    """At an outlet whose downstream cell is another basin's cell (a
    Pfafstetter split), the fine discharge in the downstream basin's first
    cell is comparable to the outlet's."""
    params, basins = world["params"], world["basins"]
    R = params.world.R
    pairs = [b for b in basins if b["downstream_basin"] >= 0 and b["outlet_downstream"] is not None and b["outlet_downstream"][0] == b["face"]]
    if not pairs:
        pytest.skip("no same-face split outlets in this world")
    ratios = []
    for b in pairs:
        f, i, j = b["outlet"]
        _, i2, j2 = b["outlet_downstream"]
        q = _fine(world, "discharge", f)
        qo = float(q[i * R : (i + 1) * R, j * R : (j + 1) * R].max())
        qd = float(q[i2 * R : (i2 + 1) * R, j2 * R : (j2 + 1) * R].max())
        ratios.append(min(qo, qd) / max(max(qo, qd), 1e-9))
    assert np.median(ratios) > 0.3


def test_stage_deterministic(world, tmp_path):
    """Running the stage again (fresh raster, one worker) reproduces every
    fine face byte for byte."""
    store, params = world["store"], world["params"]
    root2 = tmp_path / "again"
    store2 = _world_to_watersheds(root2, params)
    grid = params.coarse_grid()
    for n in ("height", "basin_id", "water_surface"):
        assert np.array_equal(store.load_field(n, grid).interior, store2.load_field(n, grid).interior)
    refine_run.run(store2, params, _log)
    for name in rz.FINE_FIELDS:
        for f in range(6):
            a = np.asarray(rz.open_fine(store.root, name, f))
            b = np.asarray(rz.open_fine(root2, name, f))
            assert np.array_equal(a, b), (name, f)
    assert store.hash_outputs(refine_run.OUTPUTS) == store2.hash_outputs(refine_run.OUTPUTS)


def test_quicklooks_and_runtime(world, tmp_path):
    store, params, info = world["store"], world["params"], world["info"]
    out = refine_run.quicklook(store, params, tmp_path / "refine.png")
    assert out is not None and out.exists()
    assert len(info["quicklooks"]) == min(refine_run.N_BASIN_QUICKLOOKS, info["n_basins"])
    assert info["n_basins"] == len(world["basins"]) and info["deaths"]["exit"] > 0
    assert world["seconds"] < 60.0
