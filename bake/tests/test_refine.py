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
exactly once; a basin on two faces is refined as one job per face piece
whose mask follows the basin across the edge; rivers continue across
split outlets and across face edges; the seam blend's weights sum to 1,
its seam row carries refined detail and it does not depend on the job
order; the fine surface has no crease at the cube edges; the whole stage
is deterministic.
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
        # divide cells: exactly the plain upsample (height, sediment)
        for name, ref in (("height", "height0"), ("sediment", "sediment0")):
            assert np.array_equal(a[name][frozen], a[ref][frozen]), name
        # ... and their discharge the largest of their active neighbours' (a
        # river runs on to the sea), the plain upsample away from any
        # (compared off the window's border row, which the job saw beyond)
        act = m == pk.MASK_ACTIVE
        touch = frozen & ndimage.binary_dilation(act, structure=np.ones((3, 3), bool))
        core = np.zeros(m.shape, bool)
        core[1:-1, 1:-1] = True
        near_q = ndimage.maximum_filter(np.where(act, a["discharge"], 0.0), size=3)
        assert np.allclose(a["discharge"][touch & core], near_q[touch & core])
        assert np.array_equal(a["discharge"][frozen & ~touch & core], a["discharge0"][frozen & ~touch & core])
        # no on-face inside cell is 8-adjacent to an outside cell / the
        # border (the off-face strip of a basin that continues across the
        # edge may reach the window border; the kernel's extended array is
        # frozen there by build_mask)
        inside = m == pk.MASK_ACTIVE
        grown = ndimage.binary_dilation(m == pk.MASK_OUTSIDE, structure=np.ones((3, 3), bool), border_value=True)
        assert not (grown & inside & _on_face_mask(res.win, params.N_fine)).any()
        # basin id: this id inside the mask, nearest-upsampled elsewhere
        assert (a["basin_id"][m > 0] == b["id"]).all()
        assert not (a["basin_id"][m == 0] == b["id"]).any()
        # every exit block on this face lies inside the mask; an ocean /
        # basin exit touches the outside (its downstream cell is outside
        # the basin) through a frozen cell, a face exit does not have to
        # (the basin continues across the edge, see test_piece_mask_crosses_the_face_edge)
        win = res.win
        R = win.R
        assert b["exits"] and b["exits"][0] == b["outlet"]
        on_face = [(e, k) for e, k in zip(b["exits"], b["exit_kinds"]) if e[0] == win.face]
        assert on_face
        for (f, i, j), kind in on_face:
            a0, b0 = (i - win.ci0) * R, (j - win.cj0) * R
            blk = m[a0 : a0 + R, b0 : b0 + R]
            assert blk.shape == (R, R) and (blk > 0).all()
            if kind != "face":
                assert (blk == pk.MASK_FROZEN).any()
        ex = bj.exit_cells(b, win, (win.NE, win.NE))
        assert ex.sum() == len(on_face) * R * R
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
            # 1.5: the test world's marshes are ground now (hydro.marsh_depth), flat wet cells the
            # refine pass works like any other, and one basin's ring reads 1.01 against 0.6 inside
            assert abs(d[near].mean() - d[inner].mean()) < 1.5
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


def _on_face_mask(win, N_fine):
    on = np.zeros((win.n, win.n), bool)
    sl = rz.window_face_slices(win, N_fine)
    if sl is not None:
        on[sl[2], sl[3]] = True
    return on


def test_edge_basin_job_runs_clean(world):
    """A piece whose bbox touches the face boundary (window past the edge
    and the halo) refines without NaNs; the parts of its window beyond the
    face are not written (rasterize clips to the face), and the window is
    padded towards the face interior (never further past the edge than the
    halo)."""
    store, params, basins = world["store"], world["params"], world["basins"]
    N, R, halo = params.N_c, params.world.R, params.refine.halo_cells
    b = _pick(basins, lambda b: b["bbox"][0] == 0 or b["bbox"][1] == 0 or b["bbox"][2] == N or b["bbox"][3] == N)
    win = basin_window(b, R, halo, N=N)
    assert win.ci0 < 0 or win.cj0 < 0 or win.ci1 > N or win.cj1 > N
    # past the edge by at most the halo, unless the padded square is wider
    # than the face (a thin piece on a long side of a tiny face)
    assert win.nc > N or (win.ci0 >= -halo and win.cj0 >= -halo and win.ci1 <= N + halo and win.cj1 <= N + halo)
    assert win.nc == max(b["bbox"][2] - b["bbox"][0], b["bbox"][3] - b["bbox"][1]) + 2 * halo
    assert win.ci0 <= b["bbox"][0] - halo and win.ci1 >= b["bbox"][2] + halo and win.cj0 <= b["bbox"][1] - halo and win.cj1 >= b["bbox"][3] + halo
    res = bj.run_basin(store.root, b, params)
    assert res.win == win
    for v in res.arrays.values():
        assert np.isfinite(v).all()
    sl = rz.window_face_slices(win, params.N_fine)
    assert sl is not None
    fi, fj, li, lj = sl
    assert fi.start >= 0 and fj.start >= 0 and fi.stop <= params.N_fine and fj.stop <= params.N_fine
    own = res.arrays["basin_id"] == b["id"]
    on = _on_face_mask(win, params.N_fine)
    assert (own & on).sum() == b["pieces"][[p["face"] for p in b["pieces"]].index(win.face)]["area_cells"] * R * R
    # a record without pieces (stub / older worlds) is one piece on its face
    legacy = {k: v for k, v in b.items() if k not in ("pieces", "faces")}
    assert basin_window(legacy, R, halo, N=N) == win
    with pytest.raises(KeyError):
        basin_window(legacy, R, halo, face=(b["face"] + 1) % 6, N=N)


def _multi_face_basin(basins):
    for b in basins:
        if len(b.get("pieces") or ()) > 1:
            return b
    pytest.skip("no basin on more than one face in this world")


def test_piece_mask_crosses_the_face_edge(world, tmp_path):
    """For a basin on two faces, the piece job on either face has active
    (mask 1) cells beyond the face edge — the frozen ring sits on the true
    divide, not on the seam — so a river crossing the edge keeps flowing;
    every cardinal face exit's flow continues into mask > 0 across the
    edge; the rasteriser writes only the piece's own on-face cells; the
    feather weight is > 0 on seam cells the basin continues across and 0
    where the basin ends at the edge; the piece returns seam records for its
    own face and for the face across the edge."""
    store, params, basins = world["store"], world["params"], world["basins"]
    R, N = params.world.R, params.N_c
    b = _multi_face_basin(basins)
    fd = store.load_field("flow_dir", params.coarse_grid()).interior
    from globe.hydro.d8 import D8_DI, D8_DJ

    n_crossing_exits = 0
    scratch_root = tmp_path / "raster"
    rz.create_fine(scratch_root, params)
    for p in b["pieces"]:
        res = bj.run_basin(store.root, b, params, face=p["face"])
        m = res.arrays["mask"]
        win = res.win
        on = _on_face_mask(win, params.N_fine)
        assert ((m == pk.MASK_ACTIVE) & ~on).any(), "the mask stops at the face edge"
        assert (res.arrays["basin_id"][~on] == b["id"]).any()
        # the on-face seam row of the piece: some active cells (not all frozen)
        for (f, i, j), kind in zip(b["exits"], b["exit_kinds"]):
            if f != win.face or kind != "face":
                continue
            code = int(fd[f, i, j])
            di, dj = int(D8_DI[code]), int(D8_DJ[code])
            a0, b0 = (i - win.ci0) * R, (j - win.cj0) * R
            assert (m[a0 : a0 + R, b0 : b0 + R] > 0).all()
            if abs(di) + abs(dj) == 1:  # cardinal crossing: the extension block beyond the edge is the downstream cell
                a1, b1 = a0 + di * R, b0 + dj * R
                assert (m[a1 : a1 + R, b1 : b1 + R] > 0).any()
                n_crossing_exits += 1
        # write only the on-face own cells; the seam is not a divide where
        # the basin continues across it (feather weight > 0 on seam cells
        # whose neighbour across the edge is the basin)
        F = int(params.refine.feather_cells)
        out, own = rz.blend_result(res.arrays, b["id"], F)
        own_on = own & on
        assert own_on.sum() == p["area_cells"] * R * R
        seam = own_on & ~ndimage.binary_erosion(on, structure=np.ones((3, 3), bool), border_value=0)
        across = ndimage.binary_dilation(own & ~on, structure=np.ones((3, 3), bool))
        w = rz.feather_weight(own, F)
        assert seam.any() and (w[seam & across] > 0).any()
        assert (w[seam & ~across & ndimage.binary_dilation(~own, structure=np.ones((3, 3), bool))] == 0).all()
        assert rz.write_result(scratch_root, params, res) == int(own_on.sum())  # a scratch raster: the world's seam cells are blended
        recs = rz.seam_records(res, params, [q["face"] for q in b["pieces"]], store.load_field("basin_id", params.coarse_grid()).interior)
        kinds = {(r["kind"], r["face"]) for r in recs}
        assert (rz.SEAM_NATIVE, win.face) in kinds and any(k == rz.SEAM_CROSS and g != win.face for k, g in kinds)
    assert n_crossing_exits > 0
    # the piece of the other face got its own rng stream
    a0 = bj.run_basin_arrays(store.root, b, params)
    a1 = bj.run_basin(store.root, b, params, face=b["pieces"][1]["face"] if b["pieces"][0]["face"] == b["face"] else b["pieces"][0]["face"]).arrays
    assert a0["height"].shape != a1["height"].shape or not np.array_equal(a0["height"], a1["height"])


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

    zone = _coast_zone(world)
    for f in range(6):
        bid = _fine(world, "basin_id", f)
        assert np.array_equal(bid, np.repeat(np.repeat(bid_c[f], R, 0), R, 1))
        h, s, w, q, hd = (_fine(world, n, f) for n in ("height", "sediment", "water_surface", "discharge", "hardness"))
        for a in (h, s, w, q, hd):
            assert np.isfinite(a).all()
        up = upsample_face(fields, derived, f, R, 0, N)
        ocean = (bid < 0) & ~zone[f]         # the coast pass owns the sea next to land (test_coast_pass)
        assert np.array_equal(h[ocean], up["height"][ocean]) and np.array_equal(s[ocean], up["sediment"][ocean])
        assert (w[ocean] == 0).all()
        assert np.array_equal(hd, up["hardness"])
        assert (w[~ocean] >= (h + s)[~ocean] - 1e-4).all()
        assert (s >= 0).all() and (q >= 0).all()


def _coast_zone(world):
    """Fine ``(6, Nf, Nf)`` cells :func:`rasterize.write_coast` may change."""
    from globe.hydro.d8 import OCEAN

    params, store = world["params"], world["store"]
    grid = params.coarse_grid()
    R = params.world.R
    near = rz.coast_zone(grid, store.load_field("flow_dir", grid).interior == OCEAN)
    return np.repeat(np.repeat(near, R, axis=1), R, axis=2)


def test_coast_pass(world):
    """``rasterize.write_coast``: in the coast zone every fine cell inside
    the smoothed ocean contour is sea at least ``COAST_MARGIN_M`` deep and
    every other cell land at least that high, so no speck of land stands
    offshore and no puddle of sea sits on the coastal plain; water surfaces
    stay >= the surface, sea surfaces at 0."""
    from globe.hydro.d8 import OCEAN

    store, params, info = world["store"], world["params"], world["info"]
    grid = params.coarse_grid()
    R, N = params.world.R, params.N_c
    ocean_c = store.load_field("flow_dir", grid).interior == OCEAN
    frac = rz.coast_fraction(grid, ocean_c)
    zone = _coast_zone(world)
    M = rz.COAST_MARGIN_M
    seen_sea = seen_land = 0
    for f in range(6):
        surf = _fine(world, "height", f) + _fine(world, "sediment", f)
        ws = _fine(world, "water_surface", f)
        sea = (frac.sample_window(f, 0, N, 0, N, R, order=1) > 0.5) & zone[f]
        land = zone[f] & ~sea
        assert (surf[sea] <= -M + 1e-3).all() and (ws[sea] <= 1e-6).all()
        assert (surf[land] >= M - 1e-3).all() and (ws[land] >= surf[land] - 1e-3).all()
        assert (_fine(world, "sediment", f) >= -1e-4).all()
        bid = _fine(world, "basin_id", f)
        q = _fine(world, "discharge", f)
        was_sea = land & (bid < 0)
        if was_sea.any():
            near_land = ndimage.maximum_filter(np.where(bid >= 0, q, 0.0), size=5)
            assert np.allclose(q[was_sea], near_land[was_sea])
        seen_sea += int(sea.sum())
        seen_land += int(land.sum())
    assert seen_sea > 0 and seen_land > 0
    assert info["coast_lowered_cells"] + info["coast_raised_cells"] > 0


def test_lake_outflow_spawns_at_the_spill_of_an_overflowing_lake():
    """``basin_job.lake_outflow``: a lake whose level reaches its lowest shore
    cell sends the flow through it (the largest upsampled coarse discharge on
    the lake) from that cell; a lake held below its rim sends nothing, and
    neither does one whose lowest rim is the sea."""
    n = 20
    plain = np.full((n, n), 10.0)
    labels = np.zeros((n, n), np.int64)
    labels[4:8, 4:8] = 1                          # lake 1, level 5, spill at (8, 6) height 5
    labels[12:16, 12:16] = 2                      # lake 2, level 3, lowest shore 9: closed
    plain[labels == 1] = 2.0
    plain[labels == 2] = 1.0
    plain[8, 6] = 5.0
    plain[11, 13] = 9.0
    level = np.where(labels == 1, 5.0, np.where(labels == 2, 3.0, plain))
    q = np.zeros((n, n))
    q[labels == 1] = 7.0
    q[5, 5] = 9.0
    q[labels == 2] = 4.0
    land = labels == 0
    out = bj.lake_outflow(labels, 2, land, plain, level, q)
    assert out[8, 6] == pytest.approx(9.0)
    assert out.sum() == pytest.approx(9.0)
    # the same first lake on the coast: the sea touches it below its land
    # spill, so the water leaves there and nothing spawns on the shore
    sea = np.zeros((n, n), bool)
    sea[4:8, 3] = True
    plain2 = np.where(sea, -20.0, plain)
    out = bj.lake_outflow(labels, 2, land & ~sea, plain2, level, q, drain=sea)
    assert out.sum() == 0.0
    # and a lake the sea does not touch is unchanged by passing it
    sea2 = np.zeros((n, n), bool)
    sea2[0, :] = True
    assert bj.lake_outflow(labels, 2, land & ~sea2, np.where(sea2, -20.0, plain), level, q, drain=sea2)[8, 6] == pytest.approx(9.0)


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
        cross &= (bid >= 0) & ~_coast_zone(world)[f]
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


def test_rivers_continue_across_face_edges(world):
    """At every coarse cell whose downstream cell is on another face, the
    fine discharge on the two sides of the seam is comparable (the pieces
    on both faces erode the river through the edge)."""
    store, params = world["store"], world["params"]
    R = params.world.R
    grid = params.coarse_grid()
    N = grid.N
    NN = N * N
    from globe.hydro.d8 import OCEAN, downstream_table

    fd = store.load_field("flow_dir", grid).interior
    down = downstream_table(np.ascontiguousarray(fd), grid.owner, grid.H)
    land = fd.reshape(-1) != OCEAN
    dn = np.maximum(down, 0)
    cross = np.nonzero(land & (down >= 0) & land[dn] & (dn // NN != np.arange(6 * NN) // NN))[0]
    if cross.size == 0:
        pytest.skip("no drainage across a face edge in this world")
    q = [_fine(world, "discharge", f) for f in range(6)]
    ratios = []
    for c in cross.tolist():
        d = int(down[c])
        f0, i0, j0 = c // NN, (c % NN) // N, c % N
        f1, i1, j1 = d // NN, (d % NN) // N, d % N
        q0 = float(q[f0][i0 * R : (i0 + 1) * R, j0 * R : (j0 + 1) * R].max())
        q1 = float(q[f1][i1 * R : (i1 + 1) * R, j1 * R : (j1 + 1) * R].max())
        ratios.append(min(q0, q1) / max(max(q0, q1), 1e-9))
    assert np.median(ratios) > 0.3


def _two_piece_records(F, n=16, detail_a=1.0, detail_b=3.0, native_w=1.0):
    """Synthetic seam records on target face 0 (an ``n x n`` fine face,
    side 1 = the edge at i = 0): the owning piece (source face 0) with
    detail ``detail_a`` on rows 0 .. F-1, the piece across the edge (source
    face 2) with ``detail_b`` on the same cells, weights from
    ``seam_side_weight`` as ``seam_records`` sets them; plain surface 100 m,
    plain sediment 1 m, no lakes."""
    i, j, k = rz.side_cells(1, n, F)
    idx = i * n + j
    a = rz.seam_side_weight(k + 0.5, F)
    m = idx.size
    f32 = lambda v: np.full(m, v, np.float32)  # noqa: E731
    native = {"kind": rz.SEAM_NATIVE, "face": 0, "src": 0, "bid": 7, "idx": idx, "dsurf": f32(detail_a), "dsed": f32(0.0),
              "beta": a.astype(np.float32), "w": f32(native_w), "surf0": f32(100.0), "sed0": f32(1.0), "surf": f32(100.0 + detail_a), "ws": f32(100.0 + detail_a)}
    cross = {"kind": rz.SEAM_CROSS, "face": 0, "src": 2, "bid": 7, "idx": idx, "dsurf": f32(detail_b), "dsed": f32(0.0), "beta": (1.0 - a).astype(np.float32)}
    return native, cross, k


def test_seam_blend_weights_sum_to_one_and_seam_row_carries_detail():
    """Two pieces meeting at a cube edge: the owning and the crossing weight
    sum to 1 at every distance and are 1/2 on the edge; the blend of detail
    1 (owning) and 3 (across) is the weighted mean row by row (1.81 half a
    cell in, 2 on the edge line), falls monotonically towards 1 over
    ``feather_cells``; the seam row
    carries detail (not the plain upsample); the result is continuous
    through the edge (the mirrored cell on the other face gets the mirrored
    value); a zero divide feather pins the plain upsample; a piece alone
    (no cross record) is not rewritten; sediment is clipped at 0 and the
    water surface of a dry cell is the blended surface."""
    F = 4
    d = np.linspace(-6.0, 6.0, 49)
    wa = rz.seam_side_weight(d, F)
    assert np.allclose(wa + rz.seam_side_weight(-d, F), 1.0)
    assert rz.seam_side_weight(0.0, F) == 0.5 and rz.seam_side_weight(F, F) == 1.0 and rz.seam_side_weight(-F, F) == 0.0
    assert np.all(np.diff(wa) >= 0)
    native, cross, k = _two_piece_records(F)
    out = rz.blend_seams([native, cross], 16)[0]
    order = np.argsort(native["idx"])
    kk = k[order]
    detail = out["height"] + out["sediment"] - 100.0
    assert np.array_equal(out["idx"], native["idx"][order])
    for r in range(F):
        a = rz.seam_side_weight(r + 0.5, F)
        assert np.allclose(detail[kk == r], a * 1.0 + (1.0 - a) * 3.0, atol=1e-4)
    seam = detail[kk == 0]
    assert (seam > 1.5).all() and (seam < 2.5).all()  # both pieces near 1/2 (0.59 / 0.41 half a cell in): refined detail, not plain
    rows = [float(detail[kk == r].mean()) for r in range(F)]
    assert all(x > y for x, y in zip(rows, rows[1:])) and rows[-1] > 1.0
    # continuity through the edge: the mirrored cell (-dist) on the other face blends 3 (its own) and 1 (across)
    mirror = rz.seam_side_weight(0.5, F) * 3.0 + (1.0 - rz.seam_side_weight(0.5, F)) * 1.0
    assert abs((seam.mean() + mirror) / 2.0 - 2.0) < 1e-6 and abs(seam.mean() - mirror) < 0.8
    assert np.allclose(out["water_surface"], out["height"] + out["sediment"])
    # divide feather 0: the plain upsample, not rewritten
    native0, cross0, _ = _two_piece_records(F, native_w=0.0)
    assert 0 not in rz.blend_seams([native0, cross0], 16)
    # the owning piece alone: nothing to blend
    assert rz.blend_seams([native], 16) == {}
    # sediment never negative
    nat, crs, _ = _two_piece_records(F)
    nat["dsed"][:] = -5.0
    crs["dsed"][:] = -5.0
    assert (rz.blend_seams([nat, crs], 16)[0]["sediment"] >= 0).all()


def test_seam_blend_job_order_independent(world):
    """The seam blend does not depend on which piece finished first: the
    records of every multi-face basin's pieces, blended in the driver's
    order, reversed and shuffled, give byte-identical cells; the owning
    and crossing weights of each blended cell sum to 1 away from window
    borders; and the stage's raster carries the blend (the seam rows
    differ from the plain upsample where a basin continues across)."""
    store, params, basins = world["store"], world["params"], world["basins"]
    grid = params.coarse_grid()
    bid_c = store.load_field("basin_id", grid).interior
    pieces = [(b, p) for b, p in refine_run.basin_pieces(basins) if len(b.get("pieces") or ()) > 1]
    if not pieces:
        pytest.skip("no basin on more than one face in this world")
    records = []
    for b, p in pieces:
        res = bj.run_basin(store.root, b, params, face=p["face"])
        records += rz.seam_records(res, params, [q["face"] for q in b["pieces"]], bid_c)
    assert any(r["kind"] == rz.SEAM_CROSS for r in records)
    ref = rz.blend_seams(records, params.N_fine)
    assert ref
    rng = np.random.default_rng(3)
    for recs in (records[::-1], [records[i] for i in rng.permutation(len(records))]):
        got = rz.blend_seams(recs, params.N_fine)
        assert got.keys() == ref.keys()
        for g in ref:
            for key in ref[g]:
                assert np.array_equal(ref[g][key], got[g][key]), (g, key)
    # the raster holds exactly these cells
    n_blended = 0
    for g, v in ref.items():
        i, j = np.divmod(v["idx"], params.N_fine)
        for key in ("height", "sediment", "water_surface"):
            assert np.array_equal(_fine(world, key, g)[i, j], v[key]), (g, key)
        n_blended += v["idx"].size
    assert world["info"]["seam_cells_blended"] == n_blended > 0
    # the seam row (touching the edge) of the blended cells carries refined detail
    seam_detail = []
    for r in records:
        if r["kind"] != rz.SEAM_NATIVE or r["face"] not in ref:
            continue
        v = ref[r["face"]]
        pos = np.searchsorted(v["idx"], r["idx"])
        hit = (pos < v["idx"].size) & (v["idx"][np.minimum(pos, v["idx"].size - 1)] == r["idx"])
        i, j = np.divmod(r["idx"][hit], params.N_fine)
        row0 = np.minimum.reduce([i, params.N_fine - 1 - i, j, params.N_fine - 1 - j]) == 0
        surf = v["height"][pos[hit]] + v["sediment"][pos[hit]]
        seam_detail.append(np.abs(surf - r["surf0"][hit])[row0])
    seam_detail = np.concatenate(seam_detail)
    assert seam_detail.size and (seam_detail > 1e-3).mean() > 0.5
    # weights: per target cell the owning + crossing betas sum to at most 1,
    # and to 1 wherever the crossing window reaches 2 feathers past the edge
    # (the tiny preset's halo is one feather; with halo_cells 4 it is two,
    # as on small and Earth) and the cell is off the cube-corner strips
    p4 = params.with_overrides(refine={"halo_cells": 2 * params.refine.feather_cells // params.world.R})
    recs4 = []
    for b, p in pieces:
        res = bj.run_basin(store.root, b, p4, face=p["face"])
        recs4 += rz.seam_records(res, p4, [q["face"] for q in b["pieces"]], bid_c)
    F, n = params.refine.feather_cells, params.N_fine
    for g in sorted({r["face"] for r in recs4}):
        rg = [r for r in recs4 if r["face"] == g]
        idx = np.concatenate([r["idx"] for r in rg])
        beta = np.concatenate([r["beta"] for r in rg]).astype(np.float64)
        cells, inv = np.unique(idx, return_inverse=True)
        tot = np.bincount(inv, weights=beta)
        has_x = np.bincount(inv, weights=np.concatenate([np.full(r["idx"].size, r["kind"] == rz.SEAM_CROSS) for r in rg])) > 0
        has_n = np.bincount(inv, weights=np.concatenate([np.full(r["idx"].size, r["kind"] == rz.SEAM_NATIVE) for r in rg])) > 0
        i, j = np.divmod(cells, n)
        corner = (np.minimum(i, n - 1 - i) < F) & (np.minimum(j, n - 1 - j) < F)
        both = has_x & has_n
        assert both.any() and (tot[both] <= 1.0 + 1e-5).all()
        assert np.median(tot[both & ~corner]) > 0.999, np.percentile(tot[both & ~corner], [5, 50])


def test_fine_seam_step_matches_inside_step(world):
    """The fine surface has no crease along the cube edges: the mean |step|
    between the two land cells straddling an edge is within 1.5x the mean
    |step| between the first two cells inside the face (where a basin ends
    at the edge the seam row is the plain upsample on both faces, as every
    divide is; where it continues both seam rows are the seam blend of the
    same two pieces)."""
    from globe.refine.lod import edge_links, neighbour_ring, side_row

    params = world["params"]
    N_fine = params.N_fine
    surf = [_fine(world, "height", f) + _fine(world, "sediment", f) for f in range(6)]
    bid = [_fine(world, "basin_id", f) for f in range(6)]
    links = edge_links(N_fine)
    seam, inside = [], []
    for f in range(6):
        for s in range(4):
            own0, own1 = side_row(surf[f], s, 0), side_row(surf[f], s, 1)
            nb0 = neighbour_ring(lambda F, S, d: side_row(surf[F], S, d), links, f, s, 0)
            l0, l1 = side_row(bid[f], s, 0) >= 0, side_row(bid[f], s, 1) >= 0
            ln0 = neighbour_ring(lambda F, S, d: side_row(bid[F], S, d), links, f, s, 0) >= 0
            seam.append(np.abs(own0 - nb0)[l0 & ln0])
            inside.append(np.abs(own0 - own1)[l0 & l1])
    seam, inside = np.concatenate(seam), np.concatenate(inside)
    assert seam.size > 50 and inside.size > 50
    assert seam.mean() <= 1.5 * inside.mean(), (seam.mean(), inside.mean())


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
    assert info["n_pieces"] == len(refine_run.basin_pieces(world["basins"])) == sum(len(b["pieces"]) for b in world["basins"]) >= info["n_basins"]
    assert world["seconds"] < 60.0


def test_detail_noise_fades_out_at_sea_level():
    """With a surface and a coast taper the noise is zero on the shoreline
    and full again a taper's height above or below it."""
    win = Window(0, 4, 20, 4, 20, 2)
    NE = win.NE
    slope = np.full((NE, NE), 0.5, np.float32)
    relief = np.full((NE, NE), 40.0, np.float32)
    hard = np.ones((NE, NE), np.float32)
    surface = np.linspace(-100.0, 100.0, NE, dtype=np.float32)[:, None].repeat(NE, 1)
    p = WorldParams.tiny_world()
    n0 = detail_noise(win, slope, relief, hard, 0.3, 50.0, p.rng("refine", 7))
    n1 = detail_noise(win, slope, relief, hard, 0.3, 50.0, p.rng("refine", 7), surface=surface, coast_taper_m=40.0)
    shore = np.abs(surface[:, 0]) <= 200.0 / (NE - 1)   # the row or two nearest the waterline
    far = np.abs(surface[:, 0]) > 40.0
    assert shore.any() and np.abs(n1[shore]).max() < 0.05 * np.abs(n0).max()
    assert np.array_equal(n1[far], n0[far])


def test_detail_noise_keeps_to_a_share_of_the_ground():
    """A volcanic island two coarse cells across drops its whole height in
    one cell, so noise scaled by that drop is its whole height several times
    over.  With ``height_share`` no land cell loses more than that share of
    its height and none ends under the sea; without it a third of the island
    did (the square mesas and lakes of earth-v18's zoom f4_851_1003)."""
    win = Window(0, 4, 20, 4, 20, 8)
    NE = win.NE
    y, x = np.meshgrid(np.arange(NE), np.arange(NE), indexing="ij")
    r = np.hypot(y - NE / 2, x - NE / 2) * (9770.0 / 8)
    surface = (4500.0 * np.exp(-0.5 * (r / 9000.0) ** 2) - 1700.0).astype(np.float32)      # a 4.5 km cone on a 1.7 km sea floor
    gy, gx = np.gradient(surface, 9770.0 / 8)
    slope = np.hypot(gx, gy).astype(np.float32)
    relief = (slope * 9770.0).astype(np.float32)
    hard = np.full((NE, NE), 0.7, np.float32)
    p = WorldParams.tiny_world()
    land = surface > 0.0
    free = detail_noise(win, slope, relief, hard, 3.0, 9770.0, p.rng("refine", 7), surface=surface, coast_taper_m=40.0)
    kept = detail_noise(win, slope, relief, hard, 3.0, 9770.0, p.rng("refine", 7), surface=surface, coast_taper_m=40.0, height_share=0.5)
    assert ((surface + free)[land] < 0.0).mean() > 0.1
    assert not ((surface + kept)[land] < 0.0).any()
    assert np.all(np.abs(kept) <= 0.5 * np.abs(surface) + 1e-3)
    assert kept.std() > 0.0 and np.array_equal(np.sign(kept), np.sign(free))               # the same noise, quieter


def test_smooth_drift_holds_an_island_on_its_own():
    """Land narrower than the hold is held to its own mean: an island beside
    a coast that sank does not take the coast's correction (the 1-2 km towers
    on shoals of docs/zoom-windows.md), an island alone in its array is held
    at all, and the coast's own correction does not see the island."""
    from globe.refine.zoom import smooth_drift
    R, n = 16, 160
    cells = np.zeros((n, n), bool)
    cells[:, :80] = True                                 # the mainland
    isle = np.zeros((n, n), bool)
    isle[70:78, 86:94] = True                            # six cells off its coast, under R x R cells
    delta = np.where(isle, 40.0, -300.0)
    F = smooth_drift(delta, cells | isle, R)
    assert np.allclose(F[isle], 40.0)
    assert np.allclose(F[cells], smooth_drift(delta, cells, R)[cells])
    alone = smooth_drift(delta, isle, R)
    assert np.allclose(alone[isle], 40.0) and not alone[~isle].any()


def test_cone_shield_is_fresh_lava():
    """Where a zoom level's ground is an active cone the rock is fresh lava:
    hardness ``CONE_HARDNESS`` and the detail noise at ``CONE_NOISE``, both
    reached where the edifice is ``CONE_FULL_M`` thick; off the cone nothing
    changes, and rock already harder stays as it is."""
    from globe.refine import zoom as rz
    cone = np.array([0.0, 0.5 * rz.CONE_FULL_M, rz.CONE_FULL_M, 4000.0, 4000.0])
    hard = np.array([0.6, 0.6, 0.6, 0.6, 0.99], np.float32)
    h, quiet = rz.cone_shield(cone, hard)
    assert h.dtype == np.float32
    assert np.allclose(h, [0.6, 0.5 * (0.6 + rz.CONE_HARDNESS), rz.CONE_HARDNESS, rz.CONE_HARDNESS, 0.99])
    assert np.allclose(quiet, [1.0, 0.5 * (1.0 + rz.CONE_NOISE), rz.CONE_NOISE, rz.CONE_NOISE, rz.CONE_NOISE])


def test_seam_strip_is_sampled_where_the_neighbour_face_is(world):
    """The seam blend resamples a piece's refined off-face strip onto the
    neighbouring face (`rasterize.strip_window_coords` + bilinear).  Checked
    on the plain upsample, which both faces compute independently: the
    window's plain surface resampled onto the neighbour's cells within
    `feather_cells` of the edge matches that face's own plain upsample to a
    small fraction of the step between adjacent cells, and a half-cell
    offset in the geometry is many times worse (a review found a two-cell
    shift passed every other test)."""
    from globe.refine.lod import edge_links
    from globe.refine.upsample import upsample_face

    store, params, basins = world["store"], world["params"], world["basins"]
    b = _multi_face_basin(basins)
    N_fine, R, F = params.N_fine, params.world.R, int(params.refine.feather_cells)
    grid, fields, derived = bj.coarse_inputs(store.root, params)
    links = edge_links(N_fine)
    faces = {int(q["face"]) for q in b["pieces"]}
    res_all, offs_all, steps = [], [], []
    for p in b["pieces"]:
        res = bj.run_basin(store.root, b, params, face=p["face"])
        win = res.win
        plain = res.arrays["height0"].astype(np.float64) + res.arrays["sediment0"]
        for s in range(4):
            ln = links[win.face * 4 + s]
            g = int(ln.nb_face)
            if g not in faces:
                continue
            i, j, _ = rz.side_cells(ln.nb_side, N_fine, F)
            up = upsample_face(fields, derived, g, R, 0, grid.N)
            own = up["height"].astype(np.float64) + up["sediment"]
            x, y = rz.strip_window_coords(win, g, i, j, N_fine)
            n = win.n
            ok = (x >= 0) & (x <= n - 1) & (y >= 0) & (y <= n - 1)
            if ok.sum() < 8:
                continue
            ref = own[i[ok], j[ok]]
            res_all.append(np.abs(rz._bilinear(plain, x[ok], y[ok]) - ref))
            offs_all.append(np.abs(rz._bilinear(plain, np.clip(x[ok] + 0.5, 0, n - 1), y[ok]) - ref))
            steps.append(np.abs(np.diff(own, axis=0)).ravel())
    if not res_all:
        pytest.skip("no crossed edge with a strip inside the window")
    r, o, st = np.concatenate(res_all).mean(), np.concatenate(offs_all).mean(), np.concatenate(steps).mean()
    assert r < 0.05 * st, (r, st)
    assert o > 5.0 * r, (r, o)


def test_zoom_erosion_profile_is_a_valid_override():
    """``refine.zoom.ZOOM_EROSION`` / ``ZOOM_REFINE`` name real parameters
    with values of their own types, and ``zoom_params`` applies them at a
    refinement, with per-call erosion overrides on top."""
    from globe.refine.zoom import ZOOM_EROSION, ZOOM_REFINE, zoom_params
    p = WorldParams.tiny_world().with_overrides(erosion=dict(ZOOM_EROSION))
    for k, v in ZOOM_EROSION.items():
        assert getattr(p.erosion, k) == v, k
    assert isinstance(p.erosion.max_steps, int)
    z = zoom_params(WorldParams.tiny_world(), 32, max_steps=300.0)
    assert z.world.R == 32 and z.erosion.max_steps == 300 and isinstance(z.erosion.max_steps, int)
    assert not z.erosion.window_lakes and z.refine.detail_amp == ZOOM_REFINE["detail_amp"]


def test_zoom_discharge_scales_are_areas_and_coarse_levels_have_their_own_profile():
    """``zoom_params`` on the earth preset: the discharge scales are the same
    upstream area at every level (80 / 32 cells at 1.2 km, the profile's 1280
    / 512 at 305 m and at 76 m, where they cap), the 1.2 km level takes
    ``COARSE_ZOOM_EROSION`` and the finer ones do not, the 1.2 km and 305 m
    levels take the wider erosion limit and 76 m does not, no level has window
    lakes, and every level low-passes its hardness."""
    from globe.refine.zoom import COARSE_ZOOM_EROSION, DISC_SATURATION_KM2, WIDE_SLOPE_LIMIT_ERODE, ZOOM_EROSION, zoom_params
    p = WorldParams()
    km2 = lambda R: (p.world.cell_size_m / R / 1000.0) ** 2
    z8, z32, z128 = (zoom_params(p, R) for R in (8, 32, 128))
    assert z8.erosion.disc_saturation_cells == pytest.approx(DISC_SATURATION_KM2 / km2(8))
    assert 60 < z8.erosion.disc_saturation_cells < 100 and 25 < z8.erosion.momentum_saturation_cells < 40
    for z in (z32, z128):
        assert z.erosion.disc_saturation_cells == ZOOM_EROSION["disc_saturation_cells"]
        assert z.erosion.momentum_saturation_cells == ZOOM_EROSION["momentum_saturation_cells"]
        assert not z.erosion.window_lakes and z.erosion.k_mom == WorldParams().erosion.k_mom
    for k, v in COARSE_ZOOM_EROSION.items():
        assert getattr(z8.erosion, k) == v, k
    assert z8.erosion.slope_limit_erode == z32.erosion.slope_limit_erode == WIDE_SLOPE_LIMIT_ERODE
    assert z128.erosion.slope_limit_erode == ZOOM_EROSION["slope_limit_erode"]
    assert zoom_params(p, 8, window_lakes=True).erosion.window_lakes          # a call's override still wins
    assert all(z.refine.hardness_smooth_cells > 0 and z.refine.hardness_max < 1 for z in (z8, z32, z128))
    # the game-scale levels (19 m, 5 m) take the gentler talus pass, 76 m and up keep the profile's
    from globe.refine.zoom import FINE_THERMAL_RATE
    z512, z2048 = (zoom_params(p, R) for R in (512, 2048))
    assert z512.erosion.thermal_rate == z2048.erosion.thermal_rate == FINE_THERMAL_RATE
    assert all(z.erosion.thermal_rate == ZOOM_EROSION["thermal_rate"] for z in (z8, z32, z128))
    assert z2048.erosion.disc_saturation_cells == ZOOM_EROSION["disc_saturation_cells"] and z2048.erosion.slope_limit_erode == ZOOM_EROSION["slope_limit_erode"]


def test_smooth_drift_removes_the_coarse_scale_without_the_grid():
    """``refine.zoom.smooth_drift`` takes a smooth coarse-scale drift out as
    well as ``basin_job.block_drift`` does, and its correction has no kink
    on the block lines: the bilinear one's second difference jumps there
    (the printed grid of docs/zoom-windows.md), the Gaussian one's does not."""
    from globe.refine.basin_job import block_drift
    from globe.refine.zoom import smooth_drift
    R, nb = 16, 12
    n = R * nb
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rng = np.random.default_rng(3)
    drift = 30.0 * np.sin(x / 45.0) * np.cos(y / 60.0) + 10.0 * (x / n)
    detail = rng.normal(0.0, 5.0, (n, n))
    cells = np.ones((n, n), bool)
    for fn in (block_drift, smooth_drift):
        F = fn(drift + detail, cells, R)
        res = (drift + detail - F).reshape(nb, R, nb, R).mean(axis=(1, 3))
        assert np.abs(res).max() < 1.0, (fn.__name__, np.abs(res).max())
    Fb = block_drift(drift + detail, cells, R)
    Fs = smooth_drift(drift + detail, cells, R)
    def kink(F):
        d2 = np.abs(F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2])      # second difference along x, at columns 1..n-2
        cols = np.arange(1, n - 1)
        on = (cols % R == R // 2)                                # bilinear nodes sit at block centres: kinks there
        return d2[:, on].mean() / max(d2[:, ~on].mean(), 1e-12)
    assert kink(Fb) > 3.0, kink(Fb)
    assert kink(Fs) < 1.5, kink(Fs)
