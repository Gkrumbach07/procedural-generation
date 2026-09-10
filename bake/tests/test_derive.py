"""Derive stage tests (PLAN.md section 11 / 15) on the tiny preset with stub
upstream stages plus a synthetic river valley / lake on fine face 0:
biome codes, vegetation range, river mask aligned with high discharge,
rivers.json geometry (inside faces, descending, widths grow with
discharge, Strahler from the fine graph and from a coarse graph), lakes.json
polygons, determinism, runtime at N_fine = 256 with an estimate for 4096."""
import dataclasses
import json
import time

import numpy as np
import pytest

from globe.climate.temperature import retarget, temperature
from globe.config import WorldParams
from globe.derive import biomes, lakes, rivers, soil
from globe.derive import run as derive_run
from globe.derive import fine as fine_mod
from globe.derive.fine import edge_neighbors, load_face, slope_magnitude, write_face
from globe.field import FaceField
from globe.io.world_store import WorldStore
from globe.stubs import stub_climate, stub_erosion, stub_hydro, stub_refine, stub_tectonics, stub_watersheds


def _log(_msg):
    pass


# --------------------------------------------------------------------------
# synthetic world
# --------------------------------------------------------------------------
def _valley_centre(i, Nf):
    return Nf / 2.0 + 6.0 * np.sin(2 * np.pi * i / Nf)


def _tributary_centre(i, Nf):
    return _valley_centre(i, Nf) + (Nf * 0.47 - i) * 1.2


def synthetic_face(Nf):
    """Face 0: a valley descending along +i with a sinuous main channel, a
    tributary joining it at i ~ 0.47 Nf, and a lake block."""
    ii, jj = np.meshgrid(np.arange(Nf), np.arange(Nf), indexing="ij")
    c = _valley_centre(ii, Nf)
    h = (900.0 - 12.0 * ii * (64.0 / Nf) + 0.8 * np.abs(jj - c)).astype(np.float32)
    q = 2.0 + 300.0 * (0.3 + ii / Nf) * np.exp(-((jj - c) ** 2) / 8.0)
    ct = _tributary_centre(ii, Nf)
    trib = (ii <= Nf * 0.47) & (ii >= Nf * 0.15)
    q = q + np.where(trib, 100.0 * np.exp(-((jj - ct) ** 2) / 8.0), 0.0)
    ws = h.copy()
    li0, li1, lj0, lj1 = int(Nf * 0.62), int(Nf * 0.75), int(Nf * 0.12), int(Nf * 0.25)
    ws[li0:li1, lj0:lj1] = h[li0:li1, lj0:lj1].max() + 3.0
    return h, q.astype(np.float32), ws, (li0, li1, lj0, lj1)


UPSTREAM_Q = 75.0  # peak discharge of the upstream reach on the neighbour face (must fall between q_low and q_thr)
EDGE_LAKE_LEVEL = 960.0
EDGE_LAKE_ID = 5


def inward_line(entry, Nf, length):
    """Cells of the neighbour face from an edge cell ``(i, j)`` straight
    inward for ``length`` cells: ``(ii, jj)`` arrays."""
    i, j = int(entry[0]), int(entry[1])
    k = np.arange(length)
    if i == 0:
        return i + k, np.full(length, j)
    if i == Nf - 1:
        return i - k, np.full(length, j)
    if j == 0:
        return np.full(length, i), j + k
    assert j == Nf - 1
    return np.full(length, i), j - k


def edge_features(params):
    """Layout of the features straddling the -i edge of face 0: the
    neighbour face ``A``, the upstream river (peak ``UPSTREAM_Q``, flowing
    towards face 0's valley at ``j = Nf/2``) as ``(ii, jj, q, h)`` arrays on
    ``A``, and the lake blocks on both faces as cell arrays."""
    Nf = params.N_fine
    f2, i2, j2 = edge_neighbors(Nf, 0, 1)[0]
    A = int(f2[0, 0])
    assert (f2 == A).all()
    jc = Nf // 2
    L = Nf // 2
    riv = []
    for j in range(jc - 4, jc + 5):
        ii, jj = inward_line((i2[0, j], j2[0, j]), Nf, L)
        d = np.arange(L) + 1  # distance from the edge
        qq = 2.0 + UPSTREAM_Q * np.exp(-((j - jc) ** 2) / 18.0) * (1.0 - 0.1 * d / L)
        hh = 900.0 + 12.0 * (64.0 / Nf) * d + 0.8 * abs(j - jc)
        riv.append((ii, jj, qq, hh))
    lj0, lj1 = int(Nf * 0.78), int(Nf * 0.90)
    depth = 3
    lake0 = np.stack(np.meshgrid(np.arange(depth), np.arange(lj0, lj1), indexing="ij"), axis=-1).reshape(-1, 2)
    lakeA = []
    for j in range(lj0, lj1):
        ii, jj = inward_line((i2[0, j], j2[0, j]), Nf, depth)
        lakeA.append(np.stack([ii, jj], axis=1))
    return {"A": A, "river": riv, "lake0": lake0, "lakeA": np.concatenate(lakeA), "jc": jc}


def build_world(path, params, drainage=None):
    store = WorldStore(path, create=True)
    grid = params.coarse_grid()
    store.init_manifest(params, grid)
    for s, fn in (
        ("tectonics", stub_tectonics), ("climate", stub_climate), ("erosion", stub_erosion),
        ("hydro", stub_hydro), ("watersheds", stub_watersheds), ("refine", stub_refine),
    ):
        fn(store, params, _log)
        store.mark_stage(s, [], {"stub": True}, 0.0, params=params)
    Nf, R = params.N_fine, params.world.R
    h, q, ws, _ = synthetic_face(Nf)
    ef = edge_features(params)
    A = ef["A"]
    # face 0: the valley; a lake block on its -i edge
    l0 = ef["lake0"]
    ws[l0[:, 0], l0[:, 1]] = EDGE_LAKE_LEVEL
    write_face(store, "height", 0, h)
    write_face(store, "sediment", 0, np.full((Nf, Nf), 0.2, np.float32))
    write_face(store, "water_surface", 0, ws)
    write_face(store, "discharge", 0, q)
    # neighbour face A: a plateau with the upstream reach and the other half of the edge lake
    hA = np.full((Nf, Nf), 1500.0, np.float32)
    qA = np.full((Nf, Nf), 2.0, np.float32)
    for ii, jj, qq, hh in ef["river"]:
        hA[ii, jj] = np.minimum(hA[ii, jj], hh)
        qA[ii, jj] = np.maximum(qA[ii, jj], qq)
    lA = ef["lakeA"]
    hA[lA[:, 0], lA[:, 1]] = 930.0
    wsA = hA.copy()
    wsA[lA[:, 0], lA[:, 1]] = EDGE_LAKE_LEVEL
    write_face(store, "height", A, hA)
    write_face(store, "sediment", A, np.full((Nf, Nf), 0.2, np.float32))
    write_face(store, "water_surface", A, wsA)
    write_face(store, "discharge", A, qA)
    # coarse lake under the edge lake (both faces, one level) + its hydro record
    hc = store.load_field("height", grid)
    wc = store.load_field("water_surface", grid)
    for f, cells in ((0, l0), (A, lA)):
        ci, cj = cells[:, 0] // R + grid.H, cells[:, 1] // R + grid.H
        hc.data[f, ci, cj] = 930.0
        wc.data[f, ci, cj] = EDGE_LAKE_LEVEL
    hc.exchange_halos()
    wc.exchange_halos()
    store.save_field(hc)
    store.save_field(wc)
    outlet = [0, int(l0[0, 0] // R), int(l0[0, 1] // R)]
    store.write_json("graph/lakes_coarse.json", {"lakes": [{"id": EDGE_LAKE_ID, "surface_m": EDGE_LAKE_LEVEL, "outlet": outlet, "faces": [0, A]}], "lake_min_depth": params.hydro.lake_min_depth})
    if drainage is not None:
        store.write_json("graph/drainage.json", drainage)
    return store


def fake_drainage(params):
    """One coarse reach (id 7, order 3) along the main valley of face 0."""
    Nf, R = params.N_fine, params.world.R
    cells = []
    for i in range(0, Nf, R):
        ci, cj = i // R, int(_valley_centre(i, Nf)) // R
        if not cells or cells[-1] != [0, ci, cj]:
            cells.append([0, ci, cj])
    return {
        "nodes": [{"id": 0, "cell": cells[0], "kind": "source", "acc": 1.0}, {"id": 1, "cell": cells[-1], "kind": "outlet", "acc": 9.0}],
        "edges": [{"id": 7, "from": 0, "to": 1, "order": 3, "length_m": 1.0, "mean_discharge": 5.0, "cells": cells}],
    }


@pytest.fixture(scope="module")
def tiny(scratch):
    params = WorldParams.tiny_world(seed=3)
    store = build_world(scratch / "derive_tiny", params)
    info = derive_run.run(store, params, _log)
    return store, params, info


# --------------------------------------------------------------------------
# unit tests of the building blocks
# --------------------------------------------------------------------------
def test_whittaker_and_overrides():
    T = np.array([-20.0, -5.0, 0.0, 10.0, 10.0, 15.0, 25.0, 25.0, 25.0], np.float32)
    P = np.array([50.0, 50.0, 50.0, 10.0, 100.0, 40.0, 10.0, 150.0, 300.0], np.float32)
    got = biomes.whittaker(T, P)
    exp = [biomes.ICE, biomes.TUNDRA, biomes.BOREAL_FOREST, biomes.DESERT, biomes.TEMPERATE_FOREST, biomes.SHRUBLAND, biomes.DESERT, biomes.TROPICAL_SEASONAL_FOREST, biomes.TROPICAL_RAINFOREST]
    assert got.tolist() == exp
    base = np.full(4, biomes.TEMPERATE_FOREST, np.uint8)
    t = np.array([True, True, False, False])
    out = biomes.apply_overrides(base, ocean=np.array([True, False, False, False]), lake=t, wetland=None, cliff=np.array([False, False, True, False]), alpine=None, riparian=np.array([False, False, True, True]))
    assert out.tolist() == [biomes.OCEAN, biomes.LAKE, biomes.CLIFF, biomes.RIPARIAN]
    assert biomes.PALETTE.shape == (biomes.N_BIOMES, 3) and len(biomes.NAMES) == biomes.N_BIOMES
    assert soil.soil_factor(np.array([0.0, 0.5, 5.0]), 1.0).tolist() == pytest.approx([0.7, 0.85, 1.0])


def test_effective_alpine_min_follows_the_achieved_relief():
    """A fixed metre threshold does not survive a planet whose relief
    scales with its size (tectonics.relief_spacings), so 0 means "a
    fraction of the achieved land relief"."""
    p = WorldParams.tiny_world()
    dp = p.derive
    land = np.linspace(0.0, 1000.0, 10001, dtype=np.float32)
    assert dp.alpine_min_m == 0.0  # the default is relative
    got = biomes.effective_alpine_min(land, dp)
    assert got == pytest.approx(dp.alpine_min_relief_frac * float(np.quantile(land, 0.999)), rel=1e-6)
    assert biomes.effective_alpine_min(np.array([], np.float32), dp) == 0.0
    fixed = dataclasses.replace(dp, alpine_min_m=800.0)
    assert biomes.effective_alpine_min(land, fixed) == 800.0
    # and it is what classify uses
    surf = np.array([300.0, 700.0], np.float32)
    T = np.full(2, -5.0, np.float32)
    P = np.full(2, 100.0, np.float32)
    zero = np.zeros(2, bool)
    out = biomes.classify(T, P, surf, np.zeros(2, np.float32), zero, zero, zero, dp, 1.6, got)
    assert out.tolist() == [biomes.TUNDRA, biomes.ALPINE]


def test_biome_temperature_is_taken_at_the_eroded_surface(tiny):
    """``climate/temperature`` is computed on the pre-erosion bedrock;
    derive re-references it to the surface it classifies (coarse and per
    fine cell), so peaks are not classified several degrees too warm."""
    store, params, info = tiny
    grid = params.coarse_grid()
    cp = params.climate
    T = store.load_field("temperature", grid)
    bed = store.load_field("bedrock", grid)
    surface_c = (store.load_field("height", grid).interior + store.load_field("sediment", grid).interior).astype(np.float32)
    land = surface_c >= 0.0
    surf_field = FaceField.from_interior(grid, surface_c, name="surface")
    # re-referencing is exactly a full recompute of climate.temperature on
    # the eroded surface (checked on the real formula: the stub climate uses
    # its own latitude law, so only the terrain is borrowed here)
    T_bed = temperature(grid, bed.data, cp)
    round_trip = retarget(retarget(T_bed, bed.data, 0.0, cp), 0.0, surf_field.data, cp)
    assert np.allclose(round_trip, temperature(grid, surf_field.data, cp), atol=1e-4)
    # the stage classified with the stored field re-referenced that way ...
    exp_c = retarget(retarget(T.interior, bed.interior, 0.0, cp), 0.0, surface_c, cp)
    assert info["T_biome_land_mean"] == pytest.approx(float(exp_c[land].mean()), abs=1e-4)
    # ... which really differs from the stale (bedrock) field
    assert np.abs(exp_c - T.interior)[land].max() > 0.1
    # fine: each face's lapse is applied at its own cells, not upsampled
    T0 = FaceField.from_interior(grid, retarget(T.interior, bed.interior, 0.0, cp), name="T0")
    for f in (0, edge_features(params)["A"]):
        surface_f = np.asarray(load_face(store, "height", f), np.float32) + np.asarray(load_face(store, "sediment", f), np.float32)
        exp = retarget(fine_mod.upsample_face(T0, f, params.world.R, order=1), 0.0, surface_f, cp)
        m = surface_f >= 0.0
        assert info["faces"][f]["T_land_mean"] == pytest.approx(float(exp[m].mean()), abs=1e-4)


def test_thinning_gives_connected_one_pixel_skeleton():
    m = np.zeros((64, 64), bool)
    ii, jj = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    m |= np.abs(jj - (20 + 0.4 * ii)) <= 3  # thick diagonal band
    m[30:34, :] = True  # a crossing band -> junctions
    sk = rivers.skeletonize(m, spur_len=4)
    assert sk.any()
    from scipy import ndimage

    _, n = ndimage.label(sk, structure=np.ones((3, 3)))
    assert n == 1
    # 1-pixel wide: no 2x2 block fully on
    blocks = sk[:-1, :-1] & sk[1:, :-1] & sk[:-1, 1:] & sk[1:, 1:]
    assert not blocks.any()
    assert sk[:, :].sum() < 0.35 * m.sum()
    off, pi, pj, deg = rivers.trace_segments(sk)
    assert off.size - 1 >= 3  # the crossing splits the diagonal
    for s in range(off.size - 1):
        a, b = off[s], off[s + 1]
        assert deg[pi[a], pj[a]] != 2 and deg[pi[b - 1], pj[b - 1]] != 2
        steps = np.hypot(np.diff(pi[a:b].astype(float)), np.diff(pj[a:b].astype(float)))
        assert (steps <= np.sqrt(2) + 1e-9).all()


def test_strahler_orders_from_segment_graph():
    starts = np.array([1, 2, 9, 5, 9])
    ends = np.array([9, 9, 7, 9, 8])  # three inflows into 9, two outflows
    o = rivers.strahler_orders(starts, ends)
    assert o.tolist() == [1, 1, 2, 1, 2]
    loop = rivers.strahler_orders(np.array([1, 2]), np.array([2, 1]))
    assert loop.tolist() == [1, 1]


def test_catmull_rom_keeps_endpoints():
    P = np.stack([np.arange(20.0), 3 * np.sin(np.arange(20.0) / 3)], axis=1)
    out, w = rivers.catmull_rom(P, np.linspace(1, 5, 20), 2.0, 2.0)
    assert np.allclose(out[0], P[0]) and np.allclose(out[-1], P[-1])
    assert out.shape[0] == 11 and w[0] == 1 and w[-1] == 5
    d = np.abs(out[:, 1] - np.interp(out[:, 0], P[:, 0], P[:, 1]))
    assert d.max() < 1.0


def test_lake_rings_close_and_turn_right_at_pinches():
    m = np.zeros((8, 8), bool)
    m[1:3, 1:3] = True
    m[3, 3] = True  # touches the block only at a corner: one 8-connected lake
    m[5:7, 5:7] = True
    r = lakes.trace_rings(m)
    assert len(r) == 2
    for ring in r:
        assert (ring[0] == ring[-1]).all()
    assert lakes.ring_area(r[0]) > 0 and abs(lakes.ring_area(r[0])) == 5.0
    assert abs(lakes.ring_area(r[1])) == 4.0
    hole = np.ones((6, 6), bool)
    hole[2:4, 2:4] = False
    r = lakes.trace_rings(hole)
    assert len(r) == 2 and lakes.ring_area(r[0]) > 0 and lakes.ring_area(r[1]) < 0


def test_threshold_from_histogram():
    q = np.concatenate([np.full(900, 1.0), np.full(100, 50.0)]).astype(np.float32)
    counts, n = rivers.log_histogram(q, np.ones(q.size, bool))
    t = rivers.threshold_from_histogram(counts, n, 0.1)
    assert 1.0 < t < 50.0
    assert rivers.threshold_from_histogram(counts, n, 0.0) == float("inf")


# --------------------------------------------------------------------------
# stage on the synthetic tiny world
# --------------------------------------------------------------------------
def test_outputs_and_biome_codes(tiny):
    store, params, info = tiny
    grid = params.coarse_grid()
    N, Nf = grid.N, params.N_fine
    b = store.load_field("biome", grid)
    assert b.dtype == np.uint8 and b.interior.shape == (6, N, N)
    surface = store.load_field("height", grid).interior + store.load_field("sediment", grid).interior
    assert (b.interior[surface < 0] == biomes.OCEAN).all()
    assert (b.interior[surface >= 0] != biomes.OCEAN).all()
    assert b.interior.max() < biomes.N_BIOMES
    for name in derive_run.OUTPUTS:
        assert store.has_field(name) or (store.root / name).exists(), name
    for f in range(6):
        bi = np.asarray(load_face(store, "biome", f))
        veg = np.asarray(load_face(store, "vegetation", f))
        rm = np.asarray(load_face(store, "river_mask", f))
        assert bi.shape == veg.shape == rm.shape == (Nf, Nf)
        assert bi.dtype == veg.dtype == rm.dtype == np.uint8
        sf = np.asarray(load_face(store, "height", f)) + np.asarray(load_face(store, "sediment", f))
        assert (bi[sf < 0] == biomes.OCEAN).all() and (bi[sf >= 0] != biomes.OCEAN).all()
        assert bi.max() < biomes.N_BIOMES
        assert (veg[sf < 0] == 0).all() and (rm[sf < 0] == 0).all()
        assert set(np.unique(rm)) <= {0, 255}
    veg0 = np.asarray(load_face(store, "vegetation", 0))
    assert 20 < np.median(veg0) < 250
    # synthetic lake block is LAKE with a wetland ring, cliffs nowhere on the gentle valley
    b0 = np.asarray(load_face(store, "biome", 0))
    li0, li1, lj0, lj1 = synthetic_face(Nf)[3]
    assert (b0[li0:li1, lj0:lj1] == biomes.LAKE).all()
    assert b0[li0 - 1, lj0] == biomes.WETLAND and b0[li1, lj1 - 1] == biomes.WETLAND
    R = params.world.R  # the outer R cells see the (unrelated) neighbour faces through the slope pads
    assert (b0[R:-R, R:-R] != biomes.CLIFF).all()
    assert info["n_rivers"] > 0 and info["n_lakes"] >= 1


def test_river_mask_aligned_with_high_discharge(tiny):
    store, params, _ = tiny
    Nf = params.N_fine
    q0 = np.asarray(load_face(store, "discharge", 0))
    rm = np.asarray(load_face(store, "river_mask", 0)) > 0
    assert rm.sum() > 0.5 * Nf  # at least the main stem
    land = (np.asarray(load_face(store, "height", 0)) + 0.2) >= 0
    assert q0[rm].mean() > 3 * q0[land].mean()
    ii, jj = np.nonzero(rm)
    d_main = np.abs(jj - _valley_centre(ii, Nf))
    d_trib = np.abs(jj - _tributary_centre(ii, Nf))
    assert (np.minimum(d_main, d_trib) <= 4).mean() > 0.9
    # riparian corridor around the mask
    b0 = np.asarray(load_face(store, "biome", 0))
    assert (b0[rm] == biomes.RIPARIAN).mean() > 0.9
    # the main stem crosses the whole face
    assert rm[:, :].any(axis=1).sum() > 0.9 * Nf


def _river_heights(store, params, r):
    Nf = params.N_fine
    pts = np.asarray(r["points"], dtype=np.float64)
    f = int(pts[0, 0])
    i = np.minimum((pts[:, 1] * Nf).astype(int), Nf - 1)
    j = np.minimum((pts[:, 2] * Nf).astype(int), Nf - 1)
    sf = np.asarray(load_face(store, "height", f)) + np.asarray(load_face(store, "sediment", f))
    q = np.asarray(load_face(store, "discharge", f))
    return pts, sf[i, j], q[i, j]


def test_rivers_json_geometry(tiny):
    store, params, _ = tiny
    rj = store.read_json("graph/rivers.json")
    rivers_ = rj["rivers"]
    assert rivers_ and [r["id"] for r in rivers_] == list(range(len(rivers_)))
    on0 = [r for r in rivers_ if r["face"] == 0]
    assert on0
    all_w, all_q = [], []
    for r in rivers_:
        pts = np.asarray(r["points"], dtype=np.float64)
        assert pts.shape[1] == 5 and len(pts) >= 2
        assert (np.diff(pts[:, 4]) <= 1e-6).all(), r["id"]  # height_m never rises downstream
        assert (pts[:, 0] == r["face"]).all()
        assert (pts[:, 1] >= 0).all() and (pts[:, 1] < 1).all() and (pts[:, 2] >= 0).all() and (pts[:, 2] < 1).all()
        assert (pts[:, 3] >= params.fine_cell_size_m * 0.99).all()
        assert r["order"] >= 1 and isinstance(r["edge_id"], int)
        _, h, q = _river_heights(store, params, r)
        if r["face"] == 0:
            # descending along the synthetic valley: no step rises more than 3 m, end lower than start
            assert np.diff(h).max() <= 3.0 + 1e-6, r["id"]
            assert h[-1] < h[0]
            all_w += list(pts[:, 3])
            all_q += list(q)
    from scipy.stats import spearmanr

    assert spearmanr(all_w, all_q).correlation > 0.5
    # the main stem is wider near its mouth than near its source
    main = max(on0, key=lambda r: len(r["points"]))
    pts = np.asarray(main["points"])
    assert pts[-3:, 3].mean() > pts[:3, 3].mean()
    # fine Strahler (no coarse graph): the tributary junction makes an order-2 reach
    assert max(r["order"] for r in on0) >= 2
    assert all(r["edge_id"] == -1 for r in on0)
    assert rj["discharge_threshold"] > 0


def test_coarse_graph_supplies_order_and_edge_id(scratch):
    params = WorldParams.tiny_world(seed=3)
    store = build_world(scratch / "derive_graph", params, drainage=fake_drainage(params))
    derive_run.run(store, params, _log)
    on0 = [r for r in store.read_json("graph/rivers.json")["rivers"] if r["face"] == 0]
    main = max(on0, key=lambda r: len(r["points"]))
    assert main["edge_id"] == 7 and main["order"] == 3


def test_lakes_json_polygons(tiny):
    store, params, _ = tiny
    Nf = params.N_fine
    lj = store.read_json("graph/lakes.json")
    assert lj["lakes"]
    li0, li1, lj0, lj1 = synthetic_face(Nf)[3]
    found = False
    for L in lj["lakes"]:
        assert [L["id"] for L in lj["lakes"]] == list(range(len(lj["lakes"])))
        poly = np.asarray(L["polygon"])
        assert len(poly) >= 5 and (poly[0] == poly[-1]).all()
        for ring in L["rings"]:
            ring = np.asarray(ring)
            assert (ring[0] == ring[-1]).all()
            f = int(ring[0, 0])
            sf = np.asarray(load_face(store, "height", f)) + np.asarray(load_face(store, "sediment", f))
            ws = np.asarray(load_face(store, "water_surface", f))
            lake = lakes.lake_mask(sf, ws, params.hydro.lake_min_depth)
            pad = np.zeros((Nf + 2, Nf + 2), bool)
            pad[1:-1, 1:-1] = lake
            ci = np.rint(ring[:, 1] * Nf).astype(int)
            cj = np.rint(ring[:, 2] * Nf).astype(int)
            # every ring corner touches a lake cell
            touch = pad[ci, cj] | pad[ci + 1, cj] | pad[ci, cj + 1] | pad[ci + 1, cj + 1]
            assert touch.all()
        if L["faces"] == [0] and L["area_cells"] == (li1 - li0) * (lj1 - lj0):
            found = True
            assert L["surface_m"] == pytest.approx(float(np.asarray(load_face(store, "water_surface", 0))[li0, lj0]))
            assert L["outlet"] is None  # stub hydro writes no lakes_coarse.json
            assert L["area_m2"] > 0
            corners = {(round(p[1] * Nf), round(p[2] * Nf)) for p in L["polygon"]}
            assert corners == {(li0, lj0), (li1, lj0), (li1, lj1), (li0, lj1)}
    assert found


def test_fine_edge_map_matches_coarse_owner():
    params = WorldParams.tiny_world()
    grid = params.coarse_grid()
    N, H, R = grid.N, grid.H, params.world.R
    Nf = N * R
    of = grid.owner_face_ij
    rng = np.arange(Nf)
    coarse_sides = ((np.full(Nf, H - 1), rng // R + H), (np.full(Nf, H + N), rng // R + H), (rng // R + H, np.full(Nf, H - 1)), (rng // R + H, np.full(Nf, H + N)))
    for f in range(6):
        for side, (f2, i2, j2) in enumerate(edge_neighbors(Nf, f, 1)):
            assert f2.shape == ((1, Nf) if side < 2 else (Nf, 1))
            o = of[f][coarse_sides[side]]
            assert (o[:, 0] == f2.ravel()).all() and (o[:, 1] == i2.ravel() // R).all() and (o[:, 2] == j2.ravel() // R).all()
            i2, j2 = i2.ravel(), j2.ravel()
            assert ((i2 == 0) | (i2 == Nf - 1) | (j2 == 0) | (j2 == Nf - 1)).all()
    # cells inside the face map to themselves
    f2, i2, j2 = fine_mod.outside_cells(Nf, 3, np.array([0, 5, Nf - 1]), np.array([7, 7, 0]))
    assert (f2 == 3).all() and i2.tolist() == [0, 5, Nf - 1] and j2.tolist() == [7, 7, 0]


def test_river_connectivity_keeps_low_upstream_reach_across_edge():
    """A river whose discharge crosses the high threshold only after a
    cube edge: per-face hysteresis drops the upstream face, the global
    decision keeps both."""
    Nf = 32
    f2, i2, j2 = edge_neighbors(Nf, 0, 1)[0]
    A, ia, ja = int(f2[0, 0]), int(i2[0, Nf // 2]), int(j2[0, Nf // 2])
    q = np.zeros((6, Nf, Nf), np.float32)
    q[0, :, Nf // 2] = np.linspace(94, 139, Nf)  # downstream on face 0 (+i)
    ii, jj = inward_line((ia, ja), Nf, Nf)
    q[A, ii, jj] = np.linspace(94, 50, Nf)  # upstream on A, weakest far from the edge
    land = np.ones((Nf, Nf), bool)
    q_low, q_thr = 40.0, 100.0
    per_face = [rivers.hysteresis_mask(q[f], land, q_thr, q_low).sum() for f in (0, A)]
    assert per_face == [Nf, 0]
    conn = rivers.RiverConnectivity(Nf, min_cells=8)
    for f in range(6):
        conn.add_face(f, q[f] > q_low, q[f] > q_thr)
    conn.finalize()
    assert conn.n_labels == 2 and conn.keep.tolist() == [True, True]
    assert conn.mask(0, q[0] > q_low).sum() == Nf and conn.mask(A, q[A] > q_low).sum() == Nf
    # a blob that never reaches the high threshold is still dropped, and so is a small kept-size union
    q2 = q.copy()
    q2[0] = 0
    conn2 = rivers.RiverConnectivity(Nf, min_cells=8)
    for f in range(6):
        conn2.add_face(f, q2[f] > q_low, q2[f] > q_thr)
    assert conn2.finalize().tolist() == [False]
    conn3 = rivers.RiverConnectivity(Nf, min_cells=3 * Nf)
    for f in range(6):
        conn3.add_face(f, q[f] > q_low, q[f] > q_thr)
    assert conn3.finalize().tolist() == [False, False]


def test_river_continues_across_face_edge(tiny):
    store, params, info = tiny
    Nf = params.N_fine
    ef = edge_features(params)
    A, jc = ef["A"], ef["jc"]
    qA = np.asarray(load_face(store, "discharge", A))
    # precondition of this test: the upstream reach lies between the two thresholds
    q_sA = rivers.smooth_discharge(qA, params.derive.discharge_smooth_cells)
    ii, jj, _, _ = ef["river"][4]  # the centre line of the upstream reach on A
    ii, jj = ii[:-2], jj[:-2]  # (the smoothing blurs the reach's far end)
    assert (q_sA[ii, jj] > info["discharge_threshold_low"]).all() and q_sA.max() < info["discharge_threshold"], (info["discharge_threshold_low"], info["discharge_threshold"])
    rmA = np.asarray(load_face(store, "river_mask", A)) > 0
    rm0 = np.asarray(load_face(store, "river_mask", 0)) > 0
    assert rmA[ii, jj].mean() > 0.8, "upstream reach missing on the neighbour face"
    assert rmA[ii[0], jj[0]] and rm0[0, jc - 1 : jc + 2].any(), "river mask must meet at the edge"
    on_A = [r for r in store.read_json("graph/rivers.json")["rivers"] if r["face"] == A]
    assert on_A
    pts = np.concatenate([np.asarray(r["points"])[:, 1:3] * Nf for r in on_A])
    d_edge = np.minimum.reduce([pts[:, 0], Nf - pts[:, 0], pts[:, 1], Nf - pts[:, 1]])
    assert d_edge.min() < 1.5  # a polyline reaches the cube edge
    # riparian corridor along the reach (alpine outranks riparian: the synthetic plateau is > alpine_min_m and cold on some seeds)
    assert np.isin(np.asarray(load_face(store, "biome", A))[ii, jj], [biomes.RIPARIAN, biomes.ALPINE]).mean() > 0.8


def test_lake_across_edge_is_one_record_with_outlet_and_hydro_id(tiny):
    store, params, _ = tiny
    Nf, R = params.N_fine, params.world.R
    ef = edge_features(params)
    A = ef["A"]
    lj = store.read_json("graph/lakes.json")
    edge_lakes = [L for L in lj["lakes"] if L["surface_m"] == pytest.approx(EDGE_LAKE_LEVEL)]
    assert len(edge_lakes) == 1, [L["faces"] for L in edge_lakes]
    L = edge_lakes[0]
    assert L["faces"] == sorted([0, A]) and len(L["rings"]) == 2
    assert L["area_cells"] == len(ef["lake0"]) + len(ef["lakeA"])
    assert L["coarse_id"] == EDGE_LAKE_ID
    assert L["outlet"] == [0, int(ef["lake0"][0, 0] // R), int(ef["lake0"][0, 1] // R)]
    faces_of_rings = sorted(int(r[0][0]) for r in L["rings"])
    assert faces_of_rings == sorted([0, A])
    # the other lakes carry no hydro record
    assert all(L2["coarse_id"] == -1 and L2["outlet"] is None for L2 in lj["lakes"] if L2 is not L)


def test_near_padded_continues_band_across_edge():
    m = np.zeros((16, 16), bool)
    top = np.zeros((3, 16), bool)
    top[2, 5] = True  # a river cell just beyond the -i edge (row i = -1)
    empty_v, empty_h = np.zeros((3, 16), bool), np.zeros((16, 3), bool)
    band = biomes.near_padded(m, [top, empty_v, empty_h, empty_h], 3)
    assert band[0, 5] and band[1, 5] and band[2, 5] and not band[3, 5]
    assert band[0, 3] and band[2, 7] and not band[0, 9] and not band[3, 3]
    assert not biomes.near(m, 3).any()
    assert np.array_equal(biomes.near_padded(m, None, 3), biomes.near(m, 3))


def test_slope_pads_remove_edge_seam():
    """A smooth function of the sphere point has a smooth slope across the
    cube edges when the stencil is fed the neighbouring faces' cells."""
    from globe.cubesphere import get_grid, to_sphere_v

    params = WorldParams.tiny_world()
    grid = params.coarse_grid()
    N, R = grid.N, params.world.R
    Nf = N * R
    c = (np.arange(Nf) + 0.5) / Nf
    U, V = np.meshgrid(c, c, indexing="ij")
    surf = np.zeros((6, Nf, Nf), np.float32)
    for f in range(6):
        p = to_sphere_v(np.full(U.shape, f), U, V)
        surf[f] = 2000.0 * p[..., 2] + 800.0 * p[..., 0] * p[..., 1] + 500.0 * p[..., 1]
    getter = lambda f2, i2, j2: surf[f2, i2, j2]
    stencil = R
    for f in range(6):
        g = fine_mod.coarse_face_array(grid, grid.metric_inv, f)
        pads = fine_mod.face_pads(getter, Nf, f, stencil)
        with_pads = slope_magnitude(surf[f], g, R, params.fine_cell_size_m, stencil, pads=pads)
        without = slope_magnitude(surf[f], g, R, params.fine_cell_size_m, stencil)
        assert np.allclose(with_pads[stencil:-stencil, stencil:-stencil], without[stencil:-stencil, stencil:-stencil])
        for edge, inner in ((with_pads[0], with_pads[stencil + 1]), (with_pads[-1], with_pads[-stencil - 2]), (with_pads[:, 0], with_pads[:, stencil + 1]), (with_pads[:, -1], with_pads[:, -stencil - 2])):
            ref = np.abs(with_pads[stencil + 1] - with_pads[2 * stencil + 2]).max() + np.abs(with_pads[:, stencil + 1] - with_pads[:, 2 * stencil + 2]).max()
            assert np.abs(edge - inner).max() < 2.0 * ref + 1e-4


def test_determinism(scratch):
    params = WorldParams.tiny_world(seed=5)
    outs = []
    for k in range(2):
        store = build_world(scratch / f"derive_det{k}", params)
        derive_run.run(store, params, _log)
        blobs = {}
        for name in derive_run.OUTPUTS:
            if store.has_field(name):
                for f in range(6):
                    blobs[f"{name}{f}"] = (store.coarse_dir / f"{name}.f{f}.npy").read_bytes()
            else:
                blobs[name] = (store.root / name).read_bytes()
        outs.append(blobs)
    assert outs[0].keys() == outs[1].keys()
    for k in outs[0]:
        assert outs[0][k] == outs[1][k], k


def test_quicklook(tiny, tmp_path):
    store, params, _ = tiny
    p = derive_run.quicklook(store, params, tmp_path / "derive.png")
    assert p.exists() and (store.quicklook_path("derive", "vegetation")).exists()
    from PIL import Image

    im = Image.open(p)
    # two orthographic hemispheres side by side with a gap between them:
    # a bit wider than 2:1, never the 4:3 of the old unfolded cube net
    w, h = im.size
    assert 2.0 < w / h < 2.2, im.size


def test_runtime_small(scratch):
    params = WorldParams.small_world(seed=1)
    store = build_world(scratch / "derive_small", params)
    Nf = params.N_fine
    assert Nf == 256
    derive_run.run(store, params, _log)  # warm numba caches
    t = time.time()
    info = derive_run.run(store, params, _log)
    dt = time.time() - t
    est = dt * (4096 / Nf) ** 2
    print(f"\nderive at N_fine={Nf}: {dt:.2f}s ({info['n_rivers']} rivers); linear estimate for N_fine=4096: {est / 60:.1f} min")
    assert dt < 30.0
