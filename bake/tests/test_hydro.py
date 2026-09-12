"""Hydro stage tests (PLAN.md section 9 / 15): D8 contract, heap, priority
flood (flat + sphere, bowls on one face and across a face edge), routing
without cycles, accumulation = precip sums per basin, drainage forest with
valid Strahler orders, lakes, land re-quantile, determinism."""
from pathlib import Path

import numpy as np
import pytest

from globe.config import WorldParams
from globe.cubesphere import Grid, get_grid, to_sphere_v
from globe.hydro import run as hydro_run
from globe.hydro.d8 import D8_OFFSETS, OCEAN, cid_fij, downstream_flat, downstream_table, neighbor_cid, neighbor_fij, neighbor_table
from globe.hydro.lakes import extract_lakes, ring_area, trace_rings
from globe.hydro.priority_flood import heap_pop, heap_push, priority_flood_flat, priority_flood_sphere
from globe.hydro.routing import flow_directions, topological_order, walk_to_sink
from globe.hydro.run import requantile_height
from globe.io.world_store import WorldStore
from globe.stubs import stub_climate, stub_erosion, stub_tectonics


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _log(_msg):
    pass


def _bake_to_erosion(scratch, name, params):
    """Stub tectonics/climate/erosion outputs (never depend on the other
    engineers' real upstream stages; PLAN 15 / DEVELOPING.md)."""
    store = WorldStore(scratch / name, create=True)
    store.init_manifest(params, params.coarse_grid())
    for s, fn in (("tectonics", stub_tectonics), ("climate", stub_climate), ("erosion", stub_erosion)):
        fn(store, params, _log)
        store.mark_stage(s, [], {"stub": True}, 0.0, params=params)
    return store


def _walk_all_land(down, land, N):
    """Every land cell must reach an ocean cell within N*N steps (no cycles)."""
    M = down.size
    limit = N * N
    for c in np.nonzero(land)[0]:
        end, steps = walk_to_sink(down, int(c), limit)
        assert steps < limit, f"cell {cid_fij(int(c), N)} did not reach the ocean"
        assert not land[end], f"walk from {cid_fij(int(c), N)} ended on land"


def _outlet_labels(down, land):
    """Label of every land cell = last land cell before the ocean (memoised walk)."""
    M = down.size
    label = np.full(M, -1, dtype=np.int64)
    for c in np.nonzero(land)[0]:
        path = []
        x = int(c)
        while label[x] < 0:
            path.append(x)
            d = int(down[x])
            if d < 0 or not land[d]:
                label[x] = x
                break
            x = d
        for p in path:
            label[p] = label[x]
    return label


# --------------------------------------------------------------------------
# D8 contract + heap
# --------------------------------------------------------------------------
def test_d8_contract():
    assert D8_OFFSETS == [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    g = get_grid(16, 4, 50.0)
    N, H, NE = g.N, g.H, g.NE
    own = g.owner
    # interior: downstream_flat == plain offset
    for k, (di, dj) in enumerate(D8_OFFSETS):
        flat = downstream_flat(own, N, H, 2, 5, 7, k)
        f, ei, ej = g.unflat_index(int(flat))
        assert (int(f), int(ei) - H, int(ej) - H) == (2, 5 + di, 7 + dj)
        assert neighbor_cid(own, N, H, 2, 5, 7, k) == (2 * N + 5 + di) * N + 7 + dj
    assert downstream_flat(own, N, H, 0, 0, 0, OCEAN) == -1
    # across an edge: the neighbour is on another face and its centre is
    # within ~1 cell of the extrapolated position
    for f in range(6):
        for i, j, k in ((N - 1, N // 2, 0), (0, N // 2, 4), (N // 2, N - 1, 2), (N // 2, 0, 6), (N - 1, N - 1, 1)):
            f2, i2, j2 = neighbor_fij(own, N, H, f, i, j, k)
            assert f2 != f and 0 <= i2 < N and 0 <= j2 < N
            di, dj = D8_OFFSETS[k]
            p = to_sphere_v(f, (i + di + 0.5) / N, (j + dj + 0.5) / N)
            q = to_sphere_v(f2, (i2 + 0.5) / N, (j2 + 0.5) / N)
            assert np.arccos(np.clip(p @ q, -1, 1)) < 1.5 * (np.pi / 2) / N


def test_heap_pops_in_key_then_sequence_order():
    rng = np.random.default_rng(3)
    n = 500
    keys = rng.integers(0, 20, n).astype(np.float64)  # many ties
    hk = np.empty(n)
    hs = np.empty(n, np.int64)
    hc = np.empty(n, np.int64)
    hn = np.zeros(1, np.int64)
    for s in range(n):
        heap_push(hk, hs, hc, hn, keys[s], s, s * 7)
    out = []
    while hn[0] > 0:
        k, c = heap_pop(hk, hs, hc, hn)
        out.append((k, c // 7))
    assert out == sorted(out)


# --------------------------------------------------------------------------
# priority flood
# --------------------------------------------------------------------------
def test_flood_flat_bowl_fills_to_spill_point():
    Hh, W = 40, 50
    ii, jj = np.meshgrid(np.arange(Hh), np.arange(W), indexing="ij")
    surface = (100.0 + 0.5 * ii).astype(np.float32)  # tilted plane
    bowl = (ii - 20) ** 2 + (jj - 25) ** 2 < 8**2
    surface[bowl] = 60.0 + 0.1 * ii[bowl]
    drain = np.zeros((Hh, W), bool)
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    res = priority_flood_flat(surface, drain)
    filled = res.filled
    assert filled.shape == surface.shape and filled.dtype == np.float32
    assert np.all(filled >= surface)
    assert np.array_equal(filled[~bowl], surface[~bowl])  # equal outside depressions
    # rim: lowest surface value among non-bowl cells adjacent to the bowl
    pad = np.zeros((Hh + 2, W + 2), bool)
    pad[1:-1, 1:-1] = bowl
    rim = np.zeros_like(bowl)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            rim |= pad[1 + di : Hh + 1 + di, 1 + dj : W + 1 + dj]
    rim &= ~bowl
    spill = surface[rim].min()
    assert np.allclose(filled[bowl], spill)
    # order is a topological key: parent popped before child, filled monotone
    order = res.order.reshape(Hh, W)
    assert (order >= 0).all()
    par = res.parent
    flat_order = res.order
    for c in np.nonzero(par >= 0)[0]:
        assert flat_order[par[c]] < flat_order[c]
        assert res.filled.reshape(-1)[par[c]] <= res.filled.reshape(-1)[c]
    assert np.all(np.diff(res.filled.reshape(-1)[res.pop_seq]) >= 0)


def test_flood_flat_active_mask_and_single_outlet():
    Hh, W = 30, 30
    ii, jj = np.meshgrid(np.arange(Hh), np.arange(W), indexing="ij")
    surface = (50.0 + 1.0 * jj).astype(np.float32)
    surface[10:15, 10:15] = 20.0  # pit inside the active window
    active = (ii - 15) ** 2 + (jj - 15) ** 2 < 12**2
    drain = np.zeros((Hh, W), bool)
    drain[15, 4] = True  # single outlet (like a refinement basin)
    res = priority_flood_flat(surface, drain, active)
    assert (res.order[active.reshape(-1)] >= 0).all()
    assert (res.order[~active.reshape(-1)] == -1).all()
    assert np.array_equal(res.filled[~active], surface[~active])
    assert (res.filled >= surface).all()
    assert np.allclose(res.filled[10:15, 10:15], surface[10, 9])  # spills over the lowest rim (column j = 9 is at 59)


def _edge_bowl_world(N=32, H=4):
    """Surface on the sphere: tilted land cap with a bowl straddling the
    +X/+Z edge; ocean where the surface < 0."""
    g = Grid(N, H, 50.0)
    c = g.interior_centers
    a = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)  # point on the +X/+Z edge
    d = c @ a
    # land cap around a, rising towards a, plus a bowl centred on a
    surface = 1000.0 * (d - 0.3) + 100.0 * c[..., 1]
    r = np.arccos(np.clip(d, -1, 1))
    bowl = r < 0.25
    surface = np.where(bowl, surface - 400.0 * (1 - r / 0.25), surface).astype(np.float32)
    return g, surface, bowl


def _reference_fill(surface, ocean, g):
    """Planchon & Darboux (2001) style fixed-point fill: start from +inf on
    land and lower every cell to max(surface, min neighbour) until nothing
    changes.  Uses only links that exist in both directions (like the
    flood kernel)."""
    N, H = g.N, g.H
    nbr = neighbor_table(N, H, g.owner)
    M = nbr.shape[0]
    recip = np.zeros_like(nbr, dtype=bool)
    for k in range(8):
        recip[:, k] = (nbr[nbr[:, k]] == np.arange(M)[:, None]).any(axis=1)
    s = surface.reshape(-1)
    oc = ocean.reshape(-1)
    filled = np.where(oc, s, np.float32(np.inf)).astype(np.float32)
    while True:
        nv = np.where(recip, filled[nbr], np.float32(np.inf)).min(axis=1)
        new = np.where(oc, s, np.maximum(s, nv)).astype(np.float32)
        if np.array_equal(new, filled):
            return filled.reshape(surface.shape)
        filled = new


def test_flood_sphere_bowl_across_face_edge():
    g, surface, bowl = _edge_bowl_world()
    ocean = surface < 0
    assert ocean.any() and (~ocean).any()
    faces_in_bowl = np.unique(np.nonzero(bowl)[0])
    assert 0 in faces_in_bowl and 4 in faces_in_bowl  # the bowl really straddles the edge
    res = priority_flood_sphere(surface, ocean, g)
    filled = res.filled
    assert filled.shape == surface.shape
    assert np.all(filled >= surface)
    land = ~ocean
    assert (res.order[land.reshape(-1)] >= 0).all()  # every land cell reached
    outside = land & ~bowl
    assert np.array_equal(filled[outside], surface[outside])
    # bowl fills to one flat level on both faces, equal to the min surface on the rim
    lvl = np.unique(filled[bowl & (filled > surface)])
    assert lvl.size == 1
    for f in (0, 4):
        assert np.isclose(filled[f][bowl[f] & (filled[f] > surface[f])].max(), lvl[0])
    # exact agreement with an independent reference fill (Planchon-Darboux
    # iteration on the same reciprocal neighbour relation)
    N, H = g.N, g.H
    own = g.owner
    assert np.array_equal(filled, _reference_fill(surface, ocean, g))
    # routing on it: no cycles, everything reaches the ocean, flats drain via parents
    fd = flow_directions(filled, ocean, res.parent, g)
    assert fd.dtype == np.uint8 and (fd[ocean] == OCEAN).all() and (fd[land] < 8).all()
    down = downstream_table(fd, own, H)
    _walk_all_land(down, land.reshape(-1), N)
    topo = topological_order(down, land.reshape(-1))
    assert topo.size == land.sum()
    # the lake drains through a single cell on the rim
    lake = filled > surface
    exits = {int(down[c]) for c in np.nonzero(lake.reshape(-1))[0] if not lake.reshape(-1)[down[c]]}
    assert len(exits) == 1


def test_flood_sphere_no_ocean_falls_back_to_lowest_cell():
    g = Grid(16, 4, 50.0)
    s = (100.0 + g.interior_centers[..., 2] * 10).astype(np.float32)
    res = priority_flood_sphere(s, np.zeros(s.shape, bool), g)
    assert (res.order >= 0).all() and (res.filled >= s).all()


# --------------------------------------------------------------------------
# lakes helpers
# --------------------------------------------------------------------------
def test_trace_rings_square_and_hole():
    m = np.zeros((10, 10), bool)
    m[2:7, 3:8] = True
    rings = trace_rings(m)
    assert len(rings) == 1
    r = rings[0]
    assert tuple(r[0]) == tuple(r[-1]) and r.shape[0] == 21  # closed, 20 unit edges
    assert set(map(tuple, r)) >= {(2, 3), (7, 3), (7, 8), (2, 8)}
    m[4, 5] = False  # hole -> second (shorter) ring
    rings = trace_rings(m)
    assert len(rings) == 2 and rings[0].shape[0] > rings[1].shape[0] == 5
    assert ring_area(rings[0]) > 0 > ring_area(rings[1])  # outer CCW, hole CW


def test_extract_lakes_serpentine_island_is_not_the_polygon():
    """A wiggly island inside a compact lake has the longer boundary; the
    polygon must still be the (shorter) outer shore, counter-clockwise."""
    from globe.cubesphere import Grid

    g = Grid(24, 2, 100.0)
    N = g.N
    lake = np.zeros((6, N, N), bool)
    lake[1, 2:20, 2:20] = True
    # serpentine island: comb of 8 teeth -> boundary 2*(8*13)+... > outer 4*18
    for t in range(8):
        lake[1, 4:17, 4 + 2 * t] = False
    lake[1, 4, 4:19] = False  # spine joining the teeth
    outer_len = 4 * 18
    ws = np.where(lake, 5.0, 0.0).astype(np.float32)
    M = 6 * N * N
    order = np.arange(M, dtype=np.int64)  # cell (1,2,2) is the first lake cell popped
    down = np.full(M, -1, dtype=np.int64)
    lakes, lab = extract_lakes(lake, ws, order, down, g)
    assert len(lakes) == 1
    L = lakes[0]
    assert L["area_cells"] == int(lake.sum())
    poly = [[u * N, v * N] for _, u, v in L["polygon"]]
    assert len(poly) == outer_len + 1  # the outer square, not the island
    assert ring_area(poly) == 18 * 18  # encloses the whole square, CCW
    assert L["rings"] == [L["polygon"]]
    ij = np.array(poly)
    assert ij.min() == 2 and ij.max() == 20


def test_a_closed_basin_below_sea_level_is_not_ocean():
    """Being under the waterline does not make a cell sea; being joined to
    the sea does.  The Caspian is 28 m down and landlocked, and a rift or a
    trapped piece of ocean floor gives the same thing here -- 504 of them,
    6.9 M km², were classified as ocean before `open_ocean` existed."""
    g = get_grid(16, 4, 1000.0)
    N = g.N
    surface = np.full((6, N, N), 100.0, np.float32)
    surface[0] = -500.0                 # face 0: the sea
    surface[1, 6:10, 6:10] = -300.0     # a walled-in basin on the far face

    ocean = hydro_run.open_ocean(surface, g, 0.02)
    assert ocean[0].all()
    assert not ocean[1].any()
    # the largest body is the sea however small it is, so there is always one
    assert hydro_run.open_ocean(surface, g, 0.99)[0].all()
    # and the flood then fills the basin to its rim: a lake, not a gulf
    filled = priority_flood_sphere(surface, ocean, g).filled
    assert np.allclose(filled[1, 6:10, 6:10], 100.0)
    assert np.allclose(filled[0], -500.0)


# --------------------------------------------------------------------------
# the stage on the stub world
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_hydro_spill(scratch):
    """The same world with the endorheic balance off, for the invariants
    that only hold when every lake overflows."""
    params = WorldParams.tiny_world(seed=5).with_overrides(hydro={"lake_evap": 0.0})
    params.hydro.river_threshold = 8.0
    store = _bake_to_erosion(scratch, "hydro_tiny_spill", params)
    info = hydro_run.run(store, params, _log)
    return store, params, info


@pytest.fixture(scope="module")
def tiny_hydro(scratch):
    params = WorldParams.tiny_world(seed=5)
    params.hydro.river_threshold = 8.0  # noise input: small basins, so a low threshold yields channels
    store = _bake_to_erosion(scratch, "hydro_tiny", params)
    info = hydro_run.run(store, params, _log)
    return store, params, info


def test_stage_outputs_and_requantile(tiny_hydro):
    store, params, info = tiny_hydro
    grid = params.coarse_grid()
    N = grid.N
    for n in ("water_surface", "flow_dir", "flow_acc"):
        assert store.has_field(n), n
    assert store.has("graph/drainage.json") and store.has("graph/lakes_coarse.json")
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    ws = store.load_field("water_surface", grid)
    fd = store.load_field("flow_dir", grid)
    acc = store.load_field("flow_acc", grid)
    assert ws.dtype == np.float32 and fd.dtype == np.uint8 and acc.dtype == np.float32
    surface = (h.interior + sed.interior).astype(np.float32)
    # the sea is the *connected* ocean, so `surface < 0` is not the mask: a
    # closed basin below sea level is land with a lake in it (open_ocean)
    ocean = fd.interior == OCEAN
    M = surface.size
    assert ocean.sum() <= (surface < 0).sum()
    assert int(((surface < 0) & ~ocean).sum()) == info["closed_basin_cells"]
    # sea level still puts exactly land_fraction of the cells above zero
    assert abs(int((surface >= 0).sum()) - round(params.world.land_fraction * M)) <= 2
    assert (ws.interior[ocean] == 0).all()
    assert (ws.interior[~ocean] >= surface[~ocean]).all()
    assert (fd.interior[ocean] == OCEAN).all() and (fd.interior[~ocean] < 8).all()
    assert (acc.interior[ocean] == 0).all()
    assert info["n_lakes"] > 0 and info["n_edges"] > 0
    # the water surface is the reference depression fill, *at most*: with
    # `hydro.lake_evap` on, a basin that cannot sustain a full pool settles
    # below its spill point (hydro/balance.py), and the exact-equality case
    # is `test_lake_evap_zero_is_the_spill_point_fill`
    ref = _reference_fill(surface, ocean, grid)
    assert (ws.interior[~ocean] <= ref[~ocean] + 1e-4).all()
    bal = info["lake_balance"]
    if bal["closed"] == 0 and bal["dry"] == 0:
        assert np.array_equal(ws.interior[~ocean], ref[~ocean])
    assert (ws.interior > surface + params.hydro.lake_min_depth)[~ocean].sum() == info["lake_cells"]


def test_requantile_exact():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(6, 8, 8)).astype(np.float32) * 100
    s = rng.random((6, 8, 8)).astype(np.float32)
    for lf in (0.3, 0.5, 0.05):
        h2, q = requantile_height(h, s, lf)
        assert h2.dtype == np.float32
        assert abs(int(((h2 + s) >= 0).sum()) - round(lf * h.size)) <= 1


def test_flow_dir_acyclic_and_reaches_ocean(tiny_hydro):
    store, params, _ = tiny_hydro
    grid = params.coarse_grid()
    fd = store.load_field("flow_dir", grid).interior
    land = (fd != OCEAN).reshape(-1)
    down = downstream_table(np.ascontiguousarray(fd), grid.owner, grid.H)
    _walk_all_land(down, land, grid.N)
    assert topological_order(down, land).size == land.sum()
    # some flow crosses face edges (cells whose downstream is on another face)
    N = grid.N
    cids = np.arange(down.size)
    cross = land & (down >= 0) & (down // (N * N) != cids // (N * N))
    assert cross.sum() > 0


def test_flow_acc_equals_precip_sum_per_basin(tiny_hydro_spill):
    """Accumulation is conservative and monotone downstream -- on a world
    where every depression overflows.  `hydro.lake_evap` deliberately breaks
    both below a closed lake (its outlet passes nothing on), which is what
    `test_a_closed_lake_passes_no_water_downstream` checks instead."""
    store, params, _ = tiny_hydro_spill
    grid = params.coarse_grid()
    fd = store.load_field("flow_dir", grid).interior
    acc = store.load_field("flow_acc", grid).interior.reshape(-1).astype(np.float64)
    precip = store.load_field("precip", grid).interior.reshape(-1).astype(np.float64)
    land = (fd != OCEAN).reshape(-1)
    down = downstream_table(np.ascontiguousarray(fd), grid.owner, grid.H)
    label = _outlet_labels(down, land)
    outlets = np.unique(label[land])
    assert outlets.size > 10
    sums = np.zeros(down.size)
    np.add.at(sums, label[land], precip[land])
    for o in outlets:
        assert abs(acc[o] - sums[o]) <= 1e-3 * sums[o] + 1e-6, o
    # accumulation includes the cell's own precip and is monotone downstream
    assert (acc[land] >= precip[land] * (1 - 1e-6)).all()
    d = down[land]
    ok = land[d]
    assert (acc[d[ok]] >= acc[np.nonzero(land)[0][ok]] * (1 - 1e-6)).all()


def test_drainage_graph_is_forest_with_valid_strahler(tiny_hydro):
    store, params, _ = tiny_hydro
    grid = params.coarse_grid()
    N = grid.N
    G = store.read_json("graph/drainage.json")
    nodes, edges = G["nodes"], G["edges"]
    assert nodes and edges
    ids = [n["id"] for n in nodes]
    assert ids == list(range(len(nodes)))
    out_deg = {}
    in_edges = {n["id"]: [] for n in nodes}
    kinds = {n["id"]: n["kind"] for n in nodes}
    for e in edges:
        out_deg[e["from"]] = out_deg.get(e["from"], 0) + 1
        in_edges[e["to"]].append(e)
        assert e["length_m"] > 0 and e["mean_discharge"] > 0 and len(e["cells"]) >= 2
        assert e["cells"][0] == nodes[e["from"]]["cell"] and e["cells"][-1] == nodes[e["to"]]["cell"]
    # forest: every node has at most one outgoing edge; outlets none, others exactly one
    for n in nodes:
        if n["kind"] == "outlet":
            assert out_deg.get(n["id"], 0) == 0
        else:
            assert out_deg.get(n["id"], 0) == 1, n
    # Strahler rule at every node, sources are 1
    for n in nodes:
        ins = in_edges[n["id"]]
        outs = [e for e in edges if e["from"] == n["id"]]
        if not outs:
            continue
        o = outs[0]["order"]
        if not ins:
            assert n["kind"] == "source" and o == 1
        else:
            m = max(e["order"] for e in ins)
            expect = m + 1 if sum(1 for e in ins if e["order"] == m) >= 2 else m
            assert o == expect, (n, ins, o)
        if n["kind"] == "junction":
            assert len(ins) >= 2
    # the graph follows flow_dir: every edge's cell chain is a downstream walk
    fd = store.load_field("flow_dir", grid).interior
    down = downstream_table(np.ascontiguousarray(fd), grid.owner, grid.H)
    for e in edges:
        for a, b in zip(e["cells"][:-1], e["cells"][1:]):
            ca = (a[0] * N + a[1]) * N + a[2]
            cb = (b[0] * N + b[1]) * N + b[2]
            assert down[ca] == cb
    # acyclic over nodes
    seen = set()
    for n in nodes:
        x = n["id"]
        path = set()
        while x is not None and x not in seen:
            assert x not in path
            path.add(x)
            outs = [e["to"] for e in edges if e["from"] == x]
            x = outs[0] if outs else None
        seen |= path


def test_lakes_json(tiny_hydro):
    store, params, _ = tiny_hydro
    grid = params.coarse_grid()
    N = grid.N
    L = store.read_json("graph/lakes_coarse.json")["lakes"]
    assert L
    h = store.load_field("height", grid).interior
    sed = store.load_field("sediment", grid).interior
    ws = store.load_field("water_surface", grid).interior
    fd = store.load_field("flow_dir", grid).interior
    surface = (h + sed).astype(np.float32)
    lake = (ws - surface) > params.hydro.lake_min_depth
    lake &= fd != OCEAN
    assert sum(l["area_cells"] for l in L) == int(lake.sum())
    down = downstream_table(np.ascontiguousarray(fd), grid.owner, grid.H)
    for l in L:
        f, i, j = l["outlet"]
        assert lake[f, i, j] and np.isclose(ws[f, i, j], l["surface_m"])
        assert l["area_m2"] > 0 and l["outlet_downstream"] is not None
        od = l["outlet_downstream"]
        assert down[(f * N + i) * N + j] == (od[0] * N + od[1]) * N + od[2]
        assert not lake[od[0], od[1], od[2]]  # the outlet cell drains out of the lake
        poly = l["polygon"]
        assert len(poly) >= 5 and poly[0] == poly[-1]
        for fu, u, v in poly:
            assert 0 <= u <= 1 and 0 <= v <= 1 and fu in l["faces"]
    assert not store.has("graph/lakes.json")  # derive's file, not hydro's
    # polygons are the counter-clockwise outer shore (positive shoelace area)
    from globe.hydro.lakes import ring_area

    for l in L:
        assert ring_area([[u * N, v * N] for _, u, v in l["polygon"]]) > 0
        assert all(ring_area([[u * N, v * N] for _, u, v in r]) > 0 for r in l["rings"])


def test_hydro_determinism(scratch):
    params = WorldParams.tiny_world(seed=9)
    params.hydro.river_threshold = 8.0
    outs = []
    for name in ("hydro_det_a", "hydro_det_b"):
        store = _bake_to_erosion(scratch, name, params)
        hydro_run.run(store, params, _log)
        blobs = {}
        for n in ("water_surface", "flow_dir", "flow_acc", "height"):
            blobs[n] = b"".join((store.coarse_dir / f"{n}.f{k}.npy").read_bytes() for k in range(6))
        for j in ("graph/drainage.json", "graph/lakes_coarse.json"):
            blobs[j] = (store.root / j).read_bytes()
        outs.append(blobs)
    assert outs[0] == outs[1]


def test_quicklook_writes_png(tiny_hydro, tmp_path):
    store, params, _ = tiny_hydro
    p = hydro_run.quicklook(store, params, tmp_path / "hydro.png")
    assert Path(p).exists() and Path(p).stat().st_size > 1000


# --------------------------------------------------------------------------
# evaporation-limited lake levels (hydro/balance.py)
# --------------------------------------------------------------------------
def _bowl_world(N=64, floor=-30.0, rim=40.0, radius=0.30):
    """One face carrying a round bowl in a plateau, the rest of the planet
    ocean.  Returns (grid, surface, ocean) on the interior."""
    grid = get_grid(N, 4, 4000.0)
    surface = np.full((6, N, N), -100.0, dtype=np.float32)
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    plateau = np.where(r < 0.45, rim, -100.0)
    bowl = np.where(r < radius, floor + (rim - floor) * (r / radius) ** 2, plateau)
    surface[0] = bowl.astype(np.float32)
    ocean = surface < 0
    ocean[0] = ocean[0] & (r >= 0.45)          # the bowl's floor is landlocked
    return grid, surface, ocean


def _solve_bowl(lake_evap, precip_scale=1.0, evap_scale=1.0, **kw):
    from globe.hydro.balance import balance_lakes
    grid, surface, ocean = _bowl_world(**kw)
    flood = priority_flood_sphere(surface, ocean, grid)
    fd = flow_directions(flood.filled, ocean, flood.parent, grid)
    down = downstream_table(fd, grid.owner, grid.H)
    topo = flood.pop_seq[::-1]
    topo = topo[~ocean.reshape(-1)[topo]]
    precip = np.full((6, grid.N, grid.N), precip_scale, dtype=np.float32)
    evap = np.full((6, grid.N, grid.N), evap_scale, dtype=np.float32)
    water, acc, info = balance_lakes(surface, flood.filled, ocean, flood.order, down, topo,
                                     precip, evap, grid, lake_evap)
    return surface, flood.filled, water, acc, info


def test_lake_evap_zero_is_the_spill_point_fill():
    """``lake_evap = 0`` must be a strict no-op: the water surface is the
    priority flood's filled DEM, which is what hydro did before the balance
    existed, so the switch cannot move a coastline by itself."""
    surface, filled, water, acc, info = _solve_bowl(0.0)
    assert np.array_equal(water, filled)
    assert info["closed"] == 0 and info["overflowing"] == 0


def test_a_basin_settles_below_its_rim_when_evaporation_can_take_the_inflow():
    """The balance: raise ``lake_evap`` and the bowl's water level falls,
    monotonically, from the spill point towards the floor."""
    surface, filled, _, _, _ = _solve_bowl(0.0)
    spill = float(filled[0].max() if False else filled[0][filled[0] > surface[0]].max())
    floor = float(surface[0].min())
    levels = []
    for le in (0.5, 2.0, 8.0, 32.0, 128.0):
        _, _, water, _, info = _solve_bowl(le)
        lv = float(water[0][water[0] > surface[0]].max()) if (water[0] > surface[0]).any() else floor
        levels.append(lv)
        assert floor <= lv <= spill
    assert levels == sorted(levels, reverse=True), levels
    assert levels[0] >= spill - 1e-3, "a weakly evaporating lake still overflows"
    assert levels[-1] < spill, "a strongly evaporating one does not"


def test_a_closed_lake_passes_no_water_downstream():
    """A lake that does not reach its spill point has no outlet, so the flow
    accumulation *at* its outlet cell is zero -- that, not ``flow_dir``, is
    what says the river is not there (see hydro/balance.py).  An overflowing
    one passes on exactly the surplus its evaporation could not take."""
    surface, filled, _, _, _ = _solve_bowl(0.0)
    dep = (filled > surface).reshape(-1)
    grid, _, ocean = _bowl_world()
    flood = priority_flood_sphere(surface, ocean, grid)
    cells = np.flatnonzero(dep)
    outlet = cells[np.argmin(np.asarray(flood.order)[cells])]

    _, _, _, acc_open, info_open = _solve_bowl(0.5)
    _, _, _, acc_shut, info_shut = _solve_bowl(128.0)
    assert info_open["overflowing"] == 1 and info_open["closed"] == 0
    assert info_shut["closed"] == 1 and info_shut["overflowing"] == 0
    assert acc_open[outlet] > 0.0        # spills into the sea
    assert acc_shut[outlet] == 0.0       # keeps every drop
    # the surplus is the inflow the full pool could not evaporate, so the
    # overflowing lake still passes on less than everything that fell on it
    assert acc_open[outlet] < acc_open.max()


def test_a_cold_lake_cannot_evaporate_and_still_overflows():
    """``evap`` is ``k_evap*max(T, 0)``, so a lake in a frozen place has no
    evaporative term at all and fills to its rim whatever ``lake_evap`` says.
    That is the right answer -- and it is why most of this world's
    depressions can never close."""
    surface, filled, water, _, info = _solve_bowl(1e6, evap_scale=0.0)
    assert np.array_equal(water, filled)
    assert info["overflowing"] == 1 and info["closed"] == 0


def test_the_balance_runs_in_the_stage(scratch):
    """End to end through ``hydro.run``: the stage reports the mix and the
    lakes it writes are no larger than the spill-point ones."""
    p = WorldParams.small_world(3)
    counts = {}
    for le in (0.0, 6.0):
        pp = p.with_overrides(hydro={"lake_evap": le})
        store = _bake_to_erosion(scratch, f"balance_{le}", pp)
        info = hydro_run.run(store, pp, _log)
        counts[le] = (info["lake_cells"], info["n_lakes"], info["lake_balance"])
    assert counts[6.0][0] <= counts[0.0][0]
    bal = counts[6.0][2]
    assert bal["depressions"] == bal["overflowing"] + bal["closed"] + bal["dry"]
    assert bal["cells_balanced"] <= bal["cells_spill"]
