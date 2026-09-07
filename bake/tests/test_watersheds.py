"""Watershed partition tests (PLAN.md section 10.1 / 15): the partition
covers all land exactly once, basins respect the size bounds (documented
exceptions carry a reason), never cross a face edge, the hierarchy is a
tree, bbox / tiles / outlets are right, and the stage is deterministic."""
import numpy as np
import pytest

from globe.config import WorldParams
from globe.cubesphere import Grid
from globe.hydro import watersheds as ws_stage
from globe.hydro.d8 import OCEAN, cid_fij, downstream_table
from globe.hydro.priority_flood import priority_flood_sphere
from globe.hydro.routing import channel_network, flow_directions
from globe.hydro import run as hydro_run
from globe.hydro.watersheds import partition
from globe.io.world_store import WorldStore
from globe.stubs import stub_climate, stub_erosion, stub_tectonics


def _log(_msg):
    pass


def _stub_world_to_hydro(path, params):
    """Stub tectonics/climate/erosion + the real hydro stage (no dependency
    on the other engineers' real upstream implementations)."""
    store = WorldStore(path, create=True)
    store.init_manifest(params, params.coarse_grid())
    for s, fn in (("tectonics", stub_tectonics), ("climate", stub_climate), ("erosion", stub_erosion)):
        fn(store, params, _log)
        store.mark_stage(s, [], {"stub": True}, 0.0, params=params)
    hydro_run.run(store, params, _log)
    store.mark_stage("hydro", hydro_run.OUTPUTS, {}, 0.0, params=params)
    return store


# --------------------------------------------------------------------------
# invariants shared by the synthetic and the stub-world checks
# --------------------------------------------------------------------------
def _check_partition(basin_id, basins, flow_dir, grid, max_cells, min_cells, R, T):
    N, H = grid.N, grid.H
    NN = N * N
    land = flow_dir != OCEAN
    ids = np.array([b["id"] for b in basins])
    assert np.array_equal(ids, np.arange(len(basins)))  # dense ids
    # covers all land exactly once, ocean is -1
    assert (basin_id[~land] == -1).all()
    assert (basin_id[land] >= 0).all()
    counts = np.bincount(basin_id[land], minlength=len(basins))
    assert counts.size == len(basins) and (counts > 0).all()
    assert sum(b["area_cells"] for b in basins) == int(land.sum())
    down = downstream_table(np.ascontiguousarray(flow_dir), grid.owner, H)
    by_id = {b["id"]: b for b in basins}
    for b in basins:
        bid = b["id"]
        sel = basin_id == bid
        f_cells = np.unique(np.nonzero(sel)[0])
        assert f_cells.size == 1 and int(f_cells[0]) == b["face"]  # never crosses a face edge
        assert b["area_cells"] == int(sel.sum()) == counts[bid]
        assert b["area_cells"] <= max_cells
        if b["area_cells"] < min_cells:
            assert b["undersized_reason"] in ("island", "edge", "max")
        # bbox (exclusive) is tight
        _, ii, jj = np.nonzero(sel)
        assert b["bbox"] == [int(ii.min()), int(jj.min()), int(ii.max()) + 1, int(jj.max()) + 1]
        # tiles: every LOD-0 tile touched by the basin's fine cells, exactly
        fi = np.concatenate([ii * R + r for r in range(R)])
        fj = np.concatenate([jj * R + r for r in range(R)])
        tiles = {(int(x), int(y)) for x, y in zip(fi // T, fj // T)}
        assert {tuple(t) for t in b["tiles"]} == tiles
        # outlet is in the basin; its downstream cell is outside (ocean / other basin / other face)
        of, oi, oj = b["outlet"]
        assert basin_id[of, oi, oj] == bid and of == b["face"]
        oc = (of * N + oi) * N + oj
        d = int(down[oc])
        if b["downstream_basin"] == -1:
            assert not land.reshape(-1)[d]
            assert b["parent"] == -1
        else:
            df, di, dj = cid_fij(d, N)
            assert b["outlet_downstream"] == [df, di, dj]
            assert basin_id[df, di, dj] == b["downstream_basin"] != bid
            assert b["parent"] == b["downstream_basin"]
            assert b["downstream_basin"] in by_id
        assert b["order"] >= 0
    # hierarchy is a tree (parent pointers acyclic, roots have parent -1)
    for b in basins:
        seen = set()
        x = b["id"]
        while x != -1:
            assert x not in seen, "parent cycle"
            seen.add(x)
            x = by_id[x]["parent"]
    assert any(b["parent"] == -1 for b in basins)
    # exits: exactly the cells whose downstream lies outside the basin,
    # outlet first, kinds right; every cell of a basin reaches one of its
    # listed exits without leaving the basin (refine floods / sinks there)
    lab = basin_id.reshape(-1)
    landf = land.reshape(-1)
    land_idx = np.nonzero(landf)[0]
    d = down[land_idx]
    is_exit = (d < 0) | (lab[np.maximum(d, 0)] != lab[land_idx])
    exit_cells = land_idx[is_exit]
    exit_of_basin = {}
    for c in exit_cells.tolist():
        exit_of_basin.setdefault(int(lab[c]), set()).add(c)
    n_multi = 0
    for b in basins:
        ex = [(f * N + i) * N + j for f, i, j in b["exits"]]
        assert ex[0] == (b["outlet"][0] * N + b["outlet"][1]) * N + b["outlet"][2]
        assert len(set(ex)) == len(ex) and set(ex) == exit_of_basin[b["id"]]
        assert len(b["exit_kinds"]) == len(ex)
        for c, kind in zip(ex, b["exit_kinds"]):
            dc = int(down[c])
            want = "ocean" if (dc < 0 or lab[dc] < 0) else ("face" if dc // NN != c // NN else "basin")
            assert kind == want and (dc < 0 or lab[dc] != b["id"])
        n_multi += len(ex) > 1
        if b["downstream_basin"] == -1:
            assert b["exit_kinds"][0] == "ocean"
    # reach an exit: label every land cell with the first exit on its path
    is_exit_full = np.zeros(lab.size, bool)
    is_exit_full[exit_cells] = True
    reach = ws_stage._label_outlets(down, ws_stage.topological_order(down, landf), is_exit_full)
    assert (reach[land_idx] >= 0).all()
    assert np.array_equal(lab[reach[land_idx]], lab[land_idx])  # exit reached lies in the same basin
    return n_multi


# --------------------------------------------------------------------------
# synthetic world: big drainage system crossing the +X/+Z edge -> splits,
# edge cuts, merges of coastal micro-basins
# --------------------------------------------------------------------------
def _synthetic_world(N=32):
    g = Grid(N, 4, 50.0)
    c = g.interior_centers
    a = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    d = c @ a
    # land cap around the +X/+Z edge rising towards it, with a valley along
    # the y = 0 great circle so that flow gathers into two big rivers that
    # cross the edge region on their way to the coast
    valley = 300.0 * np.clip(1.0 - np.abs(c[..., 1]) / 0.35, 0, 1)
    surface = (1200.0 * (d - 0.25) - valley).astype(np.float32)
    ocean = surface < 0
    res = priority_flood_sphere(surface, ocean, g)
    fd = flow_directions(res.filled, ocean, res.parent, g)
    return g, surface, fd


@pytest.fixture(scope="module")
def synthetic():
    return _synthetic_world()


def test_partition_synthetic_splits_and_edge_cuts(synthetic):
    g, surface, fd = synthetic
    N = g.N
    max_cells, min_cells = 150, 9
    R, T = 2, 16
    part = partition(fd, g, max_cells, min_cells, R=R, T=T)
    basins = part.basins
    n_multi = _check_partition(part.basin_id, basins, fd, g, max_cells, min_cells, R, T)
    land = fd != OCEAN
    assert part.info["n_splits"] > 0 and part.info["n_merges"] > 0
    assert n_multi == part.info["n_multi_exit"] > 0  # merged coastal strips have several exits
    assert len(basins) >= land.sum() // max_cells
    # channels crossing a face edge produce child basins whose outlet's
    # downstream cell is on another face
    down = downstream_table(np.ascontiguousarray(fd), g.owner, g.H)
    edge_children = [b for b in basins if b["downstream_basin"] >= 0 and b["outlet_downstream"][0] != b["face"]]
    assert edge_children
    for b in edge_children:
        of, oi, oj = b["outlet"]
        d = int(down[(of * N + oi) * N + oj])
        assert d // (N * N) != of
    # split children: outlet's downstream cell is in another basin on the same face
    split_children = [b for b in basins if b["downstream_basin"] >= 0 and b["outlet_downstream"][0] == b["face"]]
    assert split_children
    # the split point sits on a channel: Strahler order is recorded from the
    # network when given
    thr = 8.0
    acc = np.ones(fd.shape, np.float32)
    from globe.hydro.routing import accumulate, topological_order

    lf = land.reshape(-1)
    topo = topological_order(down, lf)
    acc = accumulate(acc, down, topo).astype(np.float32).reshape(fd.shape)
    acc[~land] = 0
    _, _, cell_order = channel_network(fd, acc, thr, g)
    part2 = partition(fd, g, max_cells, min_cells, cell_order=cell_order, R=R, T=T)
    assert [b["outlet"] for b in part2.basins] == [b["outlet"] for b in basins]  # order does not change the partition
    assert max(b["order"] for b in part2.basins) >= 1  # (the planar synthetic valley has no confluences)
    for b in part2.basins:
        of, oi, oj = b["outlet"]
        assert b["order"] == int(cell_order[of, oi, oj])


def test_partition_lake_cells_avoided_as_split_points(synthetic):
    g, surface, fd = synthetic
    lake = np.zeros(fd.shape, bool)
    # declare a band of cells 'lake': split outlets must avoid them
    lake[:, 10:14, :] = True
    lake &= fd != OCEAN
    part = partition(fd, g, 150, 9, lake=lake, R=2, T=16)
    for b in part.basins:
        if b["downstream_basin"] >= 0 and b["outlet_downstream"][0] == b["face"]:
            of, oi, oj = b["outlet"]
            assert not lake[of, oi, oj]


def test_partition_is_deterministic(synthetic):
    g, surface, fd = synthetic
    a = partition(fd, g, 150, 9, R=2, T=16)
    b = partition(fd, g, 150, 9, R=2, T=16)
    assert np.array_equal(a.basin_id, b.basin_id)
    assert a.basins == b.basins


# --------------------------------------------------------------------------
# the stage on the stub world (tiny preset, real hydro upstream)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_world(scratch):
    params = WorldParams.tiny_world(seed=4)
    params.hydro.river_threshold = 8.0
    store = _stub_world_to_hydro(scratch / "ws_tiny", params)
    info = ws_stage.run(store, params, _log)
    return store, params, info


def test_stage_partition_invariants(tiny_world):
    store, params, info = tiny_world
    grid = params.coarse_grid()
    bid = store.load_field("basin_id", grid)
    assert bid.dtype == np.int32
    fd = store.load_field("flow_dir", grid).interior
    basins = store.read_json("graph/basins.json")["basins"]
    wp = params.watersheds
    _check_partition(bid.interior, basins, fd, grid, wp.basin_max_cells, wp.basin_min_cells, params.world.R, params.world.T)
    assert info["n_basins"] == len(basins)
    assert set(basins[0]) >= {"id", "parent", "face", "outlet", "downstream_basin", "area_cells", "bbox", "order", "tiles"}


def test_stage_determinism(scratch, tiny_world):
    store_a, params, _ = tiny_world
    store_b = _stub_world_to_hydro(scratch / "ws_tiny_b", params)
    ws_stage.run(store_b, params, _log)
    for k in range(6):
        assert (store_a.coarse_dir / f"basin_id.f{k}.npy").read_bytes() == (store_b.coarse_dir / f"basin_id.f{k}.npy").read_bytes()
    assert (store_a.root / "graph/basins.json").read_bytes() == (store_b.root / "graph/basins.json").read_bytes()


def test_quicklook(tiny_world, tmp_path):
    store, params, _ = tiny_world
    p = ws_stage.quicklook(store, params, tmp_path / "ws.png")
    assert p.exists() and p.stat().st_size > 1000
