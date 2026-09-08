"""Tiles stage (PLAN 10.3): vertex tiles, shared columns (also across cube
edges), the LOD pyramid, neighbours, index.json, determinism."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from globe.config import STAGES, WorldParams
from globe.cubesphere import to_sphere_v
from globe.io.tiles import LAYER_SEDIMENT_MAX, encode_discharge, read_tile, tile_dir, tile_exists, tiles_per_face
from globe.io.world_store import WorldStore
from globe.pipeline import bake
from globe.refine import lod as lodmod
from globe.refine import tiles as tl

def synthetic_fine(root: Path, params: WorldParams) -> WorldStore:
    """A tiny world with hand-made fine fields: smooth terrain, an ocean, a
    lake in a spherical cap, blocky basin ids, biome bands, river lines."""
    store = WorldStore(root, create=True)
    g = params.fine_grid()
    N = params.N_fine
    p = g.interior_centers  # (6, N, N, 3)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    rng = np.random.default_rng(5)
    height = (900.0 * z * x + 350.0 * np.sin(3.0 * y) + 200.0 * np.cos(4.0 * x * z)).astype(np.float32)
    sediment = np.clip(3.0 + 3.0 * np.sin(5.0 * y) * np.cos(2.0 * z), 0.0, 6.0).astype(np.float32)
    surface = height + sediment
    cap = np.array([0.3, 0.5, 0.81])
    cap /= np.linalg.norm(cap)
    in_cap = (p @ cap) > np.cos(0.35)
    lake_level = float(np.percentile(surface[in_cap & (surface > 0)], 60.0))
    ws = np.where(surface > 0, surface, 0.0)
    ws = np.where(in_cap & (surface < lake_level) & (surface > 0), lake_level, ws).astype(np.float32)
    discharge = np.exp(4.0 * (y + 1.0)).astype(np.float32)
    hardness = (0.5 + 0.4 * x).astype(np.float32)
    bid = (np.floor((x + 1.0) * 1.5) + 3 * np.floor((y + 1.0) * 1.5) + 9 * np.floor((z + 1.0) * 1.5)).astype(np.int32)
    basin_id = np.where(surface >= 0, bid, -1).astype(np.int32)
    biome = np.where(surface < 0, 0, 1 + np.clip((z + 1) * 3, 0, 5).astype(np.int64)).astype(np.uint8)
    vegetation = rng.integers(0, 256, size=height.shape, dtype=np.uint8)
    ii = np.arange(N)
    river_mask = np.zeros(height.shape, np.uint8)
    river_mask[:, ii % 13 == 3, :] = 1
    river_mask[:, :, ii % 17 == 5] = 1
    river_mask[surface < 0] = 0
    fine = store.root / "fine"
    fine.mkdir(exist_ok=True)
    for name, arr in (
        ("height", height), ("sediment", sediment), ("water_surface", ws), ("discharge", discharge), ("hardness", hardness),
        ("basin_id", basin_id), ("biome", biome), ("vegetation", vegetation), ("river_mask", river_mask),
    ):
        for k in range(6):
            np.save(fine / f"{name}.f{k}.npy", np.ascontiguousarray(arr[k]))
    store.lake_level = lake_level
    return store


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    params = WorldParams.tiny_world(seed=0)
    root = tmp_path_factory.mktemp("tiles") / "syn"
    store = synthetic_fine(root, params)
    info = tl.run(store, params, log=lambda m: None)
    lat = tl.build_lattices(store, params, work=root / "lat", log=lambda m: None)
    return {"store": store, "params": params, "info": info, "lat": lat, "lake_level": store.lake_level}


def _lat(world, chan, lod, f):
    return np.asarray(world["lat"].load(chan, lod, f))


def _all_tiles(params):
    for lod in range(params.max_lod + 1):
        n = tiles_per_face(params.N_fine, params.world.T, lod)
        for f in range(6):
            for x in range(n):
                for y in range(n):
                    yield lod, f, x, y, n


def test_every_tile_exists_and_index(world):
    store, params = world["store"], world["params"]
    T = params.world.T
    count = 0
    for lod, f, x, y, n in _all_tiles(params):
        assert tile_exists(store.root, lod, f, x, y), (lod, f, x, y)
        count += 1
    assert world["info"]["tiles"] == count == 6 * (16 + 4 + 1)
    idx = json.loads((store.root / "tiles" / "index.json").read_text())
    assert idx["T"] == T and idx["N_fine"] == params.N_fine and idx["max_lod"] == params.max_lod
    assert abs(idx["R_planet"] - params.R_planet) < 1e-6
    assert [d["lod"] for d in idx["lods"]] == list(range(params.max_lod + 1))
    for d in idx["lods"]:
        assert d["tiles_per_face"] == tiles_per_face(params.N_fine, T, d["lod"]) and d["size"] == T + 1
    t = read_tile(store.root, 0, 2, 1, 2)
    assert t.height.shape == (T + 1, T + 1) and t.layers.shape == (T + 1, T + 1, 4) and t.flow.shape == (T + 1, T + 1, 3)
    assert t.meta["size"] == T + 1 and len(t.meta["neighbors"]) == 4
    assert not (store.root / "tiles" / "_work").exists()


def test_lod0_lattice_semantics(world):
    """Interior LOD-0 vertices reduce the 2x2 fine cells around the corner
    per channel; edge vertices reduce 2 own + 2 neighbour cells."""
    store, params = world["store"], world["params"]
    N = params.N_fine
    cells = tl.FineCells(store.root, N)
    f = 3
    h = np.asarray(cells.face("height", f), np.float64) + np.asarray(cells.face("sediment", f))
    ws = np.asarray(cells.face("water_surface", f), np.float64)
    lake = np.where((ws - h > tl.LAKE_MIN_DEPTH_M) & (ws > 0), ws, 0.0)
    sed = np.asarray(cells.face("sediment", f), np.float64)
    bid = np.asarray(cells.face("basin_id", f))
    veg = np.asarray(cells.face("vegetation", f), np.int64)
    riv = np.asarray(cells.face("river_mask", f))

    def blocks(a):
        return np.stack([a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:]], axis=-1)

    H = _lat(world, "height", 0, f)
    assert H.shape == (N + 1, N + 1)
    assert np.allclose(H[1:-1, 1:-1], blocks(h).mean(-1), atol=1e-3)
    assert np.array_equal(_lat(world, "water", 0, f)[1:-1, 1:-1], blocks(lake).max(-1).astype(np.float32))
    assert np.array_equal(_lat(world, "river", 0, f)[1:-1, 1:-1], blocks(riv).max(-1))
    s = blocks(veg).sum(-1)
    assert np.array_equal(_lat(world, "vegetation", 0, f)[1:-1, 1:-1], ((2 * s + 4) // 8).astype(np.uint8))
    ref = np.clip(np.rint(blocks(sed).mean(-1) / LAYER_SEDIMENT_MAX * 255), 0, 255).astype(np.uint8)
    assert np.abs(_lat(world, "sediment", 0, f)[1:-1, 1:-1].astype(int) - ref.astype(int)).max() <= 1
    # mode with ties -> largest value (prefers land over ocean -1)
    B = _lat(world, "basin", 0, f)
    bb = blocks(bid)
    for i, j in [(5, 7), (N // 2, N // 3), (N - 2, 9)]:
        vals, cnt = np.unique(bb[i, j], return_counts=True)
        assert B[i + 1, j + 1] == vals[cnt == cnt.max()].max()
    # an edge vertex: 2 own cells + the 2 neighbour cells flanking the same edge vertex
    links = lodmod.edge_links(N)
    ln = links[f * 4 + 0]
    k = 11
    own = h[N - 1, k - 1 : k + 1]
    nbh = np.asarray(cells.face("height", ln.nb_face), np.float64) + np.asarray(cells.face("sediment", ln.nb_face))
    row = lodmod.side_row(nbh, ln.nb_side, 0)
    row = row[::-1] if ln.flip else row
    assert abs(H[N, k] - np.concatenate([own, row[k - 1 : k + 1]]).mean()) < 1e-3


def test_shared_columns_bit_equal_before_quantisation(world):
    """A vertex on a cube edge has the same value on both faces (every
    channel, every LOD) — checked on the float/int lattices."""
    params = world["params"]
    N = params.N_fine
    links = lodmod.edge_links(N)
    n_checked = 0
    for ch in tl.CHANNELS:
        for lod in range(params.max_lod + 1):
            g = world["lat"].getter(ch.name, lod)
            for f in range(6):
                own = np.asarray(g(f))
                for s in range(4):
                    a = lodmod.side_row(own, s, 0)
                    b = lodmod.neighbour_ring(g, links, f, s, 0)
                    assert a.dtype == b.dtype and np.array_equal(a, b), (ch.name, lod, f, s)
                    n_checked += a.size
    assert n_checked > 0


def _edge_column(t, side, T):
    """Row of tile samples lying on ``side`` (in that tile's along order)."""
    if side == 0:
        return t.height[T, :], t.water[T, :], t.layers[T], t.flow[T]
    if side == 1:
        return t.height[0, :], t.water[0, :], t.layers[0], t.flow[0]
    if side == 2:
        return t.height[:, T], t.water[:, T], t.layers[:, T], t.flow[:, T]
    return t.height[:, 0], t.water[:, 0], t.layers[:, 0], t.flow[:, 0]


def _global_ids(t):
    return tl.basin_lookup(t.meta["basins"])[t.flow[..., 1]]


def test_basin_local_ids_overflow():
    """More than 255 basins: meta["basins"] is capped at 255 entries and
    flow.G = 255 decodes to -1 for ocean and overflow vertices alike."""
    ids = np.arange(300).repeat(4).reshape(60, 20).astype(np.int32)
    ids[0, 0] = -1
    ids[1, :5] = 7  # most frequent -> local index 0
    basins, local = tl.basin_local_ids(ids)
    assert len(basins) == tl.NO_BASIN and basins[0] == 7 and len(set(basins)) == 255
    assert basins[1:] == sorted(basins[1:])  # ties by id
    assert local.dtype == np.uint8 and local.shape == ids.shape
    assert local[0, 0] == tl.NO_BASIN and (local[ids == 7] == 0).all()
    assert np.array_equal(local == tl.NO_BASIN, ~np.isin(ids, basins))
    dec = tl.basin_lookup(basins)[local]
    assert dec[0, 0] == -1 and (dec[local == tl.NO_BASIN] == -1).all()
    assert np.array_equal(dec[local != tl.NO_BASIN], ids[local != tl.NO_BASIN])
    # 255 or fewer basins: exact round trip
    small = (ids % 200).astype(np.int32)
    small[0, 0] = -1
    b2, l2 = tl.basin_local_ids(small)
    assert np.array_equal(tl.basin_lookup(b2)[l2], small)


def test_shared_columns_after_png_roundtrip(world):
    """The last column of tile (x, y) is the first column of (x+1, y) — also
    across a cube edge for the last tile — to 16-bit quantisation for the
    floats and exactly for the byte channels."""
    store, params = world["store"], world["params"]
    T, N = params.world.T, params.N_fine
    links = lodmod.edge_links(N)
    crossed = 0
    for lod, f, x, y, n in _all_tiles(params):
        ta = read_tile(store.root, lod, f, x, y)
        ga = _global_ids(ta)
        for s in range(4):
            lod2, f2, x2, y2 = ta.meta["neighbors"][s]
            tb = read_tile(store.root, lod2, f2, x2, y2)
            if f2 == f:
                s2 = s ^ 1  # the opposite side
                flip = False
            else:
                ln = links[f * 4 + s]
                s2, flip = ln.nb_side, ln.flip
                crossed += 1
            ha, wa, la, fa = _edge_column(ta, s, T)
            hb, wb, lb, fb = _edge_column(tb, s2, T)
            gb = _global_ids(tb)
            ga_col = ga[T, :] if s == 0 else ga[0, :] if s == 1 else ga[:, T] if s == 2 else ga[:, 0]
            gb_col = gb[T, :] if s2 == 0 else gb[0, :] if s2 == 1 else gb[:, T] if s2 == 2 else gb[:, 0]
            if flip:
                hb, wb, lb, fb, gb_col = hb[::-1], wb[::-1], lb[::-1], fb[::-1], gb_col[::-1]
            qa = (ta.meta["height_max"] - ta.meta["height_min"]) / 65535
            qb = (tb.meta["height_max"] - tb.meta["height_min"]) / 65535
            assert np.abs(ha - hb).max() <= 0.51 * (qa + qb) + 1e-4, (lod, f, x, y, s)
            assert np.array_equal(wa > 0, wb > 0)
            if (wa > 0).any():
                qw = (ta.meta["water_max"] - ta.meta["water_min"] + tb.meta["water_max"] - tb.meta["water_min"]) / 65534
                assert np.abs(wa[wa > 0] - wb[wa > 0]).max() <= 0.51 * qw + 1e-4
            assert np.array_equal(la, lb)
            assert np.array_equal(fa[:, 0], fb[:, 0]) and np.array_equal(fa[:, 2], fb[:, 2])
            assert np.array_equal(ga_col, gb_col)
    assert crossed == 6 * 4 * sum(tiles_per_face(N, T, l) for l in range(params.max_lod + 1))


def test_lod_heights_are_strided_lod0_vertices(world):
    store, params = world["store"], world["params"]
    T, N = params.world.T, params.N_fine
    for lod in range(params.max_lod):
        for f in range(6):
            a = _lat(world, "height", lod, f)
            b = _lat(world, "height", lod + 1, f)
            assert np.array_equal(b, a[::2, ::2])
    # through the PNGs: LOD l+1 tile vs its four LOD l tiles
    for lod in range(params.max_lod):
        n = tiles_per_face(N, T, lod + 1)
        for f in range(6):
            for x in range(n):
                for y in range(n):
                    coarse = read_tile(store.root, lod + 1, f, x, y)
                    q = (coarse.meta["height_max"] - coarse.meta["height_min"]) / 65535
                    for a in range(2):
                        for b in range(2):
                            fine = read_tile(store.root, lod, f, 2 * x + a, 2 * y + b)
                            qf = (fine.meta["height_max"] - fine.meta["height_min"]) / 65535
                            sub = coarse.height[a * T // 2 : a * T // 2 + T // 2 + 1, b * T // 2 : b * T // 2 + T // 2 + 1]
                            assert np.abs(sub - fine.height[::2, ::2]).max() <= 0.51 * (q + qf) + 1e-4


def test_pyramid_pooling_rules(world):
    """Face-interior LOD l+1 vertices reduce the 3x3 LOD-l neighbourhood:
    max (water, river), mode (basin, biome; ties -> largest), mean (bytes)."""
    params = world["params"]

    def win(a, k, m):
        return a[2 * k - 1 : 2 * k + 2, 2 * m - 1 : 2 * m + 2]

    for lod in range(params.max_lod):
        for f in (0, 5):
            n1 = (params.N_fine >> (lod + 1)) + 1
            W0, W1 = _lat(world, "water", lod, f), _lat(world, "water", lod + 1, f)
            R0, R1 = _lat(world, "river", lod, f), _lat(world, "river", lod + 1, f)
            B0, B1 = _lat(world, "basin", lod, f), _lat(world, "basin", lod + 1, f)
            M0, M1 = _lat(world, "biome", lod, f), _lat(world, "biome", lod + 1, f)
            V0, V1 = _lat(world, "vegetation", lod, f), _lat(world, "vegetation", lod + 1, f)
            for k in range(1, n1 - 1):
                for m in range(1, n1 - 1):
                    assert W1[k, m] == win(W0, k, m).max()
                    assert R1[k, m] == win(R0, k, m).max()
                    for A0, A1 in ((B0, B1), (M0, M1)):
                        vals, cnt = np.unique(win(A0, k, m), return_counts=True)
                        assert A1[k, m] == vals[cnt == cnt.max()].max()
                    s = int(win(V0, k, m).astype(np.int64).sum())
                    assert V1[k, m] == (2 * s + 9) // 18
    # the lake survives the pyramid where present (max-pool keeps the surface)
    for lod in range(params.max_lod + 1):
        lake = np.concatenate([_lat(world, "water", lod, f).ravel() for f in range(6)])
        assert (lake > 0).any()
        assert np.abs(lake[lake > 0] - world["lake_level"]).max() < 1e-3


def test_water_and_layers_channels(world):
    store, params = world["store"], world["params"]
    N = params.N_fine
    cells = tl.FineCells(store.root, N)
    has_water = 0
    for lod, f, x, y, n in _all_tiles(params):
        t = read_tile(store.root, lod, f, x, y)
        has_water += bool(t.meta["has_water"])
        if t.meta["has_water"]:
            assert (t.water > 0).any()
            assert np.abs(t.water[t.water > 0] - world["lake_level"]).max() < 0.05
        # basin-local ids index meta["basins"]; 255 only for ocean
        g = t.flow[..., 1]
        assert (g[g != tl.NO_BASIN] < len(t.meta["basins"])).all()
        assert -1 not in t.meta["basins"] and len(set(t.meta["basins"])) == len(t.meta["basins"])
        ids = _lat(world, "basin", lod, f)[x * params.world.T : (x + 1) * params.world.T + 1, y * params.world.T : (y + 1) * params.world.T + 1]
        assert np.array_equal(g == tl.NO_BASIN, ids < 0)
        assert np.array_equal(_global_ids(t), ids)
        assert set(t.meta["basins"]) == set(np.unique(ids[ids >= 0]).tolist())
    assert has_water > 0
    # ocean vertices have no water (implicit sea level); discharge is log-coded
    q = cells.face("discharge", 0)
    lat_q = _lat(world, "discharge", 0, 0)
    assert np.abs(lat_q[1:-1, 1:-1].astype(int) - encode_discharge(0.25 * (q[:-1, :-1] + q[1:, :-1] + q[:-1, 1:] + q[1:, 1:])).astype(int)).max() <= 1
    for lod, f, x, y, n in _all_tiles(params):
        t = read_tile(store.root, lod, f, x, y)
        assert (t.water[t.height < -1.0] == 0).all()


def test_neighbours_consistent(world):
    """Every tile's side-s neighbour lists the tile back exactly once, and
    the neighbour is the tile containing a point just beyond that side."""
    store, params = world["store"], world["params"]
    N, T = params.N_fine, params.world.T
    crossings = 0
    for lod, f, x, y, n in _all_tiles(params):
        t = read_tile(store.root, lod, f, x, y)
        nb = t.meta["neighbors"]
        assert len(nb) == 4 and len({tuple(b) for b in nb}) == 4
        w = 1.0 / n
        for s in range(4):
            lod2, f2, x2, y2 = nb[s]
            assert lod2 == lod
            back = read_tile(store.root, lod2, f2, x2, y2).meta["neighbors"]
            assert sum(1 for b in back if b == [lod, f, x, y]) == 1, (lod, f, x, y, s, back)
            if f2 != f:
                crossings += 1
            du, dv = ((1, 0), (-1, 0), (0, 1), (0, -1))[s]
            eps = 1e-6
            u = (x + 0.5 + du * (0.5 + eps)) * w
            v = (y + 0.5 + dv * (0.5 + eps)) * w
            p = to_sphere_v(np.array([f]), np.array([u]), np.array([v]))
            f3, x3, y3 = lodmod.tile_of_point(p, lod, N, T)
            assert [int(f3[0]), int(x3[0]), int(y3[0])] == [f2, x2, y2], (lod, f, x, y, s)
    assert crossings > 0


def test_determinism(tmp_path):
    params = WorldParams.tiny_world(seed=0)
    stores = []
    for name in ("a", "b"):
        st = synthetic_fine(tmp_path / name, params)
        tl.run(st, params, log=lambda m: None)
        stores.append(st)
    files = sorted(p.relative_to(stores[0].root / "tiles") for p in (stores[0].root / "tiles").rglob("*") if p.is_file())
    assert len(files) == 6 * 21 * 5 + 1
    for rel in files:
        a = (stores[0].root / "tiles" / rel).read_bytes()
        b = (stores[1].root / "tiles" / rel).read_bytes()
        assert a == b, rel


def test_pipeline_bake_tiny(tmp_path):
    """Through the runner with the upstream stages (real or stub): missing
    derive fields are zeros, quicklooks are written, hashes recorded."""
    params = WorldParams.tiny_world(seed=3)
    log = []
    store = bake(tmp_path / "w", params, logger=log.append)
    assert all(store.stage_done(s) for s in STAGES)
    assert (store.root / "quicklook" / "tiles.png").exists() and (store.root / "quicklook" / "tiles_basins.png").exists()
    assert (store.root / "tiles" / "index.json").exists()
    npf = tiles_per_face(params.N_fine, params.world.T, 0)
    t = read_tile(store.root, 0, 1, npf - 1, 0)
    assert t.height.shape == (params.world.T + 1,) * 2
    fine = store.root / "fine"
    if not (fine / "river_mask.f0.npy").exists():
        assert not t.flow[..., 2].any()
    assert store.stage_info("tiles")["info"]["tiles"] == 6 * (npf * npf + (npf // 2) ** 2 + 1)


def _suicide_job(args):  # module level: the executor pickles the callable by reference
    os._exit(9)


def test_dead_tile_worker_raises_instead_of_hanging(tmp_path, monkeypatch):
    """A worker that dies without returning (OOM kill) must break the pool,
    not block the bake forever: ``multiprocessing.Pool.imap_unordered``
    silently replaces the worker and never yields the in-flight result."""
    import multiprocessing as mp
    import types
    from concurrent.futures.process import BrokenProcessPool

    if "fork" not in mp.get_all_start_methods():  # pragma: no cover - not linux
        pytest.skip("needs the fork start method")

    monkeypatch.setattr(tl, "_write_tiles_job", _suicide_job)  # inherited by the forked workers
    params = WorldParams.tiny_world(seed=0)
    store = types.SimpleNamespace(root=tmp_path)
    ls = types.SimpleNamespace(work=tmp_path / "_work")
    with pytest.raises(BrokenProcessPool):
        tl.write_all_tiles(store, params, ls, log=lambda m: None, workers=2)


def test_edge_links_and_extend():
    links = lodmod.edge_links(64)
    assert len(links) == 24
    assert sum(ln.flip for ln in links) == 8  # 4 of the 12 cube edges reverse the along-index
    for ln in links:
        back = links[ln.nb_face * 4 + ln.nb_side]
        assert (back.nb_face, back.nb_side, back.flip) == (ln.face, ln.side, ln.flip)
    assert links == lodmod.edge_links(1024)  # geometry only
    # tile_neighbors agrees with in-face adjacency and is an involution across edges
    for n in (1, 2, 4):
        for f in range(6):
            for x in range(n):
                for y in range(n):
                    for s, (lod2, f2, x2, y2) in enumerate(lodmod.tile_neighbors(0, f, x, y, n, links)):
                        back = lodmod.tile_neighbors(0, f2, x2, y2, n, links)
                        assert [0, f, x, y] in back
    # extend: rings and validity pattern
    own = np.arange(9, dtype=np.int32).reshape(3, 3)
    rings = [np.full(3, 10 + s, np.int32) for s in range(4)]
    E, valid = lodmod.extend(own, rings, lattice=True)
    assert E.shape == (5, 5) and np.array_equal(E[1:-1, 1:-1], own)
    assert (E[-1, 1:-1] == 10).all() and (E[0, 1:-1] == 11).all() and (E[1:-1, -1] == 12).all() and (E[1:-1, 0] == 13).all()
    assert valid.sum() == 25 - 4 - 4
    E2, valid2 = lodmod.extend(own, rings, lattice=False)
    assert valid2.sum() == 25 - 4
