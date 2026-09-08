"""Phase 0 tests: stub bake produces the layout, is deterministic, resumes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_viz import seam_discontinuity

from globe.config import PRESETS, STAGES, WorldGroup, WorldParams
from globe.cubesphere import BASES, HALF_PI, Grid, from_sphere_v
from globe.io.tiles import Tile, read_tile, tile_exists, tiles_per_face, write_tile
from globe.io.world_store import WorldStore
from globe.pipeline import bake

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bake.py"


def _run(scratch, name, params, **kw):
    return bake(scratch / name, params, logger=lambda m: None, **kw)


def _cli_bake(scratch, name, *extra, hashseed="0"):
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    subprocess.run(
        [sys.executable, str(SCRIPT), "--world", name, "--preset", "tiny", "--seed", "7", "--worlds-dir", str(scratch), *extra],
        check=True,
        capture_output=True,
        env=env,
    )
    m = json.loads((scratch / name / "manifest.json").read_text())
    return {s: m["stages"][s]["hash"] for s in STAGES if s in m["stages"]}


def test_stub_bake_layout(scratch):
    params = WorldParams.tiny_world(seed=3)
    store = _run(scratch, "layout", params)
    root = store.root
    assert (root / "manifest.json").exists()
    for n in ("bedrock", "uplift", "hardness", "plate_id", "plate_vel", "temperature", "wind", "precip", "height", "sediment", "discharge", "momentum", "water_surface", "flow_dir", "flow_acc", "basin_id", "biome"):
        assert store.has_field(n), n
    for j in ("graph/drainage.json", "graph/basins.json", "graph/lakes.json"):
        assert (root / j).exists(), j
    for s in STAGES:
        assert store.stage_done(s)
        assert (root / "quicklook" / f"{s}.png").exists()
    npf = params.N_fine // params.world.T
    assert tile_exists(root, 0, 5, npf - 1, npf - 1)
    assert tile_exists(root, params.max_lod, 0, 0, 0)
    t = read_tile(root, 0, 0, 0, 0)
    assert t.height.shape == (params.world.T + 1, params.world.T + 1)
    m = json.loads((root / "manifest.json").read_text())
    assert m["params_hash"] == params.content_hash()
    # PLAN 15: no gradient discontinuity across face edges in any stage output
    grid = params.coarse_grid()
    for n in store.field_names():
        f = store.load_field(n, grid)
        if np.issubdtype(f.dtype, np.integer):
            continue
        assert seam_discontinuity(f) < 3.0, n


def _edge_vertex_neighbour(face, side, k, N):
    """Exact cube-edge point of vertex k on ``side`` of ``face`` (a fine-cell
    corner at u or v = 0/1) and its (face, iu, iv) vertex index on the face
    that owns it (built from BASES: to_sphere(face, 1.0, v) is off by an ulp
    because tan(pi/4) != 1 in double, which flips the face tie-break)."""
    r, up, n = BASES[face]
    t = np.tan((k / N - 0.5) * HALF_PI)
    p = (n + r + t * up, n - r + t * up, n + t * r + up, n + t * r - up)[side]
    f2, u2, v2 = from_sphere_v(p / np.linalg.norm(p))
    f2, iu, iv = int(f2), float(u2) * N, float(v2) * N
    # the edge vertex is a fine-cell corner on the neighbouring face too
    assert abs(iu - round(iu)) < 1e-9 and abs(iv - round(iv)) < 1e-9, (face, side, k, iu, iv)
    return f2, int(round(iu)), int(round(iv))


def test_fine_vertex_field_agrees_across_faces():
    """The tile sample convention (globe.io.tiles): vertex values computed on
    either side of a cube edge from a smooth field agree to interpolation
    tolerance (O(h^2)) at exactly coincident vertex positions."""
    from globe.stubs import fine_vertex_field

    fn = lambda p: np.sin(3 * p[..., 0]) * p[..., 1] + p[..., 2] ** 2
    prev = None
    for N in (32, 64):
        g = Grid(N, 4, 50.0)
        V = fine_vertex_field(g, fn(g.interior_centers), "t")
        assert V.shape == (6, N + 1, N + 1)
        worst = 0.0
        for f in range(6):
            for side in range(4):
                for k in range(N + 1):
                    f2, iu, iv = _edge_vertex_neighbour(f, side, k, N)
                    if f2 == f:
                        continue  # tie-break kept this face; checked from the other side
                    va = (V[f, N, k], V[f, 0, k], V[f, k, N], V[f, k, 0])[side]
                    worst = max(worst, abs(va - V[f2, iu, iv]))
        assert worst < 2e-3, (N, worst)
        assert prev is None or worst < prev / 2.5, (N, worst, prev)  # O(h^2)
        prev = worst


def test_tile_edge_columns_are_vertex_aligned(scratch):
    """Tiles are (T+1)^2 vertex samples on fine-cell corners, so the last
    column of the last tile of a face lies exactly on the cube edge and
    coincides with a column of the neighbouring face's tile.  The shared
    values agree closely (the stub fine fields are piecewise constant, so
    the cross-face 2x2 averages differ by cubic-interpolation error at the
    EAC offset, up to ~1% of the tile range; a column mapped one vertex
    off differs by ~6% max / ~2% mean)."""
    params = WorldParams.tiny_world(seed=3)
    store = _run(scratch, "layout", params)
    Nf, T = params.N_fine, params.world.T
    npf = tiles_per_face(Nf, T, 0)
    face = 4
    n_checked = 0
    rel = []
    for y in range(npf):
        ta = read_tile(store.root, 0, face, npf - 1, y)
        for l in range(T + 1):
            f2, iu, iv = _edge_vertex_neighbour(face, 0, y * T + l, Nf)
            assert f2 != face  # +Z's u = 1 edge belongs to +X by the tie-break
            x2, y2 = min(iu // T, npf - 1), min(iv // T, npf - 1)
            tb = read_tile(store.root, 0, f2, x2, y2)
            ha = ta.height[T, l]
            hb = tb.height[iu - x2 * T, iv - y2 * T]
            rel.append(abs(ha - hb) / (ta.meta["height_max"] - ta.meta["height_min"]))
            n_checked += 1
    assert n_checked == npf * (T + 1)
    assert max(rel) < 0.03 and float(np.mean(rel)) < 0.005, (max(rel), np.mean(rel))


def test_determinism(scratch):
    p1 = WorldParams.tiny_world(seed=7)
    p2 = WorldParams.tiny_world(seed=7)
    a = _run(scratch, "det_a", p1)
    b = _run(scratch, "det_b", p2)
    for s in STAGES:
        assert a.stage_info(s)["hash"] == b.stage_info(s)["hash"], s
    c = _run(scratch, "det_c", WorldParams.tiny_world(seed=8))
    for s in STAGES:
        assert c.stage_info(s)["hash"] != a.stage_info(s)["hash"], s


def test_cli_bake_is_deterministic_across_processes(scratch):
    """PLAN 4 acceptance / PLAN 15: ``bake.py`` twice with the same seed ->
    identical hashes, in separate processes with different hash seeds, and
    equal to the in-process result."""
    a = _cli_bake(scratch, "cli_a", hashseed="1")
    b = _cli_bake(scratch, "cli_b", hashseed="4242")
    assert set(a) == set(STAGES) and a == b
    inproc = _run(scratch, "cli_inproc", WorldParams.tiny_world(seed=7))
    assert {s: inproc.stage_info(s)["hash"] for s in STAGES} == a  # CLI == in-process
    # stage range / resume through the CLI
    assert list(_cli_bake(scratch, "cli_range", "--to", "erosion")) == list(STAGES[:3])
    r = _cli_bake(scratch, "cli_range", "--from", "hydro")
    assert set(r) == set(STAGES) and r["tectonics"] == a["tectonics"]


def test_resume_and_param_guard(scratch):
    params = WorldParams.tiny_world(seed=1)
    store = _run(scratch, "resume", params, to_stage="erosion")
    assert store.stage_done("erosion") and not store.stage_done("hydro")
    h_before = store.stage_info("tectonics")["hash"]
    store = _run(scratch, "resume", params, from_stage="hydro")
    assert store.stage_done("tiles")
    assert store.stage_info("tectonics")["hash"] == h_before
    with pytest.raises(RuntimeError):
        _run(scratch, "resume", WorldParams.tiny_world(seed=2))
    with pytest.raises(RuntimeError):
        _run(scratch, "fresh", params, from_stage="erosion")  # upstream missing


def test_crashed_stage_is_not_marked_done(scratch, monkeypatch):
    import globe.erosion.run as erosion_run
    from globe.field import FaceField

    params = WorldParams.tiny_world(seed=11)
    store = _run(scratch, "crash", params)
    h_erosion = store.stage_info("erosion")["hash"]

    def broken(store, params, log=print):
        grid = params.coarse_grid()
        store.save_field(FaceField.full(grid, 1.0, name="height"))
        raise RuntimeError("boom")

    monkeypatch.setattr(erosion_run, "run", broken)
    with pytest.raises(RuntimeError, match="boom"):
        _run(scratch, "crash", params, from_stage="erosion")
    store = WorldStore(scratch / "crash")
    assert not store.stage_done("erosion") and not store.stage_done("hydro") and not store.stage_done("tiles")
    assert store.stage_done("climate")
    monkeypatch.undo()
    log = []
    store = bake(scratch / "crash", params, logger=log.append)
    assert not any("[erosion] already done" in m for m in log)
    assert all(store.stage_done(s) for s in STAGES)
    assert store.stage_info("erosion")["hash"] == h_erosion


def test_upstream_param_drift_is_detected(scratch):
    params = WorldParams.tiny_world(seed=12)
    _run(scratch, "drift", params)
    changed = WorldParams.tiny_world(seed=12)
    changed.tectonics.steps += 1
    with pytest.raises(RuntimeError, match="tectonics"):
        _run(scratch, "drift", changed, from_stage="erosion", force=True)


def test_runtime_knobs_do_not_invalidate(scratch):
    params = WorldParams.tiny_world(seed=13)
    _run(scratch, "knobs", params)
    knobs = WorldParams.tiny_world(seed=13)
    knobs.refine.workers = 3
    knobs.erosion.checkpoint_every = 1
    knobs.erosion.resume = False  # only chooses whether checkpoints/ is read
    knobs.render.view_distance_m = 1.0
    assert knobs.content_hash() == params.content_hash()
    log = []
    store = bake(scratch / "knobs", knobs, logger=log.append)
    assert all(store.stage_done(s, knobs) for s in STAGES)
    assert sum("already done" in m for m in log) == len(STAGES)


def test_rerun_clears_stale_outputs(scratch):
    params = WorldParams.tiny_world(seed=14)
    store = _run(scratch, "stale", params)
    h_tiles = store.stage_info("tiles")["hash"]
    junk = store.root / "tiles" / "L0" / "junk.txt"
    junk.write_text("stale")
    store = _run(scratch, "stale", params, from_stage="tiles")
    assert not junk.exists()
    assert store.stage_info("tiles")["hash"] == h_tiles


def test_params_validation(scratch):
    with pytest.raises(ValueError, match="power of two"):
        WorldParams.from_dict({"world": {"N_c": 1536, "R": 4, "T": 256}})  # 24 tiles per face
    with pytest.raises(ValueError, match="multiple"):
        WorldParams.from_dict({"world": {"N_c": 1000, "R": 4, "T": 256}})
    p = WorldParams.tiny_world()
    p.world = WorldGroup(N_c=48, R=2, T=16)  # post-hoc mutation: 6 tiles per face
    with pytest.raises(ValueError):
        bake(scratch / "invalid", p)
    assert not (scratch / "invalid").exists()  # validated before touching the disk
    # every preset (and the PLAN default) tiles exactly down to 1 tile per face
    for make in list(PRESETS.values()) + [lambda: WorldParams.from_dict({"world": {"N_c": 1024, "R": 4, "T": 256}})]:
        q = make()
        Nf, T = q.N_fine, q.world.T
        for lod in range(q.max_lod + 1):
            assert tiles_per_face(Nf, T, lod) * (T << lod) == Nf, (Nf, T, lod)
        assert tiles_per_face(Nf, T, q.max_lod) == 1


def test_tile_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    T = 16
    h = (rng.random((T + 1, T + 1)) * 1000 - 200).astype(np.float32)
    w = np.where(rng.random((T + 1, T + 1)) > 0.7, h + 3.0, 0).astype(np.float32)
    layers = rng.integers(0, 255, (T + 1, T + 1, 4), dtype=np.uint8)
    flow = rng.integers(0, 255, (T + 1, T + 1, 3), dtype=np.uint8)
    write_tile(tmp_path, Tile(1, 2, 3, 4, h, w, layers, flow, {"basins": [1, 2]}))
    t = read_tile(tmp_path, 1, 2, 3, 4)
    assert np.abs(t.height - h).max() < (h.max() - h.min()) / 65535 * 1.01
    assert np.array_equal(t.water > 0, w > 0)
    assert np.abs(t.water[w > 0] - w[w > 0]).max() < 0.05
    assert np.array_equal(t.layers, layers) and np.array_equal(t.flow, flow)
    assert t.meta["basins"] == [1, 2]


def test_params_yaml_roundtrip(tmp_path):
    p = WorldParams.small_world(seed=5)
    p.to_yaml(tmp_path / "p.yaml")
    q = WorldParams.from_yaml(tmp_path / "p.yaml")
    assert q.content_hash() == p.content_hash()
    with pytest.raises(KeyError):
        WorldParams.from_dict({"erosion": {"nope": 1}})
    r1 = p.rng("erosion", 3).random(4)
    r2 = WorldParams.small_world(seed=5).rng("erosion", 3).random(4)
    assert np.array_equal(r1, r2)
    # sub-key streams are distinct across key lengths and accept negative ids
    assert not np.array_equal(p.rng("refine").random(4), p.rng("refine", 0).random(4))
    assert not np.array_equal(p.rng("refine", 0).random(4), p.rng("refine", 0, 0).random(4))
    assert not np.array_equal(p.rng("refine", 1).random(4), p.rng("refine", 1, 0).random(4))
    assert np.array_equal(p.rng("refine", -1).random(4), WorldParams.small_world(seed=5).rng("refine", -1).random(4))
    assert not np.array_equal(p.rng("refine", -1).random(4), p.rng("refine", 1).random(4))
    # seed + offset aliases across stages are impossible only if offsets differ; document the layout
    assert not np.array_equal(p.rng("tectonics").random(4), p.rng("climate").random(4))
