"""Phase 0 tests: stub bake produces the layout, is deterministic, resumes."""
import json

import numpy as np
import pytest

from globe.config import STAGES, WorldParams
from globe.io.tiles import Tile, read_tile, tile_exists, write_tile
from globe.io.world_store import WorldStore
from globe.pipeline import bake


def _run(scratch, name, params, **kw):
    return bake(scratch / name, params, logger=lambda m: None, **kw)


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


def test_determinism(scratch):
    p1 = WorldParams.tiny_world(seed=7)
    p2 = WorldParams.tiny_world(seed=7)
    a = _run(scratch, "det_a", p1)
    b = _run(scratch, "det_b", p2)
    for s in STAGES:
        assert a.stage_info(s)["hash"] == b.stage_info(s)["hash"], s
    c = _run(scratch, "det_c", WorldParams.tiny_world(seed=8))
    assert c.stage_info("tectonics")["hash"] != a.stage_info("tectonics")["hash"]


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
