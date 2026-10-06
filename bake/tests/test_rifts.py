"""Rift grabens (tectonics.rift_graben_m, globe/tectonics/rifts.py): off, finalise is what it
was; on, a rift that is still opening has a narrow, deep trough along its land axis, closed
basins with sills between them, and the whole of it is in the uplift field as subsidence."""
import dataclasses

import numpy as np
import pytest
from scipy.spatial import cKDTree

from globe.config import TectonicsParams, WorldParams
import globe.tectonics.run as tect
from globe.tectonics import intraplate, rifts
from globe.tectonics.collision import interior_centers_flat

DEPTH = 2000.0


def _earth(seed=0, **kw):
    """The Earth preset on 512-cell faces with a thin cloud.  A graben has its own size in km,
    so the test needs Earth's radius and cells its floor is wider than (19.5 km here) -- on
    the toy bodies a cell is 78 km of the tectonic sphere -- but not Earth's 20000 segments.
    The 0.40 supercontinent is the one the rift tests of test_synth_dyn cut."""
    p = WorldParams()
    p.world = dataclasses.replace(p.world, seed=seed, N_c=512, cell_size_m=2.0 * 9773.0)
    # two steps are the whole run: the uplift window, which a rift's age is measured against, is those two
    p.tectonics = dataclasses.replace(p.tectonics, N_tect=64, segments=1500, steps=2, continental_fraction=0.40, rift_candidates=48, **kw)
    return p


def _finalise(sim, depth):
    sim.tp.rift_graben_m = depth
    try:
        return tect.finalise(sim)
    finally:
        sim.tp.rift_graben_m = 0.0


@pytest.fixture(scope="module")
def rifted():
    """A supercontinent with a force rift cut through it (test_synth_dyn's `_rifted`), as old
    as the uplift window, finalised with the grabens off and on."""
    sim = tect.initialise(_earth(), log=None)
    assert sim.tp is sim.params.tectonics
    for _ in range(int(sim.tp.steps)):
        sim.step()
    H = sim.heat.grid.H
    sim.heat_bg[:, H:-H, H:-H] = 0.0                   # the mantle under everything fully insulated
    ev = intraplate.rift(sim, np.random.default_rng(5), 1)
    assert ev["pairs"], ev
    pair = tuple(ev["pairs"][0])
    sim.rift_pairs[pair]["k0"] = int(sim.ref_step)     # open for the whole window...
    sim.rift_pairs[pair]["delta"] = sim.km(60.0)       # ...and 60 km by now
    return sim, pair, _finalise(sim, 0.0), _finalise(sim, DEPTH)


def test_the_default_is_off():
    assert TectonicsParams().rift_graben_m == 0.0


def test_off_the_rift_reaches_nothing_finalise_writes(rifted):
    sim, pair, off, on = rifted
    assert "rift_graben" not in off and off["_rifts"] == {}
    # ...and nothing reads the rift's books: the outputs are those of the same crust with none
    st = sim.rift_pairs.pop(pair)
    try:
        bare = _finalise(sim, 0.0)
    finally:
        sim.rift_pairs[pair] = st
    for name in ("bedrock", "uplift", "hardness"):
        assert np.array_equal(off[name].interior, bare[name].interior), name
    # on, with no rift, there is nothing to stamp
    st = sim.rift_pairs.pop(pair)
    sim.tp.rift_graben_m = DEPTH
    try:
        none = tect.finalise_bed(sim)
    finally:
        sim.rift_pairs[pair] = st
        sim.tp.rift_graben_m = 0.0
    assert none["rifts"]["rifts"] == 0 and none["rifts"]["basins"] == 0 and not none["graben"].any()
    assert np.array_equal(none["bed"], off["bedrock"].interior)


def test_a_rift_on_land_has_a_narrow_deep_graben(rifted):
    sim, pair, off, on = rifted
    info = on["_rifts"]
    assert info["rifts"] == 1 and info["rifts_on_land"] == 1 and info["basins"] >= 1, info
    g = on["rift_graben"].interior.astype(np.float64)
    bed0, bed = off["bedrock"].interior.astype(np.float64), on["bedrock"].interior.astype(np.float64)
    # away from the stamp the bedrock, the uplift and the sea level are bit for bit what they were
    away = g == 0.0
    assert away.mean() > 0.9
    assert np.array_equal(bed[away], bed0[away])
    assert np.array_equal(on["uplift"].interior[away], off["uplift"].interior[away])
    assert on["_sea_level_units"] == off["_sea_level_units"] and on["_scale_m_per_unit"] == off["_scale_m_per_unit"]
    assert np.array_equal(on["hardness"].interior, off["hardness"].interior)
    # the floors: a basin's share of the parameter's depth, on ground that was land
    lo, hi = rifts.DEPTH_SHARE
    assert lo * DEPTH <= info["deepest_m"] <= hi * DEPTH + 1e-6
    floor = g < -0.9 * lo * DEPTH
    assert floor.sum() > 20 and float(bed0[floor].min()) > 0.0
    # narrower than 100 km: the ground sunk by more than half a basin's least depth, over the basins' length
    area_km2 = sim.params.coarse_grid().interior_cell_area / 1e6
    width = float(area_km2[g < -0.5 * lo * DEPTH].sum()) / info["length_km"]
    assert 25.0 < width < 100.0, width
    # deeper than 1 km in the final bedrock: every floor cell under the ground 80-130 km from it
    centers = interior_centers_flat(sim.params.coarse_grid())
    tree = cKDTree(centers)
    R_km = sim.params.R_planet / 1000.0
    flat = bed.reshape(-1)
    for c in np.flatnonzero(floor.reshape(-1))[::7]:
        ring = np.setdiff1d(tree.query_ball_point(centers[c], 130.0 / R_km), tree.query_ball_point(centers[c], 80.0 / R_km))
        assert float(np.median(flat[ring])) - float(flat[c]) > 1000.0
    # the shoulders stand above what was there, by no more than their share of the depth
    assert 0.5 * rifts.SHOULDER * DEPTH < float(g.max()) <= rifts.SHOULDER * DEPTH + 1e-6
    # a floor under sea level is a few cells of the planet: the land fraction the erosion stage holds barely moves
    assert abs(float((bed >= 0).mean()) - float((bed0 >= 0).mean())) < 2e-3


def test_the_graben_is_all_subsidence(rifted):
    """`bedrock` less the uplift the erosion stage replays is the ground before the graben: the
    replay start has no trough for the basin fill to level."""
    sim, pair, off, on = rifted
    n_iter = int(sim.params.erosion.iterations)
    g = on["rift_graben"].interior.astype(np.float64)
    start0 = off["bedrock"].interior.astype(np.float64) - n_iter * off["uplift"].interior.astype(np.float64)
    start = on["bedrock"].interior.astype(np.float64) - n_iter * on["uplift"].interior.astype(np.float64)
    assert float(np.abs(start - start0).max()) < 0.05
    floor = g < -0.9 * rifts.DEPTH_SHARE[0] * DEPTH
    sink = (on["uplift"].interior.astype(np.float64) - off["uplift"].interior.astype(np.float64))[floor] * n_iter
    assert floor.any() and float(sink.max()) < -0.9 * rifts.DEPTH_SHARE[0] * DEPTH
    # ...and the shoulders rise during the stage as the floor sinks
    rim = g > 0.5 * rifts.SHOULDER * DEPTH
    assert float(((on["uplift"].interior.astype(np.float64) - off["uplift"].interior.astype(np.float64))[rim]).min()) > 0.0
    assert "graben" not in tect.finalise_bed(sim)


def test_a_rift_cut_inside_the_window_has_had_less_time(rifted):
    sim, pair, off, on = rifted
    k0 = sim.rift_pairs[pair]["k0"]
    assert rifts.rifting_pairs(sim) == [(pair[0], pair[1], 1.0)]
    sim.rift_pairs[pair]["k0"] = int(sim.step_index) - (int(sim.step_index) - int(sim.ref_step)) // 2
    sim.tp.rift_graben_m = DEPTH
    try:
        (a, b, share), = rifts.rifting_pairs(sim)
        fb = tect.finalise_bed(sim)
    finally:
        sim.rift_pairs[pair]["k0"] = k0
        sim.tp.rift_graben_m = 0.0
    assert share == pytest.approx(0.5)
    assert fb["rifts"]["deepest_m"] == pytest.approx(0.5 * on["_rifts"]["deepest_m"], rel=1e-6)
    assert np.allclose(fb["bed"], on["bedrock"].interior - 0.5 * on["rift_graben"].interior, atol=1e-2)
    # ...and one that has barely opened less still; held shut, it has no graben at all
    delta = sim.rift_pairs[pair]["delta"]
    try:
        sim.rift_pairs[pair]["delta"] = sim.km(0.25 * rifts.OPEN_KM)
        assert rifts.rifting_pairs(sim) == [(pair[0], pair[1], pytest.approx(0.25))]
        sim.rift_pairs[pair]["delta"] = 0.0
        assert rifts.rifting_pairs(sim) == []
    finally:
        sim.rift_pairs[pair]["delta"] = delta


def test_the_halves_of_a_rift_that_broke_through_still_count(rifted):
    sim, pair, off, on = rifted
    st = sim.rift_pairs.pop(pair)
    try:
        sim.rift_free[pair[0]] = sim.rift_free[pair[1]] = int(sim.step_index)
        assert rifts.rifting_pairs(sim) == [(min(pair), max(pair), 1.0)]
        # two plates freed on one step by different rifts are not a pair
        cut = sim.last_rift[pair[1]]
        sim.last_rift[pair[1]] = cut - 1
        try:
            assert rifts.rifting_pairs(sim) == []
        finally:
            sim.last_rift[pair[1]] = cut
        sim.rift_free[pair[0]] = sim.rift_free[pair[1]] = int(sim.step_index) - int(sim.steps_of(sim.tp.rift_free_my)) - 1
        assert rifts.rifting_pairs(sim) == []
    finally:
        sim.rift_free.pop(pair[0], None)
        sim.rift_free.pop(pair[1], None)
        sim.rift_pairs[pair] = st


def test_the_stage_saves_what_it_stamped(scratch):
    """The tectonics stage with the grabens on: the diagnostic is on disk and the stage info
    says what was stamped -- on a toy body, whose clock rifts hold no halves together, nothing."""
    from globe.io.world_store import WorldStore
    from globe.pipeline import bake

    p = WorldParams.tiny_world(2).with_overrides(tectonics={"rift_graben_m": DEPTH}, render={"viewer": False})
    bake(scratch / "rift_graben_on", p, to_stage="tectonics", logger=lambda m: None)
    store = WorldStore(scratch / "rift_graben_on")
    assert store.stage_info("tectonics")["info"]["rifts"] == {"rifts": 0, "rifts_on_land": 0, "basins": 0, "length_km": 0.0,
                                                                "deepest_m": 0.0, "cells": 0}
    g = np.stack([np.load(store.root / "diagnostics" / f"rift_graben.f{k}.npy") for k in range(6)])
    assert not g.any()
    q = WorldParams.tiny_world(2).with_overrides(render={"viewer": False})
    bake(scratch / "rift_graben_off", q, to_stage="tectonics", logger=lambda m: None)
    off = WorldStore(scratch / "rift_graben_off")
    assert off.stage_info("tectonics")["info"]["rifts"] == {} and not (off.root / "diagnostics" / "rift_graben.f0.npy").exists()
    grid = p.coarse_grid()
    for name in ("bedrock", "uplift"):
        assert np.array_equal(store.load_field(name, grid).data, off.load_field(name, grid).data), name


def test_basins_are_closed_and_strung_with_sills():
    s = np.arange(0.0, 3000.0, 5.0)
    land = s <= 2000.0                                     # 2000 km of land axis, then the shelf
    out = rifts.basins(s, land, np.random.default_rng(3), (150.0, 600.0), (40.0, 120.0), 50.0)
    assert len(out) >= 3
    for idx, e in out:
        assert e[0] < 0.05 and e[-1] < 0.05 and e.max() == 1.0             # shut at both ends
        assert 150.0 - 5.0 <= s[idx[-1]] - s[idx[0]] <= 600.0 and s[idx[-1]] <= 2000.0
    for (i0, _), (i1, _) in zip(out[:-1], out[1:]):
        assert 40.0 - 5.0 <= s[i1[0]] - s[i0[-1]] <= 120.0 + 10.0         # a sill between two
    lengths = [s[idx[-1]] - s[idx[0]] for idx, _ in out]
    assert max(lengths) > 1.3 * min(lengths)                               # and no two alike
    assert rifts.basins(s, s < 140.0, np.random.default_rng(3), (150.0, 600.0), (40.0, 120.0), 50.0) == []   # too short a stretch holds none
    # across strike: the floor, the scarp, the shoulder, and nothing beyond its reach
    d = np.array([0.0, 25.0, 40.0, 300.0])
    p = rifts.profile(d, 25.0, 15.0, 60.0)
    assert p[0] == -1.0 and p[1] == -1.0 and p[2] == pytest.approx(rifts.SHOULDER) and p[3] == 0.0
