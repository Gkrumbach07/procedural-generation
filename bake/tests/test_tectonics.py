"""Tectonics stage tests (PLAN.md section 6.5).  Runs on the tiny/small
presets; the whole file takes well under a minute once numba caches are
warm."""
from __future__ import annotations

import copy
import dataclasses
import math
import time

import numpy as np
from scipy import ndimage
import pytest

from globe.config import WorldParams
from globe.cubesphere import get_grid
from globe.field import rotation_field
from globe.io.world_store import WorldStore
from globe.pipeline import bake
from globe.tectonics import intraplate
from globe.tectonics import run as tect
from globe.tectonics.collision import (SmoothSplat, build_tree, collide, interior_centers_flat, label_map, label_map_fast,
                                        relax_segments, splat_weights, spread_collisions, weighted_quantile,
                                        wendland_support)
from globe.tectonics.plates import Plates, cluster_plates, rotate_segments
from globe.tectonics.segments import Segments, best_candidate_sphere, greedy_accept, mean_spacing
from scripts.coastline import equal_area_level, isoline_metrics, shelf_edge_level


@pytest.fixture(scope="module")
def tiny_sim():
    p = WorldParams.tiny_world()
    return tect.simulate(p, log=None)


@pytest.fixture(scope="module")
def tiny_out(tiny_sim):
    return tect.finalise(tiny_sim)


# --------------------------------------------------------------------------
# segments / plates
# --------------------------------------------------------------------------
def test_best_candidate_sampling_on_sphere():
    rng = np.random.default_rng(0)
    M = 800
    pos = best_candidate_sphere(M, rng)
    assert pos.shape == (M, 3)
    assert np.allclose(np.linalg.norm(pos, axis=1), 1.0, atol=1e-12)
    d, _ = build_tree(Segments(pos, 1.0, 0.5, 0.0, 0, 0.0)).query(pos, k=2)
    nn = d[:, 1]
    s = mean_spacing(M)
    # blue-noise-ish: nearest neighbour never much closer than a third of the mean spacing
    assert nn.min() > 0.3 * s
    assert abs(np.median(nn) / s - 0.8) < 0.3
    # deterministic
    assert np.array_equal(pos, best_candidate_sphere(M, np.random.default_rng(0)))


def test_greedy_accept_respects_spacing():
    rng = np.random.default_rng(1)
    existing = best_candidate_sphere(300, rng)
    c = rng.normal(size=(2000, 3))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    r = 0.5 * mean_spacing(300)
    acc = greedy_accept(c, existing, r)
    pts = np.concatenate([existing, c[acc]])
    d, _ = build_tree(Segments(pts, 1.0, 0.5, 0.0, 0, 0.0)).query(c[acc], k=2)
    assert d[:, 1].min() >= r - 1e-12  # every accepted candidate keeps r from everything else
    assert 0 < acc.sum() < acc.size
    # rejected candidates really were too close to something
    d_rej, _ = build_tree(Segments(pts, 1.0, 0.5, 0.0, 0, 0.0)).query(c[~acc], k=1)
    assert d_rej.max() < r


def test_rodrigues_rotation_keeps_unit_sphere_and_rigidity():
    rng = np.random.default_rng(2)
    pos = best_candidate_sphere(500, rng)
    pid = cluster_plates(pos, 4, rng)
    seg = Segments(pos, 1.0, 0.5, 0.0, pid, 0.0)
    plates = Plates(4)
    plates.update_stats(seg)
    plates.omega[:] = rng.normal(size=(4, 3)) * 0.05
    before = seg.pos.copy()
    for _ in range(20):
        rotate_segments(seg, plates)
    assert np.allclose(np.linalg.norm(seg.pos, axis=1), 1.0, atol=1e-12)
    for p in range(4):
        sel = pid == p
        g0 = before[sel] @ before[sel].T
        g1 = seg.pos[sel] @ seg.pos[sel].T
        assert np.allclose(g0, g1, atol=1e-9)  # pairwise angles preserved
    assert cluster_plates(pos, 4, np.random.default_rng(2)).shape == (500,)
    assert set(np.unique(pid)) == set(range(4))


# --------------------------------------------------------------------------
# label map / collisions
# --------------------------------------------------------------------------
def test_label_map_fast_matches_kdtree_and_covers_every_cell():
    grid = get_grid(48, 4, 100.0)
    rng = np.random.default_rng(3)
    seg = Segments(best_candidate_sphere(600, rng), 1.0, 0.5, 0.0, 0, 0.0)
    tree = build_tree(seg)
    i1, d1 = label_map(tree, grid)
    i2, d2 = label_map_fast(seg, grid, 0.7 * mean_spacing(600), tree)  # cap smaller than many distances -> exercises the fallback
    assert i1.shape == (6, grid.N, grid.N)
    assert np.array_equal(i1, i2)
    assert np.allclose(d1, d2)
    assert (i2 >= 0).all()


def test_collisions_conserve_mass_and_kill_denser():
    rng = np.random.default_rng(4)
    pos = best_candidate_sphere(800, rng)
    pid = cluster_plates(pos, 3, rng)
    seg = Segments(pos, rng.uniform(0.2, 1.0, 800), rng.uniform(0.3, 0.9, 800), 0.0, pid, 4 * np.pi / 800)
    plates = Plates(3)
    plates.update_stats(seg)
    plates.omega[:] = rng.normal(size=(3, 3)) * 0.02
    s = mean_spacing(800)
    m0 = seg.total_mass()
    tree = build_tree(seg)
    alive = np.ones(seg.M, bool)
    losers, survivors = collide(seg, tree, 1.2 * s, plates.omega, alive)
    assert losers.size > 0
    assert (~alive).sum() == losers.size
    assert (seg.plate_id[losers] != seg.plate_id[survivors]).all()
    # the loser's arrays keep the transferred amounts until compress; the live total is conserved
    live = lambda: float(seg.mass[alive].sum())
    assert live() == pytest.approx(m0, rel=1e-12)
    spread_collisions(seg, tree, losers, survivors, alive, s)
    assert live() == pytest.approx(m0, rel=1e-12)
    seg.compress(alive)
    assert seg.M == 800 - losers.size
    assert seg.total_mass() == pytest.approx(m0, rel=1e-12)
    relax_segments(seg, build_tree(seg), 0.3, 0.1, s)
    assert seg.total_mass() == pytest.approx(m0, rel=1e-12)
    assert np.allclose(seg.mass, seg.thickness * seg.density)


def test_ocean_on_ocean_subduction_has_one_polarity_per_boundary():
    """One plate dives along the whole trench, not whichever segment happens
    to win each contact.

    Every oceanic segment has the same density, so ``density[i] > density[j]``
    compared the rounding of ``mass / thickness`` and the ties fell to the
    segment index: measured over 300 steps of the Earth preset, 28.7 % of
    ocean-on-ocean collisions were exact ties and the rest were decided by
    the last bits of a division.  Both are uncorrelated with the plate, so
    each side kept half the contacts and drove fingers into the other.
    """
    rng = np.random.default_rng(11)
    pos = best_candidate_sphere(800, rng)
    pid = cluster_plates(pos, 2, rng)
    seg = Segments(pos, 0.2, 0.88, 0.0, pid, 4 * np.pi / 800)
    seg.age[pid == 0] = 300.0    # plate 0 carries the older, colder floor
    seg.age[pid == 1] = 50.0
    seg.density[::2] = np.nextafter(0.88, 1.0)   # the noise a run actually has
    seg.mass[:] = seg.thickness * seg.density
    plates = Plates(2)
    plates.update_stats(seg)
    s = mean_spacing(800)
    alive = np.ones(seg.M, bool)
    # omega = 0: nothing is approaching, so every pair takes the overlap branch
    losers, survivors = collide(seg, build_tree(seg), 1.2 * s, plates.omega, alive, overlap_fraction=1.0)
    assert losers.size > 0
    assert (seg.plate_id[losers] == 0).all()
    assert (seg.plate_id[survivors] == 1).all()


def test_rift_opens_the_cut_rather_than_shearing_along_it(tiny_sim):
    """The Euler pole of a spreading pair lies on the rift, so the halves
    move apart; about the cut's normal they would only slide along it.  And
    a new pole obeys the speed cap -- the plate moves before the force model
    is reached, so an over-fast one has already teleported the crust (the
    step that opened a rift used to turn each half 14.4 degrees)."""
    sim = tect.initialise(WorldParams.tiny_world(), log=None)
    P0 = sim.plates.P
    target = int(np.argmax(sim.plates.count))
    out = intraplate._rift_one(sim, target, sim.params.rng("tectonics", 6, 1))
    assert out["split"] == target and out["moved"] > 0
    om = sim.plates.omega
    assert np.linalg.norm(om, axis=1).max() <= sim.max_omega * (1 + 1e-12)
    a, b = sim.seg.plate_id == target, sim.seg.plate_id == P0
    ca, cb = sim.seg.pos[a].mean(axis=0), sim.seg.pos[b].mean(axis=0)
    va, vb = np.cross(om[target], ca), np.cross(om[P0], cb)
    # the halves separate: their relative velocity points along the line
    # between them (a shear would leave this at ~0)
    assert float(np.dot(va - vb, ca - cb)) > 0.0


# --------------------------------------------------------------------------
# simulation bookkeeping
# --------------------------------------------------------------------------
def test_mass_ledger_closes_and_subduction_is_a_sink(tiny_sim):
    sim = tiny_sim
    L = sim.ledger
    # driven off the declared key sets rather than a hand-written list, so a
    # new ledger term has to be classified as mass or counter to pass
    assert set(L) <= set(tect.MASS_KEYS) | set(tect.COUNTER_KEYS), sorted(set(L) - set(tect.MASS_KEYS) - set(tect.COUNTER_KEYS))
    expected = sum(L.get(k, 0.0) for k in tect.MASS_KEYS)
    assert sim.seg.total_mass() == pytest.approx(expected, rel=1e-9)
    # subduction and delamination are sinks, never sources: crust returns to
    # the mantle at a trench and under an over-thickened root, and nothing in
    # either path can add mass.  A positive value here means a kernel is
    # spreading more than was transferred, which is how the first crust-type
    # implementation leaked (spread_collisions handed neighbours the whole
    # slab while only arc_accretion of it had been accreted).
    for k in tect.SINK_KEYS:
        assert L.get(k, 0.0) <= 1e-9 * expected, k
    assert np.allclose(sim.seg.mass, sim.seg.thickness * sim.seg.density)
    assert np.allclose(np.linalg.norm(sim.seg.pos, axis=1), 1.0, atol=1e-12)
    assert (sim.seg.thickness > 0).all() and (sim.seg.density > 0).all() and (sim.seg.density <= 1).all()


def test_no_plateless_holes_after_gap_filling(tiny_sim):
    sim = tiny_sim
    idx, dist = label_map_fast(sim.seg, sim.grid, sim.r_cap)
    assert (idx >= 0).all()
    # every cell is within the gap radius (+ one cell of jitter slack) of a segment
    cell = np.pi / 2 / sim.grid.N
    assert dist.max() < sim.r_gap + 2.0 * cell
    assert (sim.seg.plate_id >= 0).all() and (sim.seg.plate_id < sim.plates.P).all()
    assert sim.plates.n_alive() >= 2


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def test_output_dtypes_and_ranges(tiny_out):
    p = WorldParams.tiny_world()
    out = tiny_out
    N = p.N_c
    assert out["bedrock"].dtype == np.float32 and out["bedrock"].interior.shape == (6, N, N)
    assert out["uplift"].dtype == np.float32
    assert out["hardness"].dtype == np.float32
    assert out["plate_id"].dtype == np.int16
    assert out["plate_vel"].dtype == np.float32 and out["plate_vel"].is_vector and out["plate_vel"].interior.shape == (6, N, N, 2)
    for k in ("bedrock", "uplift", "hardness", "plate_vel"):
        assert np.isfinite(out[k].data).all()
    bed = out["bedrock"].interior
    area = p.coarse_grid().interior_cell_area
    land = float(area[bed >= 0].sum() / area.sum())
    assert abs(land - p.world.land_fraction) < 0.03
    assert bed.max() > 0.5 * relief_target(p) and bed.min() < 0
    h = out["hardness"].interior
    assert h.min() >= 0 and h.max() <= 1 and h.std() > 0.01
    pid = out["plate_id"].interior
    assert pid.min() >= 0
    assert len(np.unique(pid)) >= 2
    assert len(np.unique(pid)) == pid.max() + 1  # compact ids
    up = out["uplift"].interior
    assert up.max() > 0
    assert (up[out["collision_zone"].interior.astype(bool)] >= 0).all()


def relief_target(p: WorldParams) -> float:
    """The metre height the 99.9th percentile of land is scaled to."""
    tp = p.tectonics
    return float(tp.relief_m) if tp.relief_m > 0 else float(tp.relief_spacings) * mean_spacing(int(tp.segments)) * p.R_planet


def test_relief_follows_the_tectonic_spacing(tiny_out):
    """The vertical scale is tied to the horizontal one: the 99.9th
    percentile of land sits at ``relief_spacings`` mean segment spacings,
    so the land slope stays well under the talus angle at every preset (a
    fixed relief_m puts kilometres on a pattern a few cells wide and the
    whole planet ends up at the angle of repose)."""
    p = WorldParams.tiny_world()
    grid = p.coarse_grid()
    bed = tiny_out["bedrock"].interior
    land = bed > 0
    top = weighted_quantile(bed[land], grid.interior_cell_area[land], 0.999)
    target = relief_target(p)
    assert abs(top - target) < 0.05 * target, (top, target)
    info = tiny_out["_land_slope"]
    slope = tiny_out["bedrock"].gradient().vec_norm().interior[land]
    assert info["land_slope_median"] == pytest.approx(float(np.median(slope)))
    assert info["land_slope_p90"] == pytest.approx(float(np.percentile(slope, 90)))
    hard = p.erosion.talus_slope_hard
    assert info["land_slope_median"] < 0.5 * hard, info  # not already at the talus angle
    # The tail is checked on `small` below, not here.  `tiny` is 300 segments
    # on a 32-cell grid -- 4.2 cells of segment spacing, and 17.6 % of its land
    # is coastline against 7.0 % on `small` -- so the continent/ocean contact,
    # which crust types made a real step, lands on nearly every land cell.
    # Measured: 11.0 % of tiny's land above the talus angle against 0.10 % of
    # small's.  That is the toy geometry, not the relief scaling.
    assert info["land_above_talus_fraction"] < 0.2, info


def test_relief_stays_under_the_talus_angle_away_from_the_coast():
    """The tail `test_relief_follows_the_tectonic_spacing` cannot check.

    Land at the angle of repose is the failure the relief scaling exists to
    prevent. Measured at the **shipping vertical scale** -- Airy isostasy,
    `height_scale_m` metres per bedrock unit on an Earth-radius body -- at a
    cheap resolution, rather than on `small`, whose `relief_spacings` scaling
    ties relief to a 4 km body and is used by nothing but the test presets.
    The difference is not marginal: 0.00 % of inland land above the talus
    angle at the Earth scale against 4.11 % on the toy one.

    Away from the coast, because the coastline is a different quantity. Crust
    types made the continent-ocean contact a real ~4 km step in bedrock
    height, which on a small body falls across a handful of cells and is
    genuinely at the angle of repose; Earth softens it with a sediment
    shelf-slope-rise ramp that tectonics does not model. Including those
    cells measures the margin geometry, excluding them measures the relief
    scaling this test is named for.
    """
    p = WorldParams().with_overrides(
        world={"N_c": 128, "cell_size_m": 9773.0, "R": 2},
        tectonics={"N_tect": 64, "segments": 1500, "steps": 300, "rift_every": 100},
    )
    out = tect.finalise(tect.simulate(p, log=None))
    bed = out["bedrock"].interior
    land = bed > 0
    slope = out["bedrock"].gradient().vec_norm().interior
    coast = np.zeros_like(land)
    for f in range(6):                     # land within 2 cells of ocean
        coast[f] = ndimage.binary_dilation(~land[f], iterations=2) & land[f]
    inland = land & ~coast
    assert inland.sum() > 0.3 * land.sum(), "too little inland to measure"
    frac = float((slope[inland] > p.erosion.talus_slope_hard).mean())
    assert frac < 0.02, (frac, out["_land_slope"])

def test_shelf_sea_level_cuts_the_continental_crust():
    """Sea level must land *inside* the continental hump, not in the trough.

    `world.land_fraction` places it by surface area, which cannot do this
    reliably once the hypsometry is bimodal: the continental fraction at the
    end of a run varies by about +-0.1 between seeds, so an area quantile
    that clears the hump on one seed falls into the near-empty gap between
    the humps on another.  Measured across three seeds of one Earth-scale
    configuration, land under 1 km came out 78.0 / 75.6 / 17.4 % -- the
    third is the miss, and it moved the ocean median from -3672 m to
    -1555 m.  `shelf_fraction` measures against the continental crust
    instead, so it cannot miss.
    """
    p = WorldParams.tiny_world().with_overrides(tectonics={"shelf_fraction": 0.275})
    sim = tect.simulate(p, log=None)
    bed = tect.finalise(sim)["bedrock"].interior
    cont = sim.seg.kind == 1
    assert cont.any() and (~cont).any(), "both crust types must survive the run"
    # the requested fraction of continental crust is drowned, and the land
    # area is whatever falls out of that rather than a forced constant
    land = bed > 0
    assert 0.05 < land.mean() < 0.60, land.mean()
    # and the two populations are still separated in the finalised bed: the
    # median land cell sits well above the median ocean cell
    assert np.median(bed[land]) - np.median(bed[~land]) > 0.5 * np.ptp(bed[~land])


def _with_tectonics(sim, **overrides):
    """The same finished simulation with other tectonics parameters, for the
    knobs that are read at finalise time only (`margin_sigma_factor`,
    `splat_knn_base`)."""
    s = copy.copy(sim)
    s.params = sim.params.with_overrides(tectonics=dict(overrides))
    s.tp = s.params.tectonics
    return s


def _with_margin_sigma(sim, f: float):
    return _with_tectonics(sim, margin_sigma_factor=f)


@pytest.fixture(scope="module")
def small_sim():
    return tect.simulate(WorldParams.small_world(), log=None)


def test_margin_ramp_raises_only_the_oceanic_side(tiny_sim, tiny_out):
    """The margin ramp re-positions the continental/oceanic step with a wide
    kernel and fills the oceanic side up to it; everything else is the
    narrow blend, bit for bit.  Off, it *is* the narrow blend."""
    sim = tiny_sim
    tp, grid, seg = sim.tp, sim.grid, sim.seg
    tree = build_tree(seg)
    blend = SmoothSplat(tree, grid, tp.splat_sigma_factor * sim.spacing, int(tp.splat_knn))
    h = seg.height() + tect.ridge_buoyancy(seg, tp)
    off = dataclasses.replace(tp, margin_sigma_factor=0.0)
    bed_off, c_off, lift_off = tect.margin_ramp(grid, blend, seg, h, off, sim.spacing)
    assert np.array_equal(bed_off, blend(h)) and not lift_off.any()
    on = dataclasses.replace(tp, margin_sigma_factor=2.5)
    bed, c, lift = tect.margin_ramp(grid, blend, seg, h, on, sim.spacing)
    # the crust-type boundary is the narrow blend either way
    assert np.array_equal(c, c_off) and np.array_equal(c, blend(seg.kind.astype(np.float64)))
    # raise-only, and it does raise something on a world with both crusts
    assert (lift >= 0).all() and lift.any()
    assert np.array_equal(bed, bed_off + lift)
    # the continental interior -- every nearest segment continental: land,
    # cratons and belts -- is untouched, so are cells the wide kernel sees
    # no differently from the narrow one
    assert not lift[c >= 1.0].any()
    # the raise sits on the oceanic side of the boundary: where the wide
    # fraction exceeds the narrow one, i.e. within a few spacings of the
    # margin on oceanic crust
    assert (c[lift > 0] < 1.0).all()
    assert np.median(c[lift > 0]) < 0.5
    # through finalise: crust_kind (the shelf mask) is identical to the
    # knob-0 run, the land fraction is the same by construction, and the
    # ramp is confined below sea level -- the land median moves by less
    # than a coarse-grid rounding of the relief
    out0 = tect.finalise(_with_margin_sigma(sim, 0.0))   # the default: the knob is off
    out1 = tect.finalise(_with_margin_sigma(sim, 2.5))
    assert np.array_equal(out1["crust_kind"].data, out0["crust_kind"].data)
    b1, b0 = out1["bedrock"].interior.astype(np.float64), out0["bedrock"].interior.astype(np.float64)
    assert abs((b1 > 0).mean() - (b0 > 0).mean()) < 0.005
    assert abs(out1["_sea_level_units"] - out0["_sea_level_units"]) < 1e-3
    land = (b0 > 0) & (b1 > 0)
    assert abs(np.median(b1[land]) - np.median(b0[land])) < 0.01 * relief_target(WorldParams.tiny_world())
    # and the sea got shallower near the margins, nowhere deeper
    assert np.median(b1[~land]) >= np.median(b0[~land])
    assert out1["_margin_raised_fraction"] > 0 and out0["_margin_raised_fraction"] == 0


def test_margin_ramp_smooths_the_shelf_edge(small_sim):
    """On `small` the shelf-edge isoline (the outer edge of the viewer's
    light band) is less convoluted with the ramp than without, while the
    coastline and the crust-type boundary are what they were (measured:
    L/sqrt(A) 7.79 -> 6.38 at the shelf edge, 12.92 -> 12.91 at the coast,
    docs/coast-fringe.md)."""
    sim = small_sim
    p = sim.params
    out0 = tect.finalise(_with_margin_sigma(sim, 0.0))
    out1 = tect.finalise(_with_margin_sigma(sim, 2.5))
    crust = out0["crust_kind"].interior.astype(bool)
    assert np.array_equal(crust, out1["crust_kind"].interior.astype(bool))
    b0, b1 = out0["bedrock"].interior.astype(np.float64), out1["bedrock"].interior.astype(np.float64)
    r = 0.5 * sim.spacing * p.N_c / (math.pi / 2)  # half a segment spacing, coarse cells
    edge = shelf_edge_level(b0, crust)
    assert np.isfinite(edge) and edge < 0
    before, after = isoline_metrics(b0, edge, r), isoline_metrics(b1, edge, r)
    assert after["ratio"] < 0.9 * before["ratio"], (before["ratio"], after["ratio"])
    coast0, coast1 = isoline_metrics(b0, 0.0, r), isoline_metrics(b1, 0.0, r)
    assert abs(coast1["ratio"] - coast0["ratio"]) < 0.02 * coast0["ratio"], (coast0["ratio"], coast1["ratio"])


def test_splat_knn_base_off_is_bit_identical(tiny_sim, tiny_out):
    """`splat_knn_base = 0` (the default) is the old path byte for byte: the
    narrow blend of the whole height, no base/excess split, through
    `margin_ramp` and through `finalise` (tiny_out is the default)."""
    sim = tiny_sim
    tp, grid, seg = sim.tp, sim.grid, sim.seg
    assert tp.splat_knn_base == 0
    tree = build_tree(seg)
    blend, base_blend = tect.splat_blends(tree, grid, tp, sim.spacing)
    assert base_blend is None
    # a count no wider than splat_knn is off too
    assert tect.splat_blends(tree, grid, dataclasses.replace(tp, splat_knn_base=int(tp.splat_knn)), sim.spacing)[1] is None
    h = seg.height() + tect.ridge_buoyancy(seg, tp)
    bed, c, lift = tect.margin_ramp(grid, blend, seg, h, tp, sim.spacing, base_blend=None)
    assert np.array_equal(bed, blend(h)) and not lift.any()
    out0 = tect.finalise(_with_tectonics(sim, splat_knn_base=0))
    for name in ("bedrock", "uplift", "hardness", "crust_kind", "collision_zone"):
        assert np.array_equal(out0[name].data, tiny_out[name].data), name
    assert out0["_splat_knn_base"] == tp.splat_knn and out0["_splat_base_fraction"] == 0.0
    # and with the knob on, the crust-type fraction (hence crust_kind and the
    # shelf mask) is still the narrow blend, and the bed is the split sum
    on = dataclasses.replace(tp, splat_knn_base=48)
    blend48, base48 = tect.splat_blends(tree, grid, on, sim.spacing)
    assert base48 is not None and base48.nb.shape[1] == min(48, seg.M)
    bed48, c48, _ = tect.margin_ramp(grid, blend48, seg, h, on, sim.spacing, base_blend=base48)
    assert np.array_equal(c48, c)
    kind = (seg.kind == 1).astype(np.float64)
    h_belt = tp.continental_thickness * tp.belt_thickness * (1.0 - tp.continental_density)
    h_base = np.where(kind > 0, np.minimum(h, h_belt), h)
    assert np.allclose(bed48, base48(h_base) + blend48(h - h_base))
    assert not np.array_equal(bed48, bed)


def _knob_pair(sim, knn_base: int):
    """(baseline, with the knob) finalised bedrock in metres, crust masks,
    belt masks, and the half-spacing opening radius in coarse cells."""
    out0 = tect.finalise(_with_tectonics(sim, splat_knn_base=0))
    out1 = tect.finalise(_with_tectonics(sim, splat_knn_base=knn_base))
    r = 0.5 * sim.spacing * sim.params.N_c / (math.pi / 2)
    return out0, out1, r


def _check_knob(sim, out0, out1, r, max_ratio, scale_tol=0.02):
    crust = out0["crust_kind"].data
    assert np.array_equal(crust, out1["crust_kind"].data)
    b0, b1 = out0["bedrock"].interior.astype(np.float64), out1["bedrock"].interior.astype(np.float64)
    assert abs((b1 > 0).mean() - (b0 > 0).mean()) < 0.005, ((b0 > 0).mean(), (b1 > 0).mean())
    # top of the land (the 99.9th land percentile in bedrock units, read
    # through the vertical scale that pins it to relief_spacings) and the
    # belts' p99 (m) within tolerance: the orogenic excess kept the narrow
    # kernel, so the peaks are where they were
    s0, s1 = out0["_scale_m_per_unit"], out1["_scale_m_per_unit"]
    assert abs(s1 - s0) < scale_tol * s0, (s0, s1)
    # (collision-zone land, the belt population docs/coast-fringe.md quotes)
    zone = out0["collision_zone"].interior.astype(bool) & (b0 > 0) & (b1 > 0)
    assert zone.any()
    p0, p1 = np.percentile(b0[zone], 99), np.percentile(b1[zone], 99)
    assert abs(p1 - p0) < 0.02 * max(p0, 1.0), (p0, p1)
    # the coast at equal area is less convoluted
    n_land = int((b0 > 0).sum())
    lv = equal_area_level(b1, n_land)
    before, after = isoline_metrics(b0, 0.0, r), isoline_metrics(b1, lv, r)
    assert abs(int((b1 > lv).sum()) - n_land) <= 1
    assert after["ratio"] < max_ratio * before["ratio"], (before["ratio"], after["ratio"])
    assert out1["_splat_knn_base"] == 48 and 0.5 < out1["_splat_base_fraction"] <= 1.0
    return before["ratio"], after["ratio"]


def test_splat_knn_base_smooths_the_coast_at_equal_area(small_sim):
    """With the base height blended from 48 neighbours (the excess from 12)
    the crust-type boundary is bit-identical, the land fraction and the
    belts' p99 are within 2 %, the top of the land (in units) moves by
    -0.5 % on `small` seed 0 and +3.7 % on seed 1 -- where the lever hurts,
    the wide base pulls a few of the highest cells down -- and the coastline
    at equal area is less convoluted: measured 12.917 -> 12.195 (0.944x) on
    seed 0 and 13.497 -> 13.108 (0.971x) on seed 1 (docs/coast-fringe.md
    section 5)."""
    out0, out1, r = _knob_pair(small_sim, 48)
    _check_knob(small_sim, out0, out1, r, 0.98, scale_tol=0.02)
    sim1 = tect.simulate(WorldParams.small_world(seed=1), log=None)
    out0, out1, r = _knob_pair(sim1, 48)
    _check_knob(sim1, out0, out1, r, 0.99, scale_tol=0.05)


def test_splat_kernel_default_is_bit_identical(tiny_sim, tiny_out):
    """`splat_kernel = 'gaussian'` (the default) is the old truncated kNN
    Gaussian byte for byte: the weights as `SmoothSplat` used to compute
    them inline, and every output of `finalise` (tiny_out is the default)."""
    sim = tiny_sim
    tp, grid, seg = sim.tp, sim.grid, sim.seg
    assert tp.splat_kernel == "gaussian"
    tree = build_tree(seg)
    sigma = tp.splat_sigma_factor * sim.spacing
    blend = SmoothSplat(tree, grid, sigma, int(tp.splat_knn))
    c = interior_centers_flat(grid)
    d, nb = tree.query(c, k=int(tp.splat_knn), workers=-1)
    w = np.exp(-(d * d) / (2.0 * float(sigma) ** 2))
    w[:, 0] = np.maximum(w[:, 0], 1e-300)
    w = w / w.sum(axis=1, keepdims=True)
    assert np.array_equal(blend.nb, nb) and np.array_equal(blend.w, w)
    assert blend.kernel == "gaussian" and blend.support_covered == 1.0
    out = tect.finalise(_with_tectonics(sim, splat_kernel="gaussian"))
    for name in ("bedrock", "uplift", "hardness", "plate_id", "plate_vel", "crust_kind", "collision_zone"):
        assert np.array_equal(out[name].data, tiny_out[name].data), name
    assert tiny_out["_splat_kernel"] == "gaussian" and tiny_out["_splat_support_covered"] == 1.0
    with pytest.raises(ValueError, match="splat_kernel"):
        SmoothSplat(tree, grid, sigma, 12, "cubic")


def _bisect_walk(p0, direction, unchanged, t_hi, eps=1e-12):
    """Two points on the unit sphere ~``eps`` apart (in the walk parameter)
    along ``p0 + t direction`` that straddle the first change of
    ``unchanged(point)``, or None if nothing changed by ``t_hi``."""
    def at(t):
        p = p0 + t * direction
        return p / np.linalg.norm(p)

    if unchanged(at(t_hi)):
        return None
    lo, hi = 0.0, t_hi
    while hi - lo > eps:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if unchanged(at(mid)) else (lo, mid)
    return at(lo), at(hi)


def test_splat_kernel_weights_are_continuous(tiny_sim):
    """Moving a query point across the moment a segment enters or leaves
    the 12 nearest changes the truncated Gaussian's value by about the
    departing neighbour's weight (a few per cent: the lace,
    docs/coast-fringe.md section 5) and the tapered and Wendland kernels'
    values by nothing measurable; the Wendland kernel is also continuous
    where a segment crosses its support radius."""
    sim = tiny_sim
    tp, seg = sim.tp, sim.seg
    tree = build_tree(seg)
    sigma = tp.splat_sigma_factor * sim.spacing
    knn = int(tp.splat_knn)
    h = wendland_support(sigma, knn, sim.spacing)
    rng = np.random.default_rng(7)
    vals = rng.random(seg.M)

    def jump(pair, kernel, sup=None):
        nb, w, _ = splat_weights(tree, np.array(pair), sigma, knn, kernel, sup)
        v = (vals[nb] * w).sum(axis=1)
        return abs(v[0] - v[1])

    # the knn-nearest set changes: walk towards the (knn+1)-th neighbour
    knn_pairs = []
    while len(knn_pairs) < 25:
        p0 = rng.normal(size=3)
        p0 /= np.linalg.norm(p0)
        _, nb0 = tree.query(p0, k=knn + 1)
        s0 = frozenset(nb0[:knn].tolist())
        pair = _bisect_walk(p0, tree.data[nb0[knn]] - tree.data[nb0[knn - 1]],
                            lambda p: frozenset(tree.query(p, k=knn)[1].tolist()) == s0, 1.0)
        if pair is not None:
            knn_pairs.append(pair)
    g = max(jump(p, "gaussian") for p in knn_pairs)
    assert g > 1e-3, g  # the test does see the truncation
    assert max(jump(p, "tapered") for p in knn_pairs) < 1e-8
    assert max(jump(p, "wendland", h) for p in knn_pairs) < 1e-8
    # a segment crosses the Wendland support radius
    sup_pairs = []
    while len(sup_pairs) < 25:
        p0 = rng.normal(size=3)
        p0 /= np.linalg.norm(p0)
        n0 = len(tree.query_ball_point(p0, h))
        pair = _bisect_walk(p0, rng.normal(size=3), lambda p: len(tree.query_ball_point(p, h)) == n0, 0.2)
        if pair is not None:
            sup_pairs.append(pair)
    assert max(jump(p, "wendland", h) for p in sup_pairs) < 1e-8


def test_wendland_gathers_the_whole_support():
    """The Wendland list holds every segment within the support radius: on
    a cloud with a cap 20x denser than the mean -- where the kNN list sized
    for the mean density falls short -- the re-query path gives the
    brute-force radius-query weights (on `small` and `tiny` the first list
    already covers every cell, the farthest listed neighbour >= 1.19 h)."""
    rng = np.random.default_rng(3)
    uni = best_candidate_sphere(400, rng)
    cap = rng.normal(size=(1600, 3)) * 0.08 + np.array([0.0, 0.0, 1.0])
    pos = np.concatenate([uni, cap / np.linalg.norm(cap, axis=1, keepdims=True)])
    tree = build_tree(Segments(pos, 1.0, 0.5, 0.0, 0, 0.0))
    s = mean_spacing(pos.shape[0])
    h = wendland_support(s, 12, s)
    pts = rng.normal(size=(500, 3)) * np.array([0.3, 0.3, 1.0]) + np.array([0.0, 0.0, 1.0])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    nb, w, covered = splat_weights(tree, pts, s, 12, "wendland", h)
    assert covered < 0.999  # the dense cap does defeat the first list
    for i in range(pts.shape[0]):
        ref = np.array(tree.query_ball_point(pts[i], h), dtype=np.int64)
        if ref.size == 0:
            assert w[i, 0] == 1.0
            continue
        q = np.linalg.norm(pos[ref] - pts[i], axis=1) / h
        wr = (1.0 - q) ** 4 * (4.0 * q + 1.0)
        got = dict(zip(nb[i][w[i] > 0].tolist(), w[i][w[i] > 0].tolist()))
        want = dict(zip(ref[wr > 0].tolist(), (wr[wr > 0] / wr.sum()).tolist()))
        assert got.keys() == want.keys(), i
        assert np.allclose([got[k] for k in want], list(want.values()), rtol=1e-10, atol=1e-15)


def test_splat_kernel_wendland_on_small_seed0(small_sim):
    """The shipping conditions (docs/coast-fringe.md sections 3, 5, 6) for
    `splat_kernel = 'wendland'` on `small` seed 0, where they hold: the
    coast at equal area is less convoluted (measured 12.917 -> 12.273,
    0.950x) with fingers not worse (1.26 -> 1.11 %), the land fraction and
    the vertical scale unchanged (+0.2 %), land / sea / belt medians and the
    belt p99 within a few per cent (+3.1 %, 0.7 % shallower, +0.1 %,
    +0.7 %), and the crust-type boundary moves on < 1 % of the cells
    (0.28 %).  `tiny` seed 1 is where it fails (fingers 6.00 -> 6.37 %,
    land and belt medians +8 %), so it is not the default."""
    sim = small_sim
    out0 = tect.finalise(_with_tectonics(sim, splat_kernel="gaussian"))
    out1 = tect.finalise(_with_tectonics(sim, splat_kernel="wendland"))
    assert out1["_splat_kernel"] == "wendland" and out1["_splat_support_covered"] > 0.999
    r = 0.5 * sim.spacing * sim.params.N_c / (math.pi / 2)
    b0, b1 = out0["bedrock"].interior.astype(np.float64), out1["bedrock"].interior.astype(np.float64)
    assert (out0["crust_kind"].interior != out1["crust_kind"].interior).mean() < 0.01
    l0, l1 = b0 > 0, b1 > 0
    assert abs(l1.mean() - l0.mean()) < 0.005
    s0, s1 = out0["_scale_m_per_unit"], out1["_scale_m_per_unit"]
    assert abs(s1 - s0) < 0.02 * s0, (s0, s1)
    lv = equal_area_level(b1, int(l0.sum()))
    before, after = isoline_metrics(b0, 0.0, r), isoline_metrics(b1, lv, r)
    assert after["ratio"] < 0.97 * before["ratio"], (before["ratio"], after["ratio"])
    assert after["fingers"] <= before["fingers"], (before["fingers"], after["fingers"])
    m0, m1 = np.median(b0[l0]), np.median(b1[l1])
    assert abs(m1 - m0) < 0.05 * m0, (m0, m1)
    q0, q1 = np.median(b0[~l0]), np.median(b1[~l1])
    assert abs(q1 - q0) < 0.02 * abs(q0), (q0, q1)
    zone = out0["collision_zone"].interior.astype(bool)
    assert np.array_equal(zone, out1["collision_zone"].interior.astype(bool))
    z0, z1 = b0[zone & l0], b1[zone & l1]
    assert abs(np.median(z1) - np.median(z0)) < 0.03 * np.median(z0)
    assert abs(np.percentile(z1, 99) - np.percentile(z0, 99)) < 0.03 * np.percentile(z0, 99)


def test_plate_vel_is_rigid_rotation_of_each_plate(tiny_sim, tiny_out):
    grid = WorldParams.tiny_world().coarse_grid()
    pv = tiny_out["plate_vel"].interior
    pid = tiny_out["plate_id"].interior
    alive = np.nonzero(tiny_sim.plates.alive)[0]
    for k, p in enumerate(alive):
        sel = pid == k
        if not sel.any():
            continue
        ref = rotation_field(grid, tiny_sim.plates.omega[p]).interior
        assert np.allclose(pv[sel], ref[sel], atol=1e-5)
    # halos are exchanged (a vector halo cell is not zero where the interior is not)
    assert np.abs(tiny_out["plate_vel"].data).max() > 0


def test_determinism():
    p = WorldParams.tiny_world()
    a = tect.finalise(tect.simulate(p, log=None))
    b = tect.finalise(tect.simulate(p, log=None))
    for k in tect.OUTPUTS:
        assert np.array_equal(a[k].data, b[k].data), k


def test_stage_runs_in_pipeline(tmp_path):
    p = WorldParams.tiny_world(seed=3)
    store = bake(tmp_path / "w", p, to_stage="tectonics", logger=lambda m: None)
    assert store.stage_done("tectonics", p)
    for name in tect.OUTPUTS:
        assert store.has_field(name)
    assert (store.quicklook_dir / "tectonics.png").exists()
    assert (store.quicklook_dir / "tectonics_plates.png").exists()
    grid = p.coarse_grid()
    pid = store.load_field("plate_id", grid)
    assert pid.dtype == np.int16 and (pid.interior >= 0).all()
    vel = store.load_field("plate_vel", grid)
    assert vel.is_vector
    stage = WorldStore(tmp_path / "w").stage_info("tectonics")["info"]
    assert isinstance(stage["segments_final"], int)
    assert 0.0 < stage["land_slope_median"] < p.erosion.talus_slope_hard  # recorded at bake time


def test_runtime_small_preset():
    """N_tect = 64, 300 steps must simulate in < 20 s (numba caches warm)."""
    p = WorldParams.small_world()
    tect.simulate(p, log=None, steps=3)  # warm-up (JIT / grid tables)
    t0 = time.time()
    sim = tect.simulate(p, log=None)
    dt = time.time() - t0
    assert sim.step_index == 300
    assert dt < 20.0, f"300 steps took {dt:.1f}s"


def test_strata_fabric_gives_hardness_structure_at_basin_scale():
    """`tectonics.strata_period/amp` lay bedrock fabric along lines of equal
    crust age (the strata the crust accreted in).

    Without it hardness is a smooth blend of age, density and boundary
    proximity — broad two-tone plateaus whose autocorrelation is still high
    a basin's width away, so rock strength has no structure for drainage to
    organise around and every continent develops the same radial network.
    The fabric both widens the contrast and shortens the correlation length.

    Measured **on land, at `small`**. On `tiny` the face is 32 cells, so a
    lag of 8 is a quarter of the world and the autocorrelation is noise
    around zero in both arms; and over the whole field the continental /
    oceanic density contrast that crust types introduced dominates the
    statistics — a real signal, but not one drainage ever sees. Land is
    where rock strength matters.
    """
    p = WorldParams.small_world(0)

    def measure(**ov):
        out = tect.finalise(tect.simulate(p.with_overrides(tectonics=ov), log=None))
        h = out["hardness"].interior
        land = out["bedrock"].interior > 0
        spread = float(np.percentile(h[land], 90) - np.percentile(h[land], 10))
        per_face = []
        for f in range(6):
            m = land[f]
            if m.sum() < 50:
                continue
            x = np.where(m, h[f].astype(np.float64), np.nan)
            x = x - np.nanmean(x)
            per_face.append(np.nanmean(x[:, :-8] * x[:, 8:]) / max(np.nanmean(x * x), 1e-12))
        return spread, float(np.mean(per_face)), h

    flat_spread, flat_ac, _ = measure(strata_amp=0.0)
    band_spread, band_ac, band = measure()

    assert band_spread > 1.5 * flat_spread, (flat_spread, band_spread)
    assert band_ac < 0.6 * flat_ac, (flat_ac, band_ac)
    assert float(band.min()) >= 0.0 and float(band.max()) <= 1.0



def test_slab_pull_drags_the_plate_towards_its_trench():
    """A subducted oceanic segment pulls its own plate towards the trench:
    the velocity the torque induces *at the trench* points from the
    plate's centre of mass to the trench, not away from it.  Continental
    losers and newborn slabs pull nothing; an old slab pulls in full."""
    from globe.tectonics.plates import slab_pull_torques
    from globe.tectonics.segments import CONTINENTAL, OCEANIC
    com = np.array([1.0, 0.0, 0.0])
    a = np.deg2rad(25.0)
    trench = np.array([np.cos(a), np.sin(a), 0.0])       # east of the centre of mass
    pos = np.stack([com, trench, trench])
    seg = Segments(pos, 1.0, 0.9, [500.0, 500.0, 0.0], [0, 0, 0], 1.0,
                   kind=np.array([OCEANIC, OCEANIC, OCEANIC], np.int8))
    plates = Plates(1)
    plates.update_stats(seg)
    plates.com[0] = com
    # old slab: full pull
    tau = slab_pull_torques(seg, np.array([1]), plates, gain=1.0, ridge_age=400.0, oceanic=OCEANIC)
    v = np.cross(tau[0], trench)                          # velocity at the trench for omega ∝ tau
    away = trench - com
    away -= away @ trench * trench
    assert v @ away > 0.99 * np.linalg.norm(v) * np.linalg.norm(away)
    # a slab at age 0 pulls nothing; a continental one never does
    assert np.allclose(slab_pull_torques(seg, np.array([2]), plates, 1.0, 400.0, OCEANIC), 0.0)
    seg.kind[1] = CONTINENTAL
    assert np.allclose(slab_pull_torques(seg, np.array([1]), plates, 1.0, 400.0, OCEANIC), 0.0)


def test_suture_welds_two_plates_and_keeps_their_momentum(tiny_sim):
    """Two plates welded by `suture` become one, the crust is untouched, and
    the merged pole is the inertia-weighted mean of the two."""
    import copy
    sim = copy.deepcopy(tiny_sim)
    plates = sim.plates
    live = np.flatnonzero(plates.alive)
    assert live.size >= 2
    a, b = int(live[0]), int(live[1])
    M, n_alive = sim.seg.M, plates.n_alive()
    Ia, Ib = plates.inertia[a], plates.inertia[b]
    expect = (Ia * plates.omega[a] + Ib * plates.omega[b]) / (Ia + Ib)
    big = a if plates.area[a] >= plates.area[b] else b
    ev = intraplate.suture(sim, a, b, np.random.default_rng(0))
    assert ev["kept"] == big and sim.seg.M == M
    assert sim.plates.n_alive() == n_alive - 1
    assert not (sim.seg.plate_id == ev["joined"]).any()
    assert np.allclose(sim.plates.omega[big], expect)


def test_continental_shortening_conserves_area_and_mass():
    """With `shortening` on, a continent-on-continent collision keeps the
    losing segment alive on the winner's plate, moves that fraction of it
    into the winner, and conserves total mass; an oceanic loser still dies."""
    from globe.tectonics.segments import CONTINENTAL, OCEANIC
    s = 0.05
    # two continental segments on different plates, head-on; a third pair oceanic-vs-continental
    pos = np.array([[1.0, 0.0, 0.0], [np.cos(0.5 * s), np.sin(0.5 * s), 0.0],
                    [0.0, 1.0, 0.0], [np.cos(0.5 * s) * 0.0 + 0.0 , np.cos(0.5 * s), np.sin(0.5 * s)]])
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    seg = Segments(pos, [1.0, 1.0, 1.0, 0.2], [0.8, 0.8, 0.8, 0.88], 100.0, [0, 1, 0, 1], 1.0,
                   kind=np.array([CONTINENTAL, CONTINENTAL, CONTINENTAL, OCEANIC], np.int8))
    plates = Plates(2)
    plates.update_stats(seg)
    # plate 1 turns towards plate 0 at both contacts
    plates.omega[0] = 0.0
    plates.omega[1] = np.array([0.0, 0.0, -0.1]) + np.array([-0.1, 0.0, 0.0])
    tree = build_tree(seg)
    alive = np.ones(4, dtype=bool)
    m0 = seg.total_mass()
    losers, survivors = collide(seg, tree, 1.2 * s, plates.omega, alive, shortening=0.5, accretion=0.15)
    assert losers.size == 2
    cc = [(lo, su) for lo, su in zip(losers, survivors) if seg.kind[lo] == CONTINENTAL][0]
    lo, su = cc
    assert alive[lo] and seg.plate_id[lo] == seg.plate_id[su]
    assert np.isclose(seg.thickness[lo], 0.5) and np.isclose(seg.thickness[su], 1.5)
    oc = [(lo, su) for lo, su in zip(losers, survivors) if seg.kind[lo] == OCEANIC][0]
    assert not alive[oc[0]]
    # live mass = initial - what the slab returned to the mantle
    assert np.isclose(seg.mass[alive].sum(), m0 - (1.0 - 0.15) * 0.2 * 0.88)
