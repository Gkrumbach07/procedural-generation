"""Tectonics stage tests (PLAN.md section 6.5).  Runs on the tiny/small
presets; the whole file takes well under a minute once numba caches are
warm."""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
import pytest

from globe.config import WorldParams
from globe.cubesphere import get_grid
from globe.field import rotation_field
from globe.io.world_store import WorldStore
from globe.pipeline import bake
from globe.tectonics import run as tect
from globe.tectonics.collision import build_tree, collide, label_map, label_map_fast, relax_segments, spread_collisions, weighted_quantile
from globe.tectonics.plates import Plates, cluster_plates, rotate_segments
from globe.tectonics.segments import Segments, best_candidate_sphere, greedy_accept, mean_spacing


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


# --------------------------------------------------------------------------
# simulation bookkeeping
# --------------------------------------------------------------------------
def test_mass_ledger_closes_and_subduction_is_a_sink(tiny_sim):
    sim = tiny_sim
    L = sim.ledger
    expected = L["initial"] + L["spawned"] + L["crystallised"] + L["subducted"] + L["delaminated"]
    assert sim.seg.total_mass() == pytest.approx(expected, rel=1e-9)
    # subduction and delamination are sinks, never sources: crust returns to
    # the mantle at a trench and under an over-thickened root, and nothing in
    # either path can add mass.  A positive value here means a kernel is
    # spreading more than was transferred, which is how the first crust-type
    # implementation leaked (spread_collisions handed neighbours the whole
    # slab while only arc_accretion of it had been accreted).
    assert L["subducted"] <= 1e-9 * expected
    assert L["delaminated"] <= 1e-9 * expected
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

