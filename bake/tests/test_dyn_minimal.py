"""dyn-minimal: the Pangaea start, the boundary forces and the insulation rift."""
import math

import numpy as np

from globe.config import WorldParams
import globe.tectonics.run as tect
from globe.tectonics import forces, intraplate
from globe.tectonics.segments import CONTINENTAL, OCEANIC


def _small(seed=0, **kw):
    p = WorldParams.small_world(seed)
    p.tectonics.initial_plates = 9
    for k, v in kw.items():
        setattr(p.tectonics, k, v)
    return p


def test_solve_reduces_to_update_omega():
    """With no boundary terms the force balance is the shipped update: omega* = gain tau / (I d)."""
    sim = tect.initialise(_small(), log=None)
    pl = sim.plates
    rng = np.random.default_rng(1)
    tau = rng.normal(size=(pl.P, 3))
    w, _ = forces.solve_omega(pl, sim.gain * tau, 0.05, sim.seg, None, 0.0, 0.0, None, 0.0, 1.0)
    want = sim.gain * tau / (np.maximum(pl.inertia, 1e-12)[:, None] * 0.05)
    assert np.allclose(w[pl.alive], want[pl.alive], rtol=1e-10, atol=0)


def test_slab_pull_does_not_depend_on_speed():
    """The census decides who goes down by crust type and polarity, so slab pull is the same
    whatever the plates' omegas (the shipped slab_pull counted this step's consumed crust)."""
    sim = tect.initialise(_small(), log=None)
    seg, pl = sim.seg, sim.plates
    sim.slab.exchange_halos()
    s_at = sim.slab.sample_sphere(seg.pos).astype(np.float64)
    sat = float(sim.tp.slab_sat_km) / sim.R_km
    t1, i1 = forces.slab_torques(seg, forces.census(seg, pl, sim.spacing), pl.P, s_at, 0.6, sat, 500.0, 0.3)
    pl.omega = pl.omega * 7.0 + 0.01
    t2, _ = forces.slab_torques(seg, forces.census(seg, pl, sim.spacing), pl.P, s_at, 0.6, sat, 500.0, 0.3)
    assert i1["n"] > 0 and np.allclose(t1, t2)


def test_pangaea_start_is_a_girdled_supercontinent():
    p = _small()
    sim = tect.initialise(p, log=None)
    seg, pl = sim.seg, sim.plates
    cont = seg.kind == CONTINENTAL
    A = np.bincount(seg.plate_id, weights=seg.ext, minlength=pl.P) / (4 * math.pi)
    assert np.all(seg.plate_id[cont] == 0) and np.all(seg.plate_id[~cont] > 0)   # one continental plate
    assert A[1:].max() <= p.tectonics.ocean_plate_max + 0.05
    # the margin floor goes down under the continent and has a slab under it
    cen = forces.census(seg, pl, sim.spacing)
    girdle = np.unique(cen["i"][cen["down_i"] & (cen["kj"] == CONTINENTAL)])
    sim.slab.exchange_halos()
    assert girdle.size and (sim.slab.sample_sphere(seg.pos[girdle]) > 0).mean() > 0.9
    conv = (cen["appr"] > 0)[(cen["ki"] != cen["kj"])]
    assert conv.mean() > 0.7          # the girdle converges at step 0 (0.92-0.96 on the Earth preset)
    # the ocean floor is aged, the oldest along the margin
    oc = seg.kind == OCEANIC
    assert seg.age[oc].max() > 0 and seg.age[girdle].mean() > np.median(seg.age[oc])
    # omegas are the force balance's, not random: the supercontinent barely moves
    v = np.linalg.norm(np.cross(pl.omega[seg.plate_id], seg.pos), axis=1)
    assert np.median(v[cont]) < 0.5 * np.median(v[oc])          # (Earth preset: ~0.03)


def test_insulation_rift_cuts_the_continent_without_a_kick():
    p = _small()
    sim = tect.initialise(p, log=None)
    H = sim.heat.grid.H
    sim.heat_bg[:, H:-H, H:-H] = 0.0                       # the mantle fully insulated
    om0 = sim.plates.omega[0].copy()
    ev = intraplate.rift(sim, np.random.default_rng(3), 2)
    assert ev["event"] == "rift" and ev["pairs"], ev
    a, b = ev["pairs"][0]
    assert a == 0 and np.allclose(sim.plates.omega[a], om0) and np.allclose(sim.plates.omega[b], om0)
    assert (a, b) in sim.rift_pairs
    # the cut stays within 80 degrees of its centre (no far side that must converge)
    seg = sim.seg
    cen = forces.census(seg, sim.plates, sim.spacing)
    m = ((cen["pi"] == a) & (cen["pj"] == b)) | ((cen["pi"] == b) & (cen["pj"] == a))
    c = np.asarray(ev["centre"])
    ang = np.degrees(np.arccos(np.clip(cen["rm"][m] @ c, -1, 1)))
    assert m.any() and ang.max() <= 85.0
