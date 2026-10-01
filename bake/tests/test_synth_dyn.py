"""synth-dyn (dyn-minimal's balance + dyn-events' event rules): the Pangaea start, the
per-segment force balance and its trial solves, the force rift, failed rifts, the
microplate life cycle, the classic dynamics of the toy presets."""
import math

import numpy as np

from globe.config import CLASSIC_DYNAMICS, TectonicsParams, WorldParams
import globe.tectonics.run as tect
from globe.tectonics import forces, intraplate
from globe.tectonics.segments import CONTINENTAL, OCEANIC


def _small_earth(seed=0, **kw):
    """`small` with the Earth preset's dynamics (the toy presets keep the shipped ones)."""
    p = WorldParams.small_world(seed)
    d = TectonicsParams()
    for k in CLASSIC_DYNAMICS:
        setattr(p.tectonics, k, getattr(d, k))
    for k, v in kw.items():
        setattr(p.tectonics, k, v)
    return p


def test_toy_presets_keep_the_shipped_dynamics():
    p = WorldParams.small_world(0)
    for k, v in CLASSIC_DYNAMICS.items():
        if k not in ("initial_plates", "rift_every"):
            assert getattr(p.tectonics, k) == v, k
    sim = tect.initialise(p, log=None)
    assert not sim.boundary_forces_on()


def test_solve_reduces_to_update_omega():
    """With no boundary terms the force balance is the shipped update: omega* = gain tau / (I d)."""
    sim = tect.initialise(_small_earth(), log=None)
    seg, pl = sim.seg, sim.plates
    rng = np.random.default_rng(1)
    tau = rng.normal(size=(pl.P, 3))
    bal = forces.assemble(seg, pl, None, gain=sim.gain, damping=0.05, grad3=None, basal_seg=seg.area * seg.mass,
                          drag_per_len=0.0, cc_per_len=0.0, slab=None, trench_per_len=0.0, rift_pairs=None,
                          rift_scale=1.0, rift_weaken=1.0)
    w = forces.solve(bal, pl, seg.plate_id, extra_tau=sim.gain * tau)
    want = sim.gain * tau / (np.maximum(pl.inertia, 1e-12)[:, None] * 0.05)
    assert np.allclose(w[pl.alive], want[pl.alive], rtol=1e-10, atol=0)


def test_release_of_a_whole_plate_is_its_terminal_velocity():
    """The trial solve (every other plate held at omega*) reproduces the global solution."""
    sim = tect.initialise(_small_earth(), log=None)
    for _ in range(3):
        sim.step()
    bal, seg = sim.balance, sim.seg
    for q in np.flatnonzero(sim.plates.alive)[:4]:
        sel = np.flatnonzero(seg.plate_id == q)
        wA, _ = forces.release(bal, seg, sel, np.zeros(sel.size, bool), np.zeros(0, np.int64), np.zeros(0, np.int64), 0.0)
        assert np.allclose(wA, bal.wstar[q], rtol=1e-8, atol=1e-14)


def test_slab_pull_does_not_depend_on_speed():
    """Who goes down is decided by crust type and polarity, so the pull is the same whatever the
    plates' omegas (the shipped slab_pull counted this step's consumed crust: negative drag)."""
    sim = tect.initialise(_small_earth(), log=None)
    seg, pl = sim.seg, sim.plates
    sim.slab.exchange_halos()
    s_at = sim.slab.sample_sphere(seg.pos).astype(np.float64)
    sat = sim.km(sim.tp.slab_sat_km)
    a = forces.slab_contrib(seg, forces.census(seg, pl, sim.spacing), s_at, 0.6, sat, 500.0, 0.3)
    pl.omega = pl.omega * 7.0 + 0.01
    b = forces.slab_contrib(seg, forces.census(seg, pl, sim.spacing), s_at, 0.6, sat, 500.0, 0.3)
    assert a["n"] > 0 and np.allclose(a["tq"], b["tq"])


def test_toy_body_converts_through_the_tectonic_radius():
    """Physical knobs convert on Earth's radius, so a 4 km body runs Earth's angular rates."""
    sim = tect.initialise(_small_earth(), log=None)
    assert abs(sim.R_km - 6371.0) < 1e-9
    for _ in range(20):
        sim.step()
    v = np.linalg.norm(np.cross(sim.plates.omega[sim.seg.plate_id], sim.seg.pos), axis=1) / sim.spacing
    assert np.median(v) > 1e-3            # dyn-minimal through R_planet: ~1e-6, frozen


def test_pangaea_start_is_a_girdled_supercontinent():
    p = _small_earth()
    sim = tect.initialise(p, log=None)
    seg, pl = sim.seg, sim.plates
    cont = seg.kind == CONTINENTAL
    A = np.bincount(seg.plate_id, weights=seg.ext, minlength=pl.P) / (4 * math.pi)
    assert np.all(seg.plate_id[cont] == 0) and np.all(seg.plate_id[~cont] > 0)   # one continental plate
    assert A[1:].max() <= p.tectonics.ocean_plate_max + 0.05
    assert A[1:].max() > 2.0 * np.median(A[1:])                                  # a power-law spread
    cen = forces.census(seg, pl, sim.spacing)
    girdle = np.unique(cen["i"][cen["down_i"] & (cen["kj"] == CONTINENTAL)])
    sim.slab.exchange_halos()
    assert girdle.size and (sim.slab.sample_sphere(seg.pos[girdle]) > 0).mean() > 0.9
    conv = (cen["appr"] > 0)[(cen["ki"] != cen["kj"])]
    assert conv.mean() > 0.7
    oc = seg.kind == OCEANIC
    assert seg.age[oc].max() > 0 and seg.age[girdle].mean() > np.median(seg.age[oc])
    v = np.linalg.norm(np.cross(pl.omega[seg.plate_id], seg.pos), axis=1)
    assert np.median(v[cont]) < 0.5 * np.median(v[oc])


def _insulated(sim):
    H = sim.heat.grid.H
    sim.heat_bg[:, H:-H, H:-H] = 0.0                   # the mantle under everything fully insulated


def test_force_rift_cuts_where_the_released_halves_open():
    sim = tect.initialise(_small_earth(rift_candidates=48), log=None)
    for _ in range(2):
        sim.step()
    _insulated(sim)
    om0 = sim.plates.omega[0].copy()
    ev = intraplate.rift(sim, np.random.default_rng(5), 1)
    assert ev["event"] == "rift" and ev["pairs"], ev
    a, b = ev["pairs"][0]
    det = ev["detail"][0]
    assert a == 0 and np.allclose(sim.plates.omega[a], om0) and np.allclose(sim.plates.omega[b], om0)   # no kick
    st = sim.rift_pairs[(a, b)]
    assert st["delta"] == 0.0 and 0.0 <= st["G0"] <= sim.tp.rift_strength_max
    assert det["conv"] <= sim.tp.rift_max_conv and det["q10_ratio"] >= sim.tp.rift_end_open
    assert det["extent_deg"] <= 2.0 * sim.tp.rift_max_angle + 1e-9
    assert det["free_cmyr"] >= sim.tp.rift_min_open_cmyr


def test_a_rift_that_does_not_open_heals():
    sim = tect.initialise(_small_earth(rift_candidates=48), log=None)
    for _ in range(2):
        sim.step()
    _insulated(sim)
    ev = intraplate.rift(sim, np.random.default_rng(5), 1)
    a, b = ev["pairs"][0]
    sim.rift_pairs[(a, b)]["k0"] = sim.step_index - int(sim.steps_of(sim.tp.rift_abort_my)) - 1
    out = intraplate.heal_failed_rifts(sim, np.random.default_rng(0))
    assert out["healed"] and (a, b) not in sim.rift_pairs and not sim.plates.alive[b]


def test_remnant_plates_are_captured():
    sim = tect.initialise(_small_earth(), log=None)
    for _ in range(3):
        sim.step()
    seg, pl = sim.seg, sim.plates
    # a remnant (below plate_min_area: one segment on small) of an ocean plate's edge on a plate of its own
    cen = sim.census_last
    edge = np.unique(cen["i"][(seg.kind[cen["i"]] == OCEANIC)])[:1]
    pid = seg.plate_id.copy()
    pid[edge] = pl.P
    sim.plates = intraplate._rebuild(seg, pid, pl.P + 1, np.random.default_rng(0), 0.0, keep=pl.omega, snap=False)
    sim.census_last = forces.census(seg, sim.plates, sim.spacing)
    ev = intraplate.micro_merge(sim, np.random.default_rng(0))
    assert any(m["plate"] == pl.P and m["remnant"] for m in ev["merges"])
    assert not sim.plates.alive[pl.P]


def test_earth_dynamics_run_without_a_speed_cap_or_far_side():
    """A short run of the whole package: nothing at the cap, every rift opens along its cut."""
    sim = tect.initialise(_small_earth(rift_deficit=0.4), log=None)
    rifts = []
    for _ in range(60):
        sim.step()
        rifts += [d for e in sim.events if e.get("event") == "rift" for d in e.get("detail", [])]
    s = np.linalg.norm(sim.plates.omega[sim.plates.alive], axis=1)
    assert (s < 0.999 * sim.max_omega).all()
    for d in rifts:
        assert d["conv"] <= sim.tp.rift_max_conv
