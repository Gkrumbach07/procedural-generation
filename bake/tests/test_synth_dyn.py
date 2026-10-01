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


# ----------------------------------------------------------------------------------------------
# landing (te/dyn): the classic dynamics pinned to trunk, the review's fixes, the spec's tests
# ----------------------------------------------------------------------------------------------
import hashlib

import pytest

from globe.config import classic_dynamics
from globe.tectonics import collision

#: sha256 of a classic-dynamics run on trunk 8fcf675 (every Segments field, the omegas and the
#: heat field, after N steps; scratchpad phase3/dyn/fingerprint.py run on a `git archive` of
#: 8fcf675).  Bit for bit, so they hold for this machine's numpy / scipy / numba: a new
#: environment regenerates them from an export of 8fcf675 the same way.
TRUNK_FINGERPRINTS = {
    ("tiny", 60): "a9dfe2c23ddf03c8ef12fa75d88208472ed1b62333bde57f0a6c17c3ec0d5b9a",
    ("small", 300): "73fcaf04c9bf3a9ff0c3dd0b4c198badbfd2e33d0100d65cef89ccf42d7a008c",
    ("earth", 300): "c3d1797c2269a9e158de95f80d16eacb9ed4eee01a3e5eabf74f774d387dfad3",
    ("earth", 610): "52c242cfe5276a4def6e35fbad918546a0d6c775192f9f4e7728631c233d25b7",   # past the step-600 clock rift
}


#: the segment fields trunk 8fcf675 had, which its fingerprints hash; fields added since
#: (te/arcs' ``terrane``, te/mass' ``shown``) are checked separately for their neutral values
TRUNK_FIELDS = ("pos", "mass", "thickness", "density", "age", "plate_id", "area", "h_ref", "kind", "craton", "weld",
                "ext", "rework")


def _fingerprint(sim, fields=None) -> str:
    """sha256 of the cloud, the plate omegas and the heat field.  ``fields`` defaults to every
    field the cloud has (a determinism check); the trunk fingerprints pass TRUNK_FIELDS."""
    h = hashlib.sha256()
    for f in (sim.seg.FIELDS if fields is None else fields):
        a = np.ascontiguousarray(getattr(sim.seg, f))
        h.update(f.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    for nm, a in (("omega", sim.plates.omega), ("heat", sim.heat.data)):
        h.update(nm.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _classic_run(preset: str, steps: int):
    from globe.config import PRESETS

    p = PRESETS[preset]()
    if preset == "earth":
        p.tectonics = classic_dynamics(p.tectonics)
    sim = tect.initialise(p, log=None)
    for _ in range(steps):
        sim.step()
    return sim


@pytest.mark.parametrize("preset,steps", [("tiny", 60), ("small", 300), ("earth", 300)])
def test_classic_dynamics_are_bit_identical_to_trunk(preset, steps):
    """CLASSIC_DYNAMICS (the toy presets, and Earth with the classic knobs) run the shipped
    dynamics bit for bit: the synth-dyn code paths are all behind knobs they leave off."""
    sim = _classic_run(preset, steps)
    assert _fingerprint(sim, TRUNK_FIELDS) == TRUNK_FINGERPRINTS[(preset, steps)]
    _assert_new_fields_neutral(sim)


def _assert_new_fields_neutral(sim):
    """The fields added since trunk stay at their neutral values under the classic dynamics:
    no arc docks (terrane 0) and the extent balance never flips a point (shown NEVER)."""
    seg = sim.seg
    extra = [f for f in seg.FIELDS if f not in TRUNK_FIELDS]
    assert set(extra) <= {"terrane", "shown"}, extra
    if "terrane" in extra:
        assert not seg.terrane.any()
    if "shown" in extra:
        assert (seg.shown == seg.NEVER).all()


@pytest.mark.slow
def test_classic_earth_is_bit_identical_to_trunk_through_its_clock_rift():
    sim = _classic_run("earth", 610)
    assert _fingerprint(sim, TRUNK_FIELDS) == TRUNK_FINGERPRINTS[("earth", 610)]
    _assert_new_fields_neutral(sim)


def _halves(sim):
    """Cut the supercontinent (plate 0) in two along a great circle through its centre, with
    no rift registered.  Returns (a, b, c, n): the plates, the cut's centre and normal."""
    seg, pl = sim.seg, sim.plates
    sel = np.flatnonzero(seg.plate_id == 0)
    c = seg.pos[sel].mean(axis=0)
    c /= np.linalg.norm(c)
    v = np.array([0.3, -0.5, 0.8])
    n = v - c * (v @ c)
    n /= np.linalg.norm(n)
    pid = seg.plate_id.copy()
    pid[sel[seg.pos[sel] @ n > 0]] = pl.P
    sim.plates = intraplate._rebuild(seg, pid, pl.P + 1, np.random.default_rng(0), 0.0, keep=pl.omega, snap=False)
    return 0, pl.P, c, n


def test_trial_solves_see_the_partition_after_a_relabel():
    """An event that relabels plates (here: one plate captured by its neighbour) leaves the
    force-phase balance describing the old partition -- the boundary drag and couplings to a
    plate that is now part of the same plate.  force_state re-solves before a trial solve, so
    releasing the merged plate uncut gives its terminal velocity again."""
    sim = tect.initialise(_small_earth(), log=None)
    for _ in range(3):
        sim.step()
    seg, pl = sim.seg, sim.plates
    cen = sim.census_last
    oo = (cen["ki"] == OCEANIC) & (cen["kj"] == OCEANIC)
    a, b = int(cen["pi"][oo][0]), int(cen["pj"][oo][0])
    stale = sim.balance
    pid = seg.plate_id.copy()
    pid[pid == b] = a
    sim.plates = intraplate._rebuild(seg, pid, pl.P, np.random.default_rng(0), 0.0, keep=pl.omega, snap=False)
    sel = np.flatnonzero(seg.plate_id == a)
    none = np.zeros(0, np.int64)
    w_stale, _ = forces.release(stale, seg, sel, np.zeros(sel.size, bool), none, none, 0.0)
    bal = sim.force_state()
    assert bal is not stale and np.array_equal(bal.pid, seg.plate_id)
    assert sim.force_state() is bal                                   # once per relabel
    w, _ = forces.release(bal, seg, sel, np.zeros(sel.size, bool), none, none, 0.0)
    assert np.allclose(w, bal.wstar[a], rtol=1e-8, atol=1e-14)
    assert not np.allclose(w_stale, bal.wstar[a], rtol=1e-3, atol=0)  # the hazard was real


def test_a_rift_clock_does_not_run_the_force_rift_twice():
    """rift_every > 0 with rift_mode 'force': the clock branch dispatched to the same force rift
    test (intraplate.rift switches on rift_mode) from the same rng stream, so it ran twice."""
    sim = tect.initialise(_small_earth(rift_every=10, rift_check_every=10), log=None)
    calls = []
    orig = intraplate.force_rifts

    def counted(s, rng):
        calls.append(s.step_index)
        return orig(s, rng)

    intraplate.force_rifts = counted
    try:
        for _ in range(31):
            sim.step()
    finally:
        intraplate.force_rifts = orig
    assert calls == [10, 20, 30]


def test_reorganise_clears_the_plate_keyed_books():
    """reorganise renumbers every plate from 0: a rift pair, a passive clock or a quiet suture
    keyed on an old id would land on an unrelated plate."""
    sim = tect.initialise(_small_earth(reorganise_every=1, reorganise_plates=6), log=None)
    sim.step()
    sim.rift_pairs[(0, 1)] = {"k0": 0, "delta": 0.0, "G0": 2.0}
    sim.micro_passive[3] = 0
    sim.suture_quiet[(1, 2)] = 0
    sim.last_rift[2] = 0
    sim.step()
    assert not sim.rift_pairs and not sim.micro_passive and not sim.suture_quiet and not sim.last_rift


def test_a_one_plate_start_still_has_its_ocean_plate():
    """initial_plates 1: the Pangaea tilings always make one ocean plate (id 1), so the plate
    table is sized by the ids, not the knob."""
    sim = tect.initialise(_small_earth(initial_plates=1), log=None)
    assert sim.plates.P >= int(sim.seg.plate_id.max()) + 1
    for _ in range(5):
        sim.step()


def test_census_reads_its_pairs_off_the_split_query():
    """Perf: the census takes its pairs from split_disconnected's wider query of the same cloud
    (plus the segments spawned since) instead of a third tree and query a step.  Every census
    of a run equals one from a fresh query, array for array."""
    sim = tect.initialise(_small_earth(rift_deficit=0.4), log=None)
    orig = forces.census
    seen = []

    def census(seg, plates, spacing, radius_factor=1.25, tree=None, pairs=None, labels=None):
        out = orig(seg, plates, spacing, radius_factor, tree, pairs, labels)
        if pairs is not None:
            ref = orig(seg, plates, spacing, radius_factor, tree, None, labels)
            for k in out:
                assert np.array_equal(np.asarray(out[k]), np.asarray(ref[k])), k
            seen.append(sim.step_index)
        return out

    forces.census = census
    try:
        for _ in range(60):
            sim.step()
    finally:
        forces.census = orig
    assert len(seen) >= 55


def _dense_polarity(plate_id, age, kind, pairs, P):
    """plate_pair_polarity as trunk computed it: P x P bincounts."""
    pi = plate_id[pairs[:, 0]].astype(np.int64)
    pj = plate_id[pairs[:, 1]].astype(np.int64)
    pol = np.zeros((P, P), dtype=np.int8)
    sel = (pi != pj) & (kind[pairs[:, 0]] == kind[pairs[:, 1]])
    if not sel.any():
        return pol
    a, b = pi[sel], pj[sel]
    swap = a > b
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    age_i, age_j = age[pairs[sel, 0]], age[pairs[sel, 1]]
    key = lo * P + hi
    cnt = np.bincount(key, minlength=P * P)
    s_lo = np.bincount(key, weights=np.where(swap, age_j, age_i), minlength=P * P)
    s_hi = np.bincount(key, weights=np.where(swap, age_i, age_j), minlength=P * P)
    k = np.nonzero(cnt)[0]
    older = s_lo[k] > s_hi[k]
    pol[k[older] // P, k[older] % P] = 1
    pol[k[~older] % P, k[~older] // P] = 1
    return pol


def test_sparse_plate_pair_polarity_is_the_dense_one():
    rng = np.random.default_rng(0)
    M = 5000
    for P in (3, 70, 700):
        for dec in (0, 1, 3):
            pid = rng.integers(0, P, M).astype(np.int32)
            age = np.round(rng.random(M) * 100, dec)            # ties included
            kind = (rng.random(M) < 0.3).astype(np.int8)
            pairs = rng.integers(0, M, (4000, 2))
            assert np.array_equal(collision.plate_pair_polarity(pid, age, kind, pairs, P),
                                  _dense_polarity(pid, age, kind, pairs, P))


def _cc_closing(sim, cen, w):
    """Length-weighted mean approach (cm/yr) over the census's C-C contacts under omegas w."""
    seg = sim.seg
    m = (cen["ki"] == CONTINENTAL) & (cen["kj"] == CONTINENTAL)
    vi = np.cross(w[cen["pi"][m]], seg.pos[cen["i"][m]])
    vj = np.cross(w[cen["pj"][m]], seg.pos[cen["j"][m]])
    return float(np.average(np.sum((vi - vj) * cen["dd"][m], axis=1), weights=cen["w"][m])) * sim.R_km * 0.1 / sim.tp.myr_per_step


def test_collisional_resistance_slows_convergence_and_never_locks_or_pulls():
    """Two continental plates pushed together close slower with the C-C drag, still close (it
    is a drag on the relative velocity, solved implicitly: it never reverses or locks), and two
    pulled apart feel nothing of it (it acts on converging contacts only)."""
    sim = tect.initialise(_small_earth(), log=None)
    a, b, c, n = _halves(sim)
    seg, pl = sim.seg, sim.plates
    e = np.cross(c, n)                       # rotating a about +e moves its edge towards b
    drag = sim.drag_per_len()

    def balance(cc):
        cen = forces.census(seg, pl, sim.spacing)
        return cen, forces.assemble(seg, pl, cen, gain=sim.gain, damping=0.05, grad3=None, basal_seg=sim.basal_seg(),
                                    drag_per_len=drag, cc_per_len=cc, slab=None, trench_per_len=0.0, rift_pairs=None,
                                    rift_scale=1.0, rift_weaken=1.0)

    for sgn in (1.0, -1.0):
        tau = np.zeros((pl.P, 3))
        tau[a], tau[b] = sgn * 1e-3 * e, -sgn * 1e-3 * e
        _, b0 = balance(0.0)
        w0 = forces.solve(b0, pl, seg.plate_id, extra_tau=tau)
        pl.omega = w0.copy()                 # the census's "converging" is the free motion's
        cen, b1 = balance(120.0 * drag)
        w1 = forces.solve(b1, pl, seg.plate_id, extra_tau=tau)
        free, held = _cc_closing(sim, cen, w0), _cc_closing(sim, cen, w1)
        if sgn > 0:
            assert b1.info["cc_pairs"] > 0
            assert 0.0 < held < 0.5 * free
        else:
            assert free < 0.0 and b1.info["cc_pairs"] == 0
            assert np.array_equal(w0, w1)


def _rifted(**kw):
    sim = tect.initialise(_small_earth(rift_candidates=48, **kw), log=None)
    for _ in range(2):
        sim.step()
    _insulated(sim)
    ev = intraplate.rift(sim, np.random.default_rng(5), 1)
    return sim, ev["pairs"][0], ev["detail"][0]


def test_a_rift_necks_slow_then_fast():
    """With its calibrated G0 a rift opens below 1.5x rift_slow while it has opened less than
    half of rift_weaken, and more than twice that once it has opened 1.5x rift_weaken."""
    sim, (a, b), det = _rifted(rift_slow_cmyr=0.5)
    st = sim.rift_pairs[(a, b)]
    assert st["G0"] > 0.0
    cms = sim.R_km * 0.1 / sim.tp.myr_per_step
    wk = sim.km(sim.tp.rift_weaken_km)
    rate = {}
    for frac in (0.0, 0.45, 1.5):
        st["delta"] = frac * wk
        w, _, _ = sim.terminal_omega(sim.grad3)
        om = sim.plates.omega
        sim.plates.omega = w
        rate[frac] = forces.rift_opening(forces.census(sim.seg, sim.plates, sim.spacing), sim.rift_pairs)[(a, b)] * cms
        sim.plates.omega = om
    slow = sim.tp.rift_slow_cmyr
    assert 0.0 < rate[0.0] < 1.5 * slow and rate[0.45] < 1.5 * slow
    assert rate[1.5] > 2.0 * rate[0.45]


def test_margin_test_needs_a_floor_and_feels_its_seed_slab():
    # the extent balance off: the seed slab's net effect on a released floor depends on the
    # margin's shape (its pulls along a curved margin partly cancel while its resistance adds
    # up), and with te/mass's balance on, this toy geometry after 3 steps has the floor closing
    # at 1.80 cm/yr without the seed slab and 1.55 with it (3.42 -> 3.77 with it off)
    sim = tect.initialise(_small_earth(extent_split=False), log=None)
    for _ in range(3):
        sim.step()
    seg = sim.seg
    co = np.flatnonzero((seg.plate_id == 0) & (seg.kind == CONTINENTAL))
    assert intraplate._margin_test(sim, 0, co, np.zeros(0, np.int64)) is None      # no floor, no margin
    # an old floor on the continent's plate: the ocean plate it shares most margin with
    cen = sim.census_last
    m = (cen["pi"] == 0) & (cen["pj"] != 0)
    nb = int(np.bincount(cen["pj"][m], weights=cen["w"][m]).argmax())
    pid = seg.plate_id.copy()
    pid[pid == nb] = 0
    sim.plates = intraplate._rebuild(seg, pid, sim.plates.P, np.random.default_rng(0), 0.0, keep=sim.plates.omega, snap=False)
    oc = np.flatnonzero((seg.plate_id == 0) & (seg.kind != CONTINENTAL))
    seg.age[oc] = sim.steps_of(200.0)
    appr = {}
    for km in (0.0, 400.0):
        sim.tp.margin_collapse_slab_km = km
        appr[km] = intraplate._margin_test(sim, 0, co, oc)["approach_cmyr"]
    assert appr[400.0] > appr[0.0] and appr[400.0] > 0.0


def test_suture_weld_joins_a_finished_collision_only():
    """Two continental plates in quiet contact for suture_time_my are one plate (the larger
    keeps it); the same pair converging at 2 cm/yr is not welded."""
    for rate, welds in ((0.0, True), (2.0, False)):
        sim = tect.initialise(_small_earth(), log=None)
        for _ in range(2):
            sim.step()
        a, b, c, n = _halves(sim)
        pl = sim.plates
        e = np.cross(c, n)
        intraplate._force_state(sim)
        cen = sim.census_last
        m = (cen["ki"] == CONTINENTAL) & (cen["kj"] == CONTINENTAL)
        # the normal rate along the contact of a unit relative rotation about e: scale it to `rate`
        unit = np.median(np.abs(np.sum(np.cross(e, sim.seg.pos[cen["i"][m]]) * cen["dd"][m], axis=1)))
        s = sim.cmyr(rate) / unit
        w = pl.omega[a].copy()
        pl.omega[a], pl.omega[b] = w + 0.5 * s * e, w - 0.5 * s * e
        A = np.bincount(sim.seg.plate_id, weights=sim.seg.ext)
        intraplate.suture_weld(sim, np.random.default_rng(0))
        sim.step_index += int(sim.steps_of(sim.tp.suture_time_my)) + 1
        ev = intraplate.suture_weld(sim, np.random.default_rng(0))
        if welds:
            keep, join = (a, b) if A[a] >= A[b] else (b, a)
            assert ev["welds"] and ev["welds"][0]["kept"] == keep and not sim.plates.alive[join]
        else:
            assert not ev["welds"] and sim.plates.alive[a] and sim.plates.alive[b]


def _microplate(sim):
    """Carve a 3-segment plate out of ocean plate X at its boundary with Y; returns (q, X, Y)."""
    seg, pl = sim.seg, sim.plates
    cen = sim.census_last
    k0 = np.flatnonzero((cen["ki"] == OCEANIC) & (cen["kj"] == OCEANIC))[0]
    X, Y, i0 = int(cen["pi"][k0]), int(cen["pj"][k0]), int(cen["i"][k0])
    inX = np.flatnonzero(seg.plate_id == X)
    qs = inX[np.argsort(np.linalg.norm(seg.pos[inX] - seg.pos[i0], axis=1))[:3]]
    pid = seg.plate_id.copy()
    q = pl.P
    pid[qs] = q
    sim.plates = intraplate._rebuild(seg, pid, pl.P + 1, np.random.default_rng(0), 0.0, keep=pl.omega, snap=False)
    sim.plates.omega[:] = 0.0
    sim.slab.interior[...] = 0.0
    return q, X, Y, qs


def _merge_after_life(sim, q):
    intraplate.micro_merge(sim, np.random.default_rng(0))
    sim.step_index += int(sim.steps_of(sim.tp.micro_life_my)) + 1
    ev = intraplate.micro_merge(sim, np.random.default_rng(0))
    return [m for m in ev["merges"] if m["plate"] == q]


def test_micro_merge_spares_slab_pulled_plates_and_converging_neighbours():
    """A passive microplate is captured by its longest neighbour (control); one hanging on a
    live slab is not passive; and one converging on its longest neighbour goes to the other."""
    def setup():
        sim = tect.initialise(_small_earth(), log=None)
        for _ in range(3):
            sim.step()
        return (sim,) + _microplate(sim)

    sim, q, X, Y, qs = setup()                                   # control: everything still
    A = 4.0 * math.pi * sim.tp.micro_area
    assert sim.tp.plate_min_area * 4 * math.pi < sim.seg.ext[qs].sum() < A
    got = _merge_after_life(sim, q)
    assert got and got[0]["into"] == X and not got[0]["remnant"]
    sim, q, X, Y, qs = setup()                                   # slab-attached: q's old floor dives under a saturated slab
    sim.seg.age[qs] = 1e6
    sim.slab.interior[...] = sim.km(sim.tp.slab_sat_km)
    assert not _merge_after_life(sim, q) and sim.plates.alive[q]
    sim, q, X, Y, qs = setup()                                   # q converges on X at 3 cm/yr
    seg = sim.seg
    cq = seg.pos[qs].mean(axis=0)
    cq /= np.linalg.norm(cq)
    d = seg.pos[seg.plate_id == X].mean(axis=0) - cq
    d -= cq * (d @ cq)
    sim.plates.omega[q] = sim.cmyr(3.0) * np.cross(cq, d / np.linalg.norm(d))
    got = _merge_after_life(sim, q)
    assert got and got[0]["into"] == Y


def test_pair_gate_leaves_the_hole_behind_a_slab_open():
    """A slab's plate dives under a continent drifting slowly the same way: the hole its front
    leaves is not a ridge.  The nearest-segment test filled it with floor on the overrider
    (the continent's edge moves away from the hole); the pair gate does not."""
    got = {}
    for gate in ("nearest", "pair"):
        sim = tect.initialise(_small_earth(), log=None)
        seg, pl = sim.seg, sim.plates
        cen = forces.census(seg, pl, sim.spacing)
        k0 = np.flatnonzero((cen["pj"] == 0) & (cen["ki"] == OCEANIC) & (cen["kj"] == CONTINENTAL))[0]
        A, c, d = int(cen["pi"][k0]), cen["rm"][k0], cen["dd"][k0]
        e = np.cross(c, d)                                       # rotation about e moves c along d
        pl.omega[:] = 0.0
        pl.omega[A] = sim.cmyr(5.0) * e
        pl.omega[0] = sim.cmyr(5.0) / 3.0 * e
        seg.compress(~((seg.plate_id == A) & (np.linalg.norm(seg.pos - c, axis=1) < 2.0 * sim.spacing)))
        pl.update_stats(seg)
        tree = collision.build_tree(seg)
        idx, dist = collision.label_map_fast(seg, sim.grid, sim.r_cap, tree)
        new, _ = collision.spawn_segments(seg, idx, dist, sim.grid, sim.r_gap, sim.r_spawn, np.random.default_rng(0),
                                          sim.heat, sim.tp.oceanic_thickness, sim.tp.oceanic_density, omega=pl.omega,
                                          tree=tree, ext=sim.spacing ** 2, void="create", pair_gate=gate == "pair")
        got[gate] = int((np.linalg.norm(new.pos - c, axis=1) < 2.5 * sim.spacing).sum()) if new.M else 0
    assert got["nearest"] > 0 and got["pair"] == 0


@pytest.mark.parametrize("preset", ["small", "earth"])
def test_zipf_ocean_tiling_is_a_connected_power_law(preset):
    """Every initial ocean plate is one piece and its share of the sphere is within
    [ocean_plate_min, ocean_plate_max] +- 0.02, the largest first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from globe.config import PRESETS
    from globe.tectonics.plates import seed_supercontinent, zipf_ocean_plates
    from globe.tectonics.run import best_candidate_sphere

    tp = (_small_earth() if preset == "small" else PRESETS["earth"]()).tectonics
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        M = int(tp.segments)
        pos = best_candidate_sphere(M, rng)
        kind = seed_supercontinent(pos, float(tp.continental_fraction), float(tp.craton_fraction), int(tp.cratons), rng)[0]
        pid = zipf_ocean_plates(pos, kind, int(tp.initial_plates), rng, lo=tp.ocean_plate_min, hi=tp.ocean_plate_max,
                                alpha=tp.ocean_plate_alpha)
        share = np.bincount(pid, minlength=int(tp.initial_plates)) / M
        assert np.all(pid[kind == CONTINENTAL] == 0) and np.all(pid[kind != CONTINENTAL] > 0)
        oc = share[1:]
        assert np.all(oc >= tp.ocean_plate_min - 0.02) and np.all(oc <= tp.ocean_plate_max + 0.02), oc
        assert oc.max() > 2.0 * np.median(oc)
        pr = cKDTree(pos).query_pairs(1.6 * math.sqrt(4 * math.pi / M), output_type="ndarray")
        for q in range(1, int(tp.initial_plates)):
            idx = np.flatnonzero(pid == q)
            loc = np.full(M, -1)
            loc[idx] = np.arange(idx.size)
            e = pr[(pid[pr[:, 0]] == q) & (pid[pr[:, 1]] == q)]
            g = coo_matrix((np.ones(e.shape[0]), (loc[e[:, 0]], loc[e[:, 1]])), shape=(idx.size, idx.size))
            assert connected_components(g, directed=False)[0] == 1, (seed, q)


def _dyn_events_run(steps, seed=2):
    sim = tect.initialise(_small_earth(seed=seed, rift_deficit=0.4), log=None)
    kinds = set()
    for _ in range(steps):
        sim.step()
        kinds |= {e.get("event") for e in sim.events
                  if e.get("pairs") or e.get("merges") or e.get("welds") or e.get("collapses") or e.get("healed")}
    return sim, kinds


def test_earth_dynamics_are_deterministic_and_keep_the_books():
    """Two runs of the dynamics with the same seed end bit for bit alike -- the events draw
    their own rng streams (6 rift, 11 collapse, 12 micro, 13 heal, 14 weld) keyed on the step --
    and with the events firing every kind's books still close."""
    # seed 0, 250 steps: with the arc and mass knobs on (te/arcs, te/mass) the first micro-merge
    # comes at step 221 on seed 0 and 281 on seed 2 (141 on seed 2 with the arcs alone, < 120
    # with the dynamics alone)
    a, kinds = _dyn_events_run(250, seed=0)
    b, _ = _dyn_events_run(250, seed=0)
    assert {"rift", "micro_merge"} <= kinds
    assert _fingerprint(a) == _fingerprint(b)
    assert max(map(abs, a.books_residual())) < 1e-9


@pytest.mark.slow
def test_earth_preset_is_deterministic():
    from globe.config import PRESETS

    fps = []
    for _ in range(2):
        p = PRESETS["earth"]()
        p.world.seed = 1
        sim = tect.initialise(p, log=None)
        for _ in range(300):
            sim.step()
        fps.append(_fingerprint(sim))
    assert fps[0] == fps[1]


def test_scorecard_observer_stays_pure_on_the_earth_dynamics():
    """The dynamics block reads the force state (slab_info), takes its own census and follows
    the rifts' opening traces: none of it may touch the run.  And it reports them."""
    from globe.tectonics import diagnostics as D

    a, _ = _dyn_events_run(90)
    b = tect.initialise(_small_earth(seed=2, rift_deficit=0.4), log=None)
    rep = D.observe(b, 90, 30)
    assert _fingerprint(a) == _fingerprint(b) and a.ledger == b.ledger
    last = rep["samples"][-1]
    for k in ("trench400_share", "largest_noarc_share", "arc_crust_share", "oc_conv_median_cmyr", "size_exponent",
              "slab_ocean_speed_cmyr", "plate_rows"):
        assert k in last["dyn"], k
    assert "cc_noarc_share" in last["window"] and "ev_micro" in last["window"]
    # convergence by ground: the census's length x rate, and what the kernel took of it
    for k in ("cc_kin_share", "cc_kernel_share", "cc_take", "sub_take"):
        assert k in last["window"], k
    ws = [s["window"] for s in rep["samples"][1:]]
    assert all(w["kin_all"] > 0.0 and 0.0 <= w["cc_kin_share"] <= 1.0 for w in ws)
    rifts = [r for r in rep["final"]["rift_list"] if r.get("G0") is not None]
    assert rifts and all("peak_open_cmyr" in r for r in rifts)
    # the books residual and the continental books by process in km3/yr (te/mass's cv_*)
    assert last["books"]["residual_max"] < 1e-9
    m = last["mass"]
    for k in ("additions_kmyr", "docked_kmyr", "accreted_kmyr", "gross_kmyr", "margin_kmyr", "growth_kmyr", "net_kmyr",
              "additions_kmyr_win", "gross_kmyr_win", "net_kmyr_win"):
        assert k in m, k
    assert abs(m["additions_kmyr"] - m["gross_kmyr"] - m["growth_kmyr"]) < 1e-9 and m["gross_kmyr"] >= 0.0
    import json
    json.dumps(rep)


def test_earth_dynamics_regression_gate():
    """The Earth preset's first 300 steps (45 My) against the synth-dyn measurements: a
    girdled Pangaea in a power-law superocean, nothing at the speed cap, the books closed."""
    from globe.config import PRESETS
    from globe.tectonics import diagnostics as D

    p = PRESETS["earth"]()
    p.world.seed = 1
    sim = tect.initialise(p, log=None)
    rep = D.observe(sim, 300, 150)
    s0, s1, s2 = rep["samples"]
    assert 55e3 <= s0["bnd"]["subduction_km"] <= 80e3                   # Pangaea's girdle ~65,500 km
    assert 0.18 <= s0["plates"]["largest_share"] <= 0.42 and s0["plates"]["top7_share"] >= 0.9
    oc = [x for x, c in zip(s0["plates"]["shares"], s0["plates"]["cont_share"]) if c < 0.2]
    assert 0.18 <= max(oc) <= 0.24                                     # the Pacific-sized plate
    assert 35.0 <= s0["ocean"]["mean_age_myr"] <= 65.0
    for s in rep["samples"]:
        assert s["plates"]["capped_share"] == 0.0
    assert s1["dyn"]["trench400_share"] >= 0.75 and s2["dyn"]["trench400_share"] >= 0.6
    assert s2["dyn"]["slab_ocean_speed_cmyr"] > 2.0 * s2["kin"]["v_cont_median_cmyr"]
    assert 0.38 <= s2["books"]["cont_extent_share"] <= 0.43
    assert max(map(abs, sim.books_residual())) < 1e-9


@pytest.mark.slow
def test_earth_dynamics_scorecard_gate_2000_steps():
    """The synth-dyn spec's landing gate, seeds 0-3 to 2000 steps (300 My): rifts without a far
    side, no plate at the cap, Earth-like plate lives, the continents kept, the supercontinent
    dispersed by 300 My, an ocean floor of Earth's age."""
    from globe.config import PRESETS
    from globe.tectonics import diagnostics as D

    dispersed = 0
    for seed in range(4):
        p = PRESETS["earth"]()
        p.world.seed = seed
        rep = D.observe(tect.initialise(p, log=None), 2000, 250)
        fin, last = rep["final"], rep["samples"][-1]
        assert (fin["rift_far_share"] or 0.0) <= 0.06, seed
        assert all(s["plates"]["capped_share"] == 0.0 for s in rep["samples"]), seed
        assert 8.0 <= fin["lifetime_median_dead_myr"] <= 25.0, seed
        assert 0.36 <= last["books"]["rendered_cont_share"] <= 0.44, seed
        assert last["books"]["ledger"]["ext_coll_cont"] >= -0.5, seed
        assert 45.0 <= last["ocean"]["mean_age_myr"] <= 85.0, seed
        dispersed += any((s["dyn"]["largest_noarc_share"] or 1.0) <= 0.7 for s in rep["samples"])
    assert dispersed >= 3


def _converging_halves(rate_cmyr, **kw):
    """The supercontinent cut in two (no rift registered), the halves closing at ``rate_cmyr``
    (median over their contact) about the cut's own pole."""
    sim = tect.initialise(_small_earth(**kw), log=None)
    for _ in range(2):
        sim.step()
    a, b, c, n = _halves(sim)
    pl = sim.plates
    e = np.cross(c, n)
    intraplate._force_state(sim)
    cen = sim.census_last
    m = (cen["ki"] == CONTINENTAL) & (cen["kj"] == CONTINENTAL)
    unit = np.median(np.abs(np.sum(np.cross(e, sim.seg.pos[cen["i"][m]]) * cen["dd"][m], axis=1)))
    s = sim.cmyr(rate_cmyr) / unit
    w = pl.omega[a].copy()
    pl.omega[a], pl.omega[b] = w + 0.5 * s * e, w - 0.5 * s * e
    if np.average(_cc_closing_rates(sim, a, b)) < 0:      # make it closing, whichever way the cut faces
        pl.omega[a], pl.omega[b] = pl.omega[b].copy(), pl.omega[a].copy()
    return sim, a, b


def _cc_closing_rates(sim, a, b):
    cen = forces.census(sim.seg, sim.plates, sim.spacing)
    m = (cen["ki"] == CONTINENTAL) & (cen["kj"] == CONTINENTAL)
    w = sim.plates.omega
    vi = np.cross(w[cen["pi"][m]], sim.seg.pos[cen["i"][m]])
    vj = np.cross(w[cen["pj"][m]], sim.seg.pos[cen["j"][m]])
    return np.sum((vi - vj) * cen["dd"][m], axis=1)


def test_contact_persistence_weld():
    """suture_persist_my: a long C-C contact held that long is welded at any closing rate below
    suture_persist_cmyr -- a pair closing at 3 cm/yr, which the quiet weld never takes."""
    for limit, welds in ((0.0, True), (2.0, False)):
        sim, a, b = _converging_halves(3.0, suture_persist_my=50.0, suture_persist_km=1500.0, suture_persist_cmyr=limit)
        intraplate.suture_weld(sim, np.random.default_rng(0))
        assert not sim.suture_quiet                                    # not quiet: 3 cm/yr
        sim.step_index += int(sim.steps_of(50.0)) + 1
        ev = intraplate.suture_weld(sim, np.random.default_rng(0))
        assert bool(ev["welds"]) == welds


def test_a_stalled_rift_lets_its_halves_go():
    """rift_stall_km: a rift that opened past half its weakening length and then stopped is
    released (two plates, an ordinary boundary, no coupling); one still opening is kept, and
    one that never opened half-way heals as before."""
    sim, (a, b), _ = _rifted(rift_stall_km=12.0)
    st = sim.rift_pairs[(a, b)]
    lim = int(sim.steps_of(sim.tp.rift_abort_my))
    sim.step_index += lim + 1
    k = sim.step_index
    st["delta"] = 0.8 * sim.km(sim.tp.rift_weaken_km)
    st["trace"] = [(k - 5, sim.km(1.0))]                          # 1 km in the last 40 My: stalled
    ev = intraplate.heal_failed_rifts(sim, np.random.default_rng(0))
    assert ev["released"] and not ev["healed"] and (a, b) not in sim.rift_pairs
    assert sim.plates.alive[a] and sim.plates.alive[b]
    sim.rift_pairs[(a, b)] = dict(st, trace=[(k - 5, sim.km(30.0))])       # still opening
    assert not intraplate.heal_failed_rifts(sim, np.random.default_rng(0))["released"]
    assert (a, b) in sim.rift_pairs
    sim.tp.rift_stall_km = 0.0                                      # off: a stalled rift holds
    sim.rift_pairs[(a, b)]["trace"] = []
    assert not intraplate.heal_failed_rifts(sim, np.random.default_rng(0))["released"]


def test_frozen_ids_stop_a_collision_chain_within_one_step():
    """collide_frozen_ids: a continental loser relabelled onto the survivor's plate no longer
    collides, in the same call, with its former plate-mates as if it were the survivor's --
    which spent the step's convergence again at every link of the chain."""
    same = {}
    for frozen in (False, True):
        sim, a, b = _converging_halves(20.0)
        seg, tp = sim.seg, sim.tp
        pid0 = seg.plate_id.copy()
        alive = np.ones(seg.M, bool)
        lo, su = tect.collide(seg, collision.build_tree(seg), sim.r_coll, sim.plates.omega, alive, tp.overlap_fraction,
                              float(tp.arc_accretion), 0.0, None, shortening=float(tp.continental_shortening),
                              weld_steps=int(tp.weld_steps), extent_min=float(tp.extent_min) * sim.spacing ** 2,
                              spent_out=[], arc_out=[], recv_out=[], books_out=[], frozen_ids=frozen)
        cc = (seg.kind[lo] == CONTINENTAL) & (seg.kind[su] == CONTINENTAL)
        assert cc.sum() > 0
        same[frozen] = int((pid0[lo] == pid0[su]).sum())
    assert same[False] > 0 and same[True] == 0


def test_cc_shortening_takes_what_the_plates_close():
    """The collision kernel's continental shortening against the plates' own convergence --
    the census's boundary length x closing rate over the C-C contacts, the yardstick the
    scorecard's window cc_take uses: on two halves closing at 1 or 8 cm/yr the shipped rule
    (live ids) spends 1.08x of it, rate-independently, and frozen ids 0.90x.  (On assembled
    contacts at 225-525 My the same state gives 0.84-0.92x live and 0.64-0.68x frozen; over
    whole Earth runs both settle near 0.7x.)"""
    takes = {}
    for rate in (1.0, 8.0):
        for frozen in (False, True):
            sim, a, b = _converging_halves(rate)
            seg, tp = sim.seg, sim.tp
            tree = collision.build_tree(seg)
            cen = forces.census(seg, sim.plates, sim.spacing, tree=tree)
            m = (cen["ki"] == CONTINENTAL) & (cen["kj"] == CONTINENTAL) & (cen["appr"] > 0)
            kin = float((cen["w"][m] * cen["appr"][m]).sum())
            sp = []
            tect.collide(seg, tree, sim.r_coll, sim.plates.omega, np.ones(seg.M, bool), tp.overlap_fraction,
                         float(tp.arc_accretion), 0.0, None, shortening=float(tp.continental_shortening),
                         weld_steps=int(tp.weld_steps), extent_min=float(tp.extent_min) * sim.spacing ** 2,
                         spent_out=sp, arc_out=[], recv_out=[], books_out=[], frozen_ids=frozen)
            takes[rate, frozen] = sp[0] / kin
    for rate in (1.0, 8.0):
        assert 0.95 <= takes[rate, False] <= 1.2, takes
        assert takes[rate, True] < takes[rate, False], takes
    assert abs(takes[1.0, False] - takes[8.0, False]) < 0.02 * takes[8.0, False]
