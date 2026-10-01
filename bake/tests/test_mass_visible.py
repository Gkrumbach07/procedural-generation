"""Continental mass and visible area (te/mass: proto/crit-mass on the synth-dyn dynamics): the
books close, collapse keeps crust as ground, the extent balance only shows ground (the child
keeps its parent's column, no ground is made) and does not flicker, margin erosion takes
ground at the overriding margin, the fold-and-thrust moves what the convergence delivered,
and the sea floor's extent does not compound with age."""
from __future__ import annotations

import math

import numpy as np
import pytest

from globe.tectonics import orogeny
from globe.tectonics.collision import build_tree, collide
from globe.tectonics.extent import shed_extent
from globe.tectonics.segments import CONTINENTAL, OCEANIC, Segments


def _pair(d_sp, ext_lo, ext_su, kind_lo, kind_su, th_lo=1.0, th_su=1.2, sp=0.02):
    """Two segments on the equator ``d_sp`` spacings apart, plate 0 (the first) moving
    east into plate 1 (the second, at rest)."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([math.cos(d_sp * sp), math.sin(d_sp * sp), 0.0])
    rho = np.array([0.804 if kind_lo == CONTINENTAL else 0.88, 0.804 if kind_su == CONTINENTAL else 0.88])
    seg = Segments(np.stack([a, b]), np.array([th_lo, th_su]), rho, 0.0, np.array([0, 1]), sp * sp,
                   kind=np.array([kind_lo, kind_su], np.int8), ext=np.array([ext_lo, ext_su]) * sp * sp)
    omega = np.zeros((2, 3))
    omega[0] = [0.0, 0.0, 0.3 * sp]               # eastward at the equator
    return seg, omega, sp


def test_extent_min_merge_conserves_the_pairs_crust():
    seg, omega, sp = _pair(0.4, 0.26, 0.26, CONTINENTAL, CONTINENTAL)   # discs overlap
    v0 = float((seg.ext * seg.thickness).sum())
    m0 = seg.crust_mass()
    alive = np.ones(2, bool)
    lo, su = collide(seg, build_tree(seg), 1.0 * sp, omega, alive, 0.5, 0.15, 0.0, None,
                     extent_min=0.25 * sp * sp, spent_out=[], arc_out=[], recv_out=[])
    assert lo.size == 1 and int(alive.sum()) == 1   # the loser merged away
    v1 = float((seg.ext * seg.thickness)[alive].sum())
    # the ground the shortening spent carried its crust into the survivor: volume is conserved
    assert abs(v1 - v0) < 1e-12 * max(v0, 1.0), (v0, v1)
    assert abs(float((seg.ext * seg.mass)[alive].sum()) - m0) < 1e-12


def test_margin_erosion_takes_ground_from_the_overriding_continent():
    seg, omega, sp = _pair(0.6, 1.0, 1.0, OCEANIC, CONTINENTAL, th_lo=0.2, th_su=1.0)
    e_su0 = float(seg.ext[1])
    alive = np.ones(2, bool)
    diag = []
    h = 1.2 / 35.0
    lo, su = collide(seg, build_tree(seg), 1.0 * sp, omega, alive, 0.5, 0.15, 0.0, None,
                     extent_min=0.25 * sp * sp, spent_out=[], arc_out=[], recv_out=[],
                     margin_erosion=h, diag_out=diag)
    assert lo.size == 1 and not alive[0] and alive[1]
    vol = h * seg.ext[0]
    g = vol / float(seg.thickness[1])               # at the column it has after the accretion
    # the accretion lands as thickness on the survivor; the erosion takes ground at its column
    assert abs((e_su0 - float(seg.ext[1])) - g) < 1e-15
    assert abs(diag[0][11] - vol) < 1e-15 and abs(diag[0][13] - g) < 1e-15


def test_collapse_keeps_the_crust_as_ground_and_spares_cratons():
    M = 4
    th = np.array([1.6, 1.6, 1.25, 0.9])
    rho = np.array([0.804, 0.804, 0.856, 0.804])
    seg = Segments(np.eye(3)[[0, 1, 2, 0]], th, rho, 0.0, 0, 1e-3, kind=np.full(M, CONTINENTAL, np.int8),
                   ext=np.full(M, 1e-3))
    base, floor = 0.92 * (1 - 0.804), 0.232 * (1 - 0.804)
    ref = Segments(seg.pos, th.copy(), rho, 0.0, 0, 1e-3, kind=np.full(M, CONTINENTAL, np.int8), ext=np.full(M, 1e-3))
    v0 = float((seg.ext * seg.thickness).sum())
    kept_v, kept_g = orogeny.collapse_orogens(seg, base, floor, 0.006, CONTINENTAL, keep=1.0)
    # the same height loss as the mantle sink...
    orogeny.relax_orogens(ref, base * 26400.0, floor * 26400.0, 26400.0, 0.006, CONTINENTAL)
    assert np.allclose(seg.thickness, ref.thickness, rtol=0, atol=1e-12)
    # ...but every bit of the crust stays: volume exact, the ground grows by what was kept
    assert abs(float((seg.ext * seg.thickness).sum()) - v0) < 1e-15
    assert kept_g > 0 and kept_v > 0
    # the craton floats at the baseline and the thin belt is below the floor: untouched
    assert seg.thickness[2] == 1.25 and seg.thickness[3] == 0.9 and seg.ext[2] == 1e-3 and seg.ext[3] == 1e-3
    # keep = 0.9 sends a tenth to the mantle
    s2 = Segments(seg.pos, th.copy(), rho, 0.0, 0, 1e-3, kind=np.full(M, CONTINENTAL, np.int8), ext=np.full(M, 1e-3))
    orogeny.collapse_orogens(s2, base, floor, 0.006, CONTINENTAL, keep=0.9)
    lost = v0 - float((s2.ext * s2.thickness).sum())
    assert abs(lost - 0.1 * (kept_v / 1.0)) < 1e-15


def _patch(n=21, sp=0.02):
    """A flat-ish lattice near (1,0,0): a continental disc on plate 0, its own sea floor
    around it, and plate 1's sea floor beyond x > 7 spacings."""
    pts, kind, plate = [], [], []
    for i in range(-n // 2 + 1, n // 2 + 1):
        for j in range(-n // 2 + 1, n // 2 + 1):
            y, z = i * sp, j * sp
            p = np.array([1.0, y, z]); p /= np.linalg.norm(p)
            r = math.hypot(i, j)
            pts.append(p)
            kind.append(CONTINENTAL if r <= 4 else OCEANIC)
            plate.append(1 if i > 7 else 0)
    pts = np.array(pts)
    kind = np.array(kind, np.int8)
    th = np.where(kind == CONTINENTAL, 1.4, 0.2)
    rho = np.where(kind == CONTINENTAL, 0.804, 0.88)
    seg = Segments(pts, th, rho, 0.0, np.array(plate), sp * sp, kind=kind, ext=np.full(len(pts), sp * sp))
    return seg, sp


def _shed(seg, area, sp, **kw):
    kw.setdefault("ext_max", 3.0)
    return shed_extent(seg, area, sp, kw.pop("ext_max"), np.random.default_rng(0), **kw)


def test_extent_split_only_shows_ground_and_grows_its_own_margin():
    """The child keeps its parent's column: crust, volume and continental ground are what
    they were -- only the point that shows them moved (proto/mass-visible thinned the child
    to belt thickness, which made ground: 5-8 sr per 8000 steps, the continents' largest
    source of it)."""
    seg, sp = _patch()
    c = seg.kind == CONTINENTAL
    centre = int(np.argmin(np.linalg.norm(seg.pos - np.array([1.0, 0, 0]), axis=1)))
    seg.ext[centre] = 2.5 * sp * sp                  # ground the map does not show
    seg.thickness[centre], seg.density[centre] = 1.6, 0.81
    seg.mass[centre] = 1.6 * 0.81
    seg.rework[centre] = 123.0
    v0 = float((seg.ext * seg.thickness)[c].sum())
    cm0 = float((seg.ext * seg.mass)[c].sum())
    g0 = float(seg.ext[c].sum())
    m_all0 = seg.crust_mass()
    n_c0, M0 = int(c.sum()), seg.M
    ev = _shed(seg, np.full(M0, sp * sp), sp, step=500)
    assert ev["split"] == 1 and ev["fail"] == 0
    c1 = seg.kind == CONTINENTAL
    assert int(c1.sum()) == n_c0 + 1 and seg.M == M0           # a continental point for a sea-floor one
    # continental volume, crust and ground exact; the planet's crust changed by the sea floor covered
    assert abs(float((seg.ext * seg.thickness)[c1].sum()) - v0) < 1e-15
    assert abs(float((seg.ext * seg.mass)[c1].sum()) - cm0) < 1e-15
    assert abs(float(seg.ext[c1].sum()) - g0) < 1e-15
    assert abs((seg.crust_mass() - m_all0) + ev["overrun_mass"]) < 1e-15
    # the child is on the parent's plate, at its margin (next to its own continental crust)
    child = seg.M - 1
    assert seg.plate_id[child] == 0 and seg.kind[child] == CONTINENTAL
    d = np.linalg.norm(seg.pos[c1] - seg.pos[child], axis=1)
    assert np.sort(d)[1] < 1.6 * sp
    # the parent's column, density and assembly age; stamped with the step
    assert seg.thickness[child] == 1.6 and seg.density[child] == 0.81 and seg.rework[child] == 123.0
    assert seg.shown[child] == 500
    # the parent gave up exactly the ground the child covers (its slot's cell)
    assert abs(seg.ext[centre] - 1.5 * sp * sp) < 1e-15 and abs(seg.ext[child] - sp * sp) < 1e-18


def test_extent_split_waits_for_a_whole_cell_of_unseen_ground():
    """Hysteresis: half a cell of unseen ground on a plate shows nothing; a parent must hold
    split_at of the child's cell itself."""
    seg, sp = _patch()
    centre = int(np.argmin(np.linalg.norm(seg.pos - np.array([1.0, 0, 0]), axis=1)))
    seg.ext[centre] = 1.5 * sp * sp
    ev = _shed(seg, np.full(seg.M, sp * sp), sp)
    assert ev["split"] == 0 and ev["merged"] == 0
    # two parents with 0.6 cells each: the plate holds 1.2 cells, neither parent split_at 0.75
    seg2, _ = _patch()
    ci = np.flatnonzero(seg2.kind == CONTINENTAL)
    seg2.ext[ci[:2]] = 1.6 * sp * sp
    assert _shed(seg2, np.full(seg2.M, sp * sp), sp)["split"] == 0
    assert _shed(seg2, np.full(seg2.M, sp * sp), sp, split_at=0.5)["split"] == 1


def test_thickness_cap_collapse_matches_delamination_and_keeps_its_share():
    from globe.tectonics.collision import delaminate
    M = 3
    th = np.array([2.8, 2.0, 3.5])
    rho = np.full(M, 0.804)
    kw = dict(kind=np.array([CONTINENTAL, CONTINENTAL, OCEANIC], np.int8), ext=np.full(M, 1e-3))
    a = Segments(np.eye(3), th.copy(), rho, 0.0, 0, 1e-3, **kw)
    b = Segments(np.eye(3), th.copy(), rho, 0.0, 0, 1e-3, **kw)
    v0 = float((a.ext * a.thickness)[:2].sum())
    kv, kg = orogeny.collapse_thick(a, 2.3, 0.05, CONTINENTAL, keep=1.0)
    delaminate(b, 2.3, 0.05)
    assert np.allclose(a.thickness[:2], b.thickness[:2], atol=1e-15)     # the same thinning...
    assert abs(float((a.ext * a.thickness)[:2].sum()) - v0) < 1e-15      # ...with the crust kept as ground
    assert a.thickness[1] == 2.0 and a.ext[1] == 1e-3                    # under the cap: untouched
    assert a.thickness[2] == 3.5                                         # continental crust only
    c = Segments(np.eye(3), th.copy(), rho, 0.0, 0, 1e-3, **kw)
    orogeny.collapse_thick(c, 2.3, 0.05, CONTINENTAL, keep=0.8)
    lost = v0 - float((c.ext * c.thickness)[:2].sum())
    assert abs(lost - 0.2 * kv) < 1e-15


def test_extent_split_takes_a_neighbours_sea_floor_when_it_has_none():
    seg, sp = _patch()
    oc = seg.kind == OCEANIC
    seg.plate_id[oc] = 1                             # every sea floor around the continent is plate 1's
    centre = int(np.argmin(np.linalg.norm(seg.pos - np.array([1.0, 0, 0]), axis=1)))
    seg.ext[centre] = 2.5 * sp * sp
    area = np.full(seg.M, sp * sp)
    a = seg.copy()
    ev = _shed(a, area, sp, active=False)
    assert ev["split"] == 0 and ev["fail"] == 1
    c = seg.kind == CONTINENTAL
    v0, g0 = float((seg.ext * seg.thickness)[c].sum()), float(seg.ext[c].sum())
    area[oc] = 1.3 * sp * sp                          # a sparser sea floor: the child covers its cell
    ev = _shed(seg, area, sp)
    assert ev["split"] == 1 and ev["active"] == 1
    child = seg.M - 1
    assert seg.plate_id[child] == 0 and abs(seg.ext[child] - 1.3 * sp * sp) < 1e-15
    c1 = seg.kind == CONTINENTAL
    assert abs(float((seg.ext * seg.thickness)[c1].sum()) - v0) < 1e-15
    assert abs(float(seg.ext[c1].sum()) - g0) < 1e-15


def test_extent_balance_retreats_a_coast_that_lost_its_ground():
    seg, sp = _patch()
    c = seg.kind == CONTINENTAL
    # a coastal segment eroded to a fifth of its cell: the plate's continents are short
    ci = np.flatnonzero(c)
    _, nb = build_tree(seg).query(seg.pos[ci], k=7)
    coastal = ci[(seg.kind[nb[:, 1:]] != CONTINENTAL).any(axis=1)]
    m = int(coastal[0])
    seg.ext[m] = 0.2 * sp * sp
    # 0.8 of a cell short: inside the band, nothing moves (hysteresis)
    assert _shed(seg.copy(), np.full(seg.M, sp * sp), sp)["merged"] == 0
    seg.ext[int(coastal[1])] = 0.2 * sp * sp         # 1.6 cells short: one cell retreats
    v0 = float((seg.ext * seg.thickness)[c].sum())
    e0 = float(seg.ext[c].sum())
    n_c0 = int(c.sum())
    ev = _shed(seg, np.full(seg.M, sp * sp), sp)
    assert ev["merged"] == 1 and ev["split"] == 0
    c1 = seg.kind == CONTINENTAL
    assert int(c1.sum()) == n_c0 - 1                     # one coastal point fewer
    assert abs(float((seg.ext * seg.thickness)[c1].sum()) - v0) < 1e-15     # crust and ground conserved
    assert abs(float(seg.ext[c1].sum()) - e0) < 1e-15
    # with the books balanced (every segment owns its cell) nothing happens
    seg2, _ = _patch()
    ev2 = _shed(seg2, np.full(seg2.M, sp * sp), sp)
    assert ev2["merged"] == 0 and ev2["split"] == 0


def test_a_retreating_coast_becomes_sea_floor_not_a_hole():
    seg, sp = _patch()
    c = seg.kind == CONTINENTAL
    ci = np.flatnonzero(c)
    _, nb = build_tree(seg).query(seg.pos[ci], k=7)
    coast = ci[(seg.kind[nb[:, 1:]] != CONTINENTAL).any(axis=1)]
    m = int(coast[0])
    seg.ext[coast[:2]] = 0.2 * sp * sp
    seg.age[seg.kind == OCEANIC] = 300.0
    M0 = seg.M
    ev = _shed(seg, np.full(seg.M, sp * sp), sp, step=77, ocean_th=0.2, ocean_rho=0.88)
    assert ev["merged"] == 1 and seg.M == M0                  # no point removed: no hole for a void fill
    assert seg.kind[m] == OCEANIC and seg.age[m] == 300.0 and abs(seg.ext[m] - sp * sp) < 1e-18
    assert seg.shown[m] == 77
    assert abs(ev["flip_mass"] - sp * sp * 0.2 * 0.88) < 1e-15


def test_a_split_parent_is_never_merged_in_the_same_pass():
    """A plate can split (a capped parent) and merge in one pass; the parent must not also
    hand its ground away, or the split's ground comes off a point that is sea floor by then."""
    seg, sp = _patch()
    c = seg.kind == CONTINENTAL
    ci = np.flatnonzero(c)
    _, nb = build_tree(seg).query(seg.pos[ci], k=7)
    coast = ci[(seg.kind[nb[:, 1:]] != CONTINENTAL).any(axis=1)]
    a, b = int(coast[0]), int(coast[1])
    seg.ext[a] = 3.5 * sp * sp                        # capped: splits whatever the balance
    for m in coast[1:]:
        seg.ext[m] = 0.3 * sp * sp                      # the rest of the coast eroded: the plate is short
    ev = _shed(seg, np.full(seg.M, sp * sp), sp)
    assert ev["split"] >= 1 and ev["merged"] >= 1
    assert (seg.ext > 0).all()


def test_a_flipped_point_is_left_alone_for_the_residence_time():
    """A child is not retreated, and a retreated coast is not split onto, within
    `residence` steps: on the label map 21 % of retreats undid a child at most 10 steps old."""
    seg, sp = _patch()
    c = seg.kind == CONTINENTAL
    ci = np.flatnonzero(c)
    _, nb = build_tree(seg).query(seg.pos[ci], k=7)
    coast = ci[(seg.kind[nb[:, 1:]] != CONTINENTAL).any(axis=1)]
    seg.ext[coast] = 0.2 * sp * sp                    # the whole coast eroded: the plate is short
    seg.shown[coast] = 90                             # ...but every coastal point was a child at step 90
    ev = _shed(seg, np.full(seg.M, sp * sp), sp, step=100, residence=67)
    assert ev["merged"] == 0
    ev = _shed(seg, np.full(seg.M, sp * sp), sp, step=200, residence=67)
    assert ev["merged"] >= 1
    # and a sea-floor point the balance made lately is not a slot
    seg2, _ = _patch()
    centre = int(np.argmin(np.linalg.norm(seg2.pos - np.array([1.0, 0, 0]), axis=1)))
    seg2.ext[centre] = 2.5 * sp * sp
    seg2.shown[seg2.kind == OCEANIC] = 95
    assert _shed(seg2.copy(), np.full(seg2.M, sp * sp), sp, step=100, residence=67)["split"] == 0
    assert _shed(seg2, np.full(seg2.M, sp * sp), sp, step=200, residence=67)["split"] == 1


def test_extent_split_needs_the_ocean_only_closure():
    """The whole-cloud closure's refund becomes geometry that overruns sea floor and deepens
    the deficit (proto/mass-visible: extent 0.40 -> 0.97 by step 4000)."""
    from globe.config import WorldParams
    from globe.tectonics import run as tect
    p = WorldParams.tiny_world()
    p.tectonics.extent_split, p.tectonics.closure_ocean_only = True, False
    sim = tect.initialise(p, log=None)
    with pytest.raises(ValueError):
        sim.step()


def test_books_close_per_kind_with_every_piece_on():
    """Collapse at a rate in My, the thickness cap, the extent balance, margin erosion,
    rifted margins stretching, the capped fold-and-thrust, the ocean-only closure with the sea
    floor relaxing to its cells -- with the Earth dynamics: each kind's books still close to
    rounding, and the sea floor's extent stays bounded."""
    from globe.config import CLASSIC_DYNAMICS, TectonicsParams, WorldParams
    from globe.tectonics import run as tect
    p = WorldParams.tiny_world()
    d = TectonicsParams()
    for k in CLASSIC_DYNAMICS:                         # the Earth preset's dynamics and mass physics
        setattr(p.tectonics, k, getattr(d, k))
    t = p.tectonics
    t.steps = 400
    t.orogen_collapse_my, t.orogen_floor_cols = 60.0, 0.02      # a low floor, so collapse fires on 300 segments
    sim = tect.simulate(p, log=None)
    rc, ro = sim.books_residual()
    assert abs(rc) < 1e-10 and abs(ro) < 1e-10, (rc, ro)
    assert abs(sum(sim.ledger[k] for k in tect.MASS_KEYS) - sim.crust_mass()) < 1e-9 * sim.crust_mass()
    assert abs(float(sim.seg.ext.sum()) - 4 * math.pi) < 1e-9
    # the split and the coast retreat move continental crust, never make or lose it: their
    # continental 'overrun' is the phase's whole change, so it must be rounding only
    assert abs(sim.books['continental']['overrun']) < 1e-12 * sim.kind_mass()[0], sim.books['continental']['overrun']
    # every piece fired
    L = sim.ledger
    assert L["cv_orogen_kept"] > 0 and L["n_split"] > 0 and L["margin_eroded"] < 0 and L["ocean_relax"] > 0, L
    assert L["rift_stretch"] > 0 and L["n_merge_coast"] > 0, L
    ocean = sim.seg.kind == OCEANIC
    assert sim.seg.ext[ocean].max() <= float(t.extent_max) * sim.spacing ** 2 * 1.01


def test_fold_and_thrust_moves_what_the_convergence_delivered():
    """A creeping contact registers as a collision every step; the foreland's fold-and-thrust
    must move crust as fast as the plates converge (`fold_cap` x what the survivor was
    handed), not up to 0.4 of each foreland column per event -- which thinned the forelands
    of creeping C-C contacts to 0.2-0.5 columns."""
    sp = 0.02
    pts, plate = [], []
    for i in range(-10, 11):
        for j in range(-10, 11):
            p = np.array([1.0, i * sp, j * sp]); p /= np.linalg.norm(p)
            pts.append(p)
            plate.append(1 if (i == 1 and j == 0) else 0)
    n = len(pts)
    base = Segments(np.array(pts), 1.0, 0.804, 300.0, np.array(plate), sp * sp, kind=np.full(n, CONTINENTAL, np.int8),
                    ext=np.full(n, sp * sp))
    lo = int(np.flatnonzero(base.plate_id == 1)[0])
    su = int(np.argmin(np.linalg.norm(base.pos - np.array([1.0, 0.0, 0.0]), axis=1)))     # beside it, on plate 0
    th_in = 1e-3
    taken = {}
    for cap in (0.0, 1.0):
        seg = base.copy()
        seg.thickness[su] += th_in                       # what the collision handed the survivor
        seg.mass[su] = seg.thickness[su] * seg.density[su]
        v0 = float((seg.ext * seg.thickness).sum())
        th0 = seg.thickness.copy()
        alive = np.ones(n, bool)
        recv = (np.array([th_in]), np.array([th_in * 0.804]))
        orogeny.shape_belt(seg, build_tree(seg), np.array([lo]), np.array([su]), alive, sp, 6.371e6, 26400.0, 0.25,
                           CONTINENTAL, received=recv, conserve_volume=True, fold_cap=cap)
        assert abs(float((seg.ext * seg.thickness).sum()) - v0) < 1e-14       # volume either way
        drop = np.maximum(th0 - seg.thickness, 0.0)
        drop[su] = 0.0
        taken[cap] = float((drop * seg.ext).sum())
    assert taken[0.0] > 50 * th_in * sp * sp                 # uncapped: the profile, whatever converged
    assert taken[1.0] <= 1.0 * th_in * sp * sp * (1 + 1e-9)   # capped: what converged


@pytest.mark.slow
def test_earth_keeps_its_continents_visible_and_in_the_books():
    """Earth preset, arc_birth 0 (the arcs track's default), 1500 steps (~225 My, through the
    girdle phase and the first breakup): the continents keep 0.40 +- 0.04 of the ground, the
    map shows at least 0.95 of it, and each kind's books close."""
    from globe.config import PRESETS
    from globe.tectonics import run as tect
    p = PRESETS["earth"]()
    p.world.seed = 1
    p.tectonics.steps = 1500
    p.tectonics.arc_birth = 0.0
    sim = tect.initialise(p, log=None)
    for _ in range(1500):
        sim.step()
    c = sim.seg.kind == CONTINENTAL
    share = float(sim.seg.ext[c].sum()) / float(sim.seg.ext.sum())
    _, cc = tect.frame_bed(sim, with_c=True)
    a = sim.grid.interior_cell_area.astype(np.float64)
    rendered = float(a[cc > 0.5].sum() / a.sum())
    assert abs(share - 0.40) < 0.04, share
    assert rendered / share > 0.95, (rendered, share)
    rc, ro = sim.books_residual()
    assert abs(rc) < 1e-12 and abs(ro) < 1e-12, (rc, ro)
