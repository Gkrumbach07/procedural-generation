"""Island arcs (te/arcs): arc crust at the volcanic front, partial arc subduction, docking
booked by kind, root foundering, docked terranes losing their dense roots, the arc ridge at
its own width, and volcanic edifices -- founded only at persistent trenches on arc crust,
stamped with their peak, and kept out of the erosion stage while they are active."""
import math

import numpy as np
import pytest

from globe.config import CLASSIC_DYNAMICS, TectonicsParams, WorldParams
import globe.tectonics.run as tect
from globe.tectonics import volcanoes as volc
from globe.tectonics.collision import build_tree, collide
from globe.tectonics.segments import CONTINENTAL, OCEANIC, Segments

ARC_OFF = dict(arc_birth=0.05, arc_front=0.0, arc_keep=-1.0, arc_dock_km=0.0, arc_max_km=0.0, terrane_relax_my=0.0,
               arc_ridge_km=0.0, volcanoes=False, hotspots=0)


def _small_earth(seed=0, **kw):
    """`small` with the Earth preset's dynamics and arcs (the toy presets keep the classic ones)."""
    p = WorldParams.small_world(seed)
    d = TectonicsParams()
    for k in CLASSIC_DYNAMICS:
        setattr(p.tectonics, k, getattr(d, k))
    for k, v in kw.items():
        setattr(p.tectonics, k, v)
    return p


def test_classic_dynamics_turn_every_arc_knob_off():
    for k, v in ARC_OFF.items():
        assert CLASSIC_DYNAMICS[k] == v, k
    tp = TectonicsParams()
    assert tp.arc_birth == 0.0 and tp.arc_front > 0 and tp.arc_dock_km > 0 and tp.arc_keep >= 0 and tp.volcanoes
    sim = tect.initialise(WorldParams.tiny_world(), log=None)
    assert sim.volc is None


@pytest.fixture(scope="module")
def arc_sim():
    p = _small_earth(1)
    p.tectonics.steps = 160
    sim = tect.initialise(p, log=None)
    sim.run(160, log=None)
    return sim


def test_books_close_with_the_arcs_on(arc_sim):
    sim = arc_sim
    L = sim.ledger
    assert set(L) <= set(tect.MASS_KEYS) | set(tect.COUNTER_KEYS)
    assert sim.crust_mass() == pytest.approx(sum(L.get(k, 0.0) for k in tect.MASS_KEYS), rel=1e-9)
    assert max(abs(r) for r in sim.books_residual()) < 1e-11
    assert L.get("arc_mantle", 0.0) == 0.0                # no coin-flip arc draws crust from the mantle
    for k in tect.SINK_KEYS:
        assert L.get(k, 0.0) <= 0.0, k
    # a dock onto a continent is a kind flip, booked on both sides
    for k in tect.CONVERSION_KEYS:
        assert sim.books["continental"][k] == -sim.books["oceanic"][k]
    assert np.allclose(sim.seg.mass, sim.seg.thickness * sim.seg.density)
    assert L.get("arc_front_moved", 0.0) > 0.0


def test_arc_crust_is_capped_and_oceanic(arc_sim):
    sim = arc_sim
    oc = sim.seg.kind == OCEANIC
    lim = sim.tp.arc_max_km / sim.tp.crust_km
    # foundering takes `delamination` of the excess a step, so a column can sit a little over
    assert float(sim.seg.thickness[oc].max()) < lim * 1.5
    assert (volc.arc_crust(sim.seg, sim.tp.oceanic_thickness)[oc] > volc.ARC_MIN_TH).any()


def _pair(kind_lo, th_lo, kind_su, th_su, s=0.05, approach=True):
    """Two segments on two plates a little closer than the collision radius, plate 0 moving
    onto plate 1 (or away)."""
    pos = np.array([[1.0, 0.0, 0.0], [math.cos(0.8 * s), math.sin(0.8 * s), 0.0]])
    seg = Segments(pos, np.array([th_lo, th_su]), np.array([0.88 if kind_lo == OCEANIC else 0.804, 0.88 if kind_su == OCEANIC else 0.804]),
                   np.array([300.0, 10.0]), np.array([0, 1], np.int32), s * s, kind=np.array([kind_lo, kind_su], np.int8))
    seg.ext[:] = s * s
    w = 0.002 if approach else -0.002
    omega = np.array([[0.0, 0.0, w], [0.0, 0.0, 0.0]])
    return seg, omega


def test_a_thick_arc_docks_onto_a_continent_and_is_booked():
    s = 0.05
    seg, omega = _pair(OCEANIC, 0.7, CONTINENTAL, 1.0, s)
    m0 = float((seg.ext * seg.mass).sum())
    alive = np.ones(2, bool)
    books, docks = [], []
    collide(seg, build_tree(seg), s, omega, alive, accretion=0.15, weld_steps=60, extent_min=0.25 * s * s,
            books_out=books, dock_out=docks, arc_dock=0.5, arc_keep=0.5, ocean_base=0.2)
    assert seg.kind[0] == CONTINENTAL and seg.terrane[0] == 1   # docked, not subducted, and marked
    assert docks[0][0] == 1.0 and books[0]["docked"][1] > 0.0
    assert books[0]["subducted"] == (0.0, 0.0)
    assert float((seg.ext * seg.mass)[alive].sum()) == pytest.approx(m0, rel=1e-12)   # shortened into the margin


def test_a_docking_arc_accretes_only_part_of_itself():
    """Of an arc docking onto a continent, arc_dock_keep of its ground accretes and the rest goes
    down with its slab: booked as subducted, the books closing over the pair."""
    s = 0.05
    seg, omega = _pair(OCEANIC, 0.7, CONTINENTAL, 1.0, s)
    w0 = seg.ext * seg.mass
    alive = np.ones(2, bool)
    books = []
    collide(seg, build_tree(seg), s, omega, alive, accretion=0.15, weld_steps=60, extent_min=0.25 * s * s,
            books_out=books, arc_dock=0.5, arc_keep=0.25, ocean_base=0.2, dock_keep=0.5)
    assert books[0]["subducted"][1] == pytest.approx(0.5 * w0[0], rel=1e-12)
    assert books[0]["docked"][1] == pytest.approx(0.5 * w0[0], rel=1e-12)
    assert float((seg.ext * seg.mass)[alive].sum()) == pytest.approx(w0.sum() - 0.5 * w0[0], rel=1e-12)


def test_a_thin_arc_goes_down_and_leaves_arc_keep_of_its_arc_crust():
    s = 0.05
    seg, omega = _pair(OCEANIC, 0.4, OCEANIC, 0.2, s)
    seg.age[:] = [300.0, 10.0]
    alive = np.ones(2, bool)
    books, recv = [], []
    th_su0 = float(seg.thickness[1])
    collide(seg, build_tree(seg), s, omega, alive, accretion=0.15, extent_min=0.25 * s * s, books_out=books,
            recv_out=recv, arc_dock=0.5, arc_keep=0.5, ocean_base=0.2)
    assert not alive[0]
    kept = 0.15 * 0.2 + 0.5 * 0.2                         # sea floor at accretion, the arc at arc_keep
    assert float(seg.thickness[1]) - th_su0 == pytest.approx(kept, rel=1e-9)
    assert books[0]["subducted"][0] == pytest.approx((1.0 - kept / 0.4) * 0.4 * 0.88, rel=1e-9)


def test_a_docked_terrane_is_never_handed_back():
    s = 0.05
    # a terrane (thick, welded) on plate 0 that the boundary's polarity would send under plate 1
    seg, omega = _pair(OCEANIC, 0.7, OCEANIC, 0.2, s)
    seg.weld[0] = 3
    alive = np.ones(2, bool)
    collide(seg, build_tree(seg), s, omega, alive, accretion=0.15, weld_steps=60, extent_min=0.25 * s * s,
            arc_dock=0.5, arc_keep=0.5, ocean_base=0.2)
    assert alive[0] and not alive[1]                      # the thin floor went down under it instead
    assert seg.plate_id[0] == 0 and seg.weld[0] == 60     # ...and the contact renewed its weld
    # a thick arc that has docked nowhere docks onto the terrane, not the other way round
    seg, omega = _pair(OCEANIC, 0.7, OCEANIC, 0.7, s)
    seg.weld[0] = 3
    alive = np.ones(2, bool)
    docks = []
    collide(seg, build_tree(seg), s, omega, alive, accretion=0.15, weld_steps=60, extent_min=0.25 * s * s,
            arc_dock=0.5, arc_keep=0.5, ocean_base=0.2, dock_out=docks)
    assert alive.all() and seg.plate_id[1] == 0 and seg.plate_id[0] == 0 and docks[0][2] == 1.0


def test_a_rifts_halves_never_dock_onto_each_other():
    s = 0.05
    seg, omega = _pair(OCEANIC, 0.7, OCEANIC, 0.2, s)
    alive = np.ones(2, bool)
    docks = []
    collide(seg, build_tree(seg), s, omega, alive, accretion=0.15, weld_steps=60, extent_min=0.25 * s * s,
            arc_dock=0.5, arc_keep=0.5, ocean_base=0.2, rift_pairs=[(0, 1)], dock_out=docks)
    assert docks[0] == (0.0, 0.0, 0.0) and seg.plate_id[0] == 0


def test_arc_front_moves_crust_volume_for_volume(arc_sim):
    sim = arc_sim
    seg = sim.seg.copy()
    tree = build_tree(seg)
    alive = np.ones(seg.M, bool)
    cen = sim.census_last
    oo = (seg.kind[cen["i"]] == OCEANIC) & (seg.kind[cen["j"]] == OCEANIC)
    lo, su = cen["i"][oo][:20], cen["j"][oo][:20]
    keep = np.unique(lo, return_index=True)[1]
    lo, su = lo[keep], su[keep]
    m = ~np.isin(lo, su)
    lo, su = lo[m], su[m]
    alive[lo] = False
    v0 = float((seg.ext * seg.mass)[alive].sum())
    rth = 0.03 * np.ones(lo.size)
    rm = rth * 0.88
    ft = volc.front_targets(seg, tree, lo, su, alive, 180.0 / sim.R_km)
    moved = volc.arc_front(seg, ft, lo, su, (rth, rm), 1.0, 0.15)
    assert moved > 0.0
    assert float((seg.ext * seg.mass)[alive].sum()) == pytest.approx(v0, rel=1e-12)
    assert (rth[ft["sel"][ft["has"]]] < 0.03).any()       # the belt is left the rest


def test_vents_only_at_persistent_trenches_on_arc_crust():
    sim = tect.initialise(_small_earth(2, volc_slab_g=2.0), log=None)   # g never reaches 2
    for _ in range(40):
        sim.step()
    assert not (sim.volc.kind != volc.HOTSPOT).any()
    assert sim.volc.events["gated_slab"] > 0
    assert (sim.volc.kind == volc.HOTSPOT).any()          # the plumes still build their chains


def test_the_stamp_keeps_each_edifice_peak():
    p = WorldParams.tiny_world()
    sim = tect.initialise(_small_earth(0), log=None)
    v = sim.volc
    grid = p.coarse_grid()
    # one vent between cell centres, standing 2600 m
    pos = np.array([[0.3, 0.2, 1.0]])
    pos /= np.linalg.norm(pos)
    v._append(pos, np.array([0]), np.array([volc.ARC_OCEAN]), 0, np.array([v.flux_ref]), np.array([-1]))
    v.jit[:] = 1.0
    H = float(v.height(0)[0])
    cone, act, who, info = volc.stamp(v, grid, np.zeros((6, grid.N, grid.N), bool), 0.0)
    assert float(cone.max()) == pytest.approx(H, rel=1e-6)                  # the peak is kept
    assert cone.reshape(-1)[volc.cell_index(grid, pos)[0]] == pytest.approx(H, rel=1e-6)
    assert np.array_equal(act, cone) and (who[cone > 0] == volc.ARC_OCEAN).all()
    cone2, _, _, _ = volc.stamp(v, grid, np.ones((6, grid.N, grid.N), bool), 0.0)
    assert not cone2.any()                                # never on continental crust


def test_the_ridge_holds_the_arc_crusts_volume(arc_sim):
    sim = arc_sim
    p = sim.params
    grid = p.coarse_grid()
    e = volc.arc_excess(sim.seg, sim.tp)
    assert (e > 0).any()
    f, info = volc.ridge_field(sim.seg, sim.tp, grid, e, 1000.0, sim.spacing, sim.R_km, float(p.R_planet))
    vol = float((f * grid.interior_cell_area).sum()) / float(p.R_planet) ** 2
    assert vol == pytest.approx(float((e * sim.seg.ext).sum()) * 1000.0, rel=1e-9)
    assert float(f.max()) <= 1000.0 * float(e.max()) * 1.05      # never above its column's relief


def test_finalise_splits_active_and_extinct_edifices(arc_sim):
    sim = arc_sim
    out = tect.finalise(sim)
    fb = tect.finalise_bed(sim)
    assert np.allclose(out["bedrock"].interior, fb["bed"], atol=1e-3)
    cone, act = fb["cone"].astype(np.float64), fb["cone_active"].astype(np.float64)
    assert (cone > 0).any() and (act <= cone + 1e-6).all()
    vact = out[tect.VOLCANO_FIELD].interior
    assert np.allclose(vact, act)
    n_iter = max(int(sim.params.erosion.iterations), 1)
    m = (cone - act) > 300.0
    if m.any():
        # an extinct edifice is replayed by the erosion stage like any other uplift...
        assert float(np.median(out["uplift"].interior[m])) >= float(np.median((cone - act)[m] / n_iter)) * 0.5
    m = act > 300.0
    if m.any():
        # ...an active one is not (the stage takes it off and puts it back on top)
        assert float(np.median(out["uplift"].interior[m])) < float(np.median(act[m] / n_iter))
    hard = out["hardness"].interior
    assert float(hard[cone > 300.0].min()) >= 0.85 - 1e-6


def test_the_erosion_stage_keeps_active_volcanoes_on_top(scratch, monkeypatch):
    """The erosion stage starts without the active edifices and puts them back on top: with
    every surface process off it ends at the bedrock -- cones and all -- up to the datum."""
    from test_erosion import _no_surface_processes, _uplift_world

    from globe.erosion import run as erosion_run
    from globe.erosion.maps import step

    n = 6
    p = WorldParams.tiny_world(3).with_overrides(erosion={
        "iterations": n, "glacial_every": 0, "checkpoint_every": 0, "quicklook_every": 0, "resume": False})
    store = _uplift_world(scratch, "volcano_on_top", p)
    _no_surface_processes(monkeypatch)
    grid = p.coarse_grid()
    bed = store.load_field("bedrock", grid)
    cone = bed.copy()
    cone.data[...] = 0.0
    cone.interior[1, 5:7, 5:7] = 3000.0
    cone.exchange_halos()
    bed.data += cone.data
    store.save_field(bed)
    cone.name = erosion_run.VOLCANO_FIELD
    store.save_field(cone)
    st = erosion_run.build_state(store, p)
    inter = st.interior
    b0 = (bed.interior.astype(np.float64) - cone.interior) / st.height_unit_m
    # the state starts without the cone (replay lowers it by the applied uplift too)
    assert float(np.max(st.height[inter][1, 5:7, 5:7] - b0[1, 5:7, 5:7])) < 1.0
    for it in range(n):
        step(st, p, it)
    info = erosion_run.restore_volcanoes(st, erosion_run.active_volcanoes(store, grid))
    d = st.height[inter] - bed.interior.astype(np.float64) / st.height_unit_m
    assert np.ptp(d) < 1e-6 and info["volcano_cells"] == 4


def test_observer_reports_the_arcs(arc_sim):
    from globe.tectonics import diagnostics as D

    p = _small_earth(1)
    sim = tect.initialise(p, log=None)
    rep = D.observe(sim, 60, 30, coarse_at=[60])
    row = rep["samples"][-1]
    a = row["arcs"]
    for k in ("ocean_arc_share", "ocean_arc_active_share", "ocean_arc_stranded_share", "docked_share",
              "docked_sea_level_shift_m"):
        assert k in a
    assert row["window"]["froth_share"] is not None and row["window"]["young_slab_share"] is not None
    isl = row["islands"]
    for k in ("crest_p50_m", "n_arc", "n_hotspot", "oo_trench_km", "vents_standing"):
        assert k in isl


def test_the_coarse_block_is_a_pure_observer():
    """The scorecard's coarse block runs finalise_bed mid-run: the run must end bit for bit
    where a plain one does (the vents, the books and the cloud)."""
    from globe.tectonics import diagnostics as D
    from globe.tectonics.segments import Segments as S

    a = tect.initialise(_small_earth(4), log=None)
    for _ in range(30):
        a.step()
    b = tect.initialise(_small_earth(4), log=None)
    D.observe(b, 30, 10, coarse_at=[10, 20])
    for f in S.FIELDS:
        assert np.array_equal(getattr(a.seg, f), getattr(b.seg, f)), f
    assert np.array_equal(a.plates.omega, b.plates.omega) and a.ledger == b.ledger
    for f in volc.Volcanoes.FIELDS:
        assert np.array_equal(getattr(a.volc, f), getattr(b.volc, f)), f


def test_docked_terranes_relax_to_the_belts_density():
    """A docked terrane loses its dense root at constant thickness until it is as light as the
    belts (the mass booked as residue); unmarked continental crust is left alone."""
    sim = tect.initialise(_small_earth(0), log=None)
    seg = sim.seg
    c = np.flatnonzero((seg.kind == CONTINENTAL) & (seg.craton == 0))[:2]
    seg.density[c] = 0.88
    seg.mass[c] = seg.thickness[c] * 0.88
    seg.terrane[c[0]] = 1
    sim.books = {kd: {k: 0.0 for k in tect.KIND_KEYS} for kd in tect.KINDS}
    sim.ledger = {k: 0.0 for k in tect.MASS_KEYS}
    sim._book("initial", *sim.kind_mass())
    th = seg.thickness[c].copy()
    for _ in range(int(8 * sim.steps_of(sim.tp.terrane_relax_my))):
        sim.relax_terranes()
    assert seg.density[c[0]] == pytest.approx(sim.tp.continental_density, abs=2e-3)
    assert seg.density[c[1]] == 0.88 and np.array_equal(seg.thickness[c], th)
    assert sim.ledger["terrane_relaxed"] < 0.0
    assert max(abs(r) for r in sim.books_residual()) < 1e-11
