"""Erosion stage tests (PLAN.md sections 8.4 / 15).

* conservation on a closed single-face window (deposit-on-exit mode),
* dendritic networks, erosion-attributable incision (against a no-erosion
  twin) + frozen divides on a flat-ish noise face with synthetic uplift
  (window mode, the refine-stage contract),
* no seam artefacts, bounded per-iteration change, sane bedrock/sediment
  and a stable coastline in the global 6-face pass (stub upstream stages),
* byte-identical determinism, checkpoint/resume through the driver,
* uplifted regions stay high relative to a no-uplift twin.

Everything runs on tiny grids (N <= 128) so the file stays well under a
minute.  Upstream data always comes from ``globe.stubs`` (never the other
engineers' real stages, docs/DEVELOPING.md).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from scipy import ndimage

from globe.config import WorldParams
from globe.erosion import glacial
from globe.erosion import particle as pk
from globe.erosion import run as erosion_run
from globe.erosion.maps import ErosionState, apply_uplift, run_iteration, step
from globe.hydro import run as hydro_run
from globe.io.world_store import WorldStore
from globe.stubs import stub_climate, stub_tectonics

try:  # PLAN 15 seam metric lives in the viz tests
    from tests.test_viz import seam_discontinuity
except ImportError:  # pragma: no cover - alternative rootdir layouts
    from test_viz import seam_discontinuity


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _log(_msg):
    pass


def noise2d(NE: int, rng: np.random.Generator, octaves: int = 5, base: float = 3.0) -> np.ndarray:
    """Smooth 2-D value noise in roughly [-1, 1] (window tests)."""
    out = np.zeros((NE, NE))
    amp, tot = 1.0, 0.0
    x = np.linspace(0, 1, NE, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for o in range(octaves):
        f = base * 2**o
        n = int(np.ceil(f)) + 2
        lat = rng.random((n + 1, n + 1))
        qx, qy = X * f, Y * f
        i0 = np.floor(qx).astype(int)
        j0 = np.floor(qy).astype(int)
        tx, ty = qx - i0, qy - j0
        tx = tx * tx * (3 - 2 * tx)
        ty = ty * ty * (3 - 2 * ty)
        v = lat[i0, j0] * (1 - tx) * (1 - ty) + lat[i0 + 1, j0] * tx * (1 - ty) + lat[i0, j0 + 1] * (1 - tx) * ty + lat[i0 + 1, j0 + 1] * tx * ty
        out += amp * (v * 2 - 1)
        tot += amp
        amp *= 0.5
    return out / tot


def make_window(N: int, mode: str, params: WorldParams, seed: int = 0, relief: float = 6.0, uplift_total: float = 8.0, iters: int = 150, deposit_on_exit: bool = False):
    """Single-face window state (flat metric, heights in cells).

    ``closed``: positive noise everywhere, halo = mask 0, no ocean, no uplift.
    ``tilt``: plane rising along +i with noise, a deep ocean strip along
    i = 0, frozen (mask 2) wall rims on the other three sides and a
    Gaussian uplift ridge near the far side.
    ``dome``: closed face with an uplift dome in the centre.
    """
    H = params.world.halo
    NE = N + 2 * H
    rng = np.random.default_rng(seed)
    cs = params.cell_size_m
    x = (np.arange(NE) - H + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.zeros((NE, NE), np.uint8)
    mask[H : H + N, H : H + N] = 1
    precip = np.ones((NE, NE), np.float32)
    upl = np.zeros((NE, NE))
    if mode == "closed":
        h = relief * (0.5 + 0.5 * noise2d(NE, rng)) + 0.5
    elif mode == "dome":
        h = relief * (0.5 + 0.5 * noise2d(NE, rng)) + 0.5
        upl = uplift_total / iters * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.18**2))
    elif mode == "tilt":
        h = relief * (0.35 * noise2d(NE, rng) + X) + 0.5
        h = np.where(X < 6.0 / N, -20.0, h)  # deep ocean strip (does not silt up in 150 iterations)
        wall = ((Y < 2.0 / N) | (Y > 1 - 2.0 / N) | (X > 1 - 2.0 / N)) & (mask == 1)
        h = np.where(wall, relief + 5.0, h)
        mask[wall] = 2
        precip[wall] = 0
        h = np.where(mask == 0, relief + 10.0, h)
        upl = uplift_total / iters * np.exp(-((X - 0.85) ** 2) / (2 * 0.12**2))
    else:
        raise ValueError(mode)
    metric = np.zeros((NE, NE, 3), np.float32)
    metric[..., 0] = 1.0
    metric[..., 2] = 1.0
    z = np.zeros((NE, NE))
    return ErosionState.window(
        h * cs, z, z, np.zeros((NE, NE, 2)), np.full((NE, NE), 0.5, np.float32), precip, np.ones((NE, NE), np.float32), upl * cs, mask, metric, metric, H, cs, deposit_on_exit=deposit_on_exit
    )


def channel_mask(state: ErosionState) -> np.ndarray:
    """Top 5 % discharge cells of the land (window face 0)."""
    q = state.discharge[state.interior][0]
    land = (state.surface()[state.interior][0] >= 0) & (state.mask[state.interior][0] == pk.MASK_ACTIVE)
    return (q > np.percentile(q[land], 95)) & land


def local_relief(surf: np.ndarray, where: np.ndarray, r: int = 3) -> np.ndarray:
    """Height of the (2r+1)² neighbourhood mean above each selected cell:
    positive where the cell sits in a valley."""
    k = np.ones((2 * r + 1, 2 * r + 1))
    k[r, r] = 0
    nb = ndimage.convolve(surf, k / k.sum(), mode="nearest")
    return (nb - surf)[where]


def discharge_metrics(state: ErosionState) -> dict:
    q = state.discharge[state.interior][0]
    land = (state.surface()[state.interior][0] >= 0) & (state.mask[state.interior][0] == pk.MASK_ACTIVE)
    ql = q[land]
    srt = np.sort(ql)[::-1]
    top1 = float(srt[: max(1, srt.size // 100)].sum() / max(srt.sum(), 1e-12))
    thr = np.percentile(ql, 95)
    m = (q > thr) & land
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    return {"top1_share": top1, "n_components": int(n), "frac_in_big": float(sizes[sizes >= 16].sum() / max(sizes.sum(), 1)), "largest": int(sizes.max()) if n else 0}


def _stub_world(scratch, name: str, params: WorldParams) -> WorldStore:
    store = WorldStore(scratch / name, create=True)
    store.init_manifest(params, params.coarse_grid())
    for s, fn in (("tectonics", stub_tectonics), ("climate", stub_climate)):
        fn(store, params, _log)
        store.mark_stage(s, [], {"stub": True}, 0.0, params=params)
    return store


# --------------------------------------------------------------------------
# 1. conservation
# --------------------------------------------------------------------------
def test_conservation_closed_window():
    p = WorldParams.small_world(0)
    st = make_window(48, "closed", p, deposit_on_exit=True)
    assert np.all(st.surface()[st.interior] > 0)  # no ocean: nothing can leave as ocean deposit
    m0 = st.total_mass()
    clamped = 0
    for it in range(20):
        s = step(st, p, it)
        assert s["particles"] > 0
        clamped += s["clamped"]
    m1 = st.total_mass()  # height + sediment + pending
    assert abs(m1 - m0) <= 1e-6 * abs(m0), (m0, m1)
    assert clamped > 0  # the live caps were exercised, and mass still balances
    assert st.sediment[st.interior].max() > 0  # particles did move mass around
    assert np.all(st.sediment >= 0)
    assert np.all(st.pending >= 0)


def test_offshore_loss_is_accounted():
    """Open coast (``tilt`` window, deposit-on-exit): the mass that leaves
    the modelled surface is exactly the load submarine fans could not
    place (``lost_offshore``) — everything else stays in height + sediment
    + pending — and nothing is parked in ``pending`` on a submerged cell."""
    p = WorldParams.small_world(0)
    st = make_window(48, "tilt", p, deposit_on_exit=True)
    m0 = st.total_mass()
    lost = 0.0
    for it in range(25):
        s = step(st, p, it)
        lost += s["lost_offshore"]
        # no deficit leaves this window: the residue is the rounding of
        # cancelling a clamped particle's later deposits (~1e-18)
        assert s["deficit_out"] <= 1e-12 * abs(m0)
    # window mode applies uplift as given (only the global state is made
    # mean-free, apply_uplift), so it is still a mass source here
    upl = float(st.uplift[st.interior][st.mask[st.interior] == pk.MASK_ACTIVE].sum()) * 25
    m1 = st.total_mass()
    assert abs((m1 + lost) - (m0 + upl)) <= 1e-6 * abs(m0), (m0, m1, lost, upl)
    sea = st.surface() < 0
    assert st.pending[sea].sum() == 0.0
    assert s["deaths"]["ocean"] > 0


def test_exit_without_deposit_loses_mass_only():
    """Default window mode: particles leaving the mask deposit nothing, so
    the total can only decrease (and does, on an open face)."""
    p = WorldParams.small_world(0)
    st = make_window(48, "closed", p, deposit_on_exit=False)
    m0 = st.total_mass()
    for it in range(10):
        step(st, p, it)
    assert st.total_mass() < m0


# --------------------------------------------------------------------------
# 2. dendritic networks, frozen divides
# --------------------------------------------------------------------------
def test_dendritic_network_and_frozen_divides():
    """Flat-ish noise + synthetic uplift, 150 iterations, window mode, run
    twice: with the default erodibility and as a no-erosion twin
    (erodibility 0: particles route and thermal erosion runs, but no
    fluvial erosion / deposition).

    * Network: the top 1 % of land cells carry > 10 % of the discharge (a
      uniform field gives 1 %) and the top-5 % cells are spatially coherent
      (>= 60 % in 8-connected components of >= 16 cells, largest >= 48;
      measured 65-78).
    * Erosion did the work: the final channel cells are *incised* — their
      local relief (17x17 neighbourhood mean minus the cell; hillslope
      creep widens valleys to several cells, so a 7x7 window would sit
      inside them) grows by more than 0.04 cells in the median (measured
      0.09), while the twin's channels are not (measured -0.06: creep and
      thermal erosion only smooth).
    * Frozen wall cells (mask 2) are sampled but never modified (refine's
      divide contract); the uplift ridge stays the highest part of the face.
    """
    p = WorldParams.small_world(0)
    res = {}
    for erod in (p.erosion.erodibility, 0.0):
        q = dataclasses.replace(p)
        q.erosion = dataclasses.replace(p.erosion, erodibility=erod)
        st = make_window(96, "tilt", q, iters=150)
        frozen = st.mask == pk.MASK_FROZEN
        assert frozen.any()
        h_frozen = st.height[frozen].copy()
        s0 = st.surface()[st.interior][0].copy()
        for it in range(150):
            step(st, q, it)
        chan = channel_mask(st)
        surf = st.surface()[st.interior][0]
        res[erod] = {
            "metrics": discharge_metrics(st),
            "incision": float(np.median(local_relief(surf, chan, r=8)) - np.median(local_relief(s0, chan, r=8))),
            "frozen_ok": bool(np.array_equal(st.height[frozen], h_frozen) and np.all(st.sediment[frozen] == 0.0)),
            "surf": surf,
        }
    m = res[p.erosion.erodibility]["metrics"]
    assert m["top1_share"] > 0.10, m
    assert m["frac_in_big"] > 0.6, m
    assert m["largest"] >= 48, m
    inc, inc0 = res[p.erosion.erodibility]["incision"], res[0.0]["incision"]
    assert inc > 0.04, (inc, inc0)
    assert inc > inc0 + 0.08, (inc, inc0)
    assert all(r["frozen_ok"] for r in res.values())
    # the uplift ridge stays the highest part of the face
    surf = res[p.erosion.erodibility]["surf"]
    N = surf.shape[0]
    ridge = surf[int(0.75 * N) : int(0.95 * N), 4:-4].mean()
    plain = surf[int(0.15 * N) : int(0.4 * N), 4:-4].mean()
    assert ridge > plain + 3.0


# --------------------------------------------------------------------------
# 3. no seams in the global pass
# --------------------------------------------------------------------------
def test_global_pass_has_no_seams_and_is_bounded(scratch):
    """30 iterations on a stub small world.  Seam metric (PLAN 15) on every
    output; per-iteration surface change bounded by the caps (particles
    <= iter_deposit / iter_erode, thermal <= 8 * thermal_max inflow, plus
    uplift); bedrock never dug below the erosion budget; sediment p99 on
    land sane; the coastline is the tectonic one (land fraction drifts by
    < 2 %: particles and mass wasting never raise a sea cell above sea
    level and never erode a land cell below it)."""
    p = WorldParams.small_world(3)
    store = _stub_world(scratch, "erosion_seam", p)
    st = erosion_run.build_state(store, p)
    ep = p.erosion
    bed = st.height.copy()
    land0 = np.mean(st.surface()[st.interior] >= 0)
    upl_max = st.uplift[st.interior].max()
    mx = 0.0
    for it in range(30):
        s0 = st.surface().copy()
        step(st, p, it)
        mx = max(mx, np.abs(st.surface() - s0)[st.interior].max())
    assert mx < ep.iter_deposit + 8 * ep.thermal_max + upl_max + 0.5, mx
    inter = st.interior
    assert (st.height - bed)[inter].min() > -(ep.iter_erode + ep.thermal_max) * 30, (st.height - bed)[inter].min()
    land = st.surface()[inter] >= 0
    assert np.percentile(st.sediment[inter][land], 99) < 5.0  # cells
    assert st.sediment[inter].max() < 30.0
    assert abs(np.mean(land) - land0) < 0.02, (land0, np.mean(land))
    assert st.pending[inter].sum() < 0.01 * abs(st.total_mass())
    f = st.fields()
    assert seam_discontinuity(f["discharge"]) < 3.0
    assert seam_discontinuity(f["height"]) < 3.0
    assert seam_discontinuity(f["sediment"]) < 3.0
    assert f["discharge"].interior.max() > 10 * f["discharge"].interior.mean()  # rivers exist
    # particles really cross faces: every face receives discharge
    assert all(f["discharge"].interior[k].max() > 0 for k in range(6))


# --------------------------------------------------------------------------
# 4. determinism + checkpoint / resume through the driver
# --------------------------------------------------------------------------
def test_determinism_two_fresh_runs(scratch):
    p = WorldParams.tiny_world(5)
    store = _stub_world(scratch, "erosion_det", p)
    outs = []
    for _ in range(2):
        st = erosion_run.build_state(store, p)
        for it in range(6):
            step(st, p, it)
        outs.append([st.height.copy(), st.sediment.copy(), st.discharge.copy(), st.momentum.copy()])
    for a, b in zip(*outs):
        assert a.tobytes() == b.tobytes()


def test_driver_checkpoint_resume(scratch):
    p = WorldParams.tiny_world(7)
    p.erosion = dataclasses.replace(p.erosion, iterations=10, checkpoint_every=5, quicklook_every=5)
    store = _stub_world(scratch, "erosion_resume", p)
    info = erosion_run.run(store, p, _log)
    assert info["iterations"] == 10
    ref = {n: store.load_field(n, p.coarse_grid()).data.copy() for n in erosion_run.OUTPUTS}
    assert (store.checkpoint_dir / "erosion_iter0005.npz").exists()
    assert (store.checkpoint_dir / "erosion_iter0010.json").exists()
    assert (store.quicklook_dir / "erosion_iter0005.png").exists()
    meta = json.loads((store.checkpoint_dir / "erosion_iter0010.json").read_text())
    assert meta["iteration"] == 10
    # drop the final checkpoint and the outputs: the driver must resume from iteration 5
    for suf in (".npz", ".json"):
        (store.checkpoint_dir / f"erosion_iter0010{suf}").unlink()
    store.clear_outputs(erosion_run.OUTPUTS)
    msgs = []
    erosion_run.run(store, p, msgs.append)
    assert any("resumed" in m and "iteration 5" in m for m in msgs), msgs
    for n in erosion_run.OUTPUTS:
        assert store.load_field(n, p.coarse_grid()).data.tobytes() == ref[n].tobytes(), n
    # a checkpoint written for different erosion params is ignored
    p2 = p.with_overrides(erosion={"friction": p.erosion.friction * 0.5})
    assert erosion_run.find_checkpoint(store, p2) is None
    assert erosion_run.find_checkpoint(store, p)[1]["iteration"] == 10
    # the kernel version is part of the checkpoint hash
    assert f":k{pk.KERNEL_VERSION}" in erosion_run._ckpt_hash(p, store)
    # resume=False recomputes from bedrock (and still lands on the same output)
    # without orphaning the checkpoints it did not read (it is a run-control
    # knob: same checkpoint hash, and excluded from the content hash)
    p3 = p.with_overrides(erosion={"resume": False})
    assert erosion_run._ckpt_hash(p3, store) == erosion_run._ckpt_hash(p, store)
    assert p3.content_hash() == p.content_hash()
    msgs = []
    erosion_run.run(store, p3, msgs.append)
    assert not any("resumed" in m for m in msgs)
    for n in erosion_run.OUTPUTS:
        assert store.load_field(n, p.coarse_grid()).data.tobytes() == ref[n].tobytes(), n


def test_checkpoints_are_pruned_and_a_shorter_run_resumes(scratch):
    """Only the two newest checkpoints of a parameter family survive, and
    ``iterations`` is not part of the checkpoint identity: a shorter rerun
    resumes from the newest checkpoint at or below its end."""
    p = WorldParams.tiny_world(11)
    p.erosion = dataclasses.replace(p.erosion, iterations=9, checkpoint_every=3, quicklook_every=0)
    store = _stub_world(scratch, "erosion_prune", p)
    erosion_run.run(store, p, _log)
    kept = sorted(q.name for q in store.checkpoint_dir.glob("erosion_iter*.npz"))
    assert kept == ["erosion_iter0006.npz", "erosion_iter0009.npz"], kept
    assert not (store.checkpoint_dir / "erosion_iter0003.json").exists()
    assert erosion_run.find_checkpoint(store, p)[1]["iteration"] == 9
    # shortening the run reuses the checkpoints (max_iteration rejects 9)
    p6 = p.with_overrides(erosion={"iterations": 6})
    assert erosion_run._ckpt_hash(p6, store) == erosion_run._ckpt_hash(p, store)
    assert erosion_run.find_checkpoint(store, p6, max_iteration=6)[1]["iteration"] == 6
    # ... and a real state knob still separates the families
    p2 = p.with_overrides(erosion={"friction": p.erosion.friction * 0.5})
    assert erosion_run._ckpt_hash(p2, store) != erosion_run._ckpt_hash(p, store)
    # the prune only ever touches its own family
    foreign = store.checkpoint_dir / "erosion_iter0001.npz"
    foreign.write_bytes(b"")
    foreign.with_suffix(".json").write_text(json.dumps({"iteration": 1, "params_hash": "other"}))
    erosion_run._prune_checkpoints(store.checkpoint_dir, erosion_run._ckpt_hash(p, store), keep=1)
    assert sorted(q.name for q in store.checkpoint_dir.glob("erosion_iter*.npz")) == ["erosion_iter0001.npz", "erosion_iter0009.npz"]


# --------------------------------------------------------------------------
# 5. uplift
# --------------------------------------------------------------------------
def test_uplift_keeps_region_high():
    p = WorldParams.small_world(0)
    iters = 60
    a = make_window(64, "dome", p, iters=iters, uplift_total=6.0)
    b = make_window(64, "dome", p, iters=iters, uplift_total=6.0)
    b.uplift[...] = 0.0
    for it in range(iters):
        step(a, p, it)
        step(b, p, it)
    N = a.N
    c = slice(N // 2 - 6, N // 2 + 6)
    sa = a.surface()[a.interior][0][c, c].mean()
    sb = b.surface()[b.interior][0][c, c].mean()
    assert sa - sb > 0.5 * 6.0, (sa, sb)  # at least half the applied uplift survives erosion
    assert a.surface()[a.interior].max() < 40.0  # and nothing blew up


def test_uplift_is_mean_free_on_the_sphere(scratch):
    """On the global state ``apply_uplift`` adds no net mass: it raises the
    cells above the active-interior mean uplift and lowers the rest by the
    same total, so the kernel's base level stays at z = 0 (the planet does
    not inflate and hydro's re-quantile has nothing to undo).  A window
    keeps its uplift as given — it owns no planetary datum."""
    p = WorldParams.small_world(0)
    grid = p.coarse_grid()
    store = _stub_world(scratch, "uplift_mean_free", p)
    f = {n: store.load_field(n, grid) for n in ("bedrock", "hardness", "precip", "evap", "uplift")}
    st = ErosionState.from_grid(grid, f["bedrock"], f["hardness"], f["precip"], f["evap"], f["uplift"], p.erosion)
    assert st.uplift[st.interior].mean() > 0.0  # the stub uplift really is a source
    m0 = st.total_mass()
    h0 = st.height.copy()
    apply_uplift(st)
    assert abs(st.total_mass() - m0) <= 1e-9 * max(abs(m0), 1.0)
    d = (st.height - h0)[st.interior]
    u = st.uplift[st.interior]
    assert np.allclose(d - d.mean(), u - u.mean(), atol=1e-12)  # only the datum moved
    assert d.max() > 0.0 > d.min()  # a real shift, not a no-op
    # the surface therefore does not drift upwards over an iteration
    w = make_window(48, "dome", p, iters=10, uplift_total=8.0)
    m0 = w.total_mass()
    apply_uplift(w)
    up = float(w.uplift[w.interior][w.mask[w.interior] == pk.MASK_ACTIVE].sum())
    assert abs((w.total_mass() - m0) - up) <= 1e-9 * max(abs(m0), 1.0)


def test_datum_is_held_through_the_run(scratch):
    """The global pass keeps ``world.land_fraction`` of the cells above sea
    level from the first iteration to the last (``maps.hold_datum``), so
    hydro's re-quantile has nothing left to correct and never drops sea
    level onto terrain that was sculpted against a different base level.

    Two things move the datum: uplift is a forcing (``apply_uplift``
    removes its mean, but only over the *active* cells) and mass leaves the
    surface for the deep ocean.  The drift is therefore real even with a
    mean-free uplift — with the hold disabled this world ends at land
    fraction 0.263 instead of 0.300, and before the mean-free uplift the
    small e2e world ended at 0.56 with a -164 m hydro shift.  The stub
    uplift is scaled to the ~1 m per iteration the real tectonics stage
    produces (the unscaled stub is 4 orders of magnitude smaller, which is
    why the other global-pass tests cannot see this).
    """
    p = WorldParams.tiny_world(3)
    p.erosion = dataclasses.replace(p.erosion, iterations=20, checkpoint_every=0, quicklook_every=0, resume=False)
    store = _stub_world(scratch, "erosion_datum", p)
    grid = p.coarse_grid()
    upl = store.load_field("uplift", grid)
    upl.data *= 1000.0
    store.save_field(upl)
    assert float(upl.interior.mean()) > 0.5  # net positive, as tectonic uplift is

    info = erosion_run.run(store, p, _log)
    assert abs(info["land_fraction"] - p.world.land_fraction) < 0.02, info["land_fraction"]
    assert "datum_drift_m" in info

    # ... and the hydro re-quantile the contract keeps as a backstop is a no-op
    hinfo = hydro_run.run(store, p, _log)
    assert abs(hinfo["height_shift_m"]) < 5.0, hinfo["height_shift_m"]
    assert abs(hinfo["land_fraction"] - p.world.land_fraction) < 0.02


# --------------------------------------------------------------------------
# 6. kernel-level contracts
# --------------------------------------------------------------------------
def test_run_iteration_stats_and_height_units():
    p = WorldParams.small_world(0)
    st = make_window(32, "closed", p, deposit_on_exit=True)
    assert st.height_unit_m == p.cell_size_m  # height_unit_m = 0 -> cell units
    s = run_iteration(st, p, (0, 1))
    assert s["particles"] == round(p.erosion.particles_per_cell * 32 * 32)
    assert sum(s["deaths"].values()) == s["particles"]
    assert s["deaths"]["ocean"] == 0  # no ocean on a closed face
    assert st.disch_track.max() == 0.0  # tracks are folded into the EMA and zeroed
    assert st.discharge.max() > 0.0
    assert np.isfinite(st.momentum).all()


def _striped_hardness_run(cover: float, N: int = 96, iters: int = 60, period: int = 16):
    """Tilted window whose hardness alternates in bands *across* the regional
    slope.  Returns (surface in soft bands, surface in hard bands).

    ``cover`` is metres, like `erosion.cover_depth` itself -- the length
    parameters are declared in metres and divided by the cell size, so that
    the same physical alluvial thickness means the same thing on a 50 m grid
    and on a 9.8 km one."""
    p = WorldParams.small_world(0).with_overrides(erosion={"cover_depth": cover})
    st = make_window(N, "tilt", p, iters=iters)
    H = p.world.halo
    jj = np.arange(st.hardness.shape[2])
    band = 0.5 + 0.45 * np.sign(np.sin(2 * np.pi * (jj - H) / period))
    st.hardness[...] = np.broadcast_to(band[None, None, :], st.hardness.shape).astype(st.hardness.dtype)
    for i in range(iters):
        step(st, p, (i,))
    surf = st.surface()[st.interior][0]
    soft = np.broadcast_to((band < 0.5)[H : H + N][None, :], surf.shape)
    return float(surf[soft].mean()), float(surf[~soft].mean())


def test_alluvial_cover_lets_hardness_shape_the_landscape():
    """``erosion.cover_depth`` is what makes the hardness field bite.

    Erodibility blends between sediment (fully erodible) and bedrock
    (scaled by hardness) over ``cover_depth`` of cover, so a thin film no
    longer shields the rock beneath it.  With the old bare-rock-only gate
    (``cover_depth = 0``) hardness reached only the 0.1 % of land cells with
    exactly zero sediment, and bands of hard and soft rock eroded to nearly
    the same level; with a cover scale the soft bands sit distinctly lower,
    which is what gives a landscape its structural grain.
    """
    soft_off, hard_off = _striped_hardness_run(0.0)
    soft_on, hard_on = _striped_hardness_run(5.0)  # 0.1 cell units at the 50 m cells this runs on
    relief_off = hard_off - soft_off  # measured -0.003 cell units: no response at all
    relief_on = hard_on - soft_on  # measured +0.139
    assert relief_on > 2.0 * relief_off, (relief_off, relief_on)
    assert relief_on > 0.08, relief_on  # soft rock really is carved down


def test_glacial_carving_makes_closed_basins_and_conserves_mass():
    """Glacial carving is the only pass that can leave a lake.

    Every other erosional term is bounded below by where the water can get
    out, so a fluvial landscape drains completely: the tectonic bedrock of a
    baked world had 1 closed depression in 456,693 land cells.  Ice flows
    uphill out of a basin, so `glacial.carve` lowers the bed under ice with
    no base-level limit at all, and puts the spoil on the margin as moraine.

    Checks the two properties the rest of the pipeline relies on: it really
    does create depressions the flood can pond in, and it moves rock rather
    than creating or destroying it (so it composes with `hold_datum`).
    """
    p = WorldParams.small_world(0).with_overrides(erosion={"glacial_rate": 1.0, "glacial_every": 10})
    st = make_window(64, "dome", p, iters=40)
    # freeze the upper half of the dome: `evap <= 0` is the kernel's `T <= 0`
    st.evap[...] = 1.0
    high = st.surface() > np.percentile(st.surface()[st.interior], 70)
    st.evap[high] = 0.0
    st.discharge[...] = 4.0  # a uniform ice flux, so the taper alone shapes the cut

    before = st.total_mass()
    depressions_before = _closed_depressions(st)
    stats = glacial.carve(st, p)
    after = st.total_mass()

    assert stats["ice_cells"] > 0 and stats["carved"] > 0.0
    assert stats["moraine_cells"] > 0
    assert abs(after - before) <= 1e-9 * max(abs(before), 1.0), (before, after)
    assert _closed_depressions(st) > depressions_before


def _closed_depressions(state) -> int:
    """Interior land cells with no strictly lower 8-neighbour."""
    surf = state.surface()[state.interior]
    lo = ndimage.minimum_filter(surf, size=3, mode="nearest")
    return int(((surf <= lo) & (surf > 0)).sum())
