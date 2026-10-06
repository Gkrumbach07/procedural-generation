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

from globe.config import WorldParams, cell_units
from globe.field import FaceField
from globe.erosion import glacial
from globe.erosion import maps as emaps
from globe.erosion import particle as pk
from globe.erosion import run as erosion_run
from globe.erosion.maps import ErosionState, apply_isostasy, apply_uplift, hold_datum, run_iteration, step, uplift_cap
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
    ``shelf``: ``tilt`` with the sea strip already filled to the deposition
    floor (nothing can ever be placed on it) and walled on its far side
    too, so no particle leaves: the runaway test for the seafloor stockpile.
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
    elif mode == "shelf":
        floor = cell_units(params.erosion, "dep_floor_m", cs)
        h = relief * (0.35 * noise2d(NE, rng) + X) + 0.5
        h = np.where(X < 12.0 / N, -floor, h)  # a shelf filled to the floor: no room for anything
        wall = ((Y < 2.0 / N) | (Y > 1 - 2.0 / N) | (X > 1 - 2.0 / N) | (X < 2.0 / N)) & (mask == 1)
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


def test_evaporation_floor_follows_the_spawn_volume():
    """A fine cell spawns rain in proportion to its area, so the spawn volume
    at R = 128 is 1/16384 of the coarse one.  The evaporation floor
    (``erosion.min_volume_frac``) is a fraction of that spawn volume, so the
    same window with its rain scaled down walks as far as before.  With the
    floor as a length (the retired ``min_volume``: 0.5 m / cell size) every
    particle of the scaled window was born below it and died of evaporation
    at once -- the 76 m zoom window of docs/zoom-windows.md eroded nothing."""
    p = WorldParams.small_world(0)
    runs = {}
    for scale in (1.0, 1.0 / 16384.0):
        st = make_window(48, "tilt", p, seed=3)
        st.precip *= np.float32(scale)
        s = step(st, p, 0)
        n = max(sum(s["deaths"].values()), 1)
        runs[scale] = (s["steps_mean"], s["deaths"]["evap"] / n)
    (steps1, evap1), (steps2, evap2) = runs[1.0], runs[1.0 / 16384.0]
    assert steps2 > 0.5 * steps1 and steps2 > 5.0, runs
    assert evap2 <= evap1 + 0.05, runs


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
    the modelled surface is exactly ``lost_offshore`` — everything else
    stays in height + sediment + pending — and ``lost_offshore`` is the
    ``offshore_writeoff`` share of what sea deaths could not place: the rest
    is parked in the death cell's ``pending``, so after every iteration the
    stockpile on submerged cells is (1 - w) / w times the write-off.  With
    ``offshore_writeoff = 1.0`` it is the old rule (nothing parked on a
    submerged cell, the whole surplus deleted), which deletes more."""
    p = WorldParams.small_world(0)
    assert 0.0 < p.erosion.offshore_writeoff < 1.0
    out = {}
    for label, pp in (("park", p), ("delete", p.with_overrides(erosion={"offshore_writeoff": 1.0}))):
        w = pp.erosion.offshore_writeoff
        st = make_window(48, "tilt", pp, deposit_on_exit=True)
        m0 = st.total_mass()
        lost = parked = 0.0
        for it in range(25):
            s = step(st, pp, it)
            lost += s["lost_offshore"]
            # no deficit leaves this window: the residue is the rounding of
            # cancelling a clamped particle's later deposits (~1e-18)
            assert s["deficit_out"] <= 1e-12 * abs(m0)
            # every sea-death surplus splits 1 - w : w between the stockpile
            # of its cell and the write-off (pending is re-injected and
            # zeroed at the start of an iteration, so this is exact)
            sea_pending = float(st.pending[st.surface() < 0].sum())
            parked += sea_pending
            assert abs(sea_pending - (1.0 - w) / w * s["lost_offshore"]) <= 1e-9 * (sea_pending + s["lost_offshore"]), (label, it)
        # window mode applies uplift as given (only the global state is made
        # mean-free, apply_uplift), so it is still a mass source here
        upl = float(st.uplift[st.interior][st.mask[st.interior] == pk.MASK_ACTIVE].sum()) * 25
        m1 = st.total_mass()
        assert abs((m1 + lost) - (m0 + upl)) <= 1e-6 * abs(m0), (label, m0, m1, lost, upl)
        assert s["deaths"]["ocean"] > 0
        out[label] = (lost, parked)
    assert out["delete"][0] > 0.0, "the old rule must lose mass offshore, or this proves nothing"
    assert out["delete"][1] == 0.0  # the old rule never parks on a submerged cell
    assert out["park"][1] > 0.0  # the new one does ...
    assert out["park"][0] < out["delete"][0], out  # ... and so deletes less


def test_seafloor_stockpile_is_bounded():
    """A shelf nothing can be placed on (``shelf`` window: the strip is
    filled to ``dep_floor_m`` below the waterline from the start and walled
    off), so every load that reaches the sea is surplus, every iteration.
    The stockpile that comes back must decay, not accumulate: each
    re-injection writes ``offshore_writeoff`` of it off, so pending on the
    seafloor is a geometric series bounded by inflow / w, where parking all
    of it would grow linearly with the iterations (the deadlock the old
    deletion rule was introduced against)."""
    p = WorldParams.small_world(0)
    w = p.erosion.offshore_writeoff
    st = make_window(48, "shelf", p, deposit_on_exit=True)
    sea0 = st.surface() < 0
    assert sea0[st.interior].any()
    m0 = st.total_mass()
    n = 40
    lost_k = np.zeros(n)
    pend_k = np.zeros(n)  # pending on submerged cells after each iteration
    for it in range(n):
        s = step(st, p, it)
        assert s["lost_offshore"] > 0.0, it  # a full shelf writes off every iteration
        lost_k[it] = s["lost_offshore"]
        pend_k[it] = float(st.pending[st.surface() < 0].sum())
        # the kept share is (1 - w) / w of the write-off, less the dust the
        # kernel writes off outright (a kept share under 1e-4 cell units)
        gap = (1.0 - w) / w * lost_k[it] - pend_k[it]
        assert -1e-9 * (pend_k[it] + lost_k[it]) <= gap <= 1e-2 * (1.0 - w) / w * lost_k[it] + 1e-9, (it, gap)
    assert np.all(st.surface()[sea0] < 0.0)  # sea never turned into land
    upl = float(st.uplift[st.interior][st.mask[st.interior] == pk.MASK_ACTIVE].sum()) * n
    m1 = st.total_mass()
    assert abs((m1 + lost_k.sum()) - (m0 + upl)) <= 1e-6 * abs(m0), (m0, m1, lost_k.sum(), upl)
    # the split gives the fresh arrivals: lost_k / w = (re-injected) P_{k-1} + I_k
    fresh = lost_k / w - np.concatenate([[0.0], pend_k[:-1]])
    assert fresh.max() > 0.0
    # geometric, not linear: never above the series ceiling, a small fraction
    # of what parking everything would have piled up, and flat over the run
    assert pend_k.max() <= (1.0 - w) / w * fresh.max() * (1.0 + 1e-9)
    assert pend_k[-1] < 0.25 * np.sum((1.0 - w) * fresh), (pend_k[-1], np.sum((1.0 - w) * fresh))
    assert pend_k[3 * n // 4:].max() < 2.0 * pend_k[n // 4: n // 2].max(), pend_k


def test_diag_hook_sees_the_change_list_and_changes_nothing():
    """``run_iteration``'s ``diag`` hook, and the change-list invariants a
    diagnostic reads it through (scripts/fork_erosion.py --deaths).

    The hook exists so a diagnostic can ask where particles stopped without
    re-tracing them; if what it is handed does not mean what it is
    documented to mean, every number built on it is wrong and nothing else
    would notice.
    """
    p = WorldParams.small_world(0)
    seen = []

    def diag(state, cl_cell, cl_vol, cl_count, cap, sp_death):
        assert cl_count.shape == sp_death.shape
        for q in range(cl_count.shape[0]):
            row = cl_vol[q * cap: q * cap + cl_count[q]]
            if row.size == 0:
                continue
            # steps come first and final deposits last, so every entry after
            # the first negative one is also negative: the death cell is the
            # first `cl_vol < 0` and there is nothing of the walk behind it
            neg = np.flatnonzero(row < 0.0)
            if neg.size:
                assert np.all(row[neg[0]:] < 0.0)
            assert np.all(row[:neg[0] if neg.size else row.size] >= 0.0)
        seen.append((int(cl_count.sum()), int((cl_vol[:cl_count.shape[0] * cap] < 0).sum())))

    st = make_window(48, "tilt", p, deposit_on_exit=True)
    ref = make_window(48, "tilt", p, deposit_on_exit=True)
    for it in range(3):
        a = run_iteration(st, p, it, diag=diag)
        b = run_iteration(ref, p, it)
        assert a["entries"] == b["entries"] and a["deaths"] == b["deaths"]
    assert seen and any(n > 0 for n, _ in seen)
    # the hook is a diagnostic: with it and without it the state is identical
    assert np.array_equal(st.height, ref.height)
    assert np.array_equal(st.sediment, ref.sediment)


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
    ``iterations`` is not part of the checkpoint identity in
    ``uplift_mode = 'stack'``: a shorter rerun resumes from the newest
    checkpoint at or below its end.  (In 'replay' it is, see
    test_uplift_replay_resume_equals_continuous_run.)"""
    p = WorldParams.tiny_world(11)
    p.erosion = dataclasses.replace(p.erosion, iterations=9, checkpoint_every=3, quicklook_every=0, uplift_mode="stack")
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


def test_uplift_cap_bounds_the_rise_and_stays_mean_free(scratch):
    """``erosion.uplift_max_m`` (maps.apply_uplift ``cap``): the tectonic
    uplift field has no ceiling -- earth-v5's fastest cell carried
    4.51 m/it for all 800 iterations, 3.6 km on a 6.7 km bedrock, and the
    highest point ended at 9862 m against a 6747 m bedrock maximum
    (docs/uplift-ceiling.md).  With the cap on, no cell rises by more than
    the cap (less the datum), the applied change is still exactly
    mass-free, cells under the cap get exactly their own uplift, cap off
    reproduces the old rule bit for bit, and a window state (its own
    synthetic field, refine's zeros) is never capped by ``step``."""
    p = WorldParams.small_world(0).with_overrides(erosion={"uplift_max_m": 2.0})  # 50 m cells: 2 m/it = 0.04 cell units (off by default since 'replay')
    grid = p.coarse_grid()
    store = _stub_world(scratch, "uplift_cap", p)
    f = {n: store.load_field(n, grid) for n in ("bedrock", "hardness", "precip", "evap", "uplift")}
    # the stub uplift is ~1e-3 m/it: scale it to ~0.3 m/it (the real
    # stage's land median is 0.07, its 2-4 km band 0.5; x1000 as
    # test_datum_is_held_through_the_run does would put the whole world at
    # 1.1-1.3 m/it, over the cap) and plant a belt core at 5 m/it,
    # sweep-od006's 8.5 m/it order of magnitude
    f["uplift"].data *= 300.0
    f["uplift"].interior[2, 20:40, 30:50] = 5.0  # metres per iteration; from_grid divides by the 50 m unit
    f["uplift"].exchange_halos()
    st = ErosionState.from_grid(grid, f["bedrock"], f["hardness"], f["precip"], f["evap"], f["uplift"], p.erosion)
    cap = cell_units(p.erosion, "uplift_max_m", st.height_unit_m)
    assert cap == pytest.approx(2.0 / 50.0) and float(p.erosion.uplift_max_m) == 2.0
    ia = st.mask[st.interior] == pk.MASK_ACTIVE
    u = st.uplift[st.interior]
    assert (u[ia] > cap).sum() >= 20 * 20  # the belt core really is above the cap
    assert (u[ia] < cap).sum() > 0.9 * ia.sum()  # and the rest of the world is not

    m0 = st.total_mass()
    h0 = st.height.copy()
    apply_uplift(st, cap)
    d = (st.height - h0)[st.interior]
    uc = np.minimum(u, cap)
    mean = float(uc[ia].mean())
    assert abs(st.total_mass() - m0) <= 1e-9 * max(abs(m0), 1.0)  # (b) mass-free
    assert d[ia].max() <= cap - mean + 1e-12  # (a) bounded
    assert np.allclose(d[ia], uc[ia] - mean, atol=1e-12)  # the capped field, less the datum
    under = ia & (u < cap)
    assert np.allclose(d[under], u[under] - mean, atol=1e-12)  # (c) under the cap: exactly its own uplift
    core = ia & (u > cap)
    assert np.allclose(d[core], cap - mean, atol=1e-12)  # the core is pinned at the cap
    # the cap bit: the same belt uncapped would have risen 2.5x as far
    assert (5.0 / 50.0) - mean > 2.0 * d[core].max()

    # (d) cap off is the old rule, bit for bit
    a = ErosionState.from_grid(grid, f["bedrock"], f["hardness"], f["precip"], f["evap"], f["uplift"], p.erosion)
    b = ErosionState.from_grid(grid, f["bedrock"], f["hardness"], f["precip"], f["evap"], f["uplift"], p.erosion)
    apply_uplift(a)
    apply_uplift(b, None)
    assert np.array_equal(a.height, b.height)
    p_off = p.with_overrides(erosion={"uplift_max_m": 0.0})
    assert uplift_cap(a, p_off.erosion) is None
    assert uplift_cap(a, p.erosion) == pytest.approx(cap)
    # and the uncapped rise at the core is the raw field less the raw mean
    da = (a.height - h0)[st.interior]
    assert np.allclose(da[core], (5.0 / 50.0) - float(u[ia].mean()), atol=1e-12)

    # (e) a window is never capped: its synthetic field (2.7-5 m/it here,
    # far above the 1 m/it ceiling) is applied as given by step()
    w = make_window(48, "dome", p, iters=10, uplift_total=8.0)
    assert uplift_cap(w, p.erosion) is None
    assert float(w.uplift[w.interior].max()) > cap
    w2 = make_window(48, "dome", p, iters=10, uplift_total=8.0)
    w2.uplift[...] = 0.0
    step(w, p, 0)
    step(w2, p, 0)
    dw = (w.surface() - w2.surface())[w.interior]
    assert dw.max() > 2.0 * cap  # the dome's uplift survived at its own rate, not the cap's


def _point_load(st, face, i, j, size=4):
    st.iso_acc = np.zeros_like(st.height)
    H = st.H
    st.iso_acc[face, H + i:H + i + size, H + j:H + j + size] = -1.0   # rock removed
    return size * size


def test_isostasy_is_regional_and_mean_free(scratch):
    """Erosional isostasy (maps.apply_isostasy): a column that lost rock
    rebounds by ``isostasy`` of it, spread by a flexural Gaussian, and the
    global mean -- a datum shift, which hold_datum owns -- is removed."""
    p = WorldParams.tiny_world(5).with_overrides(erosion={"isostasy": 0.8, "flexure_km": 0.2})
    st = erosion_run.build_state(_stub_world(scratch, "iso", p), p)
    n = _point_load(st, 0, 14, 14)
    h0, s0 = st.height.copy(), st.sediment.copy()
    info = apply_isostasy(st, p)
    dh = (st.height - h0)[st.interior]
    assert abs(dh.mean()) < 1e-9 * dh.max()                       # mean-free
    peak = np.unravel_index(np.argmax(dh), dh.shape)
    assert peak[0] == 0 and 13 <= peak[1] <= 18 and 13 <= peak[2] <= 18  # over the load
    assert int((dh > 0.01 * dh.max()).sum()) > 4 * n              # regional, not per cell
    # the positive part carries the compensated load, k * (rock removed)
    assert abs((dh - dh.min()).sum() - 0.8 * n) < 0.15 * 0.8 * n
    assert np.array_equal(st.sediment, s0)                         # rebound is rock, not cover
    assert not st.iso_acc.any()                                    # the accumulator is spent
    assert info["diffusion_steps"] > 0


def test_isostasy_has_no_seam(scratch):
    """The flexural smoothing uses the halo-exchanged Laplace-Beltrami
    operator, so a load at a face edge spreads across it smoothly."""
    p = WorldParams.tiny_world(5).with_overrides(erosion={"isostasy": 0.8, "flexure_km": 0.2})
    st = erosion_run.build_state(_stub_world(scratch, "iso_seam", p), p)
    _point_load(st, 0, 0, 12)                                      # touching face 0's i = 0 edge
    h0 = st.height.copy()
    apply_isostasy(st, p)
    f = FaceField(st.grid, (st.height - h0).astype(np.float32))
    f.exchange_halos()
    assert seam_discontinuity(f) < 4.0


def test_isostasy_smoother_is_stable_on_white_noise(scratch):
    """The regression that let the first Earth-scale isostasy run blow up to
    a 1.7e16 m peak: an explicit step that is safe on a flat grid is not on
    the cube-sphere.  A smooth point load barely excites the unstable mode,
    so it has to be white noise -- and a monotone smoother never exceeds
    the input's extreme (discrete maximum principle)."""
    p = WorldParams.tiny_world(5).with_overrides(erosion={"isostasy": 0.8, "flexure_km": 0.2})
    st = erosion_run.build_state(_stub_world(scratch, "iso_noise", p), p)
    rng = np.random.default_rng(3)
    st.iso_acc = np.zeros_like(st.height)
    st.iso_acc[st.interior] = rng.standard_normal(st.iso_acc[st.interior].shape)
    peak = float(np.abs(st.iso_acc[st.interior]).max())
    h0 = st.height.copy()
    info = apply_isostasy(st, p)
    dh = (st.height - h0)[st.interior]
    assert np.isfinite(dh).all()
    assert info["diffusion_steps"] >= 20
    assert float(np.abs(dh).max()) <= 0.8 * peak * 1.05


def test_isostasy_is_on_by_default_and_zero_turns_it_off(scratch):
    """On by default (docs/streaks-and-flats.md, "What ships"); 0 must still
    switch it off completely, with no accumulator kept."""
    assert WorldParams().erosion.isostasy == 0.8
    p = WorldParams.tiny_world(5).with_overrides(erosion={"isostasy": 0.0})
    st = erosion_run.build_state(_stub_world(scratch, "iso_off", p), p)
    step(st, p, 0)
    assert getattr(st, "iso_acc", None) is None


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
    p.erosion = dataclasses.replace(p.erosion, iterations=20, checkpoint_every=0, quicklook_every=0, resume=False, uplift_max_m=0.0)  # about the datum, not the ceiling: uncapped so the x1000 stub uplift is the forcing it documents
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


def test_shelf_mode_holds_the_bedrock_land_fraction(scratch):
    """In shelf mode (``tectonics.shelf_fraction > 0``) the land area is an
    output of tectonics, so erosion and hydro hold the bedrock's own land
    fraction (``maps.datum_land_fraction``) instead of forcing
    ``world.land_fraction`` onto it: holding 30 % lifted seed 0's 24 %-land
    Earth by ~550 m (docs/ocean-depth.md).  The stub bedrock is lowered to
    18 % land, far enough from 30 % that either rule is unmistakable."""
    p = WorldParams.tiny_world(3).with_overrides(tectonics={"shelf_fraction": 0.275})
    p.erosion = dataclasses.replace(p.erosion, iterations=10, checkpoint_every=0, quicklook_every=0, resume=False)
    store = _stub_world(scratch, "erosion_shelf_datum", p)
    grid = p.coarse_grid()
    bed = store.load_field("bedrock", grid)
    bed.data -= np.float32(np.quantile(bed.interior, 1.0 - 0.18))
    store.save_field(bed)
    target = float(np.mean(bed.interior >= 0))
    assert abs(target - p.world.land_fraction) > 0.1
    assert emaps.datum_land_fraction(p, bed.interior) == target
    assert emaps.datum_land_fraction(WorldParams.tiny_world(3), bed.interior) == p.world.land_fraction

    # the replay start holds it too, not only the loop from iteration 1
    st0 = erosion_run.build_state(store, p)
    assert abs(float(np.mean(st0.surface()[st0.interior] >= 0)) - target) < 2.0 / bed.interior.size
    # a shelf-mode state the driver did not build is refused, not held at 30 %
    st0.land_target = None
    with pytest.raises(ValueError, match="shelf mode"):
        emaps.held_land_fraction(st0, p)

    info = erosion_run.run(store, p, _log)
    assert abs(info["land_fraction"] - target) < 0.02, (info["land_fraction"], target)
    hinfo = hydro_run.run(store, p, _log)
    assert hinfo["land_fraction_target"] == target
    assert abs(hinfo["height_shift_m"]) < 5.0, hinfo["height_shift_m"]


def _uplift_world(scratch, name: str, p: WorldParams) -> WorldStore:
    """A stub world whose uplift is the size of the real stage's: the stub
    field x1000 (~1 m/it, net positive) and a 6 x 6 belt core at 5 m/it, so
    a 2 m/it cap binds on it and a double count shows."""
    store = _stub_world(scratch, name, p)
    grid = p.coarse_grid()
    upl = store.load_field("uplift", grid)
    upl.data *= 1000.0
    upl.interior[2, 10:16, 10:16] = 5.0
    upl.exchange_halos()
    store.save_field(upl)
    return store


def _no_surface_processes(monkeypatch):
    """No particles, no mass wasting: what is left of ``maps.step`` is the
    base/route/lake refreshes, uplift, isostasy (on nothing) and the datum."""
    monkeypatch.setattr(emaps, "run_iteration", lambda state, params, key, **kw: {"particles": 0, "steps_mean": 0.0, "deaths": {}})
    monkeypatch.setattr(emaps, "thermal_erosion", lambda state, params: None)


def test_uplift_replay_without_erosion_ends_at_bedrock(scratch, monkeypatch):
    """``erosion.uplift_mode = 'replay'`` (docs/uplift-replay.md): the stage
    starts from ``bedrock`` less the total the run will apply and replays it,
    so with every surface process off the surface ends at ``bedrock`` --
    cell for cell, up to one global constant (the datum hold) -- with the
    cap off and with it binding.  'stack' ends at ``bedrock`` plus the
    applied uplift: the last ``uplift_window`` tectonic steps counted twice."""
    n = 12
    p = WorldParams.tiny_world(3).with_overrides(erosion={
        "iterations": n, "glacial_every": 0, "checkpoint_every": 0, "quicklook_every": 0, "resume": False})
    assert p.erosion.uplift_mode == "replay" and p.erosion.isostasy > 0.0  # the defaults; isostasy stays on
    store = _uplift_world(scratch, "replay_no_erosion", p)
    _no_surface_processes(monkeypatch)
    grid = p.coarse_grid()
    for cap_m in (0.0, 2.0):
        q = p.with_overrides(erosion={"uplift_max_m": cap_m})
        st = erosion_run.build_state(store, q)
        bed = store.load_field("bedrock", grid).interior.astype(np.float64) / st.height_unit_m
        inter = st.interior
        u, mean = emaps._applied_uplift(st, uplift_cap(st, q.erosion))
        applied = n * (u[inter] - mean)
        if cap_m:
            assert (st.uplift[inter] > uplift_cap(st, q.erosion)).sum() == 36  # the cap binds on the core
        # the start is the reference-step crust, held at the land fraction
        d0 = st.height[inter] - (bed - applied)
        assert np.ptp(d0) < 1e-9
        assert abs(np.mean(st.surface()[inter] >= 0) - p.world.land_fraction) < 2.0 / d0.size
        assert applied.max() > 0.2  # a real replay: the core starts > 10 m below its bedrock (48 m uncapped, 12 m capped)
        m0 = st.total_mass()
        for it in range(n):
            step(st, q, it)
        d = st.height[inter] - bed
        assert np.ptp(d) < 1e-9, np.ptp(d)  # bedrock + one constant
        assert not st.sediment.any() and not st.pending.any()
        # the replayed total is mass-free: the only mass change is the datum's
        assert abs((st.total_mass() - m0) - d.size * (float(d.mean()) - float(d0.mean()))) < 1e-6

        s = q.with_overrides(erosion={"uplift_mode": "stack"})
        ss = erosion_run.build_state(store, s)
        assert np.array_equal(ss.height[inter], bed)  # stack starts at bedrock itself
        for it in range(n):
            step(ss, s, it)
        ds = ss.height[inter] - bed
        assert np.ptp(ds - applied) < 1e-9  # ... and ends at bedrock + the applied uplift
        assert np.ptp(ds) > 0.2


def test_uplift_stack_mode_is_the_old_rule_bit_for_bit(scratch, monkeypatch):
    """``uplift_mode = 'stack'`` is the behaviour before the replay existed
    (af58d28): the start state is ``ErosionState.from_grid`` on ``bedrock``
    untouched, ``start_replay`` is a no-op, and steps driven with the old
    ``apply_uplift`` body (inlined below) produce the same bytes in every
    evolving array, with the 2 m/it cap binding.  Its checkpoint identity
    still leaves ``iterations`` out."""
    p = WorldParams.tiny_world(5).with_overrides(erosion={"uplift_mode": "stack", "uplift_max_m": 2.0})
    store = _uplift_world(scratch, "stack_bit_identical", p)
    grid = p.coarse_grid()
    a = erosion_run.build_state(store, p)
    f = {k: store.load_field(k, grid) for k in ("bedrock", "hardness", "precip", "evap", "uplift")}
    b = ErosionState.from_grid(grid, f["bedrock"], f["hardness"], f["precip"], f["evap"], f["uplift"], p.erosion)
    assert a.height.tobytes() == b.height.tobytes()
    assert emaps.start_replay(a, p) is None and emaps.uplift_replay(a, p.erosion) is None
    assert a.height.tobytes() == b.height.tobytes()

    def old_apply_uplift(state, cap=None):  # erosion/maps.py at af58d28, verbatim
        act = state.mask == pk.MASK_ACTIVE
        u = state.uplift if cap is None else np.minimum(state.uplift, float(cap))
        mean = 0.0
        if state.spherical:
            iact = state.mask[state.interior] == pk.MASK_ACTIVE
            mean = float(u[state.interior][iact].mean()) if iact.any() else 0.0
        state.height[act] += u[act] - mean

    for it in range(4):
        step(a, p, it)
    monkeypatch.setattr(emaps, "apply_uplift", old_apply_uplift)
    for it in range(4):
        step(b, p, it)
    for k in ("height", "sediment", "discharge", "momentum", "pending", "base", "route"):
        assert getattr(a, k).tobytes() == getattr(b, k).tobytes(), k
    p6 = p.with_overrides(erosion={"iterations": 6})
    assert erosion_run._ckpt_hash(p6, store) == erosion_run._ckpt_hash(p, store)


def test_uplift_replay_resume_equals_continuous_run(scratch):
    """The replayed start is built once: a run resumed from its iteration-5
    checkpoint writes the same bytes as the continuous run (a second
    subtraction would put the whole run ``iterations x uplift`` low).  And a
    replay checkpoint belongs to its ``iterations``: a 6-iteration run of the
    same world must not resume from it, and stack and replay never share
    checkpoints."""
    p = WorldParams.tiny_world(7)
    p.erosion = dataclasses.replace(p.erosion, iterations=10, checkpoint_every=5, quicklook_every=0, uplift_mode="replay")
    store = _uplift_world(scratch, "replay_resume", p)
    info = erosion_run.run(store, p, _log)
    assert info["uplift_mode"] == "replay"
    assert info["replay_lowered_max_m"] > 30.0  # the 5 m/it core, less the ~1 m/it mean, over 10 iterations
    assert abs(info["land_fraction"] - p.world.land_fraction) < 0.02
    ref = {k: store.load_field(k, p.coarse_grid()).data.copy() for k in erosion_run.OUTPUTS}
    for suf in (".npz", ".json"):
        (store.checkpoint_dir / f"erosion_iter0010{suf}").unlink()
    store.clear_outputs(erosion_run.OUTPUTS)
    msgs = []
    erosion_run.run(store, p, msgs.append)
    assert any("resumed" in m and "iteration 5" in m for m in msgs), msgs
    for k in erosion_run.OUTPUTS:
        assert store.load_field(k, p.coarse_grid()).data.tobytes() == ref[k].tobytes(), k
    p6 = p.with_overrides(erosion={"iterations": 6})
    assert erosion_run._ckpt_hash(p6, store) != erosion_run._ckpt_hash(p, store)
    assert erosion_run.find_checkpoint(store, p6, max_iteration=6) is None
    ps = p.with_overrides(erosion={"uplift_mode": "stack"})
    assert erosion_run._ckpt_hash(ps, store) != erosion_run._ckpt_hash(p, store)
    assert erosion_run.find_checkpoint(store, ps) is None
    with pytest.raises(ValueError, match="uplift_mode"):
        p.with_overrides(erosion={"uplift_mode": "twice"})


def test_uplift_replay_is_a_no_op_on_a_window():
    """A refinement window (``spherical=False``) owns no planetary datum and
    starts from the coarse result: replay never touches it, whatever its
    uplift field."""
    p = WorldParams.small_world(0)
    assert p.erosion.uplift_mode == "replay"
    w = make_window(48, "dome", p, iters=10, uplift_total=8.0)
    h0 = w.height.copy()
    assert emaps.uplift_replay(w, p.erosion) is None
    assert emaps.start_replay(w, p) is None
    assert w.height.tobytes() == h0.tobytes()


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


def test_sticky_ice_keeps_a_cell_carved_below_sea_level_glaciated(scratch):
    """With erosion.glacial_sticky, a cold cell that was ice stays ice after
    its bed drops below sea level; without it, it leaves the mask (and the
    margin migrates).  Off, no state is kept at all."""
    for sticky in (False, True):
        p = WorldParams.tiny_world(2).with_overrides(erosion={"glacial_sticky": sticky})
        st = erosion_run.build_state(_stub_world(scratch, f"sticky{int(sticky)}", p), p)
        st.evap[...] = 0.0                                   # everything cold
        glacial.carve(st, p)
        if not sticky:
            assert getattr(st, "ice_prev", None) is None
            continue
        was = st.ice_prev.copy()
        cell = tuple(np.argwhere(was[:, st.H:-st.H, st.H:-st.H])[0] + np.array([0, st.H, st.H]))
        st.height[cell] = -0.05 - st.sediment[cell]         # carved below the sea
        glacial.carve(st, p)
        assert st.ice_prev[cell]                              # still glaciated


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
    p = WorldParams.small_world(0).with_overrides(erosion={"glacial_rate": 50.0, "glacial_every": 10})
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


def test_glacial_lengths_are_metres_at_every_cell_size():
    """`glacial_rate` and `glacial_max` are lengths in LENGTH_PARAMS_M: the
    same 20 m cap takes 20 m of bed at 50 m cells and at 100 m cells.  In
    cell units the cap was 20 m at the 50 m cells it was tuned on and 3909 m
    at the earth preset's 9773 m (docs/missing-relief.md).  At 50 m cells it
    is still exactly the old 0.4 cell units, so the small presets are
    unchanged."""
    carved = {}
    for cs in (50.0, 100.0):
        p = WorldParams.small_world(0).with_overrides(world={"cell_size_m": cs}, erosion={"glacial_max": 20.0})
        st = make_window(64, "dome", p, iters=40)
        st.evap[...] = 1.0
        st.evap[st.surface() > np.percentile(st.surface()[st.interior], 70)] = 0.0
        st.discharge[...] = 1e4  # an ice flux far past the cap wherever the taper is non-zero
        stats = glacial.carve(st, p)
        assert st.height_unit_m == cs
        carved[cs] = stats["max_carve"]
    assert carved[50.0] == 0.4
    assert carved[50.0] * 50.0 == pytest.approx(20.0) and carved[100.0] * 100.0 == pytest.approx(20.0)


def _closed_depressions(state) -> int:
    """Interior land cells with no strictly lower 8-neighbour."""
    surf = state.surface()[state.interior]
    lo = ndimage.minimum_filter(surf, size=3, mode="nearest")
    return int(((surf <= lo) & (surf > 0)).sum())


# --------------------------------------------------------------------------
# the sea is what the sea is connected to (maps.refresh_base)
# --------------------------------------------------------------------------
def _dig_basin(st, face, i0, j0, size, floor):
    """Punch a closed square depression of ``floor`` (cell units, negative)
    into an otherwise unbroken piece of land, and return its flat mask."""
    H = st.H
    sl = (face, slice(H + i0, H + i0 + size), slice(H + j0, H + j0 + size))
    st.height[sl] = floor
    st.sediment[sl] = 0.0
    st.exchange_halos()
    m = np.zeros_like(st.height, dtype=bool)
    m[sl] = True
    return m[st.interior].reshape(-1)


def _dry_site(st, size, pad=2):
    """A ``(size + 2*pad)`` square of dry land to dig a basin in, as
    ``(face, i0, j0)`` for the basin's own corner."""
    land = st.surface()[st.interior] >= 0.0
    w = size + 2 * pad
    for f in range(land.shape[0]):
        ok = land[f]
        for i in range(0, land.shape[1] - w):
            for j in range(0, land.shape[2] - w):
                if ok[i:i + w, j:j + w].all():
                    return f, i + pad, j + pad
    raise AssertionError("no dry site in this world")


def _land_world(scratch, name, seed=0):
    """A world whose surface is land everywhere except one real ocean, so a
    dug basin is unambiguously landlocked."""
    p = WorldParams.small_world(seed).with_overrides(hydro={"ocean_min_fraction": 0.02})
    st = erosion_run.build_state(_stub_world(scratch, name, p), p)
    return p, st


def test_refresh_base_separates_a_closed_basin_from_the_sea(scratch):
    """A depression below sea level with a rim of land around it is *not*
    sea: `refresh_base` gives it its own floor as a base level, while the
    open ocean keeps 0.  This is the classification `hydro.open_ocean`
    already made and erosion did not (docs/plates-rifts-and-water.md)."""
    p, st = _land_world(scratch, "base_basin")
    surf = st.surface()[st.interior].reshape(-1)
    assert (surf < 0).any() and (surf >= 0).any(), "the stub world needs both sea and land"
    # a basin dug in the middle of the largest contiguous piece of land
    face, i0, j0 = _dry_site(st, 6)
    floor = -4.0
    basin = _dig_basin(st, face, i0, j0, 6, floor)

    info = st.refresh_base(p.hydro.ocean_min_fraction)
    base = st.base[st.interior].reshape(-1)
    now = st.surface()[st.interior].reshape(-1)
    assert info["closed_basins"] >= 1
    assert np.allclose(base[basin], floor)              # its own floor, not sea level
    # and so the basin is *land* to the kernel: surface >= base everywhere in it
    assert (now[basin] >= base[basin]).all()
    # only a cell under the waterline ever gets an internal base level, and
    # erosion's classification is hydro's: the open ocean keeps sea level
    ocean = hydro_run.open_ocean(now.reshape(6, st.N, st.N).astype(np.float32),
                                 st.grid, p.hydro.ocean_min_fraction).reshape(-1)
    assert (now[base != 0.0] < 0.0).all()
    assert np.allclose(base[ocean], 0.0)
    assert not ocean[basin].any()                       # the dug basin is not sea
    assert int(info["sea_cells"]) == int(ocean.sum())
    assert int(info["closed_cells"]) == int(((now < 0.0) & ~ocean).sum())


def test_sea_mask_off_is_the_old_sign_rule(scratch):
    """``sea_mask_every = 0`` leaves ``base`` at zero, which makes every
    ``surface < base`` test in the kernel exactly the old ``surface < 0``:
    the switch is a true no-op, so any difference between the two arms of a
    measurement is the mask and nothing else."""
    p, st = _land_world(scratch, "base_off")
    # lake_balance writes a drawn-down lake's level into `base` too, so it
    # is switched off here as well: the test is about the sea mask alone
    p_off = p.with_overrides(erosion={"sea_mask_every": 0, "lake_balance": False})
    for it in range(3):
        step(st, p_off, it)
    assert np.all(st.base == 0.0)
    assert getattr(st, "base_at", None) is None


def test_closed_basin_erodes_as_land_and_keeps_its_sediment(scratch):
    """With the mask on, a landlocked basin below sea level takes sediment
    and none of it is written off as ``lost_offshore``; with the mask off it
    is a marine sink and the load that will not fit leaves the model."""
    out = {}
    for label, every in (("on", 1), ("off", 0)):
        p, st = _land_world(scratch, f"base_sink_{label}")
        pp = p.with_overrides(erosion={"sea_mask_every": every, "glacial_every": 0, "isostasy": 0.0})
        face, i0, j0 = _dry_site(st, 6)
        basin = _dig_basin(st, face, i0, j0, 6, -4.0)
        lost = 0.0
        for it in range(6):
            lost += float(step(st, pp, it).get("lost_offshore", 0.0))
        sed = st.sediment[st.interior].reshape(-1)
        out[label] = (lost, float(sed[basin].sum()))
    assert out["off"][0] > 0.0, "the old rule must lose mass offshore, or this proves nothing"
    assert out["on"][0] < out["off"][0]
    assert out["on"][1] > out["off"][1], out   # the basin keeps what it is given


def test_window_lakes_flag_the_depressions_the_route_fills():
    """``erosion.window_lakes``: a window's depressions deeper than
    ``hydro.lake_min_depth`` on the routing surface become flagged lakes
    (``S_RFLAG == 2`` in the packed samples), refreshed on the route's
    stride, so particles cross them instead of dying against the dams a
    window's particle pass builds (docs/zoom-windows.md).  Off, a window
    has no lake flags at all -- the refine stage's behaviour."""
    p = WorldParams.small_world(0)
    on = p.with_overrides(erosion={"window_lakes": True, "flood_every": 1})
    runs = {}
    for name, params in (("off", p), ("on", on)):
        st = make_window(32, "tilt", params, seed=5)
        H = st.H
        c = H + 16
        pit = (slice(None), slice(c - 2, c + 2), slice(c - 2, c + 2))
        st.height[pit] -= 2.0                      # a 4 x 4 pit, 2 cells deep, on the slope
        step(st, params, 0)
        runs[name] = st
    off, on_st = runs["off"], runs["on"]
    assert off.lake_flag is None or not off.lake_flag.any()
    flag = on_st.lake_flag
    assert flag is not None and flag[0, c - 1:c + 1, c - 1:c + 1].all(), flag[0, c - 3:c + 3, c - 3:c + 3]
    assert flag.sum() < 200                         # the pit and what the flood joins to it, not the slope
    assert np.all(on_st.samp[0][flag[0] > 0, pk.S_RFLAG] == 2.0)


def test_hold_datum_carries_the_base_level_with_it(scratch):
    """``hold_datum`` is a rigid shift of the datum, not a physical change,
    so everything registered to the terrain moves with it.  Sea level is 0
    in the new datum as in the old, so the open ocean's base stays 0; a
    closed basin's base is the elevation of its floor, and the floor just
    moved.  Left behind, it drifts against the terrain by the accumulated
    hold between refreshes."""
    p, st = _land_world(scratch, "base_datum")
    face, i0, j0 = _dry_site(st, 6)
    basin = _dig_basin(st, face, i0, j0, 6, -4.0)
    st.height += 1.5                  # push the planet up, so the hold has work to do
    st.refresh_base(p.hydro.ocean_min_fraction)
    b0 = st.base[st.interior].reshape(-1).copy()
    assert np.allclose(b0[basin], -2.5)          # the dug floor in the lifted datum

    q = hold_datum(st, p.world.land_fraction)
    assert q > 1.0, q    # ~1.5; the order statistic moved a little when the basin was dug
    b1 = st.base[st.interior].reshape(-1)
    assert np.allclose(b1[basin], b0[basin] - q)   # the basin floor moved with the datum
    assert np.allclose(b1[b0 == 0.0], 0.0)         # sea level did not
    # and the basin is still registered to the terrain, which is the point
    surf = st.surface()[st.interior].reshape(-1)
    assert abs(float(surf[basin].min()) - float(b1[basin].min())) < 1e-9


def test_basin_fill_turns_a_dug_basin_into_a_graded_plain(scratch):
    """`fill_basins` (erosion.basin_fill_grade) lays sediment into a closed basin up to a
    plain rising from its outlet: afterwards the flood finds no basin there, only sediment
    was added (the bed is untouched), and the plain stands between the old floor and a
    little above the old rim.  Off (0, the toy presets' value) it changes nothing."""
    from globe.erosion.maps import fill_basins
    from globe.hydro.priority_flood import priority_flood_sphere
    from globe.hydro.run import open_ocean

    p, st = _land_world(scratch, "basin_fill")
    face, i0, j0 = _dry_site(st, 6)
    H = st.H
    around = st.surface()[face, H + i0 - 2:H + i0 + 8, H + j0 - 2:H + j0 + 8]
    floor = 0.3 * float(around.min())
    basin = _dig_basin(st, face, i0, j0, 5, floor)
    inter = st.interior
    h0, s0 = st.height[inter].copy(), st.sediment[inter].copy()
    assert p.erosion.basin_fill_grade == 0.0 and fill_basins(st, p) is None       # off on the toy presets
    assert np.array_equal(st.sediment[inter], s0)

    def depth_in_basin():
        surf = np.ascontiguousarray(st.surface()[inter], dtype=np.float32)
        oc = np.asarray(open_ocean(surf, st.grid, p.hydro.ocean_min_fraction), bool).reshape(surf.shape)
        fl = priority_flood_sphere(surf, oc, st.grid)
        return (fl.filled - surf).reshape(-1)[basin] * st.height_unit_m, fl.filled.reshape(-1)[basin]

    d0, rim = depth_in_basin()
    assert d0.max() > 5.0                                    # metres of closed basin before
    p.erosion.basin_fill_grade = 0.1
    info = fill_basins(st, p)
    assert info["fill_km3"] > 0.0 and info["basin_cells"] >= int((d0 > p.erosion.basin_fill_min_m).sum())
    assert np.array_equal(st.height[inter], h0)              # the bed is not touched
    add = (st.sediment[inter] - s0).reshape(-1)
    assert add.min() >= 0.0 and add[basin].max() > 0.0
    d1, _ = depth_in_basin()
    assert d1.max() <= p.erosion.basin_fill_min_m + 1e-3     # no basin left to hold a lake
    top = st.surface()[inter].reshape(-1)[basin]
    # at most the grade times the basin's width above its old rim
    assert np.all(top <= rim + 0.1 * (st.grid.cell_size_m / 1000.0) * 40 / st.height_unit_m + 1e-6)


def test_lake_balance_makes_a_dug_basin_water_the_kernel_respects(scratch):
    """`refresh_lakes` finds a basin dug above sea level, flags it (an
    overflowing lake, `S_RFLAG == 2`) or sinks its level into `base` (a
    closed one), and from then on the kernel never erodes its bed below
    the water level."""
    p, st = _land_world(scratch, "lake_basin")
    face, i0, j0 = _dry_site(st, 6)
    H = st.H
    around = st.surface()[face, H + i0 - 2:H + i0 + 8, H + j0 - 2:H + j0 + 8]
    floor = 0.3 * float(around.min())   # well below the land around it, above sea level
    assert floor > 0.0
    basin = _dig_basin(st, face, i0, j0, 5, floor)
    st.refresh_base(p.hydro.ocean_min_fraction)
    st.refresh_route(cell_units(p.erosion, "route_eps", st.height_unit_m))
    info = st.refresh_lakes(p.hydro.lake_evap, p.hydro.lake_min_depth / st.height_unit_m, p.hydro.ocean_min_fraction)
    assert info["lake_cells"] >= int(basin.sum()) // 2
    flag = st.lake_flag[st.interior].reshape(-1)
    base = st.base[st.interior].reshape(-1)
    water = (flag[basin] > 0) | (base[basin] > floor + 1e-6)
    assert water.mean() > 0.8, "the dug basin should be under water either way"
    before = st.surface()[st.interior].reshape(-1)[basin].copy()
    for it in range(4):
        step(st, p, it)
    after = st.surface()[st.interior].reshape(-1)[basin]
    # the bed may fill (delta, settling) but is never cut below its floor
    assert (after >= before - 1e-6).all()


def test_erosions_lake_balance_runs_on_the_evaporation_that_does_not_stop_at_freezing(scratch):
    """``hydro.pet_t0``: erosion's own lake balance takes ``lake_pet`` where
    the state has one (``erosion.run.build_state``), not the kernel's
    ``evap``.  In a frozen world ``evap`` is zero and every basin stands full;
    with the law they settle or dry as their rain allows."""
    p, st = _land_world(scratch, "lake_pet")
    face, i0, j0 = _dry_site(st, 6)
    H = st.H
    around = st.surface()[face, H + i0 - 2:H + i0 + 8, H + j0 - 2:H + j0 + 8]
    _dig_basin(st, face, i0, j0, 5, 0.3 * float(around.min()))
    st.evap[...] = 0.0
    min_depth = p.hydro.lake_min_depth / st.height_unit_m

    def lakes():
        st.refresh_base(p.hydro.ocean_min_fraction)
        st.refresh_route(cell_units(p.erosion, "route_eps", st.height_unit_m))
        return st.refresh_lakes(64.0, min_depth, p.hydro.ocean_min_fraction)

    frozen = lakes()
    assert frozen["lake_cells"] > 0 and frozen["closed"] == 0 and frozen["dry"] == 0
    st.lake_pet = np.full(st.evap.shape, 1.0, dtype=np.float32)
    thawed = lakes()
    assert thawed["closed"] + thawed["dry"] > 0 and thawed["lake_cells"] < frozen["lake_cells"]
    assert (st.evap == 0.0).all()                                 # the kernel's field is not the balance's


def test_the_cold_follows_the_ground(scratch):
    """``erosion.climate_at_surface`` (``ErosionState.temperature`` /
    ``cold`` / ``balance_evaporation``): the ice line and the lakes'
    evaporation read the climate's temperature moved by the lapse rate to
    the surface as it stands.  Ground taken down into the warmth stops being
    ice, ground built up into the cold starts, and without the switch the
    climate's own field rules whatever the surface does."""
    from globe.erosion import glacial
    from globe.hydro.balance import insolation
    p, st = _land_world(scratch, "cold_ground")
    u = st.height_unit_m
    z0 = (st.surface() * u).astype(np.float32)
    assert st.temperature() is None and np.array_equal(st.cold(0.0), st.evap <= 0.0)      # no temperature: `evap` as before
    top = float(np.percentile(z0[st.interior][z0[st.interior] > 0.0], 60))
    assert top > 260.0
    st.temp0 = np.where(z0 > top, -1.0, 4.0).astype(np.float32)                           # the high ground just under freezing
    st.temp_z, st.temp_lapse, st.temp_k = z0.copy(), 6.5, 1.0 / 28.0
    st.evap[...] = (st.temp_k * np.maximum(st.temp0, 0.0)).astype(st.evap.dtype)
    high = st.temp0 < 0.0
    assert np.array_equal(st.cold(0.0), high)                                              # the climate's own field...
    st.height[high] -= 250.0 / u                                                           # ...which a 250 m cut does not move
    assert np.array_equal(st.cold(0.0), high)
    st.temp_follow = True
    assert not st.cold(0.0)[high].any()                                                    # 250 m down is 1.6 C warmer: no ice left
    land = (st.mask == pk.MASK_ACTIVE) & (st.surface() > 0.0)
    assert not (glacial.ice_mask(st, 0.0) & high).any()
    st.sediment[~high & land] += 800.0 / u                                                 # the low ground built 800 m up: 5.2 C colder
    assert st.cold(0.0)[~high & land].all() and glacial.ice_mask(st, 0.0)[~high & land].all()
    # the lakes' evaporation reads the same temperature
    st.pet_law = (28.0, 17.8, insolation(st.grid.latitude()).astype(np.float32))
    warm = st.balance_evaporation()
    st.temp_follow = False
    assert (st.balance_evaporation()[high] < warm[high]).all()                             # the cut ground evaporates more than the climate's field says
    # ice needs snow (erosion.ice_aridity): cold ground is ice where its rain is not all the year can take
    st.temp_follow = True
    cold = st.cold(0.0)
    rain = np.where(np.arange(st.evap.shape[2])[None, None, :] % 2 == 0, 5.0, 0.01) * np.ones(st.evap.shape)
    st.ice_snow = (1.5, 3.0, rain)
    assert np.array_equal(st.cold(0.0), cold & (rain > 1.0)) and (st.cold(0.0) & cold).any()


def test_a_lake_keeps_its_rivers_sediment_and_fills_towards_a_plain(scratch):
    """`erosion.lake_fill`: in a lake with room the load its shore cannot take
    is parked on the lake (`lake_load`), never more than the lake's room, and
    the lake's refresh lays it towards a plain that rises from the outlet --
    where the least is needed first.  A full lake and a frozen one take
    nothing; a load parked where the lake has gone goes on as `pending`."""
    from globe.erosion import maps as emaps
    p, st = _land_world(scratch, "lake_fill")
    face, i0, j0 = _dry_site(st, 6)
    H = st.H
    around = st.surface()[face, H + i0 - 2:H + i0 + 8, H + j0 - 2:H + j0 + 8]
    floor = 0.3 * float(around.min())
    basin = _dig_basin(st, face, i0, j0, 5, floor)
    inter = st.interior

    def put(arr, flat_cells, values):               # interior flat ids -> the extended array
        f, i, j = np.unravel_index(flat_cells, (6, st.N, st.N))
        arr[f, i + H, j + H] += values

    p = p.with_overrides(hydro={"lake_evap": 0.0}, erosion={"lake_fill": True, "basin_fill_grade": 0.1})   # every depression overflows
    st.evap[...] = 1.0                                                                                # ... and none is frozen
    fill = emaps.lake_fill_args(st, p)
    assert fill is not None and fill[0] > 0.0
    min_depth = p.hydro.lake_min_depth / st.height_unit_m

    def refresh():
        st.refresh_base(p.hydro.ocean_min_fraction)
        st.refresh_route(cell_units(p.erosion, "route_eps", st.height_unit_m))
        return st.refresh_lakes(p.hydro.lake_evap, min_depth, p.hydro.ocean_min_fraction, fill)

    info = refresh()
    flag = st.lake_flag[inter].reshape(-1)
    ids = st.lake_id[inter].reshape(-1)
    assert (flag[basin] == 2).mean() > 0.8 and info["load_lakes"] >= 1 and info["load_room"] > 0.0
    mine = int(np.bincount(ids[basin][ids[basin] >= 0]).argmax())             # the dug basin's lake
    cells = np.flatnonzero(ids == mine)
    room0 = float(st.lake_room[mine])
    assert room0 > 0.0 and (flag[ids >= 0] == 2).all() and (ids[flag != 2] == -1).all()
    # the kernel parks on lakes that take load only, and never more than a lake's room
    for it in range(3):
        run_iteration(st, p, it)
    parked = st.lake_load[inter].reshape(-1)
    assert parked.sum() > 0.0 and parked.min() >= 0.0 and not (parked[ids < 0] > 0).any()
    took = np.bincount(ids[ids >= 0], weights=parked[ids >= 0], minlength=st.lake_room.size)
    assert (st.lake_room >= -1e-12).all() and took[mine] <= room0 + 1e-9
    # lay a known load: a quarter of the lake's room
    st.lake_load[...] = 0.0
    surf = st.surface()[inter].reshape(-1).copy()
    put(st.lake_load, cells[0], 0.25 * room0)
    off = int(np.flatnonzero((flag == 0) & (st.mask[inter].reshape(-1) == pk.MASK_ACTIVE))[0])
    put(st.lake_load, off, 3.0)                                    # a load where there is no lake
    pend0 = float(st.pending[inter].sum())
    info = refresh()
    rise = st.surface()[inter].reshape(-1) - surf
    assert info["load_laid"] == pytest.approx(0.25 * room0, rel=1e-3) and info["load_stray"] == pytest.approx(3.0)
    assert float(st.lake_load.sum()) == 0.0 and float(st.pending[inter].sum()) == pytest.approx(pend0 + 3.0)
    assert rise[cells].sum() == pytest.approx(0.25 * room0, rel=1e-3) and (rise[cells] >= 0).all()
    assert 0 < (rise[cells] > 0).sum() < cells.size                # part of the lake, not all of it
    # a frozen lake takes nothing
    st.evap[...] = 0.0
    info = refresh()
    assert info["load_lakes"] == 0 and (st.lake_flag[inter].reshape(-1)[basin] <= 1).all() and (st.lake_id < 0).all()
    st.evap[...] = 1.0
    # filled to its plain the lake is ground that drains: nothing of it is water any more
    info = refresh()
    mine = int(np.bincount(st.lake_id[inter].reshape(-1)[basin][st.lake_id[inter].reshape(-1)[basin] >= 0]).argmax())
    put(st.lake_load, np.flatnonzero(st.lake_id[inter].reshape(-1) == mine)[0], float(st.lake_room[mine]))
    info = refresh()
    assert (st.lake_flag[inter].reshape(-1)[basin] == 0).mean() > 0.8


def test_offshore_writeoff_is_range_checked():
    import pytest
    from globe.config import WorldParams
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            WorldParams.tiny_world().with_overrides(erosion={"offshore_writeoff": bad}).validate()
    WorldParams.tiny_world().with_overrides(erosion={"offshore_writeoff": 1.0}).validate()


def test_slope_erode_cap_is_soillibs_downhill_limit():
    """``particle.slope_erode_cap``: soillib's pit-free limit, ``coef x
    sqrt(2) x`` the Godunov downhill slope.  The bottom of a pit gets 0 (it
    can never be deepened), a cell loses at most that share of its drop to
    its lowest axis neighbour, uphill neighbours do not count, and off the
    array the exit slope stands in."""
    h = np.full((1, 7, 7), 5.0)
    s = np.zeros_like(h)
    h[0, 2, 2] = 1.0                    # a one-cell pit
    h[0, 4, 5] = 3.0                    # (4, 4) drains to it, 2 below
    h[0, 4, 3] = 9.0                    # uphill of (4, 4): ignored
    mask = np.ones((1, 7, 7), np.uint8)
    out = np.empty_like(h)
    pk.slope_erode_cap(h, s, mask, 0.25, 0.02, out)
    k = 0.25 * np.sqrt(2.0)
    assert out[0, 2, 2] == 0.0
    assert out[0, 4, 4] == pytest.approx(k * 2.0)
    assert out[0, 0, 3] == pytest.approx(k * 0.02)     # the array edge: the exit slope
    mask[0, 3, 3] = 0
    pk.slope_erode_cap(h, s, mask, 0.25, 0.02, out)
    assert out[0, 3, 3] == 0.0


def test_slope_limit_erode_bounds_an_iterations_erosion():
    """With ``erosion.slope_limit_erode`` no cell loses more in one iteration
    than the cap computed on the surface it started from; without it the
    same particles take more somewhere (the limit binds)."""
    base = WorldParams.small_world(0).with_overrides(erosion={"erodibility": 1.0, "k_disc": 10.0, "disc_exponent": 0.0,
                                                              "max_erode": 1e9, "iter_erode": 1e9})
    losses = {}
    for name, coef in (("off", 0.0), ("on", 0.02)):
        p = base.with_overrides(erosion={"slope_limit_erode": coef})
        st = make_window(48, "tilt", p, seed=11, relief=12.0)
        s0 = (st.height + st.sediment).copy()
        cap = np.empty_like(st.height)
        pk.slope_erode_cap(st.height, st.sediment, st.mask, 0.02, float(p.erosion.slope_limit_exit), cap)
        run_iteration(st, p, 0)
        loss = s0 - (st.height + st.sediment)
        act = st.mask == pk.MASK_ACTIVE
        losses[name] = (loss[act], cap[act])
    loss_on, cap = losses["on"]
    assert np.all(loss_on <= cap + 1e-9), float((loss_on - cap).max())
    loss_off, _ = losses["off"]
    assert (loss_off > cap + 1e-6).sum() > 10


def test_discharge_scales_in_cells_of_upstream_rain():
    """``disc_saturation_cells`` / ``momentum_saturation_cells`` put the
    entrainment saturation and the momentum push's half strength at a number
    of cells of upstream rain, whatever the rain volume per cell (a zoom
    window's precip is the planet's over R^2); inflow spawn weight
    (``ErosionState.inflow_volume``) is not rain and does not count."""
    for scale in (1.0, 1e-4):
        p = WorldParams.small_world(0).with_overrides(erosion={"disc_saturation_cells": 1280.0, "momentum_saturation_cells": 512.0})
        st = make_window(32, "tilt", p, seed=2)
        st.precip *= np.float32(scale)
        idx, w = emaps.spawn_weights(st)            # the cells rain falls on (land, active, precip > 0)
        stats = run_iteration(st, p, 0)
        n_cells = idx.size
        rain = float(w.sum()) / n_cells
        assert stats["disc_saturation"] == pytest.approx(1280.0 * rain, rel=1e-3)
        assert stats["mom_scale"] == pytest.approx(stats["volume"] / (512.0 * rain), rel=1e-3)
    st = make_window(32, "tilt", p, seed=2)
    idx, w = emaps.spawn_weights(st)
    extra = float(w.sum())
    st.precip.reshape(-1)[idx] *= np.float32(2.0)            # as much again enters as inflow
    st.inflow_volume = extra
    stats2 = run_iteration(st, p, 0)
    assert stats2["disc_saturation"] == pytest.approx(1280.0 * extra / n_cells, rel=1e-3)


def _flat_metric(n: int) -> np.ndarray:
    """(g_ii, g_ij, g_jj) of a flat square grid -- off-diagonal 0, not 1."""
    return np.stack([np.ones((n, n)), np.zeros((n, n)), np.ones((n, n))], axis=-1)


def _basin_window(n: int = 60, precip: float = 1.0):
    """A slope falling along i with a square basin dug into it."""
    surf = 80.0 - 1.0 * np.repeat(np.arange(n, dtype=np.float64)[:, None], n, 1)
    surf[20:30, 20:30] -= 25.0
    mask = np.ones((n, n), np.uint8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    st = emaps.ErosionState.window(surf.copy(), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n, 2)), np.ones((n, n)),
                                   np.full((n, n), precip), np.ones((n, n)), np.zeros((n, n)), mask,
                                   _flat_metric(n), _flat_metric(n), 1, 1.0)
    st.refresh_route(0.001)
    return st, surf


def test_window_lakes_balance_a_depression_against_its_evaporation():
    """``window_lake_evap``: a basin fed by a catchment still fills to its
    spill, one fed by almost nothing holds a smaller lake at the level where
    evaporation matches the inflow, and the water it does hold is what the
    kernel sees (the routing surface) -- the dry bed above it stays at the
    spill level so a particle can still climb out."""
    st, surf = _basin_window(precip=1.0)
    spill = st.refresh_lakes_window(0.5, 0.02)
    assert spill["overflowing"] == 1 and spill["closed"] == 0
    full = int(st.lake_flag[0].sum())
    assert full == 100                                        # the whole basin

    st2, _ = _basin_window(precip=0.001)
    route_before = st2.route[0].copy()
    small = st2.refresh_lakes_window(0.5, 0.02)
    assert small["closed"] == 1 and small["overflowing"] == 0
    wet = st2.lake_flag[0] > 0
    assert 0 < int(wet.sum()) < full                           # a lake, but not the whole basin
    level = float((st2.route[0])[wet].max())
    assert level < float(route_before[wet].max())              # the water stands below the spill
    assert (st2.route[0][wet] <= route_before[wet] + 1e-9).all()
    dry = (st2.height[0] + st2.sediment[0] < route_before) & ~wet
    assert np.array_equal(st2.route[0][dry], route_before[dry])  # the dry bed keeps the spill route
    assert float(st2.base[0].max()) == 0.0                     # a window lake is never a sink

    # no evaporation: the spill-point fill, as before
    st3, _ = _basin_window(precip=0.001)
    st3.refresh_lakes_window(0.5, 0.0)
    assert int(st3.lake_flag[0].sum()) == full


def _plain_window(n: int = 64, seed: int = 3):
    surf = 40.0 - 0.02 * np.repeat(np.arange(n, dtype=np.float64)[:, None], n, 1)
    surf += 0.12 * ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(n, n)), 4.0)
    mask = np.ones((n, n), np.uint8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    return surf, mask


def _run_window(surf, mask, rate, its=25, ppc=0.3):
    from globe.refine.basin_job import basin_erosion_params
    from globe.refine.zoom import zoom_params

    n = surf.shape[0]
    ep = basin_erosion_params(zoom_params(WorldParams(), 32), np.random.default_rng(5))
    ep.lateral_rate = rate
    ep.max_steps = 60
    st = emaps.ErosionState.window(surf.copy(), np.full((n, n), 1.0), np.zeros((n, n)), np.zeros((n, n, 2)), np.ones((n, n)),
                                   np.ones((n, n)), np.ones((n, n)), np.zeros((n, n)), mask,
                                   _flat_metric(n), _flat_metric(n), 1, 30.0)
    for it in range(its):
        emaps.step(st, ep, it, particles_per_cell=ppc, rng_stage="refine")
    return st


def test_change_list_cap_is_one_formula_for_the_kernel_and_its_caller():
    """``trace_particles`` lays out a particle's slice with
    :func:`particle.change_list_cap` and ``apply_changes`` is handed the same
    number: if the two ever disagree the apply reads uninitialised entries
    and indexes the terrain with whatever they hold."""
    for max_steps in (40, 500):
        for lateral in (False, True):
            cap = int(pk.change_list_cap(max_steps, lateral))
            assert cap >= (2 * max_steps if lateral else max_steps) + 2 * pk.SPREAD + 1
    surf, mask = _plain_window(48)
    st = _run_window(surf, mask, 0.0, its=2)
    assert np.isfinite(st.height).all()


def test_lateral_erosion_cuts_the_bank_on_the_outside_of_a_bend():
    """``erosion.lateral_rate``: where a particle is cutting *and* turning it
    also takes material off the cell across the flow on the outside of the
    bend.  Those entries carry volume -2, so :func:`particle.apply_changes`
    erodes them under the same caps without tracking discharge or moving the
    particle's path; with the rate at 0 none are written and the pass is what
    it was."""
    surf, mask = _plain_window()
    seen = {}

    def count(rate):
        n = surf.shape[0]
        from globe.refine.basin_job import basin_erosion_params
        from globe.refine.zoom import zoom_params

        ep = basin_erosion_params(zoom_params(WorldParams(), 32), np.random.default_rng(5))
        ep.lateral_rate = rate
        ep.max_steps = 60
        st = emaps.ErosionState.window(surf.copy(), np.full((n, n), 1.0), np.zeros((n, n)), np.zeros((n, n, 2)), np.ones((n, n)),
                                       np.ones((n, n)), np.ones((n, n)), np.zeros((n, n)), mask,
                                       _flat_metric(n), _flat_metric(n), 1, 30.0)
        tally = [0, 0]

        def diag(state, cl_cell, cl_vol, cl_count, cap, sp_death):
            for p in range(cl_count.size):
                v = cl_vol[p * cap: p * cap + cl_count[p]]
                tally[0] += v.size
                tally[1] += int((v == -2.0).sum())

        emaps.run_iteration(st, ep, 0, particles_per_cell=0.3, rng_stage="refine", diag=diag)
        return tally, st.height[0] + st.sediment[0]

    (entries0, lat0), h0 = count(0.0)
    (entries1, lat1), h1 = count(50.0)
    assert lat0 == 0 and entries0 > 100
    assert lat1 > 0.01 * entries1, (lat1, entries1)        # a real share of the steps cut a bank
    assert not np.array_equal(h0, h1)                       # and it moves ground
