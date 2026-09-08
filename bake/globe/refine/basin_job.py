"""One basin -> fine arrays over its window (PLAN 10.2 steps 1-4); the
multiprocessing target of the refine driver.

:func:`run_basin` is pure (window in, arrays out) and deterministic: every
random number comes from one ``params.rng("refine", basin_id)`` stream —
the detail noise draws first, then the erosion iterations continue the
same generator (a small ``ErosionParams`` subclass hands that generator
to the kernel's ``run_iteration``), so the result depends only on
``(seed, params, basin)`` and never on which worker ran it.

Erosion setup (``globe.erosion`` window mode, ``spherical=False``,
``F = 1``, heights in fine-cell units):

* ``mask``: 0 outside the basin (nearest-upsampled ``basin_id`` != this
  basin: other basins, other faces, ocean), 1 inside, 2 frozen divide
  (inside cells 8-adjacent to a 0 cell or the array border).  Every exit
  cell's downstream cell is outside the basin, so each exit's ``R x R``
  fine block touches a mask-0 cell through frozen cells: a particle
  following the flow out through the outlet or any other exit dies on
  the first mask-0 cell (``deposit_on_exit`` is off, so its load leaves
  the basin like water into the ocean) — every exit is a sink.
* ``discharge`` / ``momentum`` start from the upsampled coarse maps (same
  volume units: a fine cell spawns ``precip / R²``), so the fine rivers
  continue the coarse ones and the EMA relaxes from there.
* lakes: fine cells whose upsampled lake depth exceeds
  ``hydro.lake_min_depth`` get no rain and an evaporation multiplier
  that empties a particle within two steps (``volume *= 0.01`` per step,
  the kernel's floor), so a particle reaching a lake dies of evaporation
  *inside the mask* and the kernel deposits its whole load at the death
  cell ("deposit and stop", PLAN 8.2) — the only way to stop particles
  with the published kernel API without a mask-0 hole (which would
  discard the load).  A lake is a sink with a *ceiling*.  Lake cells are
  the cells the local priority flood of the *plain* upsample (same
  drains as the final flood) covers deeper than ``lake_min_depth``
  within a coarse cell of a coarse lake (upsampled lake depth > 0); the
  bilinearly smeared coarse water surface itself is not usable as a
  level, it varies by tens of metres over a lake in a steep valley.
  After every iteration (and once after the detail noise)
  :func:`settle_lakes` takes whatever stands above the ceiling ``plain
  fine water surface - max(lake_min_depth, depth / 2)`` in a lake cell
  (the shoreline delta the kernel piled up, thermal inflow, noise),
  spreads it as sediment over the remaining capacity of the same lake
  (8-connected component) and discards the rest
  (``stats["lake_discarded_m"]``, the same non-conserving semantics as
  the ocean).  After the final flood a filled cell left shallower than
  ``lake_min_depth`` (its sill eroded) gives its deposit back down to
  the plain floor, so the coarse lakes keep water surfaces consistent
  with the coarse ones (PLAN 10.2 step 4) instead of being buried.  With
  ``erosion.evap_rate == 0`` lakes fall back to mask 0 (load discarded).
* ``uplift = 0``; thermal erosion on; the kernel's routing surface
  (``erosion.flood_every``) is refreshed inside the window as in the
  global pass.
* coarse-scale drift removal (:func:`block_drift`): the refine pass
  exports mass through the exits and relaxes the coarse cliffs, which
  lowers the basin interior by a *coarse-scale* amount (10-35 m on the
  small preset) while the divide ring is pinned to the plain upsample —
  every divide became a crest.  After the iterations the per-coarse-cell
  mean of ``surface - plain`` over the basin's active non-lake cells is
  interpolated bilinearly to the fine grid and subtracted (Jacobi
  passes until every block mean is below 5 cm), so the refined surface
  agrees with the coarse one at coarse-cell scale by construction (LOD-2
  agreement, PLAN 15) and only sub-coarse detail remains; lowering takes
  sediment first, raising goes to bedrock.

Then a local priority flood (``hydro.priority_flood.priority_flood_flat``)
over the basin cells with *every* exit cell (all fine cells of each coarse
exit) and every ocean cell of the window as drains gives the fine
``water_surface``.
"""
from __future__ import annotations

import dataclasses
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import ErosionParams, WorldParams
from ..erosion import particle as pk
from ..erosion.maps import ErosionState, height_unit, step
from ..hydro.priority_flood import priority_flood_flat
from ..io.world_store import WorldStore
from .upsample import COARSE_INPUTS, Window, basin_window, coarse_derived, detail_noise, upsample_window

#: arrays a job returns over its fine window (n x n), plus ``mask`` (u8),
#: ``lake`` (bool, the lake cells of the module docstring) and the
#: plain-upsample references ``height0``, ``sediment0``,
#: ``water_surface0``, ``discharge0`` used by the feathering
RESULT_FIELDS = ("height", "sediment", "water_surface", "discharge", "hardness", "basin_id")


@dataclass
class BasinResult:
    id: int
    win: Window
    arrays: dict[str, np.ndarray]
    stats: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    @property
    def mask(self) -> np.ndarray:
        return self.arrays["mask"]

    def surface(self) -> np.ndarray:
        return self.arrays["height"] + self.arrays["sediment"]


# --------------------------------------------------------------------------
# per-process cache of the coarse inputs
# --------------------------------------------------------------------------
_CACHE: dict[str, tuple] = {}


def coarse_inputs(root: str | Path, params: WorldParams) -> tuple:
    """``(grid, fields, derived)`` of the world at ``root`` — loaded once
    per process (workers keep it across jobs)."""
    key = str(Path(root).resolve())
    hit = _CACHE.get(key)
    if hit is None:
        grid = params.coarse_grid()
        store = WorldStore(root)
        fields = {n: store.load_field(n, grid) for n in COARSE_INPUTS}
        derived = coarse_derived(fields)
        _CACHE.clear()
        hit = (grid, fields, derived)
        _CACHE[key] = hit
    return hit


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def build_mask(basin_id_up: np.ndarray, bid: int) -> np.ndarray:
    """uint8 mask codes from the nearest-upsampled basin ids: 0 outside,
    1 inside, 2 inside cells 8-adjacent to an outside cell or to the array
    border (frozen divides)."""
    inside = np.asarray(basin_id_up) == int(bid)
    core = ndimage.binary_erosion(inside, structure=np.ones((3, 3), bool), border_value=0)
    mask = np.zeros(inside.shape, dtype=np.uint8)
    mask[inside] = pk.MASK_FROZEN
    mask[core] = pk.MASK_ACTIVE
    return mask


@dataclass
class BasinErosionParams(ErosionParams):
    """``ErosionParams`` that hands one persistent generator to the kernel
    (``run_iteration`` calls ``params.rng(stage, *key)`` every iteration;
    the stream is ``params.rng("refine", basin_id)``)."""

    def rng(self, stage: str, *key) -> np.random.Generator:
        return self._gen


def basin_erosion_params(params: WorldParams, gen: np.random.Generator) -> BasinErosionParams:
    ep = BasinErosionParams(**dataclasses.asdict(params.erosion))
    ep._gen = gen
    return ep


#: numba threads a job uses: one per this many extended-window cells, capped
#: at the process's thread budget.  The kernel's parallel regions (one per
#: particle chunk, one per thermal / pack pass) are fork-joins whose
#: overhead dominates on small windows: on a 280² window 4 threads were
#: 2x *slower* than 1, and a whole small-preset refine with 4 threads per
#: job took 30x the single-thread time.
CELLS_PER_THREAD = 400_000
_THREAD_BUDGET: int | None = None


def thread_budget() -> int:
    """Threads this process may use for a job (set by :func:`pool_init`;
    numba's current count the first time otherwise)."""
    global _THREAD_BUDGET
    if _THREAD_BUDGET is None:
        import numba

        _THREAD_BUDGET = int(numba.get_num_threads())
    return _THREAD_BUDGET


def job_threads(NE: int, budget: int | None = None) -> int:
    b = thread_budget() if budget is None else int(budget)
    return max(1, min(b, (int(NE) * int(NE)) // CELLS_PER_THREAD))


def exit_cells(basin: dict, win: Window, shape: tuple[int, int]) -> np.ndarray:
    """Bool ``shape`` array (extended window) of the fine cells of every
    exit cell of the basin (``outlet`` first in the record; all treated
    alike)."""
    out = np.zeros(shape, dtype=bool)
    R = win.R
    for f, i, j in basin.get("exits") or [basin["outlet"]]:
        if int(f) != win.face:
            continue
        a, b = win.coarse_to_ext(int(i), int(j))
        if 0 <= a < shape[0] - R + 1 and 0 <= b < shape[1] - R + 1:
            out[a : a + R, b : b + R] = True
    return out


def settle_lakes(height: np.ndarray, sediment: np.ndarray, idx: np.ndarray, lab: np.ndarray, cap: np.ndarray, n_lakes: int) -> float:
    """Lake ceiling (module docstring), in place on the *flat views*
    ``height`` / ``sediment`` (any unit; ``cap`` in the same unit).
    ``idx``: flat indices of the lake cells, ``lab``: their lake number
    (``0 .. n_lakes-1``), ``cap``: the ceiling per lake cell.  Material
    above the ceiling is removed (sediment first, then bedrock) and spread
    as sediment over the same lake's remaining room, proportionally to
    the room of each cell; what does not fit is discarded.  Returns the
    discarded amount (cell-sum, same unit)."""
    if idx.size == 0:
        return 0.0
    surf = height[idx] + sediment[idx]
    excess = np.maximum(surf - cap, 0.0)
    if not (excess > 0.0).any():
        return 0.0
    room = np.maximum(cap - surf, 0.0)
    ex_tot = np.bincount(lab, weights=excess, minlength=n_lakes)
    room_tot = np.bincount(lab, weights=room, minlength=n_lakes)
    take_s = np.minimum(sediment[idx], excess)
    sediment[idx] -= take_s
    height[idx] -= excess - take_s
    frac = np.where(room_tot > 0.0, np.minimum(ex_tot / np.maximum(room_tot, 1e-300), 1.0), 0.0)
    sediment[idx] += room * frac[lab]
    return float(np.sum(ex_tot - np.minimum(ex_tot, room_tot)))


def local_flood(surface: np.ndarray, drain: np.ndarray, flood_active: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Water surface of ``surface`` (``NE x NE`` f32) drained through
    ``drain`` cells over ``flood_active`` cells (module docstring); cells
    the flood does not reach keep ``surface``; ``>= surface`` everywhere."""
    ws = surface.copy()
    if flood_active.any() and drain.any():
        fr = priority_flood_flat(surface, drain, flood_active)
        reached = (fr.order.reshape(surface.shape) >= 0) & (mask > 0)
        ws[reached] = fr.filled[reached]
    return np.maximum(ws, surface).astype(np.float32)


def peak_rss_mb() -> float:
    """Peak resident size of this process so far (MB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _bilinear_from_blocks(D: np.ndarray, R: int) -> np.ndarray:
    """Bilinear interpolation of per-block values ``D (nb, nb)`` (nodes at
    the block centres) to the ``(nb R, nb R)`` fine lattice (clamped at
    the outer half-blocks)."""
    nb = D.shape[0]
    x = (np.arange(nb * R) + 0.5) / R - 0.5
    k0 = np.clip(np.floor(x).astype(np.int64), 0, max(nb - 2, 0))
    t = np.clip(x - k0, 0.0, 1.0)
    if nb == 1:
        return np.repeat(np.repeat(D, R, 0), R, 1).astype(np.float64)
    G = (1.0 - t)[:, None] * D[k0, :] + t[:, None] * D[k0 + 1, :]
    return (1.0 - t)[None, :] * G[:, k0] + t[None, :] * G[:, k0 + 1]


def block_drift(delta: np.ndarray, cells: np.ndarray, R: int, tol: float = 0.05, max_passes: int = 64) -> np.ndarray:
    """Smooth (bilinear from coarse-cell nodes) field ``F`` such that the
    mean of ``delta - F`` over the ``cells`` of every ``R x R`` block (a
    coarse cell; the array is block-aligned) vanishes: Jacobi passes of
    "interpolate the block means of the residual" until the largest block
    residual is below ``tol`` (units of ``delta``) or ``max_passes`` (a
    block-to-block alternating drift contracts by only ~0.75 per pass at
    R = 2, smooth drifts by ~0.1).  Blocks without ``cells`` do not
    constrain ``F`` (their node value comes from the neighbours).
    Deterministic; float64 ``(NE, NE)``."""
    NE = delta.shape[0]
    nb = NE // R
    assert nb * R == NE, (NE, R)
    d = np.where(cells, delta, 0.0).astype(np.float64)
    cnt = cells.reshape(nb, R, nb, R).sum(axis=(1, 3)).astype(np.float64)
    has = cnt > 0
    F = np.zeros((NE, NE), dtype=np.float64)
    if not has.any():
        return F
    for _ in range(int(max_passes)):
        res = (d - np.where(cells, F, 0.0)).reshape(nb, R, nb, R).sum(axis=(1, 3))
        res = np.where(has, res / np.maximum(cnt, 1.0), 0.0)
        if float(np.abs(res).max()) < tol:
            break
        # blocks without cells: average of the constrained 8-neighbours (keeps the ramp smooth into the ring)
        if not has.all():
            num = ndimage.uniform_filter(res * has, size=3, mode="constant") * 9.0
            den = ndimage.uniform_filter(has.astype(np.float64), size=3, mode="constant") * 9.0
            res = np.where(has, res, np.where(den > 0, num / np.maximum(den, 1e-300), 0.0))
        F += _bilinear_from_blocks(res, R)
    return F


# --------------------------------------------------------------------------
# the job
# --------------------------------------------------------------------------
def run_basin(root: str | Path, basin: dict, params: WorldParams, log=None, threads: int | None = None) -> BasinResult:
    """Refine one basin (see module docstring).  Returns the fine window
    arrays (``n x n``, kernel halo stripped): ``height``, ``sediment``,
    ``water_surface``, ``discharge``, ``hardness``, ``basin_id`` (this id
    inside the mask, the nearest-upsampled ids elsewhere), ``mask``,
    ``lake`` and the plain-upsample references ``height0``, ``sediment0``,
    ``water_surface0``, ``discharge0``.  ``threads``: numba threads for
    the kernel (default :func:`job_threads` of the window; the result does
    not depend on it); the previous count is restored afterwards."""
    import numba

    t0 = time.time()
    bid = int(basin["id"])
    rp = params.refine
    grid, fields, derived = coarse_inputs(root, params)
    win = basin_window(basin, params.world.R, rp.halo_cells)
    H, n, NE = win.H, win.n, win.NE
    prev_threads = int(numba.get_num_threads())
    n_threads = job_threads(NE) if threads is None else max(1, int(threads))
    try:
        numba.set_num_threads(n_threads)
    except ValueError:  # more than the process was launched with
        n_threads = prev_threads
    try:
        return _run_basin(root, basin, params, log, t0, bid, rp, grid, fields, derived, win, H, n, NE, up_threads=n_threads)
    finally:
        numba.set_num_threads(prev_threads)


def _run_basin(root, basin, params, log, t0, bid, rp, grid, fields, derived, win, H, n, NE, up_threads) -> BasinResult:
    rss0 = peak_rss_mb()
    up = upsample_window(fields, derived, win, grid)
    t_up = time.time() - t0
    rss_up = peak_rss_mb()
    R = win.R

    mask = build_mask(up["basin_id"], bid)
    active = mask == pk.MASK_ACTIVE
    gen = params.rng("refine", bid)
    noise = detail_noise(win, up["slope"], up["relief"], up["hardness"], rp.detail_amp, grid.cell_size_m, gen)
    del up["slope"], up["relief"]
    noise_max = float(np.abs(noise[active]).max()) if active.any() else 0.0
    height = up["height0"].astype(np.float32)
    height[active] += noise[active]
    del noise
    sediment = up["sediment0"].copy()
    # lakes (module docstring): plain-upsample flood with the final drains,
    # lake cells, their ceiling (metres) and their 8-connected lake number
    lake_min = float(params.hydro.lake_min_depth)
    plain = (up["height0"] + up["sediment0"]).astype(np.float32)
    ws0 = (plain + up["depth"]).astype(np.float32)
    ocean = up["basin_id"] < 0
    flood_active = (mask > 0) | ocean
    drain = exit_cells(basin, win, (NE, NE)) | ocean
    wsp = local_flood(plain, drain, flood_active, mask)
    lake = (wsp - plain > lake_min) & (up["depth"] > 0.0) & active
    del up["depth"]
    lake_labels, n_lakes = ndimage.label(lake, structure=np.ones((3, 3), bool))
    lake_idx = np.flatnonzero(lake)
    lake_lab = (lake_labels.reshape(-1)[lake_idx] - 1).astype(np.int64)
    lake_floor_m = plain.reshape(-1)[lake_idx].astype(np.float64)
    lake_cap_m = wsp.reshape(-1)[lake_idx] - np.maximum(lake_min, 0.5 * (wsp - plain).reshape(-1)[lake_idx])
    del lake_labels, wsp, plain
    lake_discarded = 0.0
    ep = basin_erosion_params(params, gen)
    precip = np.where(active & ~lake, up["precip"], np.float32(0.0)).astype(np.float32)
    del up["precip"]
    evap = up.pop("evap")
    if ep.dt * ep.evap_rate > 0.0:
        evap[lake] = np.float32(2.0 / (ep.dt * ep.evap_rate))  # particle volume hits the kernel floor: dies, deposits
        lake_discarded += settle_lakes(height.reshape(-1), sediment.reshape(-1), lake_idx, lake_lab, lake_cap_m.astype(np.float32), n_lakes)
    else:
        mask[lake] = pk.MASK_OUTSIDE  # no evaporation: lakes become holes (load discarded)
        active = mask == pk.MASK_ACTIVE
        lake_idx = lake_idx[:0]
        lake_lab = lake_lab[:0]
        lake_cap_m = lake_cap_m[:0]
        lake_floor_m = lake_floor_m[:0]
    unit = height_unit(params.fine_grid(), ep)
    n_active = int(active.sum())
    n_iter = int(rp.refine_iterations) if n_active > 0 else 0
    deaths = {k: 0 for k in pk.DEATH_NAMES}
    particles = 0
    entries = 0
    clamped = 0
    t1 = time.time()
    if n_iter > 0:
        state = ErosionState.window(
            height, sediment, up["discharge"], up.pop("momentum"), up["hardness"], precip, evap,
            np.zeros((NE, NE), dtype=np.float64), mask, up.pop("metric"), up.pop("metric_inv"), H, unit, deposit_on_exit=False,
        )
        del height, sediment, precip, evap  # the state owns / copied them
        lake_cap_u = lake_cap_m / unit
        for it in range(n_iter):
            st = step(state, ep, it, particles_per_cell=rp.particles_per_cell, rng_stage="refine")
            particles += int(st.get("particles", 0))
            entries += int(st.get("entries", 0))
            clamped += int(st.get("clamped", 0))
            for k, v in (st.get("deaths") or {}).items():
                deaths[k] += int(v)
            lake_discarded += settle_lakes(state.height.reshape(-1), state.sediment.reshape(-1), lake_idx, lake_lab, lake_cap_u, n_lakes) * unit
            if log is not None:
                log(f"  basin {bid} iter {it + 1}/{n_iter}: {st.get('particles', 0)} particles, mean {st.get('steps_mean', 0.0):.0f} steps, {st.get('seconds_total', 0.0):.2f}s")
        height = state.height_m()[0].astype(np.float32)
        sediment = state.sediment_m()[0].astype(np.float32)
        discharge = state.discharge[0].astype(np.float32)
        pending = float(state.pending[0].sum() * unit)
        del state
    else:
        discharge = up["discharge"].copy()
        pending = 0.0
    rss_erode = peak_rss_mb()
    t_erode = time.time() - t1

    # coarse-scale drift removal (module docstring): block means of
    # surface - plain over the active non-lake cells -> 0
    t2 = time.time()
    plain = up["height0"] + up["sediment0"]
    drift_cells = active & ~lake
    drift = block_drift((height + sediment).astype(np.float64) - plain, drift_cells, R)
    drift_max = float(np.abs(drift[drift_cells]).max()) if drift_cells.any() else 0.0
    f = np.where(drift_cells, drift, 0.0).astype(np.float32)  # lake floors keep their settled level
    lower = np.maximum(f, np.float32(0.0))
    take_s = np.minimum(sediment, lower)
    sediment -= take_s
    height -= lower - take_s
    height -= np.minimum(f, np.float32(0.0))
    del f, lower, take_s, drift, plain
    lake_discarded += settle_lakes(height.reshape(-1), sediment.reshape(-1), lake_idx, lake_lab, lake_cap_m.astype(np.float32), n_lakes)
    t_drift = time.time() - t2

    # local priority flood: drains = every exit's fine cells + ocean cells
    t2 = time.time()
    surface = (height + sediment).astype(np.float32)
    ws = local_flood(surface, drain, flood_active, mask)
    # lake give-back: a filled lake cell left shallower than lake_min_depth
    # returns its deposit down to the plain floor (lowering a cell inside a
    # depression never changes the flood level, so ws stays valid)
    if lake_idx.size:
        hf, sf = height.reshape(-1), sediment.reshape(-1)
        cur = hf[lake_idx] + sf[lake_idx]
        target = np.maximum(lake_floor_m, np.minimum(cur, ws.reshape(-1)[lake_idx] - lake_min))
        give = np.maximum(cur - target, 0.0).astype(np.float32)
        take_s = np.minimum(sf[lake_idx], give)
        sf[lake_idx] -= take_s
        hf[lake_idx] -= give - take_s
        lake_discarded += float(give.sum())
        del surface, cur, target, give, take_s
    del drain, flood_active, ocean
    t_flood = time.time() - t2

    sl = (slice(H, H + n), slice(H, H + n))
    out_bid = np.where(mask > 0, np.int32(bid), up["basin_id"]).astype(np.int32)
    arrays = {
        "height": np.ascontiguousarray(height[sl]),
        "sediment": np.ascontiguousarray(sediment[sl]),
        "water_surface": np.ascontiguousarray(ws[sl]),
        "discharge": np.ascontiguousarray(discharge[sl]),
        "hardness": np.ascontiguousarray(up["hardness"][sl]),
        "basin_id": np.ascontiguousarray(out_bid[sl]),
        "mask": np.ascontiguousarray(mask[sl]),
        "lake": np.ascontiguousarray(lake[sl]),
        "height0": np.ascontiguousarray(up["height0"][sl]),
        "sediment0": np.ascontiguousarray(up["sediment0"][sl]),
        "water_surface0": np.ascontiguousarray(ws0[sl]),
        "discharge0": np.ascontiguousarray(up["discharge"][sl]),
    }
    stats = {
        "id": bid,
        "face": win.face,
        "area_cells": int(basin.get("area_cells", 0)),
        "window": [win.ci0, win.cj0, win.ci1, win.cj1],
        "n_fine": n,
        "active_cells": n_active,
        "frozen_cells": int((mask == pk.MASK_FROZEN).sum()),
        "lake_cells": int(lake.sum()),
        "lakes": int(n_lakes),
        "lake_discarded_m": float(lake_discarded),
        "iterations": n_iter,
        "particles": particles,
        "steps_mean": entries / max(particles, 1),
        "deaths": deaths,
        "clamped": clamped,
        "pending_m": pending,
        "noise_max_m": noise_max,
        "drift_max_m": drift_max,
        "threads": int(up_threads),
        "seconds": time.time() - t0,
        "seconds_upsample": t_up,
        "seconds_erosion": t_erode,
        "seconds_drift": t_drift,
        "seconds_flood": t_flood,
        "peak_rss_mb": [rss0, rss_up, rss_erode, peak_rss_mb()],  # process peak before, after the upsample, after the iterations, at the end
    }
    return BasinResult(bid, win, arrays, stats)


# --------------------------------------------------------------------------
# quicklook of one basin window
# --------------------------------------------------------------------------
def basin_quicklook(res: BasinResult, path: str | Path, cell_size_m: float) -> Path:
    """Hillshaded window surface, lakes, discharge, frozen divide ring
    (white) and the mask boundary (dark)."""
    from ..viz import quicklook as ql

    surf = res.surface()[None].astype(np.float64)
    img = ql.render_height(surf, 0.0, cell_size_m)
    m = res.mask[None]
    lake = ((res["water_surface"] - res.surface()) > 0.05)[None] & (m > 0)
    img = ql.overlay(img, lake, (60, 120, 220), 0.9)
    q = res["discharge"][None].astype(np.float64)
    lq = np.log1p(np.maximum(q, 0))
    if lq.max() > 0:
        thr = float(np.percentile(lq[m > 0], 90)) if (m > 0).any() else 0.0
        wgt = np.clip((lq - thr) / max(lq.max() - thr, 1e-9), 0, 1)
        img = ql.overlay(img, (wgt > 0) & (m > 0), (30, 80, 255), 0.95, weight=0.35 + 0.65 * wgt)
    img = ql.overlay(img, m == pk.MASK_FROZEN, (255, 255, 255), 0.6)
    outside = m == 0
    img[outside] = (img[outside].astype(np.float32) * 0.55).astype(np.uint8)
    return ql.save_face(path, img[0])


# --------------------------------------------------------------------------
# multiprocessing targets
# --------------------------------------------------------------------------
def run_basin_arrays(root: str | Path, basin: dict, params: WorldParams, threads: int | None = None) -> dict[str, np.ndarray]:
    """:func:`run_basin` returning only the window arrays (a picklable
    pool target for determinism checks: the same basin run in-process and
    in a worker, with any thread count, must give byte-identical arrays)."""
    return run_basin(root, basin, params, threads=threads).arrays


def pool_init(n_threads: int, root: str | Path | None = None, params: WorldParams | None = None) -> None:
    """Worker initialiser: share the machine's cores between the workers
    (the job thread budget) and, when ``root``/``params`` are given, load
    the coarse inputs once up front (the worker keeps them across jobs).
    Workers are *spawned* (see ``refine.run``): numba's OpenMP threading
    layer is not fork-safe once the parent has run a parallel kernel."""
    import numba

    global _THREAD_BUDGET
    n_threads = max(1, int(n_threads))
    try:
        numba.set_num_threads(n_threads)
    except ValueError:
        n_threads = int(numba.get_num_threads())
    _THREAD_BUDGET = n_threads
    if root is not None and params is not None:
        coarse_inputs(root, params)


def job(args: dict) -> dict:
    """Pool target (importable, picklable arguments).

    ``{"kind": "basin", "root", "params": WorldParams, "basin": record,
    "quicklook": path or None}`` runs :func:`run_basin` and rasterises the
    result into the ``fine/`` memmaps (:func:`rasterize.write_result`);
    ``{"kind": "face", "root", "params", "face"}`` writes the plain
    upsample of a whole face (:func:`rasterize.write_base_face`).  Returns
    the job's stats dict (``kind`` set)."""
    from . import rasterize

    kind = args.get("kind", "basin")
    root = args["root"]
    params = args["params"]
    if kind == "face":
        t0 = time.time()
        grid, fields, derived = coarse_inputs(root, params)
        cells = rasterize.write_base_face(root, params, int(args["face"]), fields, derived)
        return {"kind": "face", "face": int(args["face"]), "cells": cells, "seconds": time.time() - t0}
    if kind != "basin":
        raise ValueError(f"unknown job kind {kind!r}")
    res = run_basin(root, args["basin"], params)
    written = rasterize.write_result(root, params, res)
    res.stats["written_cells"] = written
    qp = args.get("quicklook")
    if qp:
        try:
            basin_quicklook(res, qp, params.fine_cell_size_m)
            res.stats["quicklook"] = str(qp)
        except Exception as e:  # a picture must never fail a bake
            res.stats["quicklook_error"] = repr(e)
    res.stats["kind"] = "basin"
    return res.stats


__all__ = ["BasinResult", "RESULT_FIELDS", "CELLS_PER_THREAD", "coarse_inputs", "build_mask", "settle_lakes", "local_flood", "block_drift", "BasinErosionParams", "basin_erosion_params", "thread_budget", "job_threads", "exit_cells", "run_basin", "run_basin_arrays", "basin_quicklook", "pool_init", "job"]
