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
  cell of the basin therefore borders a mask-0 cell — particles leaving
  through the outlet or any other exit die there (``deposit_on_exit`` is
  off, so their load leaves the basin like water into the ocean).
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
  discard the load).  With ``erosion.evap_rate == 0`` lakes fall back to
  mask 0 (load discarded).
* ``uplift = 0``; thermal erosion on; the kernel's routing surface
  (``erosion.flood_every``) is refreshed inside the window as in the
  global pass.

Then a local priority flood (``hydro.priority_flood.priority_flood_flat``)
over the basin cells with *every* exit cell (all fine cells of each coarse
exit) and every ocean cell of the window as drains gives the fine
``water_surface``.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import ErosionParams, WorldParams
from ..erosion import particle as pk
from ..erosion.maps import ErosionState, height_unit, step
from ..field import FaceField
from ..hydro.priority_flood import priority_flood_flat
from ..io.world_store import WorldStore
from .upsample import COARSE_INPUTS, Window, basin_window, coarse_derived, detail_noise, upsample_window

#: arrays a job returns over its fine window (n x n), plus ``mask`` (u8)
#: and the plain-upsample references ``height0``, ``sediment0``,
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


# --------------------------------------------------------------------------
# the job
# --------------------------------------------------------------------------
def run_basin(root: str | Path, basin: dict, params: WorldParams, log=None) -> BasinResult:
    """Refine one basin (see module docstring).  Returns the fine window
    arrays (``n x n``, kernel halo stripped): ``height``, ``sediment``,
    ``water_surface``, ``discharge``, ``hardness``, ``basin_id`` (this id
    inside the mask, the nearest-upsampled ids elsewhere), ``mask`` and
    the plain-upsample references ``height0``, ``sediment0``,
    ``water_surface0``, ``discharge0``."""
    t0 = time.time()
    bid = int(basin["id"])
    rp = params.refine
    grid, fields, derived = coarse_inputs(root, params)
    win = basin_window(basin, params.world.R, rp.halo_cells)
    R, H, n, NE = win.R, win.H, win.n, win.NE
    up = upsample_window(fields, derived, win, grid)
    t_up = time.time() - t0

    mask = build_mask(up["basin_id"], bid)
    active = mask == pk.MASK_ACTIVE
    gen = params.rng("refine", bid)
    noise = detail_noise(win, up["slope"], up["relief"], up["hardness"], rp.detail_amp, grid.cell_size_m, gen)
    height = up["height0"].astype(np.float32)
    height[active] += noise[active]
    sediment = up["sediment0"].copy()
    surface0 = height + sediment
    lake = (up["depth"] > float(params.hydro.lake_min_depth)) & active
    ep = basin_erosion_params(params, gen)
    precip = np.where(active & ~lake, up["precip"], np.float32(0.0)).astype(np.float32)
    evap = up["evap"].copy()
    if ep.dt * ep.evap_rate > 0.0:
        evap[lake] = np.float32(2.0 / (ep.dt * ep.evap_rate))  # particle volume hits the kernel floor: dies, deposits
    else:
        mask[lake] = pk.MASK_OUTSIDE  # no evaporation: lakes become holes (load discarded)
        active = mask == pk.MASK_ACTIVE
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
            height, sediment, up["discharge"], up["momentum"], up["hardness"], precip, evap,
            np.zeros((NE, NE), dtype=np.float64), mask, up["metric"], up["metric_inv"], H, unit, deposit_on_exit=False,
        )
        for it in range(n_iter):
            st = step(state, ep, it, particles_per_cell=rp.particles_per_cell, rng_stage="refine")
            particles += int(st.get("particles", 0))
            entries += int(st.get("entries", 0))
            clamped += int(st.get("clamped", 0))
            for k, v in (st.get("deaths") or {}).items():
                deaths[k] += int(v)
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
    t_erode = time.time() - t1

    # local priority flood: drains = every exit's fine cells + ocean cells
    t2 = time.time()
    surface = (height + sediment).astype(np.float32)
    ocean = up["basin_id"] < 0
    flood_active = (mask > 0) | ocean
    drain = exit_cells(basin, win, (NE, NE)) | ocean
    ws = surface.copy()
    if flood_active.any() and drain.any():
        fr = priority_flood_flat(surface, drain, flood_active)
        reached = (fr.order.reshape(NE, NE) >= 0) & (mask > 0)
        ws[reached] = fr.filled[reached]
    ws = np.maximum(ws, surface).astype(np.float32)
    t_flood = time.time() - t2

    sl = (slice(H, H + n), slice(H, H + n))
    out_bid = np.where(mask > 0, np.int32(bid), up["basin_id"]).astype(np.int32)
    ws0 = (up["height0"] + up["sediment0"] + up["depth"]).astype(np.float32)
    arrays = {
        "height": np.ascontiguousarray(height[sl]),
        "sediment": np.ascontiguousarray(sediment[sl]),
        "water_surface": np.ascontiguousarray(ws[sl]),
        "discharge": np.ascontiguousarray(discharge[sl]),
        "hardness": np.ascontiguousarray(up["hardness"][sl]),
        "basin_id": np.ascontiguousarray(out_bid[sl]),
        "mask": np.ascontiguousarray(mask[sl]),
        "height0": np.ascontiguousarray(up["height0"][sl]),
        "sediment0": np.ascontiguousarray(up["sediment0"][sl]),
        "water_surface0": np.ascontiguousarray(ws0[sl]),
        "discharge0": np.ascontiguousarray(up["discharge"][sl]),
        "surface_init": np.ascontiguousarray(surface0[sl]),
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
        "iterations": n_iter,
        "particles": particles,
        "steps_mean": entries / max(particles, 1),
        "deaths": deaths,
        "clamped": clamped,
        "pending_m": pending,
        "noise_max_m": float(np.abs(noise[active]).max()) if n_active else 0.0,
        "seconds": time.time() - t0,
        "seconds_upsample": t_up,
        "seconds_erosion": t_erode,
        "seconds_flood": t_flood,
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
# multiprocessing target
# --------------------------------------------------------------------------
def pool_init(n_threads: int) -> None:
    """Worker initialiser: share the machine's cores between workers."""
    import numba

    try:
        numba.set_num_threads(max(1, int(n_threads)))
    except ValueError:
        pass


def job(args: dict) -> dict:
    """Pool target.  ``args`` (picklable): ``{"kind": "basin", "root",
    "params": WorldParams, "basin": record, "quicklook": path or None}``
    runs :func:`run_basin` and rasterises the result into the ``fine/``
    memmaps (``rasterize.write_result``); ``{"kind": "face", "root",
    "params", "face"}`` writes the plain upsample of a whole face
    (``rasterize.write_base_face``).  Returns a stats dict."""
    from . import rasterize

    kind = args.get("kind", "basin")
    root = args["root"]
    params = args["params"]
    if kind == "face":
        t0 = time.time()
        grid, fields, derived = coarse_inputs(root, params)
        cells = rasterize.write_base_face(root, params, int(args["face"]), fields, derived)
        return {"kind": "face", "face": int(args["face"]), "cells": cells, "seconds": time.time() - t0}
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


__all__ = ["BasinResult", "RESULT_FIELDS", "coarse_inputs", "build_mask", "BasinErosionParams", "basin_erosion_params", "exit_cells", "run_basin", "basin_quicklook", "pool_init", "job"]
