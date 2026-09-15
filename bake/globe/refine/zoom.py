"""Zoom windows: refinement far below the planet grid (docs/zoom-windows.md).

The refine stage's basin job was written for R = 2 (4.9 km on the earth
preset).  Measured on catchments of earth-v9 at R = 32 and 128 (305 m and
76 m) it needs two things changed, both kept here rather than in the stage
until zoom windows are one:

* :data:`ZOOM_EROSION` -- the particle kernel with McDonald's own
  SimpleHydrology settings.  The planet's (tuned on 9.8 km cells, with metre
  caps that bind hard there) leave a 76 m window either smooth, or dammed
  with pits, or incising without limit; these give the best river network
  of everything measured and the first branching valleys.
* :data:`ZOOM_REFINE` -- detail noise strong enough to give erosion relief
  to organise.
* :func:`smooth_drift` -- a drift correction without the coarse grid in it.
  ``basin_job.block_drift`` interpolates per-coarse-cell means bilinearly:
  its gradient jumps on every block line, which prints the coarse grid into
  a hillshade and cuts pits where a channel crosses a kink (190 lakes
  against 7 on a pit-free surface at 305 m).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

#: erosion overrides for a zoom window: McDonald's SimpleHydrology (master,
#: read 2026-09-15) mapped onto the kernel's cell units.  His heights are
#: fractions of an 80-cell ``mapscale``, so his constants are cell units as
#: ours are.  deposition 0.1, momentum transfer 1 and the discharge EMA 0.1
#: are already the kernel's; the rest:
ZOOM_EROSION = {
    "erodibility": 1.0,          # c_eq = (1 + entrainment erf(0.4 q)) x drop
    "k_disc": 10.0,              # entrainment 10
    "disc_exponent": 0.0,        # erf, not a power law
    "disc_saturation": 32.0,     # his 1 / 0.4 is in particle volume per cycle; ours is closer to upstream area (a guess)
    "max_steps": 500,            # maxAge 500: bounds a walk, and the cost of a window with it
    "min_volume_frac": 0.01,     # minVol 0.01 of a unit spawn volume
    "friction": 0.0,             # full inertia: speed accumulates, the step is normalised
    "evap_rate": 0.001 / 1.2,    # evapRate 0.001 per step; the kernel multiplies by dt
    "max_erode": 1e9,            # no caps: his mass transfer is effD x (c_eq - sediment), unbounded
    "iter_erode": 1e9,
    "iter_deposit": 1e9,
    "thermal_max": 1e9,
    "thermal_rate": 0.8,         # settling 0.8
    "talus_slope_soft": 0.8,     # maxdiff 0.01 x mapscale 80 = 0.8 cell slope
    "talus_slope_hard": 0.8,
    "creep_rate": 0.0,           # none; creep is a diffusion in cell units and erases the relief
    # lakes in erosion (ErosionState.refresh_lakes_window): pools fill with the
    # load dropped at their shores -- lake area 9 -> 4 % of a catchment over
    # 300 iterations on a branching network.  McDonald removed pools from his
    # 2023 model as ill-posed; measured here they help
    "window_lakes": True,
    "flood_every": 5,
    # discharge scales in cells, as McDonald's are (his discharge is the
    # volume of 512 particles per cycle over a 512^2 map): erf(0.4 q) is
    # ~1280 cells of upstream rain, the momentum push half strength at ~512.
    # A fixed disc_saturation 32 is 3,300 km^2 of rain on the earth preset:
    # no stream in a zoom window reached the boost, water ran off as sheets.
    # 305 m mountain catchment, 150 iterations: 3 km relief p90 652 -> 1071 m,
    # a branching incised network; with the momentum scale lakes 2540 -> 1889
    "disc_saturation_cells": 1280.0,
    "momentum_saturation_cells": 512.0,
    # gravity against inertia as his: speed renormalised to sqrt(2) a step,
    # gravity 1 (ours: unit step, dt 1.2): slope_gain 1 / (1.2 sqrt 2)
    "slope_gain": 0.589,
    # soillib's pit-free limits (erosion <= 0.25 L downhill slope, deposition
    # <= 0.25 L 0.3 per iteration): lakes 1915 -> 512 at 150 iterations and
    # 630 at 400 with the relief back (p90 751 m); the erosion limit is what
    # does it (deposition limit alone 1574)
    "slope_limit_erode": 0.25,
    "slope_limit_deposit": 0.25,
}

#: refine overrides for a zoom window: detail noise at 3x the stage's
#: amplitude -- the upsample has no relief below a parent cell, and
#: McDonald's erosion organises noise into branching valleys (on a mountain
#: catchment the drainage-relief prototype cut parallel corduroy instead)
ZOOM_REFINE = {
    "detail_amp": 3.0,
}


def zoom_params(params, R: int, **erosion):
    """``params`` (a WorldParams) for a zoom window at refinement ``R``: the
    zoom erosion profile and refine overrides, then any ``erosion`` given."""
    eo = dict(ZOOM_EROSION)
    eo.update(erosion)
    if "max_steps" in eo:
        eo["max_steps"] = int(eo["max_steps"])
    return params.with_overrides(world={"R": int(R)}, refine=dict(ZOOM_REFINE), erosion=eo)


def smooth_drift(delta: np.ndarray, cells: np.ndarray, R: int, tol: float = 0.05, max_passes: int = 16) -> np.ndarray:
    """Smooth field ``F`` removing the coarse-scale part of ``delta`` over
    ``cells``: repeated normalised Gaussian low-passes (sigma = ``R``, one
    coarse cell) of the residual ``delta - F``, until the largest residual
    mean over a coarse block of ``cells`` is below ``tol`` or ``max_passes``.
    Same signature and meaning as ``basin_job.block_drift`` (subtract ``F``),
    but ``F`` has no block structure, so it neither prints the coarse grid
    nor bends a channel at a block line.  float64, ``delta.shape``."""
    d0 = np.where(cells, delta, 0.0).astype(np.float64)
    w = cells.astype(np.float64)
    F = np.zeros(delta.shape, dtype=np.float64)
    if not cells.any():
        return F
    sig = float(max(R, 1))
    norm = ndimage.gaussian_filter(w, sig, mode="nearest")
    n0, n1 = (delta.shape[0] // R) * R, (delta.shape[1] // R) * R
    cnt = w[:n0, :n1].reshape(n0 // R, R, n1 // R, R).sum(axis=(1, 3))
    for _ in range(int(max_passes)):
        res = d0 - np.where(cells, F, 0.0)
        blk = res[:n0, :n1].reshape(n0 // R, R, n1 // R, R).sum(axis=(1, 3))
        full = cnt >= 0.5 * R * R
        if not full.any() or float(np.abs(blk[full] / cnt[full]).max()) < tol:
            break
        F += ndimage.gaussian_filter(res, sig, mode="nearest") / np.maximum(norm, 1e-6)
    return F


__all__ = ["ZOOM_EROSION", "ZOOM_REFINE", "smooth_drift", "zoom_params"]
