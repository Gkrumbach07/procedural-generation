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
    # McDonald's water layer: with `window_lake_evap` (set per level by
    # `zoom_params`, the planet's coarse-cell rate over the fine cells) a
    # depression holds what its catchment brings less what evaporates off it,
    # instead of filling to its rim, so a divot with no catchment is not a lake.
    # Off at every level (2026-09-16): the lake treatment drops a particle's load
    # on the shore and leaves the bed untouched, so a pit grows a rim and never
    # fills.  On two earth-v9 windows, holes more than 5 m below all eight
    # neighbours fell 6-160x with it off at 305 m and 76 m (lakes up to 861 m
    # deep over a planet level whose lakes there are <= 100 m), relief and
    # valley depth unchanged within run-to-run noise; at 1.2 km 32.9 -> 0.14
    # per 10^4 land cells.  The routing flood still carries water across a
    # depression, and the finish floods the lakes
    "window_lakes": False,
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

#: discharge scales as upstream *area* (km^2): ``zoom_params`` turns them
#: into cells at each level, capped at ZOOM_EROSION's cell counts.  120 and 48
#: km^2 are the 305 m level's 1280 and 512 cells, the settings the branching
#: networks were measured at, and the 76 m level stays at its caps.  At 1.2 km
#: the fixed cell counts meant 1,910 and 764 km^2: every stream under that
#: eroded as sheet flow and the planet level had no valleys.  On three 1.2 km
#: windows of earth-v9 (scratch/sweep, 80 iterations) 80 / 32 cells took a
#: mountain's valley depth p90 from 85 to 468 m and its 3.7 km relief p90 from
#: 237 to 410 m
DISC_SATURATION_KM2 = 120.0
MOMENTUM_SATURATION_KM2 = 48.0

#: erosion overrides for a level whose cells are at least COARSE_ZOOM_CELL_M
#: (the 1.2 km planet level), measured on three 1.2 km windows against the
#: zoom profile (scratch/sweep):
#: * k_mom 0.5: half the stored momentum push, which builds sediment ridges
#:   along the particles' streaks (crest -11.7 -> -1.8 m).  Not below: at
#:   305 m / 76 m it cut relief and valley depth 10-40 % with no ridges to
#:   remove (streak crests already incised, -6 to -48 m on a mountain).
COARSE_ZOOM_CELL_M = 600.0
COARSE_ZOOM_EROSION = {
    "k_mom": 0.5,
}

#: soillib's erosion limit where a cell is at least WIDE_SLOPE_LIMIT_CELL_M
#: (1.2 km and 305 m): 0.25 held gentle mountains at 1.2 km to ~1 cm of
#: incision an iteration; 0.5 took a plateau's valley depth 11 -> 18 m and its
#: 3.7 km relief p90 62 -> 103 m, and at 305 m a mountain's 1 km relief p50
#: 132 -> 154 m and 6 km valley depth p90 825 -> 941 m, with holes near zero
#: (1.0 brought the speckle lakes back at 1.2 km).  At 76 m it stays 0.25:
#: 0.5 printed an axis-aligned stair texture on a steep mountain (axis /
#: diagonal spectral power at 2-4 cells 2.4 -> 6.3, straight risers 18 -> 50
#: per 10^4 cells).
WIDE_SLOPE_LIMIT_CELL_M = 150.0
WIDE_SLOPE_LIMIT_ERODE = 0.5

#: refine overrides for a zoom window: detail noise at 3x the stage's
#: amplitude -- the upsample has no relief below a parent cell, and
#: McDonald's erosion organises noise into branching valleys (on a mountain
#: catchment the drainage-relief prototype cut parallel corduroy instead).
#: Hardness low-passed over 1.5 coarse cells and capped at 0.85
#: (basin_job.refined_hardness): tectonics' strata bands printed rings of
#: stipple and rectangular caprock terraces below the coarse grid -- on a
#: ringed window the roughness of hard against soft bands went 1.64 -> 1.04x,
#: and in a controlled 1.2 km rebake grid-aligned riser edges went 344 -> 0
HARDNESS_SMOOTH_CELLS = 1.5
HARDNESS_MAX = 0.85
ZOOM_REFINE = {
    "detail_amp": 3.0,
    "hardness_smooth_cells": HARDNESS_SMOOTH_CELLS,
    "hardness_max": HARDNESS_MAX,
}


def zoom_params(params, R: int, **erosion):
    """``params`` (a WorldParams) for a zoom window at refinement ``R``: the
    zoom erosion profile with its discharge scales in area
    (:data:`DISC_SATURATION_KM2`), the coarse-level overrides where a cell is
    at least :data:`COARSE_ZOOM_CELL_M`, the wider erosion limit where it is
    at least :data:`WIDE_SLOPE_LIMIT_CELL_M`, and the refine overrides; then any
    ``erosion`` given."""
    eo = dict(ZOOM_EROSION)
    cell_m = float(params.world.cell_size_m) / float(R)
    cell_km2 = (cell_m / 1000.0) ** 2
    eo["disc_saturation_cells"] = min(float(ZOOM_EROSION["disc_saturation_cells"]), DISC_SATURATION_KM2 / cell_km2)
    eo["momentum_saturation_cells"] = min(float(ZOOM_EROSION["momentum_saturation_cells"]), MOMENTUM_SATURATION_KM2 / cell_km2)
    if cell_m >= WIDE_SLOPE_LIMIT_CELL_M:
        eo["slope_limit_erode"] = WIDE_SLOPE_LIMIT_ERODE
    if cell_m >= COARSE_ZOOM_CELL_M:
        eo.update(COARSE_ZOOM_EROSION)
    eo.setdefault("window_lake_evap", float(params.hydro.lake_evap) / float(R * R))
    eo.update(erosion)
    if "max_steps" in eo:
        eo["max_steps"] = int(eo["max_steps"])
    return params.with_overrides(world={"R": int(R)}, refine=dict(ZOOM_REFINE), erosion=eo)


def drain_noise(surface: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """The detail noise's own closed depressions, filled to their spill level
    (a priority flood draining to the sea and the array's border): the amount
    to add to the surface, 0 outside them.

    The upsample of a coarser level drains everywhere -- it is the surface a
    flood already shaped -- but ridged noise on top of it invents basins: at
    R = 8 the noise puts 14 % of a tile's land in closed depressions over 2 m
    deep, and 80 iterations drain only part of them (2.6 % left; 200
    iterations, 2.5x the cost, leave 1.2 %).  Filling them first leaves 0.8 %
    and the same relief, and the lakes that remain are the parent's.
    """
    from ..hydro.priority_flood import priority_flood_flat

    drain = np.asarray(ocean, bool).copy()
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    fr = priority_flood_flat(np.ascontiguousarray(surface, np.float32), drain, None)
    return np.maximum(fr.filled.reshape(surface.shape) - surface, 0.0)


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


__all__ = ["ZOOM_EROSION", "ZOOM_REFINE", "COARSE_ZOOM_EROSION", "COARSE_ZOOM_CELL_M", "WIDE_SLOPE_LIMIT_CELL_M", "WIDE_SLOPE_LIMIT_ERODE", "DISC_SATURATION_KM2", "MOMENTUM_SATURATION_KM2",
           "drain_noise", "smooth_drift", "zoom_params"]
