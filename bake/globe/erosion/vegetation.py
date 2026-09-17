"""Vegetation that grows with the terrain as it erodes, after McDonald's
SimpleHydrology trees, as a canopy cover per cell in [0, 1] so it means the
same on a 76 m cell as on a 5 m one.

Each iteration (:func:`grow`):

* a bare cell the ground can carry a forest on gets saplings at random
  (``seed``), and trees spread from their neighbours (``spread``);
* cover grows towards the cell's capacity (``grow``, logistic);
* trees die at random (``death``), and quickly (``kill``) wherever the cover
  stands above the capacity: in a pool, in a stream's channel, on ground
  too steep to hold soil, past the tree line.

The capacity (:func:`capacity`) is the climate's forest (the biome's
vegetation factor, the tree line in lapse-corrected temperature), the slope
(the viewer's rock threshold, which rises as cells shrink), the share of the
cell a channel leaves (a stream's width grows as the square root of its
upstream area, so a 5 m cell of a creek is all water and a 76 m one mostly
bank), standing water, and the soil's moisture: ridges and dry spurs carry
less, the ground along the streams more.

The cover feeds back into the erosion as roots (:func:`roots`): the share of
the particles' exchange with the bed they hold back
(``erosion.particle.trace_particles``), McDonald's ``depositionRate *
(1 - treedensity)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

#: biome codes whose vegetation factor is decided by the height, not the climate
#: (ice, tundra, alpine, cliff) or by the water (riparian, wetland, lake, ocean) at a
#: coarse cell: a fine cell below the tree line, or on dry ground, takes these instead
_BIOME_OVERRIDE = {0: 0.6, 1: 0.65, 2: 0.65, 12: 0.65, 13: 0.65, 14: 0.85, 15: 0.85, 16: 0.6}


@dataclass(frozen=True)
class VegParams:
    #: share of the bed exchange a full canopy holds back
    root_hold: float = 0.5
    #: logistic growth per iteration
    grow: float = 0.12
    #: growth per iteration from the neighbours' cover (seeds falling nearby)
    spread: float = 0.08
    #: chance per iteration that a bare cell of full capacity gets saplings
    seed: float = 0.004
    #: cover a seeding starts at (x capacity)
    sapling: float = 0.2
    #: random death per iteration
    death: float = 0.01
    #: share of the cover above capacity that dies per iteration
    kill: float = 0.5
    #: mean annual temperature (C) of the tree line, and the half-width of its fade
    tree_line_c: float = -1.0
    tree_line_fade_c: float = 3.0
    #: standing water deeper than this (m) drowns the trees
    pool_m: float = 0.5
    #: channel width (m) = width_k * sqrt(upstream km^2)
    width_k: float = 3.0
    #: soil moisture: dry at this upstream area (km^2), wet at the second
    dry_km2: float = 0.0005
    wet_km2: float = 0.1
    #: capacity on the driest ground (x the wettest)
    dry_share: float = 0.7


def rock_slope_deg(cell_m: float) -> float:
    """The slope past which a cell of ``cell_m`` holds no soil: ~35 degrees at 5 m,
    lower on coarser cells, whose mean slope hides the cliffs (the viewer's
    satellite threshold)."""
    return float(np.clip(35.0 * (max(cell_m, 5.0) / 5.0) ** -0.25, 4.0, 38.0))


def climate_forest(biome: np.ndarray, veg_factor: np.ndarray) -> np.ndarray:
    """The forest the climate allows per cell, from the (coarse) biome codes:
    their vegetation factor, the height-made and water-made classes replaced
    (:data:`_BIOME_OVERRIDE`) so the fine terrain decides those."""
    table = np.asarray(veg_factor, np.float32).copy()
    for code, v in _BIOME_OVERRIDE.items():
        if code < table.size:
            table[code] = v
    b = np.clip(np.asarray(biome, np.int64), 0, table.size - 1)
    return table[b]


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def capacity(p: VegParams, forest: np.ndarray, temp_c: np.ndarray, surface_m: np.ndarray, pool_m: np.ndarray | None,
             area_km2: np.ndarray, cell_m: float, active: np.ndarray) -> np.ndarray:
    """The cover each cell can carry now (float32, 0 off ``active``).
    ``forest`` from :func:`climate_forest`, ``temp_c`` the mean annual
    temperature at the cell's height, ``pool_m`` the standing water depth (None
    = none), ``area_km2`` the upstream area."""
    gy, gx = np.gradient(np.asarray(surface_m, np.float64), cell_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    thr = rock_slope_deg(cell_m)
    k = forest * (1.0 - _smoothstep(0.75 * thr, 1.15 * thr, slope))
    k = k * _smoothstep(p.tree_line_c - p.tree_line_fade_c, p.tree_line_c + p.tree_line_fade_c, temp_c)
    a = np.maximum(area_km2, 1e-9)
    moist = np.clip(np.log(a / p.dry_km2) / math.log(p.wet_km2 / p.dry_km2), 0.0, 1.0)
    k = k * (p.dry_share + (1.0 - p.dry_share) * moist)
    k = k * (1.0 - np.clip(p.width_k * np.sqrt(a) / cell_m, 0.0, 1.0))
    if pool_m is not None:
        k = k * (1.0 - _smoothstep(0.5 * p.pool_m, p.pool_m, pool_m))
    return np.where(active, np.clip(k, 0.0, 1.0), 0.0).astype(np.float32)


@njit(cache=True, parallel=True)
def _grow(v, cap, u, grow, spread, seed, sapling, death, kill, out):
    n0, n1 = v.shape
    for i in prange(n0):
        for j in range(n1):
            k = cap[i, j]
            x = v[i, j]
            if k <= 0.0:
                out[i, j] = x * (1.0 - kill)
                continue
            s = 0.0
            c = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    a, b = i + di, j + dj
                    if 0 <= a < n0 and 0 <= b < n1:
                        s += v[a, b]
                        c += 1

            nb = s / c if c > 0 else 0.0
            if x < k:
                x += (grow * x + spread * nb) * (k - x)
                if u[i, j] < seed * k and x < sapling * k:
                    x = sapling * k
            else:
                x -= kill * (x - k)
            x *= 1.0 - death
            out[i, j] = min(max(x, 0.0), 1.0)


def grow(p: VegParams, cover: np.ndarray, cap: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One iteration of the cover (float32, same shape)."""
    out = np.empty(cover.shape, np.float32)
    u = rng.random(cover.shape, dtype=np.float32)
    _grow(np.asarray(cover, np.float32), np.asarray(cap, np.float32), u, np.float32(p.grow), np.float32(p.spread), np.float32(p.seed),
          np.float32(p.sapling), np.float32(p.death), np.float32(p.kill), out)
    return out


def roots(p: VegParams, cover: np.ndarray) -> np.ndarray:
    """The kernel's roots field (``(1, NE, NE)`` float32) of a window's cover."""
    return np.ascontiguousarray((p.root_hold * np.clip(cover, 0.0, 1.0)).astype(np.float32)[None])


__all__ = ["VegParams", "rock_slope_deg", "climate_forest", "capacity", "grow", "roots"]
