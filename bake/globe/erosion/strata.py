"""Beds: the bedrock's hardness as the erosion cuts down through layered rock.

Tectonics hands the stage one hardness a cell, and its fabric
(``tectonics.strata_amp``) is banded in map view only: hard and soft belts
along the lines of equal crust age, the same rock however deep the stage
cuts.  That is rock standing on end.  Half the land's rock is not: it is
sedimentary cover, laid flat or nearly, a few hundred metres of sandstone or
limestone on as much shale, and what erosion makes of it is the scarp --
the hard bed holds a plateau up until the soft one under it is cut away, the
edge retreats, and the country is steps (the Colorado Plateau, the cuestas
of the Paris basin, the Niagara escarpment).  For that the hardness has to
change *with depth*, at the cell, while the stage runs.

So the state counts the rock each cell has lost (``ErosionState.eroded``:
what the particles, the mass wasting and the ice took off the bedrock, not
its change of height -- uplift, the rebound and the datum move the column
without wearing it), and the hardness the kernel sees is tectonics' own
plus the bed's at that depth:

    hardness = clip(hard0 + strata_amp * bed(eroded + warp), 0, top)

``bed`` is +1 in a hard bed and -1 in a soft one, by a table of bed tops
drawn once for the world (:func:`bed_table`).  ``warp`` (metres,
:func:`warp`) is the structure: where the same bed lies deeper or
shallower, a smooth field of the place on the sphere.  With none every
contact would come to the surface at one depth of erosion the world over;
with it the beds dip a few metres a kilometre, and their outcrops are belts
that curve with the structure.

Measured before this existed (earth-v32, one 1.2 km window of sedimentary
cover re-eroded with flat beds 80-260 m thick): nothing to see.  A planet
level cuts its bedrock in its channels only (median 0 m, 128 m at the 95th
percentile, in its 80 iterations); the relief is the coarse stage's, which
is where the beds have to be.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..derive.geology import _noise3

#: rng sub-key of the bed table and the structure
STRATA_KEY = 4_815_162
#: the table reaches this deep (m): more rock than the stage takes off anywhere
TABLE_DEPTH_M = 40_000.0
#: no bed makes the rock harder than this (at 1 the kernel cannot erode it at all)
HARD_TOP = 0.95


def bed_table(seed: int, bed_m: float) -> tuple[np.ndarray, np.ndarray]:
    """``(tops, sign)``: the depth (m) of each bed's base below the datum,
    increasing, and what the bed does to the hardness -- hard and soft in
    turn, each 0.4 to 1.6 of ``bed_m`` thick and 0.6 to 1 of full strength.
    One table for the world: a formation is the same rock where it crops out
    again."""
    rng = np.random.default_rng([int(seed), STRATA_KEY])
    n = int(math.ceil(TABLE_DEPTH_M / max(0.4 * float(bed_m), 1.0))) + 2
    thick = rng.uniform(0.4, 1.6, n) * float(bed_m)
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * rng.uniform(0.6, 1.0, n)
    return np.cumsum(thick), sign


def bed(depth_m: np.ndarray, tops: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """The bed at ``depth_m`` below the datum: its ``sign`` (float32, the
    shape of ``depth_m``).  Above the datum is the first bed."""
    k = np.clip(np.searchsorted(tops, np.asarray(depth_m, np.float64), side="right"), 0, tops.size - 1)
    return sign[k].astype(np.float32)


@njit(cache=True, parallel=True)
def _warp_kernel(p, freq, seed, out):
    n = p.shape[0]
    for c in prange(n):
        x, y, z = p[c, 0] * freq, p[c, 1] * freq, p[c, 2] * freq
        out[c] = (_noise3(x, y, z, seed) + 0.5 * _noise3(2.0 * x + 17.3, 2.0 * y - 5.1, 2.0 * z + 9.7, seed + 1)) / 1.5


def warp(grid, seed: int, warp_m: float, warp_km: float) -> np.ndarray:
    """The structure: how much deeper (m) the beds lie at each cell,
    ``+-warp_m`` over ``warp_km`` (extended ``(6, NE, NE)`` float32).  A
    value noise of the place on the sphere, two octaves, so it has no seam at
    a cube edge and is the same whatever the grid."""
    if float(warp_m) <= 0.0:
        return np.zeros((6, grid.NE, grid.NE), np.float32)
    p = np.ascontiguousarray(grid.centers.reshape(-1, 3), np.float64)
    out = np.empty(p.shape[0], np.float64)
    freq = float(grid.R_planet) / max(float(warp_km) * 1000.0, 1.0)
    _warp_kernel(p, freq, int(seed) + STRATA_KEY, out)
    return (float(warp_m) * out).reshape(6, grid.NE, grid.NE).astype(np.float32)


def hardness(hard0: np.ndarray, eroded_m: np.ndarray, phase_m: np.ndarray, amp: float, tops: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Tectonics' hardness with the bed's at the depth the cell has been cut
    to (float32, see the module docstring).  Rock tectonics made harder than
    :data:`HARD_TOP` keeps what it had."""
    h0 = np.asarray(hard0, np.float32)
    h = h0 + np.float32(amp) * bed(np.asarray(eroded_m, np.float64) + np.asarray(phase_m, np.float64), tops, sign)
    return np.clip(h, 0.0, np.maximum(h0, np.float32(HARD_TOP))).astype(np.float32)


def setup(state, grid, params) -> None:
    """Give the state its beds (``erosion.strata_amp`` > 0): the hardness it
    arrived with, the rock lost so far (none), and the table and structure."""
    ep = params.erosion
    amp = float(getattr(ep, "strata_amp", 0.0))
    if amp <= 0.0:
        return
    seed = int(params.world.seed)
    tops, sign = bed_table(seed, float(ep.strata_bed_m))
    state.hard0 = state.hardness.copy()
    state.eroded = np.zeros_like(state.height)
    state.strata = (amp, tops, sign, warp(grid, seed, float(ep.strata_warp_m), float(ep.strata_warp_km)))


def refresh(state) -> None:
    """The hardness of the beds now at the surface (in place)."""
    amp, tops, sign, phase = state.strata
    state.hardness[...] = hardness(state.hard0, state.eroded.astype(np.float64) * float(state.height_unit_m), phase, amp, tops, sign)


__all__ = ["STRATA_KEY", "HARD_TOP", "bed_table", "bed", "warp", "hardness", "setup", "refresh"]
