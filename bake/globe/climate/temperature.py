"""Temperature and evaporation (PLAN.md section 7).

``T(lat, h) = T_eq − k_lat·(|lat| / (π/2))^1.5 − lapse·max(h, 0)/1000``

Latitude is measured from the +Z pole (:meth:`Grid.latitude`) and
normalised so that ``|lat| = π/2`` (the pole) contributes exactly ``k_lat``:
with the PLAN 14 defaults a sea-level pole sits at ``28 − 45 = −17 °C``.
``lapse`` is °C per km of altitude above sea level (bedrock below sea
level is treated as sea level).

Evaporation is the dimensionless multiplier ``evap = k_evap·max(T, 0)``
(docs/DEVELOPING.md): ~1 at ``T_eq`` at sea level, 0 where ``T ≤ 0``; the
erosion kernel decays particles by ``volume *= 1 − dt·evap_rate·evap``.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import ClimateParams
from ..cubesphere import Grid


def temperature(grid: Grid, height_m: np.ndarray, cp: ClimateParams) -> np.ndarray:
    """Surface temperature (°C) on every extended cell.  ``height_m`` is the
    extended ``(6, NE, NE)`` surface (metres, sea level 0).  Returns float32
    ``(6, NE, NE)``; halos are analytic (no exchange needed) as long as
    ``height_m``'s halo is filled."""
    lat = grid.latitude()  # radians, extended
    lat_n = np.abs(lat) / (0.5 * math.pi)
    h = np.maximum(np.asarray(height_m, dtype=np.float64), 0.0)
    T = cp.T_eq - cp.k_lat * lat_n**1.5 - cp.lapse * h / 1000.0
    return T.astype(np.float32)


def retarget(temperature_c: np.ndarray, from_height_m, to_height_m, cp: ClimateParams) -> np.ndarray:
    """The same temperature field referenced to another surface: remove the
    lapse term of ``from_height_m`` and apply it at ``to_height_m`` (both in
    metres, below sea level counts as sea level).  ``retarget(T, h, 0)`` is
    the sea-level field, ``retarget(T0, 0, h)`` puts it back at ``h``.

    ``derive`` needs this because ``climate`` runs before ``erosion`` and its
    ``temperature`` therefore sits on the *bedrock* surface, hundreds of
    metres away from the terrain the biomes are classified on."""
    dh = np.maximum(np.asarray(from_height_m, dtype=np.float64), 0.0) - np.maximum(np.asarray(to_height_m, dtype=np.float64), 0.0)
    return (np.asarray(temperature_c, dtype=np.float64) + cp.lapse * dh / 1000.0).astype(np.float32)


def evaporation(temperature_c: np.ndarray, cp: ClimateParams) -> np.ndarray:
    """``k_evap·max(T, 0)`` as float32 (same shape as the input)."""
    return (cp.k_evap * np.maximum(np.asarray(temperature_c, dtype=np.float64), 0.0)).astype(np.float32)


__all__ = ["temperature", "retarget", "evaporation"]
