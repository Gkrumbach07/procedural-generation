"""Soil (PLAN.md section 11): the fine ``sediment`` depth from erosion is
the alluvium map and ``hardness`` the rock type; both are already fine
fields written by refine and packed into ``layers.png`` by tiles.  This
module only adds the small derived quantities the derive stage uses:

* :func:`soil_factor` — multiplier on the vegetation density (thin soil on
  bare rock supports less growth);
* :func:`alluvium_mask` — deep sediment on gentle ground (valley floors,
  fans, deltas);
* :func:`soil_class` — a 4-class summary (rock / thin / soil / alluvium)
  for debugging or later tile layers.
"""
from __future__ import annotations

import numpy as np

ROCK = 0
THIN = 1
SOIL = 2
ALLUVIUM = 3
SOIL_CLASS_NAMES = ("rock", "thin", "soil", "alluvium")


def soil_factor(sediment: np.ndarray, full_depth_m: float = 1.0, floor: float = 0.7) -> np.ndarray:
    """``floor + (1 - floor) * min(sediment / full_depth_m, 1)`` in
    ``[floor, 1]`` (float32)."""
    s = np.asarray(sediment, dtype=np.float32)
    t = np.clip(s / np.float32(max(full_depth_m, 1e-6)), 0.0, 1.0)
    return (np.float32(floor) + np.float32(1.0 - floor) * t).astype(np.float32)


def alluvium_mask(sediment: np.ndarray, slope: np.ndarray, min_depth_m: float = 0.5, max_slope: float = 0.3) -> np.ndarray:
    """Deep sediment (``>= min_depth_m``) on gentle ground (``slope <=
    max_slope`` rise/run)."""
    return (np.asarray(sediment, dtype=np.float32) >= np.float32(min_depth_m)) & (np.asarray(slope, dtype=np.float32) <= np.float32(max_slope))


def soil_class(sediment: np.ndarray, hardness: np.ndarray | None, slope: np.ndarray, thin_depth_m: float = 0.1, min_depth_m: float = 0.5, max_slope: float = 0.3) -> np.ndarray:
    """uint8: 0 rock (no sediment, or hard rock with < ``thin_depth_m``),
    1 thin (< ``thin_depth_m``), 2 soil, 3 alluvium (:func:`alluvium_mask`)."""
    s = np.asarray(sediment, dtype=np.float32)
    out = np.full(s.shape, SOIL, dtype=np.uint8)
    thin = s < np.float32(thin_depth_m)
    out[thin] = THIN
    rock = s <= 0.0
    if hardness is not None:
        rock |= thin & (np.asarray(hardness, dtype=np.float32) > 0.6)
    out[rock] = ROCK
    out[alluvium_mask(s, slope, min_depth_m, max_slope)] = ALLUVIUM
    return out


__all__ = ["ROCK", "THIN", "SOIL", "ALLUVIUM", "SOIL_CLASS_NAMES", "soil_factor", "alluvium_mask", "soil_class"]
