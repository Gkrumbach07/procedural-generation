"""Where a lake stands in its life, and the plants that go with it.

A lake fills in.  A young one is deep and clear, its bed bare rock or coarse
sediment and its shores steep; as its inflow lays sediment down it shallows,
its water warms and greens, reeds take its margins and weed its shallows,
until it is a marsh and then meadow.  :func:`lake_life` scores each lake of a
zoom level on that path (``age`` in [0, 1]) from what the bake leaves:

* how deep it is (``water_surface - surface``): shallow is old;
* how much sediment lies under it against the land around (the level's median):
  a basin choked with more than its share of alluvium is old;
* how much water it takes for its size (the flood tree's area at its
  deepest cell over its own area): a lake fed hard fills fast;
* how warm it is: a cold lake stays clear far longer than a warm one.

From the age come the plants, per cell:

* ``emergent`` -- reeds and sedges standing out of the water at the margin,
  in water shallower than :data:`REED_M` (deeper as the lake ages) and
  denser the older it is;
* ``submerged`` -- weed beds on the bed, over the depth light reaches
  (:data:`PHOTIC_M`, shallower in an old lake's greener water).

Both are cover in [0, 1], like the land's canopy (:mod:`globe.erosion.vegetation`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

#: a lake deeper than this (metres, mean) counts as young
YOUNG_DEPTH_M = 8.0
#: reeds stand in water up to this deep, x (1 + age)
REED_M = 0.7
#: light reaches this far down in a young lake's water; an old lake's is greener
PHOTIC_M = 7.0
#: the coldest and warmest mean temperature for a lake's growth (C)
COLD_C, WARM_C = 2.0, 16.0


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lake_life(surface: np.ndarray, water_surface: np.ndarray, ocean: np.ndarray, sediment: np.ndarray,
              flux: np.ndarray, temp_c: np.ndarray, cell_m: float, rain_cell: float, min_depth: float = 0.5) -> dict:
    """``{age, emergent, submerged, depth, lakes}`` over the arrays given: the
    per-cell age of the lake each cell belongs to (0 off water), the plants on
    it, its depth, and a list of the lakes with their measures."""
    surface = np.asarray(surface, np.float32)
    depth = np.maximum(np.asarray(water_surface, np.float32) - surface, 0.0)
    lake = (depth > float(min_depth)) & ~np.asarray(ocean, bool)
    age = np.zeros(surface.shape, np.float32)
    lakes: list[dict] = []
    if lake.any():
        lab, n = ndimage.label(lake, structure=np.ones((3, 3), bool))
        idx = np.arange(1, n + 1)
        cells = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
        mean_depth = ndimage.mean(depth, lab, idx)
        mean_sed = ndimage.mean(np.asarray(sediment, np.float32), lab, idx)
        max_flux = ndimage.maximum(np.asarray(flux, np.float64), lab, idx)
        mean_T = ndimage.mean(np.asarray(temp_c, np.float32), lab, idx)
        area_km2 = cells * (cell_m / 1000.0) ** 2
        catch_km2 = np.asarray(max_flux) / max(float(rain_cell), 1e-30) * (cell_m / 1000.0) ** 2
        md = np.asarray(mean_depth, np.float64)
        shallow = 1.0 - _smoothstep(0.0, YOUNG_DEPTH_M, md)
        land_sed = np.asarray(sediment, np.float32)[~lake & ~np.asarray(ocean, bool)]
        med = float(np.median(land_sed)) if land_sed.size else 0.0
        infill = _smoothstep(1.0, 3.0, np.asarray(mean_sed, np.float64) / max(med, 0.1))
        supply = np.clip(np.log10(np.maximum(catch_km2 / np.maximum(area_km2, 1e-9), 1.0)) / 2.5, 0.0, 1.0)
        warmth = _smoothstep(COLD_C, WARM_C, np.asarray(mean_T, np.float64))
        score = np.clip(0.45 * shallow + 0.3 * infill + 0.15 * supply + 0.10 * warmth, 0.0, 1.0)
        table = np.concatenate([[0.0], score]).astype(np.float32)
        age = table[lab]
        for k in range(n):
            lakes.append({"cells": int(cells[k]), "area_km2": float(area_km2[k]), "mean_depth_m": float(md[k]),
                          "sediment_m": float(mean_sed[k]), "catchment_km2": float(catch_km2[k]),
                          "temp_c": float(mean_T[k]), "age": float(score[k])})
    grow = _smoothstep(COLD_C - 4.0, WARM_C, np.asarray(temp_c, np.float32))
    reed_depth = REED_M * (1.0 + age)
    emergent = np.where(lake, (0.25 + 0.65 * age) * grow * (1.0 - _smoothstep(0.0, reed_depth, depth)), 0.0)
    photic = PHOTIC_M * (1.0 - 0.55 * age)
    submerged = np.where(lake, (0.20 + 0.70 * age) * grow * (1.0 - _smoothstep(0.0, photic, depth)), 0.0)
    return {"age": age.astype(np.float32), "emergent": np.clip(emergent, 0.0, 1.0).astype(np.float32),
            "submerged": np.clip(submerged, 0.0, 1.0).astype(np.float32), "depth": depth, "lakes": lakes}


def level_lake_life(zdir: Path, R: int, params=None) -> dict:
    """:func:`lake_life` of a saved level (``<world>/zoom/<name>/L{R}.npz``),
    with the climate's temperature at the level's own surface."""
    from ..config import WorldParams
    from ..io.world_store import WorldStore
    from . import bake as zb

    zdir = Path(zdir)
    root = zdir.parent.parent
    if params is None:
        params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    res = zb.load_level(zdir, int(R))
    a = res.arrays
    surf = a["height"] + a["sediment"]
    clim = zb.climate_inputs(root, zb.zoom_params(params, res.geo.R), res.geo.win, params.coarse_grid())
    if clim is None:
        raise ValueError(f"{root}: no climate to age a lake with")
    rain = res.stats.get("rain_cell") or 0.0
    if rain <= 0.0:
        raise ValueError(f"{zdir}/L{R}: no rain_cell to scale its flux by")
    return lake_life(surf, a["water_surface"], a["ocean"], a["sediment"], a["flux"],
                     zb.temperature_at(params, clim["temp0"], surf), float(res.stats["cell_m"]), float(rain),
                     float(params.hydro.lake_min_depth))


__all__ = ["YOUNG_DEPTH_M", "REED_M", "PHOTIC_M", "COLD_C", "WARM_C", "lake_life", "level_lake_life"]
