"""Biomes of a zoom level, classified on the level's own terrain.

The planet's classes are a 9.8 km cell's (or a 1.2 km one's): a coarse
cell that holds a lake is ``lake`` all over, and every 5 m cell of a zoom
under it was drawn as water-coloured ground with no forest.  A level
classifies itself the way ``derive`` classifies the planet
(:func:`globe.derive.biomes.classify`: the Whittaker base on temperature
and rain, then riparian, alpine, cliff, wetland, lake and ocean over it),
from

* the temperature at the level's own height (the climate's, its lapse
  moved from the pre-erosion bedrock to the level's surface),
* the rain (the climate's wetness, sampled),
* the level's water: its lakes (the flood surface), its ocean, its streams
  (the flood-tree flux) -- with the riparian and wetland bands in metres
  (:data:`RIPARIAN_M`, :data:`WETLAND_M`), not the planet's coarse cells,
* the level's slope, cliffs past the vegetation's rock slope for cells of
  its size (:func:`globe.erosion.vegetation.rock_slope_deg`).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage

#: a stream whose upstream area passes this (km^2) has a riparian band
RIPARIAN_KM2 = 5.0
#: riparian band either side of such a stream, metres (at least a cell)
RIPARIAN_M = 60.0
#: wetland band around a lake of at least WETLAND_LAKE_KM2, metres (at least a cell):
#: a 5 m level's pools and puddles would otherwise ring a quarter of it in wetland
WETLAND_M = 50.0
WETLAND_LAKE_KM2 = 0.01

_WET: dict = {}


def _wetness_field(root: Path, params):
    """The climate's wetness (1 = the land mean rain rate) as a coarse field,
    per world root."""
    from ..derive import biomes as bm
    from ..field import FaceField
    from ..refine import basin_job as bj

    key = str(Path(root).resolve())
    if key not in _WET:
        grid, fields, derived = bj.coarse_inputs(root, params)
        P = fields["precip"]
        area_factor = (grid.interior_cell_area / grid.cell_size_m ** 2).astype(np.float32)
        surf = derived["surface"].interior
        wet = bm.wetness(P.interior, area_factor, params.climate.precip_mean, land=surf >= 0.0)
        _WET.clear()
        _WET[key] = (FaceField.from_interior(grid, wet.astype(np.float32), name="wetness"),
                     bm.effective_alpine_min(surf[surf >= 0.0], params.derive))
    return _WET[key]


def level_biomes(root: Path, params, geo, arrays: dict, cell_m: float, rain_cell: float) -> np.ndarray:
    """uint8 biome codes over a level's work array (``arrays`` as
    :func:`globe.zoom.bake.load_level` holds them: height, sediment,
    water_surface, flux, ocean)."""
    from ..derive import biomes as bm
    from ..erosion import vegetation as veg
    from ..refine.upsample import sample
    from . import bake as zb

    root = Path(root)
    dp = params.derive
    grid = params.coarse_grid()
    surface = (np.asarray(arrays["height"], np.float32) + np.asarray(arrays["sediment"], np.float32))
    ocean = np.asarray(arrays["ocean"], bool)
    clim = zb.climate_inputs(root, params, geo.win, grid)
    if clim is None:
        raise ValueError(f"{root}: no temperature or biome fields to classify a level with")
    T = zb.temperature_at(params, clim["temp0"], surface)
    wet_f, alpine_min = _wetness_field(root, params)
    wet = np.maximum(sample(wet_f, geo.win, order=1), 0.0)
    Pcm = bm.precip_cm(wet, dp.precip_scale_cm, dp.precip_gamma, getattr(dp, "precip_max_cm", 0.0))
    gy, gx = np.gradient(surface.astype(np.float64), cell_m)
    slope = np.hypot(gx, gy).astype(np.float32)
    cliff_slope = math.tan(math.radians(1.15 * veg.rock_slope_deg(cell_m)))
    ws = np.maximum(np.asarray(arrays["water_surface"], np.float32), surface)
    lake = (ws - surface > float(params.hydro.lake_min_depth)) & ~ocean
    area_km2 = np.asarray(arrays["flux"], np.float64) / max(float(rain_cell), 1e-30) * (cell_m / 1000.0) ** 2
    stream = (area_km2 >= RIPARIAN_KM2) & ~ocean & ~lake
    river_near = bm.near(stream, max(1, int(round(RIPARIAN_M / cell_m))))
    lab, n = ndimage.label(lake, structure=np.ones((3, 3), bool))
    big = np.bincount(lab.ravel(), minlength=n + 1) * (cell_m / 1000.0) ** 2 >= WETLAND_LAKE_KM2
    big[0] = False
    lake_near = bm.near(big[lab], max(1, int(round(WETLAND_M / cell_m))))
    return bm.classify(T, Pcm, surface, slope, lake, river_near, lake_near, dp, cliff_slope, alpine_min, ocean=ocean)


def level_biomes_of(zdir: Path, R: int, params=None) -> np.ndarray:
    """:func:`level_biomes` of a saved level (``<world>/zoom/<name>/L{R}.npz``),
    for levels baked before the bake classified them."""
    from ..config import WorldParams
    from ..io.world_store import WorldStore
    from . import bake as zb

    zdir = Path(zdir)
    root = zdir.parent.parent
    if params is None:
        params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    res = zb.load_level(zdir, int(R))
    rain = res.stats.get("rain_cell") or 0.0
    if rain <= 0.0:
        raise ValueError(f"{zdir}/L{R}: no rain_cell to scale its flux by")
    return level_biomes(root, params, res.geo, res.arrays, float(res.stats["cell_m"]), float(rain))


__all__ = ["RIPARIAN_KM2", "RIPARIAN_M", "WETLAND_M", "WETLAND_LAKE_KM2", "level_biomes", "level_biomes_of"]
