"""Biomes (PLAN.md section 11): a Whittaker-style lookup on (temperature,
precipitation) with elevation / slope / water overrides, and the
vegetation density.

Biome code table (``uint8``; the same table is appended to
docs/DEVELOPING.md — change both):

| code | name                     | rule                                                  |
|-----:|--------------------------|-------------------------------------------------------|
|  0   | ocean                    | ``surface < 0``                                       |
|  1   | ice                      | T < -12 degC                                          |
|  2   | tundra                   | T < -2, or T < 5 and P < 20 cm                        |
|  3   | boreal_forest            | T < 5, P >= 20                                        |
|  4   | temperate_grassland      | 5 <= T < 20, 25 <= P < 60, T < 13                     |
|  5   | temperate_forest         | 5 <= T < 20, 60 <= P < 180                            |
|  6   | temperate_rainforest     | 5 <= T < 20, P >= 180                                 |
|  7   | desert                   | T >= 5 and P < 25 (P < 30 when T >= 20)               |
|  8   | shrubland                | 5 <= T < 20, 25 <= P < 60, T >= 13                    |
|  9   | savanna                  | T >= 20, 30 <= P < 100                                |
| 10   | tropical_seasonal_forest | T >= 20, 100 <= P < 220                               |
| 11   | tropical_rainforest      | T >= 20, P >= 220                                     |
| 12   | alpine                   | override: surface >= alpine_min_m and T < alpine_T    |
| 13   | cliff                    | override: slope > cliff_slope (rise/run)              |
| 14   | riparian                 | override: within riparian_cells of a channel / river  |
| 15   | wetland                  | override: within wetland_cells of a lake              |
| 16   | lake                     | override: lake cell (water_surface - surface > depth) |

``P`` is annual precipitation in cm derived from the climate ``precip``
volume: ``wetness = precip / (cell_area / cell_size_m^2) / mean``, the
rain rate normalised by its mean over land (``climate.precip_mean`` by
contract; the measured land mean is used when ``land`` is given, so a
slightly off upstream normalisation does not shift the biomes), and
``P = min(precip_scale_cm * wetness ** precip_gamma, precip_max_cm)``.
The default ``gamma = 0.5`` compresses the strongly skewed climate output
(orographic coasts carry 10-30x the mean, dry interiors 0.05x) into the
Whittaker range: a cell at the mean gets ``precip_scale_cm`` (100 cm,
temperate forest), a quarter of the mean 50 cm (grassland / savanna), a
sixteenth 25 cm (desert), 4x the mean 200 cm (rainforest).
Override precedence, lowest to highest: riparian < alpine < cliff <
wetland < lake < ocean.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..field import FaceField

OCEAN = 0
ICE = 1
TUNDRA = 2
BOREAL_FOREST = 3
TEMPERATE_GRASSLAND = 4
TEMPERATE_FOREST = 5
TEMPERATE_RAINFOREST = 6
DESERT = 7
SHRUBLAND = 8
SAVANNA = 9
TROPICAL_SEASONAL_FOREST = 10
TROPICAL_RAINFOREST = 11
ALPINE = 12
CLIFF = 13
RIPARIAN = 14
WETLAND = 15
LAKE = 16
N_BIOMES = 17

#: (code, name, RGB colour for quicklooks, vegetation factor in [0, 1])
BIOMES = [
    (OCEAN, "ocean", (20, 50, 110), 0.0),
    (ICE, "ice", (240, 245, 255), 0.0),
    (TUNDRA, "tundra", (170, 190, 160), 0.15),
    (BOREAL_FOREST, "boreal_forest", (40, 90, 60), 0.65),
    (TEMPERATE_GRASSLAND, "temperate_grassland", (170, 190, 90), 0.4),
    (TEMPERATE_FOREST, "temperate_forest", (60, 130, 60), 0.85),
    (TEMPERATE_RAINFOREST, "temperate_rainforest", (20, 100, 70), 1.0),
    (DESERT, "desert", (230, 210, 140), 0.05),
    (SHRUBLAND, "shrubland", (160, 150, 80), 0.3),
    (SAVANNA, "savanna", (200, 180, 80), 0.45),
    (TROPICAL_SEASONAL_FOREST, "tropical_seasonal_forest", (80, 150, 50), 0.85),
    (TROPICAL_RAINFOREST, "tropical_rainforest", (10, 110, 40), 1.0),
    (ALPINE, "alpine", (150, 140, 130), 0.15),
    (CLIFF, "cliff", (110, 100, 95), 0.03),
    (RIPARIAN, "riparian", (50, 140, 90), 0.9),
    (WETLAND, "wetland", (90, 140, 120), 0.6),
    (LAKE, "lake", (70, 130, 230), 0.0),
]
NAMES = tuple(b[1] for b in BIOMES)
PALETTE = np.array([b[2] for b in BIOMES], dtype=np.uint8)
VEG_FACTOR = np.array([b[3] for b in BIOMES], dtype=np.float32)
assert [b[0] for b in BIOMES] == list(range(N_BIOMES))


# --------------------------------------------------------------------------
# precipitation units
# --------------------------------------------------------------------------
def wetness(precip: np.ndarray, area_factor: np.ndarray, precip_mean: float, land: np.ndarray | None = None) -> np.ndarray:
    """Climate ``precip`` volume -> dimensionless wetness, 1 at the land
    *mean* rain rate.  ``area_factor = cell_area / cell_size_m**2`` (same
    shape or broadcastable).  The reference is ``precip_mean`` (the
    climate contract: land mean rate = ``climate.precip_mean``); with
    ``land`` given it is the measured mean rate over land instead (equal to
    ``precip_mean`` up to the upstream stage's rounding)."""
    rate = np.asarray(precip, dtype=np.float32) / np.maximum(np.asarray(area_factor, dtype=np.float32), 1e-6)
    ref = float(max(precip_mean, 1e-9))
    if land is not None:
        land = np.asarray(land, dtype=bool)
        if land.any():
            m = float(np.mean(rate[land]))
            if m > 0:
                ref = m
    return (rate / np.float32(ref)).astype(np.float32)


def precip_cm(wet: np.ndarray, precip_scale_cm: float, precip_gamma: float, precip_max_cm: float | None = None) -> np.ndarray:
    """Wetness -> annual precipitation in cm for the Whittaker lookup:
    ``min(precip_scale_cm * wetness ** precip_gamma, precip_max_cm)``
    (no cap when ``precip_max_cm`` is None / <= 0)."""
    w = np.maximum(np.asarray(wet, dtype=np.float32), 0.0)
    P = np.float32(precip_scale_cm) * np.power(w, np.float32(precip_gamma))
    if precip_max_cm is not None and precip_max_cm > 0:
        P = np.minimum(P, np.float32(precip_max_cm))
    return P.astype(np.float32)


# --------------------------------------------------------------------------
# lookup + overrides
# --------------------------------------------------------------------------
def whittaker(T: np.ndarray, P_cm: np.ndarray) -> np.ndarray:
    """Base biome from mean temperature (degC) and precipitation (cm/yr);
    the rules of the module table (no overrides).  uint8, same shape."""
    T = np.asarray(T, dtype=np.float32)
    P = np.asarray(P_cm, dtype=np.float32)
    cold = T < 5.0
    temperate = (T >= 5.0) & (T < 20.0)
    tropical = T >= 20.0
    conds = [
        T < -12.0,
        T < -2.0,
        cold & (P < 20.0),
        cold,
        temperate & (P < 25.0),
        temperate & (P < 60.0) & (T < 13.0),
        temperate & (P < 60.0),
        temperate & (P < 180.0),
        temperate,
        tropical & (P < 30.0),
        tropical & (P < 100.0),
        tropical & (P < 220.0),
        tropical,
    ]
    codes = [
        ICE, TUNDRA, TUNDRA, BOREAL_FOREST,
        DESERT, TEMPERATE_GRASSLAND, SHRUBLAND, TEMPERATE_FOREST, TEMPERATE_RAINFOREST,
        DESERT, SAVANNA, TROPICAL_SEASONAL_FOREST, TROPICAL_RAINFOREST,
    ]
    return np.select(conds, codes, default=TEMPERATE_FOREST).astype(np.uint8)


def apply_overrides(base: np.ndarray, ocean, lake, wetland, cliff, alpine, riparian) -> np.ndarray:
    """Stamp the override biomes onto ``base`` (uint8) in precedence order
    (later wins): riparian < alpine < cliff < wetland < lake < ocean.  Any
    argument may be ``None``."""
    out = np.array(base, dtype=np.uint8, copy=True)
    for mask, code in ((riparian, RIPARIAN), (alpine, ALPINE), (cliff, CLIFF), (wetland, WETLAND), (lake, LAKE), (ocean, OCEAN)):
        if mask is not None:
            out[np.asarray(mask, dtype=bool)] = code
    return out


def effective_cliff_slope(slope_land: np.ndarray, dp) -> float:
    """``max(dp.cliff_slope, (1 - dp.cliff_max_fraction) land quantile)``:
    cliffs never cover more than ``cliff_max_fraction`` of the land, so a
    small, steep test planet is not all cliff."""
    s = np.asarray(slope_land, dtype=np.float32).ravel()
    thr = float(dp.cliff_slope)
    frac = float(dp.cliff_max_fraction)
    if s.size and 0.0 < frac < 1.0:
        thr = max(thr, float(np.quantile(s, 1.0 - frac)))
    return thr


def classify(T, P_cm, surface, slope, lake, river_near, lake_near, dp, cliff_slope: float | None = None) -> np.ndarray:
    """Full biome classification of one array (coarse ``(6, N, N)`` or one
    fine face ``(Nf, Nf)``): Whittaker base plus the overrides of the module
    table.  ``dp`` is ``params.derive``; ``cliff_slope`` overrides
    ``dp.cliff_slope`` (see :func:`effective_cliff_slope`).  ``lake`` /
    ``river_near`` / ``lake_near`` are bool masks (``river_near`` = within
    ``riparian_cells`` of a channel; ``lake_near`` = within
    ``wetland_cells`` of a lake)."""
    surface = np.asarray(surface, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    ocean = surface < 0.0
    alpine = (surface >= np.float32(dp.alpine_min_m)) & (T < np.float32(dp.alpine_T))
    cs = dp.cliff_slope if cliff_slope is None else cliff_slope
    cliff = np.asarray(slope, dtype=np.float32) > np.float32(cs)
    lake = np.asarray(lake, dtype=bool)
    wetland = np.asarray(lake_near, dtype=bool) & ~lake
    riparian = np.asarray(river_near, dtype=bool)
    return apply_overrides(whittaker(T, P_cm), ocean, lake, wetland, cliff, alpine, riparian)


# --------------------------------------------------------------------------
# neighbourhoods
# --------------------------------------------------------------------------
def near(mask: np.ndarray, cells: int) -> np.ndarray:
    """Cells within chessboard distance ``cells`` of a True cell of the 2-D
    ``mask`` (within one face; the mask cells themselves included)."""
    m = np.asarray(mask, dtype=bool)
    if cells <= 0 or not m.any():
        return m.copy()
    if m.all():
        return m.copy()
    d = ndimage.distance_transform_cdt(~m, metric="chessboard")
    return d <= int(cells)


def near_padded(mask: np.ndarray, pads, cells: int) -> np.ndarray:
    """:func:`near` for one fine face whose ``mask`` is continued beyond
    the four sides by ``pads`` (``fine.face_pads`` layout with depth
    ``>= cells``: ``(top (d, Nf), bottom (d, Nf), left (Nf, d), right (Nf,
    d))`` bool), so the band does not stop at a cube edge.  The pad
    corners (beyond two edges) are left empty: a mask cell there would
    only matter within ``cells`` of a cube corner."""
    m = np.asarray(mask, dtype=bool)
    if cells <= 0 or pads is None:
        return near(m, cells)
    top, bottom, left, right = (np.asarray(p, dtype=bool) for p in pads)
    d = top.shape[0]
    Nf = m.shape[0]
    big = np.zeros((Nf + 2 * d, Nf + 2 * d), dtype=bool)
    big[d : d + Nf, d : d + Nf] = m
    big[:d, d : d + Nf] = top
    big[d + Nf :, d : d + Nf] = bottom
    big[d : d + Nf, :d] = left
    big[d : d + Nf, d + Nf :] = right
    return near(big, cells)[d : d + Nf, d : d + Nf]


def near_faces(mask6: np.ndarray, cells: int, grid) -> np.ndarray:
    """Coarse ``(6, N, N)`` version of :func:`near` that sees across face
    edges through the halo (exact while ``cells <= grid.H``)."""
    m = np.asarray(mask6, dtype=bool)
    if cells <= 0 or not m.any():
        return m.copy()
    ff = FaceField.from_interior(grid, m.astype(np.uint8), name="mask", exchange=True)
    H, N = grid.H, grid.N
    out = np.empty_like(m)
    for f in range(6):
        ext = near(ff.data[f].astype(bool), cells)
        out[f] = ext[H : H + N, H : H + N]
    return out


# --------------------------------------------------------------------------
# vegetation
# --------------------------------------------------------------------------
def vegetation(codes: np.ndarray, P_cm: np.ndarray, slope: np.ndarray, soil_factor, dp, cliff_slope: float | None = None) -> np.ndarray:
    """Vegetation density 0..255 (uint8):
    ``255 * sqrt(min(P/100, 1)) * sqrt(1 - min(slope/cliff_slope, 1)) *
    biome_factor * soil_factor`` (PLAN 11: precip x (1 - slope) x biome
    factor; the square roots keep moderately dry / sloped land from going
    bare too quickly).  ``soil_factor`` may be ``None``."""
    P = np.asarray(P_cm, dtype=np.float32)
    s = np.asarray(slope, dtype=np.float32)
    wet = np.sqrt(np.clip(P / np.float32(100.0), 0.0, 1.0))
    cs = dp.cliff_slope if cliff_slope is None else cliff_slope
    flat = np.sqrt(np.clip(1.0 - s / np.float32(max(cs, 1e-6)), 0.0, 1.0))
    v = wet * flat * VEG_FACTOR[np.asarray(codes, dtype=np.uint8)]
    if soil_factor is not None:
        v = v * np.asarray(soil_factor, dtype=np.float32)
    return np.clip(np.rint(v * 255.0), 0, 255).astype(np.uint8)


def colorize(codes: np.ndarray) -> np.ndarray:
    """Biome codes -> RGB uint8 (``PALETTE``; unknown codes magenta)."""
    c = np.asarray(codes, dtype=np.int64)
    pal = np.vstack([PALETTE, np.array([[255, 0, 255]], dtype=np.uint8)])
    return pal[np.where((c >= 0) & (c < N_BIOMES), c, N_BIOMES)]


__all__ = [
    "BIOMES", "NAMES", "PALETTE", "VEG_FACTOR", "N_BIOMES",
    "OCEAN", "ICE", "TUNDRA", "BOREAL_FOREST", "TEMPERATE_GRASSLAND", "TEMPERATE_FOREST", "TEMPERATE_RAINFOREST",
    "DESERT", "SHRUBLAND", "SAVANNA", "TROPICAL_SEASONAL_FOREST", "TROPICAL_RAINFOREST", "ALPINE", "CLIFF",
    "RIPARIAN", "WETLAND", "LAKE",
    "wetness", "precip_cm", "whittaker", "apply_overrides", "effective_cliff_slope", "classify", "near", "near_padded", "near_faces", "vegetation", "colorize",
]
