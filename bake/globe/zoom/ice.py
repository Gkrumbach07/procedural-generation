"""What the ice left below the coarse grid: scoured ground, and the lakes in it.

Most of Earth's lakes are small, and most of the small ones lie on ground an
ice sheet has crossed: a third of a million lakes over a square kilometre,
nine tenths of them north of 45 degrees, a tenth of the Canadian Shield and
of Finland under water.  Ice does not erode down a valley towards an outlet.
It quarries the rock under it wherever the rock is weak -- along joints,
dykes and shear zones, at every scale -- and leaves hollows that no river
could cut and none has yet drained ("deranged" drainage).

The coarse grid has the ice's large work: ``erosion.glacial`` deepens the
bed under the ice by its flux, with no base level, and the basins it leaves
are the planet's big cold lakes (half of earth-v24's lake area).  It cannot
have the small work.  A coarse cell is 95 km2, and a level below it adds no
lakes of its own (globe/zoom/parent_lakes.py), so the planet at 1.2 km had
56 lakes on a cold wet lowland of 1.5 million km2, none under 100 km2 that
the coarse grid did not hand it.

So the first level below the coarse grid lets the ice finish its work.  On
glaciated ground -- the climate's ice, tapered in from its margin as the
coarse pass tapers its carve -- the bed is lowered by up to :data:`SCOUR_M`
times the rock's *grain* (:func:`grain`): a ridged fractal field on the
sphere standing for the fracture spacing the model does not carry, deepest
along its lineaments, less in hard rock.  No base level: that is what makes
it ice.  The hollows it leaves hold water where the climate lets a small
lake keep any (:func:`coarse_ice`), each at its own spill point, and they
are the level's own lakes with a cause (``parent_lakes.level_lakes`` with
``cut``): a depression is the ice's where it was dry ground before the cut.

Measured on a cold wet lowland of earth-v24 at 1.2 km (face 4, 1.5 million
km2, 100 % ice ground; the scour applied to the finished level): see
docs/lakes-in-erosion.md, section 8.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy import ndimage

from ..derive.geology import _noise3

#: the deepest the ice quarries below the ground it found, metres: where the grain is weakest,
#: in soft rock.  Shield lakes are 5-15 m deep on average and the relief of scoured ground
#: ("knock and lochan") is tens of metres
SCOUR_M = 40.0
#: the longest wavelength of the rock's grain, km; its octaves halve down to two cells of the level
GRAIN_KM = 20.0
#: each octave's share of the one above: 0.7 gives a lake census N(> A) ~ A^-1, as Earth's
GRAIN_GAIN = 0.7
#: rng sub-key of the grain
ICE_KEY = 6_310_077
#: the share of its own catchment a small lake of the lake country covers (the Shield, Finland:
#: 10-15 %): what the water balance of a hollow is struck against (:func:`coarse_ice`)
LAKE_COUNTRY_SHARE = 0.15


@njit(cache=True, parallel=True)
def _grain_kernel(p, f0, octaves, gain, seed):
    out = np.empty(p.shape[0], np.float32)
    for k in prange(p.shape[0]):
        x, y, z = p[k, 0], p[k, 1], p[k, 2]
        a, f, v, tot = 1.0, f0, 0.0, 0.0
        for o in range(octaves):
            n = _noise3(x * f, y * f, z * f, seed + o)
            v += a * (1.0 - abs(n))                       # ridged: 1 on the noise's zero lines, the rock's lineaments
            tot += a
            a *= gain
            f *= 2.0
        out[k] = v / tot
    return out


def grain_octaves(cell_m: float) -> int:
    """Octaves of the grain a level with ``cell_m`` cells resolves: from
    :data:`GRAIN_KM` down to two cells."""
    return max(1, int(math.floor(math.log2(max(GRAIN_KM * 1000.0 / (2.0 * float(cell_m)), 1.0)))) + 1)


def grain(p: np.ndarray, radius_m: float, cell_m: float, seed: int) -> np.ndarray:
    """How weak the rock is against ice at unit vectors ``p`` (``(..., 3)``),
    0..1: ridged value noise on the sphere, seamless across the cube's faces
    and the same at every level (a finer level adds octaves below the ones a
    coarser one has).  float32, ``p.shape[:-1]``."""
    q = np.ascontiguousarray(np.asarray(p, np.float64).reshape(-1, 3))
    f0 = float(radius_m) / (GRAIN_KM * 1000.0)
    return _grain_kernel(q, f0, grain_octaves(cell_m), GRAIN_GAIN, int(seed) + ICE_KEY).reshape(np.asarray(p).shape[:-1])


def coarse_ice(fields: dict, temperature, params) -> dict:
    """``ice`` (float32, 0..1) and ``ice_wet`` (u8) on the coarse grid.

    ``ice`` is how fully the ice worked a cell: 0 off the glaciated ground
    (the climate's ``evap`` above ``erosion.ice_evap`` -- ``temperature``
    above that line with ``erosion.climate_at_surface`` -- or sea), rising to 1
    ``erosion.glacial_ramp`` cells in from its margin, as the coarse pass's
    own carve does (``erosion.glacial.ice_depth``).

    ``ice_wet`` marks where a hollow the ice left holds water: a lake that
    covers :data:`LAKE_COUNTRY_SHARE` of its own catchment gets at least what
    it evaporates, by the water balance the planet's lakes are struck with
    (``hydro.lake_evap`` / ``land_evap`` / ``pet_t0``).  Cold wet ground
    keeps its lakes; a cold desert's hollows are dry rock."""
    from ..field import FaceField
    from ..hydro.balance import aridity, budyko_evaporation, potential_evaporation

    grid = fields["evap"].grid
    ep, hp = params.erosion, params.hydro
    evap = np.asarray(fields["evap"].data, np.float32)
    sea = fields["basin_id"].data < 0
    T = np.asarray(temperature.data, np.float64)
    follow = bool(getattr(ep, "climate_at_surface", False))
    # the ice line erosion ended with: the climate's own, or at the surface (erosion.climate_at_surface;
    # `temperature` is then hydro.run.surface_temperature's)
    ice = ((T <= float(ep.ice_evap) / max(float(params.climate.k_evap), 1e-12)) if follow else (evap <= float(ep.ice_evap))) & ~sea
    t0 = float(getattr(hp, "pet_t0", 0.0))
    pet = potential_evaporation(T, grid.latitude(), float(params.climate.T_eq), t0) if t0 > 0.0 else evap
    land_evap = float(getattr(hp, "land_evap", 0.0))
    dry = float(getattr(ep, "ice_aridity", 0.0))
    # the rain as a depth, in the land-mean rain's (the balance's own measure, hydro/balance.py)
    rain = np.asarray(FaceField.from_interior(grid, (fields["precip"].interior / (grid.interior_cell_area / float(grid.cell_size_m) ** 2)).astype(np.float32)).data, np.float64)
    if dry > 0.0:
        ice &= aridity(pet, rain, land_evap) < dry            # ice needs snow (erosion.ice_aridity)
    ramp = float(max(int(ep.glacial_ramp), 0) + 1)
    taper = np.zeros(ice.shape, np.float32)
    for f in range(ice.shape[0]):
        if ice[f].any():
            taper[f] = np.clip(ndimage.distance_transform_edt(ice[f]) / ramp, 0.0, 1.0)
    rain = np.maximum(rain, 1e-9)
    room = 1.0 / LAKE_COUNTRY_SHARE - 1.0
    if land_evap > 0.0:
        a = aridity(pet, rain, land_evap)                 # potential evaporation over rain
        wet = (1.0 - budyko_evaporation(a)) * room + 1.0 >= a
    elif float(hp.lake_evap) > 0.0:
        wet = rain * (room + 1.0) >= float(hp.lake_evap) * pet
    else:
        wet = np.ones(ice.shape, bool)
    # the planet's lakes keep their shores (as from the detail noise, parent_lakes.lake_quiet): how
    # far a cell is from the nearest of them, in coarse cells, and that lake's level
    lake = ((fields["water_surface"].data - (fields["height"].data.astype(np.float32) + fields["sediment"].data.astype(np.float32)))
            > max(float(hp.lake_min_depth), float(getattr(hp, "marsh_depth", 0.0)))) & ~sea
    far = np.full(ice.shape, np.float32(1e6))
    near = np.zeros(ice.shape, np.float32)
    for f in range(ice.shape[0]):
        if lake[f].any():
            d, (ii, jj) = ndimage.distance_transform_edt(~lake[f], return_indices=True)
            far[f] = d
            near[f] = fields["water_surface"].data[f][ii, jj]
    return {"ice": FaceField(grid, taper, name="ice"), "ice_wet": FaceField(grid, (wet & ice).astype(np.uint8), name="ice_wet"),
            "lake_far": FaceField(grid, far, name="lake_far"), "lake_near": FaceField(grid, near, name="lake_near")}


_COARSE: dict = {}


def coarse_ice_of(root, params, fields: dict) -> dict:
    """:func:`coarse_ice` of the world at ``root``, kept for the process."""
    from pathlib import Path

    from ..hydro.run import surface_temperature
    from ..io.world_store import WorldStore

    key = str(Path(root).resolve())
    if key not in _COARSE:
        surface = fields["height"].data.astype(np.float32) + fields["sediment"].data.astype(np.float32)
        _COARSE.clear()
        _COARSE[key] = coarse_ice(fields, surface_temperature(WorldStore(root), params, surface), params)
    return _COARSE[key]


def beside_lakes(surface: np.ndarray, far: np.ndarray, near: np.ndarray, taper_m: float, reach_cells: float) -> np.ndarray:
    """The share of the scour the planet's lakes leave beside them: none on
    ground below a lake's level within ``reach_cells`` (coarse cells) of it,
    rising to all of it ``taper_m`` above the level or ``reach_cells`` away
    (``far``: coarse cells to the nearest lake cell, ``near``: its level).
    A sill the ice cut would drop the lake behind it."""
    w = np.clip(1.0 - np.asarray(far, np.float64) / float(reach_cells), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    t = np.clip((np.asarray(surface, np.float64) - np.asarray(near, np.float64)) / max(float(taper_m), 1e-9), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return 1.0 - w * (1.0 - t)


def scour(ice: np.ndarray, weak: np.ndarray, hardness: np.ndarray, surface: np.ndarray, coast_taper_m: float, quiet: np.ndarray | None = None) -> np.ndarray:
    """Metres the ice lowers a level's ground: :data:`SCOUR_M` times the
    ice's share ``ice`` (0..1), the rock's weakness ``weak`` (:func:`grain`)
    and ``1 - hardness / 2`` (hard rock resists the ice too, as on the coarse
    grid), fading to nothing at sea level over ``coast_taper_m`` and never
    taking land under the sea, and times ``quiet`` where given (the share the
    parent's lakes leave beside them, ``parent_lakes.lake_quiet``).  float32."""
    s = np.asarray(surface, np.float64)
    dz = SCOUR_M * np.asarray(ice, np.float64) * np.asarray(weak, np.float64) * (1.0 - 0.5 * np.clip(np.asarray(hardness, np.float64), 0.0, 1.0))
    if float(coast_taper_m) > 0.0:
        t = np.clip(s / float(coast_taper_m), 0.0, 1.0)
        dz *= t * t * (3.0 - 2.0 * t)
    if quiet is not None:
        dz *= quiet
    return np.where(s > 0.0, np.minimum(dz, np.maximum(s - 1.0, 0.0)), 0.0).astype(np.float32)


#: the planet's lakes keep the ground below their level unscoured out to this many coarse cells
LAKE_REACH = 2.0


def lower(height: np.ndarray, sediment: np.ndarray, cut: np.ndarray) -> None:
    """Take ``cut`` metres off a level's ground in place: its sediment
    first, then the rock under it."""
    take = np.minimum(np.maximum(sediment, 0.0), cut)
    sediment -= take
    height -= cut - take


def window_cut(root, params, grid, fields: dict, win, surface: np.ndarray, hardness: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """``(cut, wet)`` over a window's work array (``refine.upsample.sample``'s
    extended array; ``surface`` and ``hardness`` on it): the metres the ice
    lowers its ground (:func:`scour`) and where its hollows hold water, or
    ``(None, None)`` where the window has no glaciated ground."""
    from ..cubesphere import to_sphere_v
    from ..refine.upsample import sample

    ci = coarse_ice_of(root, params, fields)
    share = sample(ci["ice"], win, order=1)
    if not (share > 0.0).any():
        return None, None
    Nf = grid.N * win.R
    a0, a1, b0, b1 = win.fine_ext()
    U, V = np.meshgrid((np.arange(a0, a1) + 0.5) / Nf, (np.arange(b0, b1) + 0.5) / Nf, indexing="ij")
    weak = grain(to_sphere_v(np.full(U.shape, win.face), U, V), grid.R_planet, grid.cell_size_m / win.R, int(params.world.seed))
    taper_m = float(params.refine.coast_taper_m)
    quiet = beside_lakes(surface, sample(ci["lake_far"], win, order=1), sample(ci["lake_near"], win, order=0), taper_m, LAKE_REACH)
    return scour(share, weak, hardness, surface, taper_m, quiet), sample(ci["ice_wet"], win) > 0


def rows_cut(root, params, grid, fields: dict, face: int, R: int, i0: int, i1: int, surface: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """:func:`window_cut` over the fine rows of the coarse rows ``[i0, i1)``
    of a whole face of a planet level (``surface`` ``((i1 - i0) R, N R)``),
    with the planet's own hardness (``fields["hardness"]``, as the levels see
    it).  Every cell's value is its own, whatever rows it is asked with."""
    from ..cubesphere import to_sphere_v

    ci = coarse_ice_of(root, params, fields)
    N = grid.N
    share = ci["ice"].sample_window(face, i0, i1, 0, N, R, order=1)
    if not (share > 0.0).any():
        return None, None
    n = N * R
    U, V = np.meshgrid((np.arange(i0 * R, i1 * R) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="ij")
    weak = grain(to_sphere_v(np.full(U.shape, face), U, V), grid.R_planet, grid.cell_size_m / R, int(params.world.seed))
    hard = fields["hardness"].sample_window(face, i0, i1, 0, N, R, order=1)
    taper_m = float(params.refine.coast_taper_m)
    quiet = beside_lakes(surface, ci["lake_far"].sample_window(face, i0, i1, 0, N, R, order=1), ci["lake_near"].sample_window(face, i0, i1, 0, N, R, order=0), taper_m, LAKE_REACH)
    return scour(share, weak, hard, surface, taper_m, quiet), ci["ice_wet"].sample_window(face, i0, i1, 0, N, R) > 0


def face_cut(root, params, grid, fields: dict, face: int, R: int, surface: np.ndarray, strip: int = 64) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """:func:`rows_cut` over a whole face (``surface`` ``(N R, N R)``), a
    strip of coarse rows at a time; ``(None, None)`` where the face has no
    glaciated ground."""
    N = grid.N
    n = N * R
    cut = wet = None
    for i0 in range(0, N, strip):
        i1 = min(N, i0 + strip)
        rs = slice(i0 * R, i1 * R)
        c, w = rows_cut(root, params, grid, fields, face, R, i0, i1, surface[rs])
        if c is None:
            continue
        if cut is None:
            cut, wet = np.zeros((n, n), np.float32), np.zeros((n, n), bool)
        cut[rs], wet[rs] = c, w
    return cut, wet


__all__ = ["SCOUR_M", "GRAIN_KM", "GRAIN_GAIN", "LAKE_COUNTRY_SHARE", "LAKE_REACH", "grain", "grain_octaves", "coarse_ice", "coarse_ice_of", "beside_lakes", "scour", "lower",
           "window_cut", "rows_cut", "face_cut"]
