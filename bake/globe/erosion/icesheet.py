"""Ice with a thickness: a sheet in balance with its climate.

The glacial pass had its ice by a line: ground whose year averages under
freezing, with snow enough (``ErosionState.cold``).  That is where ice
*could* start.  An ice sheet is not there: it is a body.  It stands two or
three kilometres over its bed, so its own surface is twenty degrees colder
than the ground it covers; it flows, from where more snow falls than melts
out to where it melts; and it lives or dies by its summers, not by its
year's mean -- north-east Siberia is colder than Greenland and has no ice
sheet, because its summers are warm and its snow is thin.

Three pieces, each a few lines of physics with its constants Earth's:

**The balance** (:func:`balance`), metres of water a year at a surface of a
given height.  The year swings about its mean by the climate's seasonal
range (``climate.seasonal_range``, the ``temp_range`` field).  Snow is the
precipitation of the part of the year under ``ice_snow_c``; melt is
``ice_ddf_mm`` millimetres for every degree-day above freezing.  With 4 mm
a degree-day it gives back Ohmura's (1992) relation between a glacier's
summer and its precipitation at the line where the two cancel.

**The shape** (:func:`geometry`).  Ice is nearly plastic: it flows until the
stress at its bed is its yield stress and no further, which makes its
surface a parabola from the margin, ``sqrt(2 h0 d)`` over the ground a
distance ``d`` inside it, with ``h0`` = yield stress / (density x gravity),
``ice_yield_m`` -- 8 m is 72 kPa; Greenland's profile is 10, the softer-bedded
Laurentide's was 4.5.  The ground it stands on is the ground smoothed over 50 km
(:data:`BED_SMOOTH_FRAC`), and a mountain higher than the ice stands through it.

**The extent** (:func:`settle`).  The balance is summed down the ice's own
surface, each cell handing what it has to every lower neighbour by the
slope (:func:`_spread`), and never less than nothing: where the sum runs
out the ice ends.  Bare ground the ice reaches becomes ice, a ring a round,
and ice the flux no longer reaches melts; the shape and the flow are solved
in turn until neither moves.  Started from nothing it is an advance from
the snow line.  Started from a larger sheet it is a thaw, and what is left
is not what would have grown: a sheet's height keeps it, which is why there
is a Greenland.

Measured on earth-v33's last ground before any of this was in the bake
(the prototype; docs/thick-ice.md): 8.8 % of the land under ice today, 1.9 km
thick on average, 55 m of sea level in it -- Earth has 10 %, about 2 km and
66 m -- and a last maximum's worth, a quarter of the land, at about 10 C
colder.  None of it tuned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..field import FaceField
from ..hydro.d8 import cid_fij, neighbor_cid

#: the ground under a sheet is read smoothed over this share of the planet's radius (a Gaussian's
#: sigma): 50 km on Earth
BED_SMOOTH_FRAC = 0.00785
#: binomial passes over the ice's rise above its bed (:func:`geometry`)
SURFACE_SMOOTH = 3
#: a year of days
YEAR_DAYS = 365.0
#: ice against water, by weight
ICE_DENSITY = 0.917
#: a sheet has settled when its area moves less than this share of the land for QUIET rounds
SETTLE_TOL = 2.0e-4
QUIET = 6


@dataclass
class IceClimate:
    """What an ice sheet needs of the climate, on the coarse grid's interior
    ``(6, N, N)``: the year's mean ``t0`` (C) at the height ``z0`` (m) it was
    read at, half the seasonal range ``half`` (C), the precipitation ``rain``
    (m of water a year), and the constants."""
    t0: np.ndarray
    z0: np.ndarray
    half: np.ndarray
    rain: np.ndarray
    lapse: float                 # C per metre
    ddf: float                   # m of water per degree-day
    snow_c: float


def balance(clim: IceClimate, z: np.ndarray, cool: float) -> np.ndarray:
    """Metres of water a year gained (lost) at a surface of height ``z`` (m)
    in a climate ``cool`` C colder than the climate's own: the snow of the
    cold part of the year less the melt of its degree-days, the year a sine
    about its mean."""
    tm = clim.t0 - float(cool) - clim.lapse * (np.asarray(z, np.float64) - clim.z0)
    half = np.maximum(clim.half, 0.25)
    th = np.arccos(np.clip(-tm / half, -1.0, 1.0))                     # the half-angle of the year above freezing
    pdd = YEAR_DAYS / math.pi * (tm * th + half * np.sin(th))
    snowy = 1.0 - np.arccos(np.clip((clim.snow_c - tm) / half, -1.0, 1.0)) / math.pi
    return clim.rain * snowy - clim.ddf * pdd


@njit(cache=True)
def _spread(z, w, ice, order, owner, N, H):
    """The balance ``w`` (volume a year, flat) summed down the surface ``z``:
    each ice cell passes what it holds to its lower neighbours by their
    slope, bare ground keeps what reaches it and passes nothing on, and no
    cell holds less than nothing.  ``order``: the land cells, highest first."""
    acc = np.zeros(z.size)
    sl = np.zeros(8)
    nb = np.zeros(8, np.int64)
    for t in range(order.size):
        c = order[t]
        v = acc[c] + w[c]
        if v <= 0.0:
            acc[c] = 0.0
            continue
        acc[c] = v
        if not ice[c]:
            continue
        f, i, j = cid_fij(c, N)
        tot = 0.0
        for k in range(8):
            n = neighbor_cid(owner, N, H, f, i, j, k)
            nb[k] = n
            dz = z[c] - z[n]
            s = dz / (1.41421356 if k % 2 else 1.0) if dz > 0.0 else 0.0
            sl[k] = s
            tot += s
        if tot <= 0.0:
            continue
        for k in range(8):
            if sl[k] > 0.0:
                acc[nb[k]] += v * sl[k] / tot
    return acc


def smooth_bed(grid, bed: np.ndarray) -> np.ndarray:
    """The ground a sheet's base sees: ``bed`` (interior, m, the sea floor as
    0) smoothed over :data:`BED_SMOOTH_FRAC` of the planet's radius (never
    over more than ten cells: a body of few cells is not smoothed flat)."""
    from ..climate.wind import smooth_field

    f = FaceField.from_interior(grid, np.maximum(np.asarray(bed, np.float32), 0.0))
    f.exchange_halos(linear=True)
    sigma = min(BED_SMOOTH_FRAC * float(grid.R_planet) / float(grid.cell_size_m), 10.0)
    passes = max(int(round(2.0 * sigma * sigma)), 1)
    return np.maximum(smooth_field(f, passes).interior.astype(np.float64), 0.0)


def geometry(grid, mask: np.ndarray, bed: np.ndarray, bed_s: np.ndarray, h0: float) -> tuple[np.ndarray, np.ndarray]:
    """``(surface, thickness)`` (interior, m) of plastic ice on ``mask``: the
    smoothed ground plus ``sqrt(2 h0 d)``, ``d`` from half a cell at the
    margin; where the ground is higher than that it stands bare of
    thickness (a nunatak) and the surface is the ground."""
    from ..climate.precipitation import coast_distance_m

    H, N = grid.H, grid.N
    I = (slice(None), slice(H, H + N), slice(H, H + N))
    m = FaceField.from_interior(grid, np.asarray(mask, np.float32))
    m.exchange_halos()
    d = coast_distance_m(grid, m.data < 0.5, max_rounds=48)[I].astype(np.float64) - 0.5 * float(grid.cell_size_m)
    # the distance is a chamfer's, faceted along the grid's lines and ridged at a cube edge: on a dome
    # two kilometres high the facets are straight stripes in the light.  Two passes take them out
    from ..climate.wind import smooth_field

    rise = FaceField.from_interior(grid, np.sqrt(2.0 * float(h0) * np.maximum(d, 0.0)).astype(np.float32))
    rise.exchange_halos(linear=True)
    rise = smooth_field(rise, SURFACE_SMOOTH).interior.astype(np.float64)
    z = np.where(mask, np.maximum(bed, bed_s + rise), bed)
    return z, z - bed


def settle(grid, clim: IceClimate, bed: np.ndarray, sea: np.ndarray, cool: float, h0: float, mask: np.ndarray | None = None,
           fed: np.ndarray | None = None, max_rounds: int = 400, bed_s: np.ndarray | None = None, log=None) -> dict:
    """The sheet in balance with the climate ``cool`` C colder than today,
    on the ground ``bed`` (interior, m) with ``sea`` (interior bool) where
    the ice ends in water.  ``mask``: the ice to start from (None: none, and
    the sheet grows from the snow line); ``fed``: the cells the flux reached
    in the round before, from the call before (a cell melts only once the
    flux has left it twice running -- the lines of flow shift with the
    margin and its last cell would flicker).  At most ``max_rounds`` rounds;
    a sheet that has not settled goes on from where it is at the next call.

    Returns ``mask``, ``thickness`` (m), ``surface`` (m), ``flux`` (m3 of
    water a year through each cell), ``fed``, ``rounds``, ``settled``."""
    N, H = grid.N, grid.H
    land = ~np.asarray(sea, bool)
    bed = np.where(land, np.maximum(np.asarray(bed, np.float64), 0.0), np.minimum(np.asarray(bed, np.float64), 0.0))
    area = np.asarray(grid.interior_cell_area, np.float64)
    a_land = float((area * land).sum())
    if bed_s is None:
        bed_s = smooth_bed(grid, bed)
    mask = np.zeros(bed.shape, bool) if mask is None else (np.asarray(mask, bool) & land)
    owner = grid.owner
    land_ids = np.flatnonzero(land.reshape(-1))
    quiet, a_old, rounds, settled = 0, float((area * mask).sum()), 0, False
    tol = max(SETTLE_TOL * a_land, 2.5 * float(area.mean()))      # (on a body of few cells, the cells' own size)
    z = thick = q = None
    for rounds in range(1, int(max_rounds) + 1):
        z, thick = geometry(grid, mask, bed, bed_s, h0)
        w = (balance(clim, z, cool) * area).reshape(-1)
        zf = z.reshape(-1)
        order = land_ids[np.argsort(-zf[land_ids], kind="stable")]
        q = _spread(zf, w, np.ascontiguousarray(mask.reshape(-1)), order, owner, N, H).reshape(bed.shape)
        now = land & (q > 0.0)
        new = now | (mask & fed) if fed is not None else now
        fed = now
        a_new = float((area * new).sum())
        quiet = quiet + 1 if abs(a_new - a_old) < tol else 0
        a_old = a_new
        mask = new
        if quiet >= QUIET:
            settled = True
            break
    z, thick = geometry(grid, mask, bed, bed_s, h0)
    if log is not None:
        s = stats(grid, mask, thick, sea)
        log(f"[ice] {float(cool):+.1f} C: ice on {100.0 * s['land_share']:.1f} % of the land, {s['thickness_mean_m']:.0f} m thick on average "
            f"({s['thickness_max_m']:.0f} at most), {s['sea_level_m']:.0f} m of sea level; {rounds} rounds{'' if settled else ' (not settled)'}")
    return {"mask": mask, "thickness": thick.astype(np.float32), "surface": z.astype(np.float32), "flux": q, "fed": fed, "rounds": rounds, "settled": settled}


def stats(grid, mask: np.ndarray, thickness: np.ndarray, sea: np.ndarray) -> dict:
    """The sheet in numbers: its share of the land, its thickness, its
    volume and the sea level that volume of water is."""
    area = np.asarray(grid.interior_cell_area, np.float64)
    sea = np.asarray(sea, bool)
    m = np.asarray(mask, bool)
    vol = float((np.asarray(thickness, np.float64) * area * m).sum())
    a_ice = float((area * m).sum())
    return {"land_share": a_ice / max(float((area * ~sea).sum()), 1.0),
            "thickness_mean_m": vol / a_ice if a_ice > 0.0 else 0.0,
            "thickness_max_m": float(np.asarray(thickness)[m].max()) if m.any() else 0.0,
            "volume_km3": vol / 1.0e9,
            "sea_level_m": ICE_DENSITY * vol / max(float((area * sea).sum()), 1.0)}


def climate_of(grid, temperature: np.ndarray, ground: np.ndarray, temp_range: np.ndarray, precip: np.ndarray, land: np.ndarray, params) -> IceClimate:
    """The :class:`IceClimate` of a world: ``temperature`` (C) as read at
    ``ground`` (m), ``temp_range`` (C), and ``precip`` (the climate's volume
    a cell) as a depth, the land's mean taken as ``climate.land_rain_mm``.
    All interior ``(6, N, N)``."""
    cp, ep = params.climate, params.erosion
    area = np.asarray(grid.interior_cell_area, np.float64)
    depth = np.asarray(precip, np.float64) / (area / float(grid.cell_size_m) ** 2)
    land = np.asarray(land, bool)
    mean = float((area * land * depth).sum() / max(float((area * land).sum()), 1.0))
    rain = float(cp.land_rain_mm) / 1000.0 * depth / max(mean, 1e-12)
    return IceClimate(t0=np.asarray(temperature, np.float64), z0=np.maximum(np.asarray(ground, np.float64), 0.0),
                      half=0.5 * np.asarray(temp_range, np.float64), rain=rain, lapse=float(cp.lapse) / 1000.0,
                      ddf=float(ep.ice_ddf_mm) / 1000.0, snow_c=float(ep.ice_snow_c))


# --------------------------------------------------------------------------
# the erosion stage's sheet
# --------------------------------------------------------------------------
#: rounds the stage's first sheet is given: it grows from nothing
FIRST_ROUNDS = 300


def setup(state, grid, store, params) -> None:
    """Give the state its ice sheet's climate (``erosion.ice_sheet``): the
    year's mean it already carries (``temp0`` at ``temp_z``), the seasonal
    range (the climate's ``temp_range``; worked out here for a world baked
    before the field) and the rain as a depth.  The sheet itself is grown
    at the first iteration (:func:`update`)."""
    if not bool(getattr(params.erosion, "ice_sheet", False)) or state.temp0 is None:
        return
    I = state.interior
    bedrock = store.load_field("bedrock", grid)
    if store.has_field("temp_range"):
        rng = store.load_field("temp_range", grid).data
    else:
        from ..climate.temperature import seasonal_range

        rng = seasonal_range(grid, bedrock.data < 0.0, params.climate)
    state.ice_clim = climate_of(grid, state.temp0[I], state.temp_z[I], np.asarray(rng)[I], store.load_field("precip", grid).interior,
                                bedrock.interior >= 0.0, params)
    state.sheet = None
    state.sheet_h = None
    state.sheet_fed = None
    state.sheet_at = None
    state.sea_drop = 0.0             # how far today's sea stands above the stage's (cell units; sea_fall)
    state.sea_ref = None             # the ice as a depth of sea (m) when the stage's climate was today's


def ground(state) -> tuple[np.ndarray, np.ndarray]:
    """``(bed, sea)`` of the state as the sheet sees them (interior, metres
    and bool): the surface, and the kernel's own sea -- under its base level
    where that is 0 -- less what the ice already stands on (a bed the ice
    has cut below the sea is still under the ice)."""
    from . import particle as pk

    I = state.interior
    surf = (state.height[I] + state.sediment[I]).astype(np.float64) * float(state.height_unit_m)
    sea = ((surf < 0.0) & (state.base[I] == 0.0)) | (state.mask[I] != pk.MASK_ACTIVE)
    if getattr(state, "sheet", None) is not None:
        sea &= ~(state.sheet[I] & (state.mask[I] == pk.MASK_ACTIVE))
    return surf, sea


def update(state, params, rounds: int | None = None, cool: float | None = None, log=None) -> dict:
    """Bring the state's sheet up to its climate -- ``cool`` C colder than
    the climate's own, the stage's ``ErosionState.ice_age`` when None -- on
    the ground as it stands, in at most ``rounds`` rounds (``erosion.ice_rounds``;
    the first sheet of a stage gets :data:`FIRST_ROUNDS`).  Sets
    ``state.sheet`` (extended bool, what :meth:`ErosionState.cold` returns),
    ``sheet_h`` (extended float32, metres) and ``sheet_fed``; returns
    :func:`settle`'s result with :func:`stats` in it."""
    grid, ep = state.grid, params.erosion
    I = state.interior
    bed, sea = ground(state)
    first = getattr(state, "sheet", None) is None
    n = int(rounds) if rounds is not None else (FIRST_ROUNDS if first else int(ep.ice_rounds))
    res = settle(grid, state.ice_clim, bed, sea, float(state.ice_age) if cool is None else float(cool), float(ep.ice_yield_m),
                 mask=None if first else state.sheet[I], fed=None if first else state.sheet_fed, max_rounds=n, log=log)
    state.sheet = np.ascontiguousarray(FaceField.from_interior(grid, res["mask"].astype(np.float32)).data > 0.5)
    state.sheet_h = np.ascontiguousarray(FaceField.from_interior(grid, res["thickness"]).data)
    state.sheet_fed = res["fed"]
    state.sheet_at = int(state.iteration)
    # the ice passing through each cell, as water a year per metre of the cell's width (m2/yr): what the
    # glacial pass can carve by (erosion.ice_carve_flux)
    flux = np.where(res["mask"], np.maximum(res["flux"], 0.0), 0.0) / float(grid.cell_size_m)
    state.sheet_flux = np.ascontiguousarray(np.where(state.sheet, np.maximum(FaceField.from_interior(grid, flux.astype(np.float32)).data, 0.0), 0.0))
    res["stats"] = stats(grid, res["mask"], res["thickness"], sea)
    return res


def sea_fall(state, params, sea_level_m: float) -> float:
    """The sea falls as the ice takes its water (``erosion.ice_sea_share``):
    sets ``state.sea_drop`` (cell units), how far today's sea stands above
    the stage's, from the sheet's volume as a depth of sea ``sea_level_m``,
    and returns it in metres.

    The reference is the sheet as it stood when the stage's climate was
    today's (``ice_age`` 0: its volume then is ``state.sea_ref``); before
    that the sea is not moved.  Past it the sea is lower by the share of the
    extra ice: 1 is the last maximum's 120 m all through the ice age, and the
    0.5 of the parameter's comment the mean of cycles that spend as long
    thawed as frozen -- the sheet the stage carries is the cold end of each
    (``glacial.ice_cooling``), the sea it cuts its coasts against is not.

    Nothing is shifted here: :func:`maps.hold_datum` holds today's coast at
    ``sea_drop`` above the stage's sea from the next iteration on, so the
    shelf comes out of the water as the ice grows and the rivers cross it."""
    share = float(getattr(params.erosion, "ice_sea_share", 0.0))
    if share <= 0.0:
        state.sea_drop = 0.0
        return 0.0
    if float(state.ice_age) < 0.0 or getattr(state, "sea_ref", None) is None:
        if float(state.ice_age) >= 0.0:
            state.sea_ref = float(sea_level_m)
        state.sea_drop = 0.0
        return 0.0
    drop = share * max(float(sea_level_m) - float(state.sea_ref), 0.0)
    state.sea_drop = drop / float(state.height_unit_m)
    return drop


__all__ = ["sea_fall", "BED_SMOOTH_FRAC", "ICE_DENSITY", "FIRST_ROUNDS", "IceClimate", "balance", "smooth_bed", "geometry", "settle", "stats", "climate_of",
           "setup", "ground", "update"]
