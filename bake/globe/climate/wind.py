"""Prevailing wind (PLAN.md section 7): three-cell circulation by latitude
band plus a deflection around high terrain, expressed as a contravariant
cell-component vector field (cells per advection step).

Circulation
-----------
With the +Z pole, ``east = normalize(ẑ × p)`` and ``north = p × east`` form
the local geographic frame.  The zonal sign by ``|lat|`` is

* easterlies (−east) 0–30°, westerlies (+east) 30–60°, easterlies 60–90°,

blended with smoothsteps of width ``band_blend_deg`` at 30° and 60°, so the
speed dips through zero across the band edges (the doldrums / horse
latitudes are calm belts).  A meridional component of relative size
``wind_meridional`` completes the surface branches of the three cells:
equatorward under the trades and polar easterlies, poleward under the
westerlies — its sign flips across the equator (``tanh(lat / band_blend)``,
so it vanishes on the equator where the two hemispheres' trades converge).
The zonal component itself is mirror-symmetric, as on Earth.

Within ``pole_taper_deg`` of a pole, where ``east`` is undefined, the speed
is tapered smoothly to zero.

Deflection
----------
Given the surface gradient ``∇h`` (m/m, of the ocean-clipped and lightly
smoothed surface), the wind is nudged along ``ĝ⊥ = p × ĝ`` — the direction
*around* the obstacle — on the side it already leans towards::

    t   = k·|∇h| / (1 + k·|∇h|)                 (saturating, k = wind_deflection)
    s   = clip((ŵ · ĝ⊥) / 0.25, −1, 1)          (smooth sign: which way round)
    w'  = |w| · normalize(ŵ + t·s·ĝ⊥)

so ranges steer the flow a little (≈13° at slope 1 for k = 0.3) without
ever changing its speed, and a wind blowing straight up or down a slope is
left alone (its neighbours split around).

Units
-----
Tangent vectors on the unit sphere are in radians; ``speed`` cells of
``cell_size_m`` correspond to ``speed·cell_size_m / R_planet`` radians.
:func:`tangent3_to_cells` / :func:`cells_to_tangent3` convert between 3-D
tangent vectors and cell components on every extended cell via the exact
Jacobian least squares (:func:`globe.cubesphere.jacobian_v`).
"""
from __future__ import annotations

import math

import numpy as np

from ..config import ClimateParams
from ..cubesphere import Grid, jacobian_v
from ..field import FaceField

_Z = np.array([0.0, 0.0, 1.0])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component-wise ``a × b`` on (..., 3) arrays (np.cross is several
    times slower on large arrays)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float64)
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


# --------------------------------------------------------------------------
# frames and component conversion (reusable by other stages)
# --------------------------------------------------------------------------
def geographic_frame(grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(east, north, cos_lat)`` on every extended cell, each ``(6, NE, NE, 3)``
    float64 unit tangent vectors (``cos_lat`` is ``(6, NE, NE)``).  At the
    poles (``cos_lat == 0``) east/north are an arbitrary orthonormal pair."""
    p = grid.centers
    east = _cross(_Z[None, None, None, :], p)
    n = np.linalg.norm(east, axis=-1)
    east = east / np.maximum(n, 1e-12)[..., None]
    # exact pole: pick any tangent direction
    bad = n < 1e-12
    if bad.any():
        alt = _cross(np.array([1.0, 0.0, 0.0])[None, None, None, :], p)
        alt /= np.maximum(np.linalg.norm(alt, axis=-1), 1e-12)[..., None]
        east[bad] = alt[bad]
    north = _cross(p, east)
    return east, north, n


_JAC_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _jacobians(grid: Grid):
    """``(Ju, Jv)`` (6, NE, NE, 3) on every extended cell, cached per grid
    geometry (≈300 MB at N = 1024; computed once per process)."""
    key = (grid.N, grid.H, grid.cell_size_m, grid.R_planet)
    hit = _JAC_CACHE.get(key)
    if hit is not None:
        return hit
    U, V = grid._uv_grid
    ju = np.empty((6, grid.NE, grid.NE, 3))
    jv = np.empty_like(ju)
    for f in range(6):
        ju[f], jv[f] = jacobian_v(np.full(U.shape, f), U, V)
    _JAC_CACHE.clear()  # keep at most one grid's tables resident
    _JAC_CACHE[key] = (ju, jv)
    return ju, jv


def tangent3_to_cells(grid: Grid, w3: np.ndarray) -> np.ndarray:
    """3-D tangent vectors ``(6, NE, NE, 3)`` (radians on the unit sphere)
    -> contravariant cell components ``(6, NE, NE, 2)`` float64: the least
    squares solution of ``w3 = (a·Ju + b·Jv) / N`` (exact for tangent
    vectors)."""
    ju, jv = _jacobians(grid)
    guu = np.sum(ju * ju, -1)
    guv = np.sum(ju * jv, -1)
    gvv = np.sum(jv * jv, -1)
    wu = np.sum(w3 * ju, -1)
    wv = np.sum(w3 * jv, -1)
    det = guu * gvv - guv * guv
    out = np.empty(w3.shape[:3] + (2,), dtype=np.float64)
    out[..., 0] = (gvv * wu - guv * wv) / det * grid.N
    out[..., 1] = (guu * wv - guv * wu) / det * grid.N
    return out


def cells_to_tangent3(grid: Grid, ab: np.ndarray) -> np.ndarray:
    """Inverse of :func:`tangent3_to_cells`: cell components ``(6, NE, NE, 2)``
    -> 3-D tangent vectors ``(6, NE, NE, 3)`` (radians per step)."""
    ju, jv = _jacobians(grid)
    ab = np.asarray(ab, dtype=np.float64)
    return (ab[..., :1] * ju + ab[..., 1:] * jv) / grid.N


def cells_per_radian(grid: Grid) -> float:
    """``R_planet / cell_size_m``: multiply radians by this to get cells."""
    return grid.R_planet / grid.cell_size_m


def smooth_field(field: FaceField, passes: int = 1) -> FaceField:
    """``passes`` applications of a separable 1-2-1 (3×3 binomial) filter
    with a halo exchange (cubic) after each pass — a cheap seam-free
    smoothing for scalar fields.  Returns a new float32 field."""
    f = field.copy()
    if f.data.dtype != np.float32:
        f = f.astype(np.float32)
    for _ in range(int(passes)):
        d = f.data
        a = np.empty_like(d)
        a[:, 1:-1, :] = 0.25 * d[:, :-2, :] + 0.5 * d[:, 1:-1, :] + 0.25 * d[:, 2:, :]
        a[:, 0, :] = d[:, 0, :]
        a[:, -1, :] = d[:, -1, :]
        b = np.empty_like(d)
        b[:, :, 1:-1] = 0.25 * a[:, :, :-2] + 0.5 * a[:, :, 1:-1] + 0.25 * a[:, :, 2:]
        b[:, :, 0] = a[:, :, 0]
        b[:, :, -1] = a[:, :, -1]
        f.data[...] = b
        f.exchange_halos()
    return f


# --------------------------------------------------------------------------
# circulation
# --------------------------------------------------------------------------
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def circulation_profile(lat_rad: np.ndarray, cp: ClimateParams) -> tuple[np.ndarray, np.ndarray]:
    """``(zonal, meridional)`` unit-speed components of the three-cell surface
    circulation as functions of latitude (radians, +Z pole): ``zonal`` is
    −1 under the trades and polar easterlies, +1 under the westerlies
    (smoothly blended); ``meridional`` is ``wind_meridional·zonal·sign(lat)``
    with a smooth sign.  Speed factor is ``hypot(zonal, meridional) /
    hypot(1, wind_meridional)`` — exactly 1 at band centres."""
    lat_deg = np.degrees(np.asarray(lat_rad, dtype=np.float64))
    a = np.abs(lat_deg)
    w = max(float(cp.band_blend_deg), 1e-6)
    zonal = -1.0 + 2.0 * _smoothstep((a - (30.0 - 0.5 * w)) / w) - 2.0 * _smoothstep((a - (60.0 - 0.5 * w)) / w)
    merid = cp.wind_meridional * zonal * np.tanh(lat_deg / w)
    norm = math.hypot(1.0, cp.wind_meridional)
    return zonal / norm, merid / norm


def storm_belt(lat_rad: np.ndarray, cp: ClimateParams) -> np.ndarray:
    """1 where the surface branches of the circulation *converge*, 0 where
    they diverge, by latitude.

    The mean wind dips through zero at 60 degrees exactly as it does at 30,
    but the two calms are opposites.  Under the horse latitudes the air
    subsides and it is dry.  Along the polar front the westerlies' poleward
    branch meets the polar easterlies' equatorward one and the air rises:
    the belt is Earth's storm track, and its zonal-mean wind is weak because
    its weather is cyclones.  Read as a calm, the front was a dry ring
    (precipitation.py, ``climate.storm_front``).

    The convergence is that of the meridional branch on the sphere,
    ``-(1/cos lat) d(v cos lat)/dlat > 0``: the whole westerly band and the
    front (poleward flow converges on a sphere), and the trades' meeting on
    the equator; not the 30 degree belt, and not the polar cap.  Only where
    the wind is slower than at a band centre does it change anything, which
    is the front: its sign turns 0.05 degrees inside the front's poleward
    edge, where the wind is within 0.2 % of full speed again."""
    lat = np.asarray(lat_rad, dtype=np.float64)
    e = math.radians(0.05)
    v = lambda x: circulation_profile(x, cp)[1] * np.cos(x)  # noqa: E731
    return (v(lat + e) < v(lat - e)).astype(np.float64)


def storminess(lat_rad: np.ndarray, cp: ClimateParams) -> np.ndarray:
    """0..1 by latitude: how far the weather, and not the mean wind, carries
    the moisture (precipitation.py, ``climate.eddy_reach_frac``).

    Cyclones belong to the extratropics: 1 poleward of the horse latitudes,
    blended in across them as the wind's own bands are (``band_blend_deg``).
    The trades are the one wind that is its own weather -- steady, under
    subsiding air -- and there it is 0, which is what keeps their deserts.
    On the equator they converge and the air rises again: the profile's own
    ``sech^2(lat / band_blend)``, the convergence of its ``tanh``."""
    lat = np.degrees(np.asarray(lat_rad, dtype=np.float64))
    w = max(float(cp.band_blend_deg), 1e-6)
    extratropics = _smoothstep((np.abs(lat) - (30.0 - 0.5 * w)) / w)
    return np.maximum(extratropics, 1.0 / np.cosh(lat / w) ** 2)


def circulation_wind3(grid: Grid, cp: ClimateParams, lat: np.ndarray | None = None) -> np.ndarray:
    """Undeflected wind as 3-D tangent vectors (radians per step) on every
    extended cell: ``wind_speed`` cells at band centres.  ``lat``: the
    latitude the circulation's bands are read at (radians, extended), where
    it is not the cells' own -- a season's, the cells moved with the sun
    (:func:`seasonal_wind3`)."""
    east, north, cos_lat = geographic_frame(grid)
    zonal, merid = circulation_profile(grid.latitude() if lat is None else lat, cp)
    taper = _smoothstep(cos_lat / math.sin(math.radians(max(cp.pole_taper_deg, 1e-3))))
    speed_rad = cp.wind_speed / cells_per_radian(grid)
    amp = speed_rad * taper
    return (zonal * amp)[..., None] * east + (merid * amp)[..., None] * north


def deflect_wind3(grid: Grid, w3: np.ndarray, surface: FaceField, cp: ClimateParams) -> np.ndarray:
    """Steer ``w3`` (3-D tangent, any units) around high terrain (see module
    docstring).  ``surface`` is the height field in metres with halos
    exchanged; only land (``max(h, 0)``) deflects.  Speed is preserved."""
    k = float(cp.wind_deflection)
    if k <= 0.0:
        return w3
    hs = FaceField(grid, np.maximum(surface.data, 0.0).astype(np.float32), name="hs")
    if cp.deflection_smooth > 0:
        hs = smooth_field(hs, cp.deflection_smooth)
    grad = hs.gradient()  # contravariant cells; d·E = m/m
    g3 = cells_to_tangent3(grid, grad.data) * grid.R_planet  # m/m, 3-D
    gmag = np.linalg.norm(g3, axis=-1)
    ghat = g3 / np.maximum(gmag, 1e-12)[..., None]
    gperp = _cross(grid.centers, ghat)
    wmag = np.linalg.norm(w3, axis=-1)
    what = w3 / np.maximum(wmag, 1e-30)[..., None]
    t = k * gmag / (1.0 + k * gmag)
    s = np.clip(np.sum(what * gperp, -1) / 0.25, -1.0, 1.0)
    d = what + (t * s)[..., None] * gperp
    d /= np.maximum(np.linalg.norm(d, axis=-1), 1e-30)[..., None]
    out = d * wmag[..., None]
    # keep the exact (possibly zero) vector where the gradient is zero
    return np.where((gmag > 0)[..., None], out, w3)


#: how far the turn of the planet swings a season's inflow off the straight line into a heated
#: continent (degrees; to the right in the north, the left in the south, nothing on the equator)
MONSOON_TURN_DEG = 30.0


def season_latitude(grid: Grid, cp: ClimateParams, sign: float) -> np.ndarray:
    """The latitude a season's circulation is read at (radians, extended):
    the cells' own less ``sign x season_shift_deg`` -- ``sign`` +1 the
    northern summer, -1 the southern, 0 an equinox.  The three cells follow
    the sun: the belt where the trades meet stands some 7 degrees into the
    summer hemisphere, and the dry belts and the storm tracks with it."""
    return np.asarray(grid.latitude(), np.float64) - float(sign) * math.radians(float(cp.season_shift_deg))


def seasonal_wind3(grid: Grid, cp: ClimateParams, sign: float, temp_range: np.ndarray) -> np.ndarray:
    """A season's undeflected wind (3-D tangent, radians per step): the
    circulation at the season's latitude (:func:`season_latitude`) plus the
    monsoon -- air flows into the continent the season has heated and out of
    the one it has cooled, up the gradient of the season's temperature
    anomaly (half ``temp_range``, C, positive in the summer hemisphere,
    smoothed over ``monsoon_smooth_frac`` of the radius), ``monsoon`` of the
    band wind for every C per 1000 km, never more than the band wind, turned
    :data:`MONSOON_TURN_DEG` by the planet's spin."""
    w3 = circulation_wind3(grid, cp, season_latitude(grid, cp, sign))
    k = float(getattr(cp, "monsoon", 0.0))
    if k <= 0.0 or sign == 0:
        return w3
    lat = np.asarray(grid.latitude(), np.float64)
    a = FaceField(grid, (float(sign) * 0.5 * np.asarray(temp_range, np.float64) * np.sign(lat)).astype(np.float32), name="season")
    a.exchange_halos(linear=True)
    sigma = float(cp.monsoon_smooth_frac) * grid.R_planet / grid.cell_size_m
    a = smooth_field(a, min(max(int(round(2.0 * sigma * sigma)), 1), 400))
    g3 = cells_to_tangent3(grid, a.gradient().data) * grid.R_planet * 1.0e6          # C per 1000 km
    speed = cp.wind_speed / cells_per_radian(grid)
    m3 = k * speed * g3
    turn = math.radians(MONSOON_TURN_DEG) * np.tanh(np.degrees(lat) / 10.0)
    m3 = m3 * np.cos(turn)[..., None] - _cross(grid.centers, m3) * np.sin(turn)[..., None]
    m3 *= np.minimum(1.0, abs(speed) / np.maximum(np.linalg.norm(m3, axis=-1), 1e-30))[..., None]
    return w3 + m3


def seasonal_wind(grid: Grid, surface: FaceField, cp: ClimateParams, sign: float, temp_range: np.ndarray, name: str = "wind") -> FaceField:
    """:func:`wind_field` of a season (:func:`seasonal_wind3`, deflected)."""
    w3 = deflect_wind3(grid, seasonal_wind3(grid, cp, sign, temp_range), surface, cp)
    f = FaceField(grid, tangent3_to_cells(grid, w3).astype(np.float32), is_vector=True, name=name)
    f.exchange_halos()
    return f


def wind_field(grid: Grid, surface: FaceField, cp: ClimateParams, name: str = "wind") -> FaceField:
    """The stage's wind: circulation + deflection, as a contravariant vector
    ``FaceField`` (cells per advection step) with halos exchanged."""
    w3 = circulation_wind3(grid, cp)
    w3 = deflect_wind3(grid, w3, surface, cp)
    ab = tangent3_to_cells(grid, w3).astype(np.float32)
    f = FaceField(grid, ab, is_vector=True, name=name)
    f.exchange_halos()  # exact rotation: makes halos consistent with the owners
    return f


__all__ = [
    "geographic_frame",
    "tangent3_to_cells",
    "cells_to_tangent3",
    "cells_per_radian",
    "smooth_field",
    "circulation_profile",
    "storm_belt",
    "storminess",
    "circulation_wind3",
    "deflect_wind3",
    "wind_field",
    "season_latitude",
    "seasonal_wind3",
    "seasonal_wind",
    "MONSOON_TURN_DEG",
]
