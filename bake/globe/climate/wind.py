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


def circulation_wind3(grid: Grid, cp: ClimateParams) -> np.ndarray:
    """Undeflected wind as 3-D tangent vectors (radians per step) on every
    extended cell: ``wind_speed`` cells at band centres."""
    east, north, cos_lat = geographic_frame(grid)
    zonal, merid = circulation_profile(grid.latitude(), cp)
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
    "circulation_wind3",
    "deflect_wind3",
    "wind_field",
]
