"""Orographic precipitation by semi-Lagrangian moisture advection
(PLAN.md section 7).

State: moisture ``m`` (extended array, float32).  Initially ``m_ocean``
over ocean and ``m_ocean·exp(−d_coast/L_base)`` on land (``d_coast`` from a
seam-free chamfer distance, :func:`coast_distance_m`): the semi-Lagrangian
gather never updates a cell whose wind vanishes, so the cells on the
divergence lines of the circulation (the calm belts at 30°, the poles)
keep their initial value forever and feed it to everything downstream
of them; the distance-decayed guess is the physically sensible value for
that subsiding air (the ring cells of a plain ``m = 0`` start dry out
whole polar continents and converge only through numerical diffusion,
which is resolution-dependent and slow).  Every sweep:

1. exchange ``m``'s halos (bilinear — monotone, keeps ``m`` in
   ``[0, m_ocean]``);
2. for every interior cell sample ``m`` bilinearly at the departure point
   ``(i, j) − wind·dt`` (cell components, so the departure point is in
   extended-index space; ``|wind|·dt`` must stay below ``H − 1`` cells);
3. rain out ``rain = m·frac`` with a per-cell fraction that is
   parameterised in **physical length**, so a coastline-to-interior
   distance or a mountain flank of a given size rains out the same
   fraction of the moisture at any grid resolution / step size::

       frac = min(1, 1 − exp(−step_m / L_base) + k_oro·(1 − exp(−rise_m / H_oro)))

   ``step_m = max(|wind|, calm_floor·wind_speed)·dt·cell_size_m`` is the
   distance the air travels in the step (floored: moisture also rains out
   of air that barely moves, so the calm belts are not zero-rain lines),
   ``L_base`` the e-folding fetch of the background rain-out
   (``moisture_reach_m``, or ``moisture_reach_frac × R_planet`` when 0 —
   continents scale with the planet, so the fraction is the robust
   default across the presets), ``rise_m = max(0, dt·wind·∇h)`` the metres
   climbed in the step (``FaceField.directional_derivative`` of the
   ocean-clipped, lightly smoothed surface) and ``H_oro`` the orographic
   scale height: with ``k_oro = 1`` an air parcel keeps ``exp(−Δh/H_oro)``
   of its moisture after climbing ``Δh`` however many cells the climb
   spans (PLAN 7's ``k_oro·max(0, wind·∇h)`` is the small-rise limit with
   ``k_oro_plan = k_oro/H_oro`` per metre; the exponential keeps a single
   coarse cell from raining out more than ``k_oro``);
4. ``m ← m − rain`` on land, ``m ← m_ocean`` over ocean.

The sweeps are repeated until the moisture field is stationary
(``n_advect = 0``: the maximum change over land between two sweeps falls
below ``advect_tol·m_ocean``, capped at ``n_advect_max_factor·N/(wind_speed·dt)``
sweeps — the moisture front moves ``wind_speed·dt`` cells per sweep) or
for exactly ``n_advect`` sweeps, and the rain of the last sweep is the
result (the steady-state rain per sweep; PLAN 7's accumulated ``precip``
divided by the sweep count converges to the same field).

The sweep is one numba kernel (pure gather: output cells depend only on
the previous state, so the parallel schedule cannot change the result);
halo exchange is the ``FaceField`` gather.  Per sweep cost is O(6 N²).

Post-processing (:func:`finalise_precip`): the rain gets ``precip_smooth``
3×3 binomial passes (spillover — orographic rain drifts a few cells past
where it is triggered), is normalised to a land mean of 1, a background
``precip_floor`` is added on land, the latitude prior (ITCZ wet band,
subtropical dry bands, see :func:`latitude_prior`) is applied, the result
is converted to a *volume* per cell (``× cell_area / cell_size_m²``) and
rescaled so its mean over land equals ``precip_mean`` (docs/DEVELOPING.md).
Ocean cells carry their (small) physical rain — the same scale, no floor —
so they never outshine the land; downstream stages mask to land anyway.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..config import ClimateParams
from ..cubesphere import Grid
from ..field import FaceField, bilinear_ext


@njit(cache=True, parallel=True)
def _sweep(m, wind, frac, ocean, H, N, dt, m_ocean, m_out, rain_out):
    """One semi-Lagrangian sweep over the interior of every face (gather)."""
    F = m.shape[0]
    for r in prange(F * N):
        f = r // N
        i = r - f * N + H
        mf = m[f]
        for jj in range(N):
            j = jj + H
            a = wind[f, i, j, 0]
            b = wind[f, i, j, 1]
            madv = bilinear_ext(mf, i - a * dt, j - b * dt)
            rain = madv * frac[f, i, j]
            rain_out[f, i, j] = rain
            if ocean[f, i, j]:
                m_out[f, i, j] = m_ocean
            else:
                m_out[f, i, j] = madv - rain


def orographic_rise(surface: FaceField, wind: FaceField, smooth: int = 0) -> np.ndarray:
    """``max(0, wind·∇h)``: rise of ``max(surface, 0)`` (after ``smooth``
    3×3 binomial passes) in metres over one unit step along ``wind``
    (extended array, float32; valid 1..NE−2)."""
    hs = FaceField(surface.grid, np.maximum(surface.data, 0.0).astype(np.float32), name="hs")
    if smooth > 0:
        from .wind import smooth_field

        hs = smooth_field(hs, smooth)
    d = hs.directional_derivative(wind).data
    return np.maximum(d, 0.0).astype(np.float32)


@njit(cache=True, parallel=True)
def _chamfer_pass(d):
    """Two-pass 3-4 chamfer distance transform on every face of an extended
    array in place (units: 1/3 cell); seeds are the cells that hold 0."""
    F, NE, _ = d.shape
    for f in prange(F):
        df = d[f]
        for i in range(NE):
            for j in range(NE):
                v = df[i, j]
                if i > 0:
                    if df[i - 1, j] + 3.0 < v:
                        v = df[i - 1, j] + 3.0
                    if j > 0 and df[i - 1, j - 1] + 4.0 < v:
                        v = df[i - 1, j - 1] + 4.0
                    if j < NE - 1 and df[i - 1, j + 1] + 4.0 < v:
                        v = df[i - 1, j + 1] + 4.0
                if j > 0 and df[i, j - 1] + 3.0 < v:
                    v = df[i, j - 1] + 3.0
                df[i, j] = v
        for i in range(NE - 1, -1, -1):
            for j in range(NE - 1, -1, -1):
                v = df[i, j]
                if i < NE - 1:
                    if df[i + 1, j] + 3.0 < v:
                        v = df[i + 1, j] + 3.0
                    if j < NE - 1 and df[i + 1, j + 1] + 4.0 < v:
                        v = df[i + 1, j + 1] + 4.0
                    if j > 0 and df[i + 1, j - 1] + 4.0 < v:
                        v = df[i + 1, j - 1] + 4.0
                if j < NE - 1 and df[i, j + 1] + 3.0 < v:
                    v = df[i, j + 1] + 3.0
                df[i, j] = v


def coast_distance_m(grid: Grid, ocean: np.ndarray, max_rounds: int = 16) -> np.ndarray:
    """Approximate (3-4 chamfer, ≤ ~8 % error) distance in metres from every
    cell to the nearest ocean cell, seam-free: the per-face chamfer passes
    are repeated with a halo exchange in between until nothing changes.
    Extended float32 array; 0 over ocean."""
    NE = grid.NE
    big = 3.0 * 4.0 * NE
    d = FaceField(grid, np.where(np.asarray(ocean, dtype=bool), 0.0, big).astype(np.float32), name="dist")
    for _ in range(max_rounds):
        before = d.data.copy()
        _chamfer_pass(d.data)
        d.exchange_halos(linear=True)
        if np.array_equal(before, d.data):
            break
    return (d.data / 3.0 * grid.cell_size_m).astype(np.float32)


def moisture_reach_m(grid: Grid, cp: ClimateParams) -> float:
    """E-folding fetch (metres) of the background rain-out: ``moisture_reach_m``
    if positive, else ``moisture_reach_frac × R_planet``."""
    if cp.moisture_reach_m > 0:
        return float(cp.moisture_reach_m)
    return float(cp.moisture_reach_frac) * grid.R_planet


def rainout_fraction(grid: Grid, surface: FaceField, wind: FaceField, cp: ClimateParams) -> np.ndarray:
    """Per-cell fraction of the arriving moisture that rains out in one
    sweep (extended float32 array, see the module docstring)."""
    L = moisture_reach_m(grid, cp)
    dt = float(cp.dt)
    speed_m = wind.vec_norm().data.astype(np.float64)
    step_m = np.maximum(speed_m, float(cp.calm_floor) * abs(float(cp.wind_speed)) * grid.cell_size_m) * dt
    rise = orographic_rise(surface, wind, int(cp.rise_smooth)).astype(np.float64) * dt
    f_base = 1.0 - np.exp(-step_m / L)
    f_oro = float(cp.k_oro) * (1.0 - np.exp(-rise / max(float(cp.oro_height_m), 1e-6)))
    return np.minimum(f_base + f_oro, 1.0).astype(np.float32)


def max_sweeps(grid: Grid, cp: ClimateParams) -> int:
    """Sweep cap of the automatic mode: ``n_advect_max_factor·N/(wind_speed·dt)``."""
    per = max(abs(float(cp.wind_speed) * float(cp.dt)), 1e-6)
    return max(1, int(math.ceil(float(cp.n_advect_max_factor) * grid.N / per)))


def advect_precip(grid: Grid, surface: FaceField, wind: FaceField, cp: ClimateParams, n_advect: int | None = None, log=None, info: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run the moisture advection.  ``surface`` (metres, halos exchanged;
    ocean = ``surface < 0``) and ``wind`` (contravariant cells/step, halos
    exchanged).  ``n_advect``: sweep count; ``None`` = ``cp.n_advect``;
    ``0`` = sweep until stationary (see the module docstring).  Returns
    ``(rain, moisture)``: rain per sweep per cell of the last sweep and the
    final moisture (extended float32; halo cells of ``rain`` are 0).
    ``info`` (optional dict) receives ``n_sweeps``, ``converged`` (None in
    fixed-count mode) and ``delta`` (last measured max moisture change)."""
    n = int(cp.n_advect if n_advect is None else n_advect)
    H, N = grid.H, grid.N
    sl = (slice(None), slice(H, H + N), slice(H, H + N))
    ocean = np.ascontiguousarray(surface.data < 0.0)
    frac = np.ascontiguousarray(rainout_fraction(grid, surface, wind, cp))
    w = np.ascontiguousarray(wind.data.astype(np.float32))
    dist = coast_distance_m(grid, ocean)
    m0 = float(cp.m_ocean) * np.exp(-dist / moisture_reach_m(grid, cp))
    m = FaceField(grid, np.where(ocean, float(cp.m_ocean), m0).astype(np.float32), name="moisture")
    m_out = m.zeros_like()
    rain = np.zeros((6, grid.NE, grid.NE), dtype=np.float32)
    dt = float(cp.dt)
    if abs(cp.wind_speed * dt) > H - 1 and log is not None:
        log(f"[climate] warning: wind_speed*dt = {cp.wind_speed * dt:.2f} cells exceeds the halo ({H - 1}); departure points are clamped")
    auto = n <= 0
    n_max = max_sweeps(grid, cp) if auto else n
    tol = float(cp.advect_tol) * float(cp.m_ocean)
    check = 8  # convergence test every `check` sweeps (a 6N² reduction)
    land_i = ~ocean[sl]
    k = 0
    delta = math.inf
    converged = False
    while k < n_max:
        m.exchange_halos(linear=True)
        _sweep(m.data, w, frac, ocean, H, N, dt, float(cp.m_ocean), m_out.data, rain)
        m, m_out = m_out, m
        k += 1
        if auto and k % check == 0:
            d = np.abs(m.data[sl] - m_out.data[sl])
            delta = float(d[land_i].max()) if land_i.any() else 0.0
            if delta < tol:
                converged = True
                break
    if info is not None:
        info.update({"n_sweeps": k, "converged": bool(converged) if auto else None, "delta": None if delta == math.inf else delta})
    if log is not None:
        L = moisture_reach_m(grid, cp)
        msg = f"stationary (max moisture change {delta:.1e})" if converged else (f"cap {n_max} reached (max moisture change {delta:.1e})" if auto else "fixed count")
        log(f"[climate] advection: {k} sweeps, {msg}; reach {L / 1000:.2f} km = {L / grid.cell_size_m:.0f} cells, oro scale height {cp.oro_height_m:.0f} m")
    return rain, m.data


def latitude_prior(lat_rad: np.ndarray, cp: ClimateParams) -> np.ndarray:
    """Multiplicative latitude prior: ``(1 + itcz_strength·G(lat; 0, itcz_width))
    · (1 − dry_band_strength·G(|lat|; dry_band_deg, dry_band_width))`` with
    Gaussian ``G`` (peak 1, sigma in degrees); clipped to ≥ 0."""
    lat = np.degrees(np.asarray(lat_rad, dtype=np.float64))
    wet = 1.0 + cp.itcz_strength * np.exp(-0.5 * (lat / max(cp.itcz_width_deg, 1e-6)) ** 2)
    dry = 1.0 - cp.dry_band_strength * np.exp(-0.5 * ((np.abs(lat) - cp.dry_band_deg) / max(cp.dry_band_width_deg, 1e-6)) ** 2)
    return np.maximum(wet * dry, 0.0)


def finalise_precip(grid: Grid, rain: np.ndarray, land: np.ndarray, cp: ClimateParams) -> np.ndarray:
    """Rain (extended, any positive scale) -> the ``precip`` field: volume per
    cell per erosion iteration (float32, extended) whose mean over interior
    land cells is exactly ``precip_mean``.  ``land`` may be an extended
    ``(6, NE, NE)`` or an interior ``(6, N, N)`` bool mask; the background
    ``precip_floor`` is added on land only."""
    H, N = grid.H, grid.N
    sl = (slice(None), slice(H, H + N), slice(H, H + N))
    r = np.asarray(rain, dtype=np.float64)
    if cp.precip_smooth > 0:
        from .wind import smooth_field

        rf = FaceField(grid, r.astype(np.float32), name="rain")
        rf.exchange_halos(linear=True)
        r = smooth_field(rf, int(cp.precip_smooth)).data.astype(np.float64)
    land = np.asarray(land, dtype=bool)
    if land.shape[1] == N:  # interior mask -> extended (halo cells: not land)
        land_e = np.zeros((6, grid.NE, grid.NE), dtype=bool)
        land_e[sl] = land
    else:
        land_e = land
    land_i = land_e[sl]
    sel = land_i if land_i.any() else np.ones_like(land_i)
    scale = r[sl][sel].mean()
    r = r / scale if scale > 0 else r
    r = (r + np.where(land_e, float(cp.precip_floor), 0.0)) * latitude_prior(grid.latitude(), cp)
    vol = r * grid.cell_area.astype(np.float64) / grid.cell_size_m**2
    mean_land = vol[sl][sel].mean()
    if mean_land > 0:
        vol *= float(cp.precip_mean) / mean_land
    return vol.astype(np.float32)


def precipitation(grid: Grid, surface: FaceField, wind: FaceField, cp: ClimateParams, log=None) -> tuple[FaceField, dict]:
    """Full precipitation stage: returns the ``precip`` FaceField (halos
    exchanged) and an info dict."""
    ainfo: dict = {}
    rain, moisture = advect_precip(grid, surface, wind, cp, log=log, info=ainfo)
    land = surface.data >= 0.0
    vol = finalise_precip(grid, rain, land, cp)
    f = FaceField(grid, vol, name="precip")
    f.exchange_halos(linear=True)
    li = land[:, grid.H : grid.H + grid.N, grid.H : grid.H + grid.N]
    inter = f.interior
    pl = inter[li] if li.any() else np.zeros(1, np.float32)
    info = {
        "n_sweeps": int(ainfo.get("n_sweeps", 0)),
        "advect_converged": ainfo.get("converged"),
        "moisture_reach_m": moisture_reach_m(grid, cp),
        "precip_land_mean": float(pl.mean()),
        "precip_land_median": float(np.median(pl)),
        "precip_land_max": float(pl.max()),
        "land_dry_fraction": float((pl < 0.1 * cp.precip_mean).mean()),
        "moisture_land_mean": float(moisture[:, grid.H : grid.H + grid.N, grid.H : grid.H + grid.N][li].mean()) if li.any() else 0.0,
    }
    return f, info


__all__ = ["advect_precip", "coast_distance_m", "orographic_rise", "rainout_fraction", "moisture_reach_m", "max_sweeps", "latitude_prior", "finalise_precip", "precipitation"]
