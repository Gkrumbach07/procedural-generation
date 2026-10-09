"""Climate stage tests (PLAN.md section 7): temperature, wind bands,
orographic precipitation, latitude prior, seams, determinism, runtime.

Upstream data is synthetic (analytic island planets) or the tectonics
stub; the real tectonics stage is never required."""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from globe.climate import run as climate_run
from globe.climate.precipitation import advect_precip, cold_air, eddy_resupply, finalise_precip, rain_drift_passes, latitude_prior, max_sweeps, moisture_reach_m, orographic_rise, precipitation, rainout_fraction
from globe.climate.temperature import evaporation, temperature
from globe.climate.wind import cells_to_tangent3, circulation_profile, geographic_frame, storm_belt, storminess, tangent3_to_cells, wind_field
from globe.config import WorldParams
from globe.cubesphere import get_grid
from globe.field import FaceField
from globe.io.world_store import WorldStore
from globe.pipeline import bake
from globe.stubs import stub_tectonics

try:  # PLAN 15 seam metric lives in the viz tests
    from tests.test_viz import seam_discontinuity
except ImportError:  # pragma: no cover
    from test_viz import seam_discontinuity


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _params(N: int, n_advect: int = 40) -> WorldParams:
    p = WorldParams.small_world()
    p.world.N_c = N
    p.world.T = max(16, N // 4)
    p.climate = dataclasses.replace(p.climate, n_advect=n_advect)
    p.validate()
    return p


def _island_bedrock(grid) -> FaceField:
    """A continent (Gaussian cap) with a ridge across it, ocean elsewhere."""
    p = grid.centers
    c = np.array([1.0, 0.3, 0.2])
    c /= np.linalg.norm(c)
    d = np.arccos(np.clip(p @ c, -1.0, 1.0))
    h = 3000.0 * np.exp(-((d / 0.6) ** 2)) - 800.0 + 1500.0 * np.exp(-(((p[..., 2] - 0.1) / 0.08) ** 2)) * np.exp(-((d / 0.9) ** 2))
    f = FaceField(grid, h.astype(np.float32), name="bedrock")
    f.exchange_halos()
    return f


def _flat_ocean(grid) -> FaceField:
    return FaceField.full(grid, -100.0, name="bedrock")


def _geo_components(grid, wind: FaceField):
    """(zonal, meridional) wind in cells per step on every extended cell."""
    east, north, _ = geographic_frame(grid)
    w3 = cells_to_tangent3(grid, wind.data) * grid.R_planet / grid.cell_size_m
    return np.sum(w3 * east, -1), np.sum(w3 * north, -1)


# --------------------------------------------------------------------------
# temperature / evaporation
# --------------------------------------------------------------------------
def test_temperature_decreases_with_latitude_and_altitude():
    p = _params(32)
    g = p.coarse_grid()
    cp = p.climate
    T0 = temperature(g, np.zeros((6, g.NE, g.NE), np.float32), cp)
    lat = np.abs(g.latitude())
    # monotone in |lat|: sort by latitude and check T is non-increasing (ties allowed)
    order = np.argsort(lat.ravel())
    assert np.all(np.diff(T0.ravel()[order]) <= 1e-4)
    assert T0.max() == pytest.approx(cp.T_eq, abs=0.05)  # equator, sea level
    # the formula: |lat| = pi/2 (a pole) contributes exactly k_lat (28 - 45 = -17 C)
    expect = cp.T_eq - cp.k_lat * (lat / (np.pi / 2)) ** 1.5
    assert np.allclose(T0, expect, atol=1e-4)
    assert T0.min() < cp.T_eq - 0.9 * cp.k_lat  # the polar cells are within 10 % of the pole value
    # altitude: 1 km costs `lapse` degrees; below sea level costs nothing
    T1 = temperature(g, np.full((6, g.NE, g.NE), 1000.0, np.float32), cp)
    assert np.allclose(T0 - T1, cp.lapse, atol=1e-4)
    Tm = temperature(g, np.full((6, g.NE, g.NE), -500.0, np.float32), cp)
    assert np.allclose(Tm, T0)
    ev = evaporation(T0, cp)
    assert ev.dtype == np.float32 and ev.min() >= 0
    assert np.allclose(ev, cp.k_evap * np.maximum(T0, 0))


# --------------------------------------------------------------------------
# wind
# --------------------------------------------------------------------------
def test_wind_bands_sign_and_magnitude():
    p = _params(64)
    g = p.coarse_grid()
    cp = p.climate
    w = wind_field(g, _flat_ocean(g), cp)  # flat: no deflection
    assert w.is_vector and np.isfinite(w.data).all()
    zonal, merid = _geo_components(g, w)
    lat = np.degrees(g.latitude())
    H, N = g.H, g.N
    sl = (slice(None), slice(H, H + N), slice(H, H + N))
    zonal, merid, lat = zonal[sl], merid[sl], lat[sl]
    for lo, hi, zsign in ((5, 25, -1), (35, 55, +1), (65, 85, -1)):
        for hemi in (+1, -1):
            sel = (lat * hemi > lo) & (lat * hemi < hi)
            assert sel.any()
            assert np.all(zsign * zonal[sel] > 0.5 * cp.wind_speed), (lo, hi, hemi)
            # surface branch: equatorward under easterlies, poleward under westerlies
            assert np.all(zsign * hemi * merid[sel] > 0), (lo, hi, hemi)
    # equator: purely zonal
    eq = np.abs(lat) < 1.0
    assert np.abs(merid[eq]).max() < 0.05 * cp.wind_speed
    speed = w.vec_norm().interior / g.cell_size_m
    assert speed.max() <= cp.wind_speed * 1.001
    assert np.median(speed) == pytest.approx(cp.wind_speed, rel=0.02)
    # band centres are exactly wind_speed
    centre = (np.abs(np.abs(lat) - 45.0) < 3.0) | (np.abs(np.abs(lat) - 15.0) < 3.0)
    assert np.allclose(speed[centre], cp.wind_speed, rtol=1e-2)  # tanh(15/6) = 0.987 on the meridional part


def test_wind_component_roundtrip_and_profile():
    g = get_grid(24, 4, 50.0)
    rng = np.random.default_rng(0)
    ab = rng.normal(size=(6, g.NE, g.NE, 2))
    back = tangent3_to_cells(g, cells_to_tangent3(g, ab))
    assert np.allclose(ab, back, atol=1e-9)
    cp = WorldParams().climate
    z, m = circulation_profile(np.radians(np.array([0.0, 15.0, 45.0, 75.0, -45.0])), cp)
    assert np.sign(z).tolist() == [-1, -1, 1, -1, 1]
    assert m[0] == 0.0 and m[2] > 0 and m[4] < 0


def test_wind_deflection_preserves_speed_and_is_finite():
    p = _params(64)
    g = p.coarse_grid()
    bed = _island_bedrock(g)
    cp = p.climate
    w = wind_field(g, bed, cp)
    w0 = wind_field(g, bed, dataclasses.replace(cp, wind_deflection=0.0))
    assert np.isfinite(w.data).all()
    s, s0 = w.vec_norm().interior, w0.vec_norm().interior
    assert np.allclose(s, s0, rtol=1e-3, atol=1e-3)
    # the deflection changes directions somewhere on the ridge, by a bounded angle
    a3 = cells_to_tangent3(g, w.data)
    b3 = cells_to_tangent3(g, w0.data)
    cosang = np.sum(a3 * b3, -1) / np.maximum(np.linalg.norm(a3, axis=-1) * np.linalg.norm(b3, axis=-1), 1e-30)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    assert ang.max() > 3.0 and ang.max() < 60.0


def test_wind_halos_have_no_seams():
    p = _params(64)
    g = p.coarse_grid()
    w = wind_field(g, _island_bedrock(g), p.climate)
    assert seam_discontinuity(w) < 3.0  # |wind|
    zonal, _ = _geo_components(g, w)
    assert seam_discontinuity(FaceField(g, zonal.astype(np.float32), name="zonal")) < 3.0
    # halo cells hold the same 3-D vector as the analytic extended field
    w0 = wind_field(g, _flat_ocean(g), p.climate)
    from globe.climate.wind import circulation_wind3

    analytic = circulation_wind3(g, p.climate)
    got = cells_to_tangent3(g, w0.data)
    err = np.linalg.norm(got - analytic, axis=-1) / np.linalg.norm(analytic, axis=-1).max()
    assert err[:, : g.H, :].max() < 0.05 and err[:, :, -g.H :].max() < 0.05  # corner blocks are the least accurate


# --------------------------------------------------------------------------
# precipitation
# --------------------------------------------------------------------------
def test_single_ridge_windward_wetter_than_lee():
    """One ridge on the equatorial +X face under a uniform easterly-to-westerly
    wind: windward slope > 2x lee slope (PLAN 7 hand check)."""
    p = _params(64, n_advect=60)
    g = p.coarse_grid()
    cp = dataclasses.replace(p.climate, wind_deflection=0.0)
    N, H = g.N, g.H
    # ocean everywhere, a ridge along i (constant j band) in the middle of face 0
    h = np.full((6, g.NE, g.NE), -50.0, np.float32)
    jj = np.arange(g.NE) - H
    ridge = 1500.0 * np.exp(-(((jj - N / 2) / 5.0) ** 2))
    band = (jj >= N // 4) & (jj <= 3 * N // 4)
    h[0] += np.where(band[None, :], ridge[None, :] + 100.0, 0.0)
    bed = FaceField(g, h, name="bedrock")
    bed.exchange_halos()
    # uniform wind: pure +east (on face 0 that is along -j)
    east, _, _ = geographic_frame(g)
    w3 = east * (cp.wind_speed * g.cell_size_m / g.R_planet)
    wind = FaceField(g, tangent3_to_cells(g, w3).astype(np.float32), is_vector=True, name="wind")
    wind.exchange_halos()
    rise = orographic_rise(bed, wind)
    rain, _ = advect_precip(g, bed, wind, cp)
    face = rain[0, H : H + N, H : H + N]
    land = h[0, H : H + N, H : H + N] >= 0
    up = rise[0, H : H + N, H : H + N] > 1.0  # windward slope
    dh = bed.directional_derivative(wind).data[0, H : H + N, H : H + N]
    lee = land & (dh < -1.0)
    assert up.sum() > 50 and lee.sum() > 50
    assert face[up & land].mean() > 2.0 * face[lee].mean()
    assert rain.min() >= 0.0


def test_precip_properties_and_latitude_prior():
    p = _params(64, n_advect=40)
    g = p.coarse_grid()
    bed = _island_bedrock(g)
    cp = p.climate
    fields, info = climate_run.compute(g, bed, p, log=None)
    pr = fields["precip"]
    assert pr.dtype == np.float32 and np.isfinite(pr.data).all()
    assert pr.interior.min() >= 0.0
    land = bed.interior >= 0.0
    assert land.any() and (~land).any()
    assert pr.interior[land].mean() == pytest.approx(cp.precip_mean, rel=1e-5)
    assert info["precip_land_mean"] == pytest.approx(cp.precip_mean, rel=1e-5)
    # wet equatorial band: land inside the ITCZ is wetter than land in the dry band
    lat = np.degrees(g.latitude())[:, g.H : g.H + g.N, g.H : g.H + g.N]
    eq = land & (np.abs(lat) < cp.itcz_width_deg)
    dry = land & (np.abs(np.abs(lat) - cp.dry_band_deg) < cp.dry_band_width_deg)
    assert eq.any() and dry.any()
    assert pr.interior[eq].mean() > 1.5 * pr.interior[dry].mean()
    # the prior itself
    lp = latitude_prior(np.radians(np.array([0.0, cp.dry_band_deg, -cp.dry_band_deg, 60.0])), cp)
    assert lp[0] == pytest.approx(1.0 + cp.itcz_strength, rel=1e-2)  # x the dry-band tail
    assert lp[1] == pytest.approx(1.0 - cp.dry_band_strength, rel=0.05) and lp[2] == lp[1]
    assert lp.min() >= 0.0
    # volume: rain x cell_area / cell_size^2 -- a uniform rain gives precip proportional to cell area
    vol = finalise_precip(g, np.ones((6, g.NE, g.NE)), land, dataclasses.replace(cp, itcz_strength=0.0, dry_band_strength=0.0, precip_floor=0.0))
    ratio = vol / (g.cell_area / g.cell_size_m**2)
    assert np.allclose(ratio[:, 1:-1, 1:-1], ratio[0, g.H, g.H], rtol=1e-5)
    # evap contract
    ev = fields["evap"]
    assert np.allclose(ev.data, cp.k_evap * np.maximum(fields["temperature"].data, 0))


def test_ocean_precip_has_no_floor():
    """Ocean cells hold their rain on the land scale but without the land
    floor (docs/DEVELOPING.md): a uniform rain gives ocean/land = 1/(1+floor)."""
    p = _params(32, n_advect=0)
    g = p.coarse_grid()
    bed = _island_bedrock(g)
    land = bed.interior >= 0.0
    cp = dataclasses.replace(p.climate, itcz_strength=0.0, dry_band_strength=0.0, precip_floor=0.25)
    vol = finalise_precip(g, np.ones((6, g.NE, g.NE)), land, cp)
    ratio = vol[:, g.H : g.H + g.N, g.H : g.H + g.N] / (g.cell_area[:, g.H : g.H + g.N, g.H : g.H + g.N] / g.cell_size_m**2)
    assert np.allclose(ratio[land], ratio[land][0], rtol=1e-5) and np.allclose(ratio[~land], ratio[~land][0], rtol=1e-5)
    assert ratio[~land][0] / ratio[land][0] == pytest.approx(1.0 / 1.25, rel=1e-5)
    fields, info = climate_run.compute(g, bed, p, log=None)
    assert info["advect_converged"] is True and fields["precip"].interior.min() >= 0.0


def test_rainout_is_parameterised_in_physical_length():
    """The same planet (same R_planet, same relief) at two resolutions:
    the per-sweep rain-out fraction along a step scales with the step
    length and the stationary moisture over land agrees at the continent
    scale (finding: reach must not be a fixed number of cells)."""
    res = {}
    for N, cell in ((32, 100.0), (64, 50.0)):
        p = WorldParams.small_world()
        p.world.N_c, p.world.cell_size_m, p.world.T = N, cell, max(16, N // 4)
        p.climate = dataclasses.replace(p.climate, n_advect=0, wind_deflection=0.0)
        p.validate()
        g = p.coarse_grid()
        bed = _island_bedrock(g)
        wind = wind_field(g, bed, p.climate)
        frac = rainout_fraction(g, bed, wind, p.climate)
        info = {}
        rain, m = advect_precip(g, bed, wind, p.climate, info=info)
        assert info["converged"] and info["n_sweeps"] <= max_sweeps(g, p.climate)
        land = bed.interior >= 0.0
        # flat-ocean cells at the band centres: frac = 1 - exp(-cell/L) exactly
        L = moisture_reach_m(g, p.climate)
        assert L == pytest.approx(p.climate.moisture_reach_frac * g.R_planet)
        H = g.H
        from scipy import ndimage

        open_ocean = np.stack([ndimage.binary_erosion(~land[f], iterations=3) for f in range(6)])  # no smoothed relief leaks in
        oc = open_ocean & (np.abs(np.degrees(g.latitude())[:, H : H + N, H : H + N] - 45.0) < 4.0)
        assert oc.sum() > 5
        assert np.allclose(frac[:, H : H + N, H : H + N][oc], 1.0 - np.exp(-cell * p.climate.wind_speed / L), rtol=0.02)
        res[N] = (float(m[:, H : H + N, H : H + N][land].mean()), g.R_planet)
    assert res[32][1] == pytest.approx(res[64][1])  # same planet
    assert res[32][0] == pytest.approx(res[64][0], rel=0.15)  # same moisture reach in metres
    assert 0.2 < res[64][0] < 0.9  # the interior is neither bone dry nor saturated


#: the weather the mean wind leaves out, at the values its tests were measured with (config.ClimateParams' defaults since earth-v32)
WEATHER = {"storm_front": 1.0, "eddy_reach_frac": 0.15, "cold_air_k": 0.067}


def _front_against_its_sides(grid, bed: FaceField, cp) -> float:
    """Mean rain depth on the 60 degree line of the northern hemisphere over
    the mean of the bands 6-10 degrees either side of it."""
    H, N = grid.H, grid.N
    sl = (slice(None), slice(H, H + N), slice(H, H + N))
    f, _ = precipitation(grid, bed, wind_field(grid, bed, cp), cp)
    r = (f.data.astype(np.float64) / grid.cell_area * grid.cell_size_m**2)[sl]
    lat = np.degrees(grid.latitude())[sl]
    sides = 0.5 * (r[(lat > 50) & (lat < 54)].mean() + r[(lat > 66) & (lat < 70)].mean())
    return float(r[(lat > 58.5) & (lat < 61.5)].mean() / sides)


def test_the_polar_front_is_no_calm():
    """The mean wind dips through zero at 60 degrees as it does at 30, but
    there the surface branches converge: it is the storm track, and with
    ``climate.storm_front`` its rain-out is a band centre's.  Read as a calm
    it was a ring with half the rain over an open sea."""
    p = _params(64, n_advect=0)
    grid = p.coarse_grid()
    cp = dataclasses.replace(p.climate, **WEATHER)
    # the equator and the westerlies' whole band converge; the horse latitudes and the polar cap diverge
    assert storm_belt(np.radians([0.0, 30.0, 45.0, 60.0, 85.0, -60.0, -30.0]), cp).tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    sea = _flat_ocean(grid)
    assert _front_against_its_sides(grid, sea, cp) > 0.97
    assert _front_against_its_sides(grid, sea, dataclasses.replace(cp, storm_front=0.0)) < 0.6


def test_the_weather_resupplies_the_air_the_mean_wind_starves():
    """A continent across the polar front: the mean wind creeps towards the
    front's stagnation line raining as it goes and arrives dry, so the line
    was a desert through the continent.  With ``climate.eddy_reach_frac``
    the air is resupplied from the nearest sea and what is left is the
    interior being farther from one (20 degrees against 12: 0.76 by the
    moisture reach)."""
    p = _params(64, n_advect=0)
    grid = p.coarse_grid()
    cp = dataclasses.replace(p.climate, **WEATHER)
    lat = np.degrees(grid.latitude())
    bed = FaceField(grid, np.where((lat > 40.0) & (lat < 80.0), 200.0, -500.0).astype(np.float32), name="bedrock")
    bed.exchange_halos()
    assert _front_against_its_sides(grid, bed, cp) > 0.7
    assert _front_against_its_sides(grid, bed, dataclasses.replace(cp, storm_front=0.0, eddy_reach_frac=0.0)) < 0.45
    # each half does its part: the front's step without the supply, the supply without the step
    assert _front_against_its_sides(grid, bed, dataclasses.replace(cp, eddy_reach_frac=0.0)) < 0.6
    assert _front_against_its_sides(grid, bed, dataclasses.replace(cp, storm_front=0.0)) < 0.6
    # no supply, no change to the sweep: the share is exactly zero
    assert not eddy_resupply(grid, wind_field(grid, bed, cp), dataclasses.replace(cp, eddy_reach_frac=0.0)).any()


def test_the_trades_are_their_own_weather_and_cold_air_is_dry():
    """The resupply is the extratropics' and the equator's, not the trades';
    and below freezing the air holds less of the sea's moisture, so the rain
    over an open sea falls off towards the pole instead of staying level."""
    p = _params(64, n_advect=0)
    grid = p.coarse_grid()
    cp = dataclasses.replace(p.climate, **WEATHER)
    st = storminess(np.radians([0.0, 15.0, 27.0, 33.0, 45.0, 60.0, 85.0, -15.0, -45.0]), cp)
    assert st[0] == 1.0 and st[1] < 0.05 and st[7] < 0.05            # the equator's convergence; next to nothing under the trades
    assert st[2] < 0.01 and np.all(st[3:7] == 1.0) and st[8] == 1.0   # blended in across the horse latitudes, 1 to the pole
    H, N = grid.H, grid.N
    sl = (slice(None), slice(H, H + N), slice(H, H + N))
    lat = np.degrees(grid.latitude())[sl]
    ca = cold_air(grid, cp)[sl]
    t_sea = cp.T_eq - cp.k_lat * (np.abs(lat) / 90.0) ** 1.5
    assert np.all(ca[t_sea >= 0.0] == 1.0) and np.all(ca[t_sea < -1.0] < 1.0)
    assert ca.min() == pytest.approx(np.exp(cp.cold_air_k * t_sea.min()), rel=1e-6)
    assert np.all(cold_air(grid, dataclasses.replace(cp, cold_air_k=0.0)) == 1.0)
    sea = _flat_ocean(grid)

    def polar_over_mid(c):
        f, _ = precipitation(grid, sea, wind_field(grid, sea, c), c)
        r = (f.data.astype(np.float64) / grid.cell_area * grid.cell_size_m**2)[sl]
        return float(r[(lat > 78) & (lat < 84)].mean() / r[(lat > 42) & (lat < 54)].mean())

    assert polar_over_mid(cp) < 0.7
    assert polar_over_mid(dataclasses.replace(cp, cold_air_k=0.0)) > 0.95


def test_rain_drifts_while_it_falls():
    """``climate.rain_drift_frac`` spreads the rain over the distance it
    drifts: a length, so the passes follow the cell size, and none on a body
    whose cells are wider than the drift.  The land's mean stays what
    ``precip_mean`` says; the rain that the climb put on one face of a ridge
    is spread over both."""
    p = _params(96, n_advect=0)
    grid = p.coarse_grid()
    cp = dataclasses.replace(p.climate, rain_drift_frac=0.03)
    cells = 0.03 * grid.R_planet / grid.cell_size_m
    assert rain_drift_passes(grid, cp) == round(2.0 * cells * cells) > 0
    assert rain_drift_passes(grid, p.climate) == 0
    assert rain_drift_passes(grid, dataclasses.replace(cp, rain_drift_frac=0.2 * grid.cell_size_m / grid.R_planet)) == 0
    bed = _island_bedrock(grid)
    H, N = grid.H, grid.N
    sl = (slice(None), slice(H, H + N), slice(H, H + N))
    land = bed.data[sl] >= 0.0
    out = {}
    for name, c in (("off", p.climate), ("on", cp)):
        f, _ = precipitation(grid, bed, wind_field(grid, bed, c), c)
        out[name] = f.data[sl].astype(np.float64)
        assert out[name][land].mean() == pytest.approx(c.precip_mean, rel=1e-5)
    assert out["on"][land].max() < 0.9 * out["off"][land].max()
    assert out["on"][land].std() < out["off"][land].std()


def test_fixed_sweep_count_reaches_the_stationary_rain():
    """Explicit n_advect (>= the auto count) gives the same steady state."""
    p = _params(64, n_advect=0)
    g = p.coarse_grid()
    bed = _island_bedrock(g)
    wind = wind_field(g, bed, p.climate)
    info = {}
    rain_auto, _ = advect_precip(g, bed, wind, p.climate, info=info)
    rain_fix, _ = advect_precip(g, bed, wind, p.climate, n_advect=2 * info["n_sweeps"])
    diff = np.abs(rain_auto - rain_fix)
    # the stopping rule (max change per sweep < advect_tol) leaves the slow modes next to the
    # calm belts (|wind| -> 0 at 30/60 degrees; moisture creeps across them) a few % short;
    # the bulk of the field is stationary
    windy = wind.vec_norm().data / g.cell_size_m > 0.5 * p.climate.wind_speed
    print(f"auto {info['n_sweeps']} sweeps: max rain change windy {diff[windy].max() / rain_fix.max():.2e}, all {diff.max() / rain_fix.max():.2e}")
    assert diff[windy].max() < 0.05 * rain_fix.max()
    assert diff.max() < 0.1 * rain_fix.max()
    assert np.percentile(diff[windy], 99) < 0.02 * rain_fix.max()


def test_precip_seams():
    p = _params(64, n_advect=40)
    g = p.coarse_grid()
    fields, _ = climate_run.compute(g, _island_bedrock(g), p, log=None)
    assert seam_discontinuity(fields["precip"]) < 3.0
    assert seam_discontinuity(fields["temperature"]) < 3.0


# --------------------------------------------------------------------------
# pipeline, determinism, runtime
# --------------------------------------------------------------------------
def _stub_world(root, params) -> WorldStore:
    store = WorldStore(root, create=True)
    store.init_manifest(params, params.coarse_grid())
    stub_tectonics(store, params, lambda m: None)
    store.mark_stage("tectonics", [], {"stub": True}, 0.0, params=params)
    return store


def test_stage_runs_in_pipeline_and_is_deterministic(tmp_path):
    p = WorldParams.tiny_world(seed=3)
    hashes = []
    for k in range(2):
        store = _stub_world(tmp_path / f"w{k}", p)
        store = bake(store.root, p, from_stage="climate", to_stage="climate", logger=lambda m: None)
        assert store.stage_done("climate", p)
        for name in climate_run.OUTPUTS:
            assert store.has_field(name)
        assert (store.quicklook_dir / "climate.png").exists()
        assert (store.quicklook_dir / "climate_temperature.png").exists()
        hashes.append(store.stage_info("climate")["hash"])
        info = store.stage_info("climate")["info"]
        assert info["precip_land_mean"] == pytest.approx(p.climate.precip_mean, rel=1e-5)
    assert hashes[0] == hashes[1]  # byte-identical outputs
    grid = p.coarse_grid()
    w = store.load_field("wind", grid)
    assert w.is_vector and w.dtype == np.float32
    for name in ("temperature", "precip", "evap"):
        assert store.load_field(name, grid).dtype == np.float32
    # byte-level comparison of the files themselves
    for name in climate_run.OUTPUTS:
        for f in range(6):
            a = (tmp_path / "w0" / "coarse" / f"{name}.f{f}.npy").read_bytes()
            b = (tmp_path / "w1" / "coarse" / f"{name}.f{f}.npy").read_bytes()
            assert a == b, (name, f)


def test_runtime_advection_256():
    """n_advect = 200 at N_c = 256: seconds, not minutes (auto mode at
    N_c = 1024 needs ~2-3 N sweeps, see the report)."""
    p = _params(256, n_advect=200)
    g = p.coarse_grid()
    bed = _island_bedrock(g)
    wind = wind_field(g, bed, p.climate)
    advect_precip(g, bed, wind, p.climate, n_advect=2)  # JIT warm-up
    t0 = time.time()
    rain, _ = advect_precip(g, bed, wind, p.climate)
    dt = time.time() - t0
    print(f"advection N_c=256 n_advect=200: {dt:.2f}s")
    assert np.isfinite(rain).all()
    assert dt < 20.0, f"200 sweeps took {dt:.1f}s"


def test_the_year_swings_with_the_sun_and_the_distance_from_the_sea():
    """``temperature.seasonal_range`` (the ``temp_range`` field): the warmest
    month less the coldest.  Nothing on the equator, where the two solstices
    have the same sun, and most towards the poles; a fifth as much over the
    sea as deep inland (``season_sea`` / ``season_land``), and on land more
    the farther from the coast.  Earth: 8-10 C over the sea at 45 degrees,
    35-45 C in the middle of a continent there."""
    from globe.climate.temperature import daily_sun, seasonal_range

    tilt = np.radians(23.44)
    lat = np.radians(np.array([0.0, 45.0, 66.56, 90.0]))
    s, w = daily_sun(lat, tilt), daily_sun(lat, -tilt)
    assert s[0] == pytest.approx(w[0]) and w[3] == 0.0 and w[2] == pytest.approx(0.0, abs=1.0)   # no swing on the equator; the polar night
    assert 370.0 < s[1] - w[1] < 395.0 and s[3] > s[1] > s[0]                                    # 45 degrees: some 380 W/m2; a summer pole has the most sun there is
    p = _params(32)
    grid = p.coarse_grid()
    bed = _island_bedrock(grid)
    ocean = bed.data < 0.0
    rng = seasonal_range(grid, ocean, p.climate)
    latd = np.degrees(np.abs(grid.latitude()))
    assert rng.shape == ocean.shape and rng.min() >= 0.0
    sea45 = ocean & (latd > 43.0) & (latd < 47.0)
    assert 7.0 < np.median(rng[sea45]) < 10.0 and rng[ocean & (latd < 2.0)].max() < 1.0
    all_land = seasonal_range(grid, np.zeros_like(ocean), dataclasses.replace(p.climate, season_reach_frac=1e-6))
    assert 39.0 < np.median(all_land[(latd > 43.0) & (latd < 47.0)]) < 45.0                      # the middle of a continent
    land = ~ocean
    if land.any():
        assert (rng[land] >= rng[ocean].min()).all() and rng[land].max() <= all_land.max() + 1e-3
    fields, info = climate_run.compute(grid, bed, p)
    assert np.array_equal(fields["temp_range"].data, rng) and "temp_range" in climate_run.OUTPUTS and info["T_range_land_median"] is not None


def test_the_years_rain_is_its_seasons():
    """``climate.seasons`` (``precipitation.seasonal_precipitation``): a
    summer, a winter and an equinox twice, each on its own wind with the
    circulation's belts moved with the sun.  With no shift and no monsoon the
    seasons are the year, and the stage's rain is what one sweep gave; with
    them the rain is another field of the same land mean, and
    ``precip_summer`` says which half of the year a cell's rain falls in."""
    from globe.climate.wind import season_latitude, seasonal_wind

    p = _params(32)
    grid = p.coarse_grid()
    bed = _island_bedrock(grid)
    land = bed.interior >= 0.0
    year, _ = climate_run.compute(grid, bed, p)
    assert "precip_summer" not in year
    cp = dataclasses.replace(p.climate, seasons=True)
    lat = np.asarray(grid.latitude())
    assert np.allclose(season_latitude(grid, cp, 1.0), lat - np.radians(7.0)) and np.array_equal(season_latitude(grid, cp, 0.0), lat)
    same = dataclasses.replace(cp, season_shift_deg=0.0, monsoon=0.0)
    f0, _ = climate_run.compute(grid, bed, dataclasses.replace(p, climate=same))
    assert np.allclose(f0["precip"].interior, year["precip"].interior, rtol=2e-4, atol=1e-6)        # no seasons in it: the yearly stage's rain
    assert np.allclose(f0["precip_summer"].interior[land], 0.5, atol=1e-4)
    f1, info = climate_run.compute(grid, bed, dataclasses.replace(p, climate=cp))
    pr, sm = f1["precip"].interior, f1["precip_summer"].interior
    assert pr[land].mean() == pytest.approx(p.climate.precip_mean, rel=1e-5) and pr.min() >= 0.0
    assert sm.min() >= 0.0 and sm.max() <= 1.0 and sm[land].std() > 0.02
    assert not np.allclose(pr[land], year["precip"].interior[land], rtol=0.05)                       # the seasons move the rain
    assert np.array_equal(f1["wind"].data, year["wind"].data) and np.array_equal(f1["temperature"].data, year["temperature"].data)
    assert 0.0 <= info["summer_wet_land"] <= 1.0 and "winter_wet_land" in info
    # the monsoon: a season's wind is not the equinox's, and no faster than a band wind allows twice over
    w = seasonal_wind(grid, bed, cp, 1.0, f1["temp_range"].data)
    assert not np.allclose(w.data, year["wind"].data) and np.isfinite(w.data).all()
