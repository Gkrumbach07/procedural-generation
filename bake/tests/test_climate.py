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
from globe.climate.precipitation import advect_precip, finalise_precip, latitude_prior, max_sweeps, moisture_reach_m, orographic_rise, rainout_fraction
from globe.climate.temperature import evaporation, temperature
from globe.climate.wind import cells_to_tangent3, circulation_profile, geographic_frame, tangent3_to_cells, wind_field
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
