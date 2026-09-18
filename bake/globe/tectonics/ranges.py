"""Ranges inside the orogenic belts (``tectonics.ranges_amp``).

The belts leave tectonics as smooth swells: the segment cloud is 160 km
apart on the earth preset and the splat that rasterises it blurs by ~137 km,
so nothing under a few hundred km survives, and a belt 500-1,700 km wide
reads as one mountain.  A real orogen's high ground is ranges and basins:
the Cordillera is ~1,500 km across and made of ranges tens of km wide
running along the belt.  Erosion cannot make those -- it carves what it is
handed.

This adds them as oriented noise: a sparse sum of Gabor kernels whose
carrier runs *across* the belt, so its crests run along it.  The across-belt
direction is the principal axis of the smoothed structure tensor of the
belt height on the tect grid (the belt's own gradient, averaged so a plateau
top takes its flanks' orientation), and each kernel takes the orientation at
its own centre, so the pattern bends with the belt without the shear a
global phase would have.  Kernels sit at hashed positions on the unit
sphere, so the field is continuous across cube faces.  Where the belt has no
clear strike (coherence ~ 0: a knot of crossing belts) the kernels are
round, and the ranges become a massif.

The amplitude is ``ranges_amp`` times the belt's height above
``ranges_base_m``: plains and cratons are untouched, and a 3 km belt with
``ranges_amp`` 0.4 gets ranges ~1.2 km above and basins ~1.2 km below its
swell.  The noise is zero mean, and the caller re-derives sea level after it.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..cubesphere import from_sphere_v
from ..field import FaceField
from .collision import gaussian_smooth


def strike_frame(bed_t: np.ndarray, grid, sigma_cells: float) -> tuple[np.ndarray, np.ndarray]:
    """``(n, coherence)`` on ``grid`` (the tect grid): the across-belt unit
    vector (6, N, N, 3) and the structure tensor's coherence (6, N, N) in
    [0, 1], from the gradient of ``bed_t`` (6, N, N) smoothed over
    ``sigma_cells``.  Seamless: the gradient reads the exchanged halos and the
    tensor is smoothed in global coordinates."""
    H, N = grid.H, grid.N
    f = FaceField.from_interior(grid, np.asarray(bed_t, np.float64), exchange=True)
    d = f.data
    C = grid.centers
    di = d[:, 2:, 1:-1] - d[:, :-2, 1:-1]
    dj = d[:, 1:-1, 2:] - d[:, 1:-1, :-2]
    ei = C[:, 2:, 1:-1] - C[:, :-2, 1:-1]
    ej = C[:, 1:-1, 2:] - C[:, 1:-1, :-2]
    a11 = (ei * ei).sum(-1)
    a12 = (ei * ej).sum(-1)
    a22 = (ej * ej).sum(-1)
    det = np.maximum(a11 * a22 - a12 * a12, 1e-30)
    a = (a22 * di - a12 * dj) / det
    b = (a11 * dj - a12 * di) / det
    g = a[..., None] * ei + b[..., None] * ej                        # (6, NE-2, NE-2, 3), tangent
    g = g[:, H - 1:H - 1 + N, H - 1:H - 1 + N]
    comps = {}
    for p, q in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        ff = FaceField.from_interior(grid, g[..., p] * g[..., q], exchange=True)
        comps[p, q] = gaussian_smooth(ff, float(sigma_cells)).interior
    T = np.empty((6, N, N, 3, 3), np.float64)
    for (p, q), v in comps.items():
        T[..., p, q] = v
        T[..., q, p] = v
    w, V = np.linalg.eigh(T)                                         # ascending
    n = V[..., 2]
    x = grid.interior_centers
    n = n - x * (n * x).sum(-1, keepdims=True)                       # onto the tangent plane
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    coh = (w[..., 2] - w[..., 1]) / np.maximum(w[..., 2] + w[..., 1], 1e-30)
    return n, np.clip(coh, 0.0, 1.0)


@njit(cache=True, parallel=True)
def _gabor_eval(pts, amp_mask, kx, kn, kc, kw, kphi, klam, starts, counts, lat_n, lat_s, elong, out):
    """Sum of kernels at ``pts`` (M, 3) where ``amp_mask``; kernels bucketed
    in a lattice of ``lat_n``^3 cells of side ``lat_s`` over [-1, 1]^3."""
    M = pts.shape[0]
    two_pi = 2.0 * math.pi
    for m in prange(M):
        if not amp_mask[m]:
            out[m] = 0.0
            continue
        px, py, pz = pts[m, 0], pts[m, 1], pts[m, 2]
        ix = int((px + 1.0) / lat_s)
        iy = int((py + 1.0) / lat_s)
        iz = int((pz + 1.0) / lat_s)
        s = 0.0
        for dx in range(-1, 2):
            cx = ix + dx
            if cx < 0 or cx >= lat_n:
                continue
            for dy in range(-1, 2):
                cy = iy + dy
                if cy < 0 or cy >= lat_n:
                    continue
                for dz in range(-1, 2):
                    cz = iz + dz
                    if cz < 0 or cz >= lat_n:
                        continue
                    b = (cx * lat_n + cy) * lat_n + cz
                    for k in range(starts[b], starts[b] + counts[b]):
                        ddx = px - kx[k, 0]
                        ddy = py - kx[k, 1]
                        ddz = pz - kx[k, 2]
                        un = ddx * kn[k, 0] + ddy * kn[k, 1] + ddz * kn[k, 2]
                        # along strike: t = x_k x n_k
                        tx = kx[k, 1] * kn[k, 2] - kx[k, 2] * kn[k, 1]
                        ty = kx[k, 2] * kn[k, 0] - kx[k, 0] * kn[k, 2]
                        tz = kx[k, 0] * kn[k, 1] - kx[k, 1] * kn[k, 0]
                        ut = ddx * tx + ddy * ty + ddz * tz
                        a_n = klam[k]
                        a_t = a_n * (1.0 + (elong - 1.0) * kc[k])
                        e = (un / a_n) ** 2 + (ut / a_t) ** 2
                        if e > 1.6:
                            continue
                        s += kw[k] * math.exp(-math.pi * e) * math.cos(two_pi * un / a_n + kphi[k])
        out[m] = s


def range_noise(pts: np.ndarray, mask: np.ndarray, n_t: np.ndarray, coh_t: np.ndarray, N_t: int, rng: np.random.Generator,
                wavelength: float, elongation: float, density: float = 10.0, lam_jitter: float = 0.4,
                angle_jitter_deg: float = 12.0, strength_wavelength: float = 0.0) -> np.ndarray:
    """Oriented Gabor noise at unit vectors ``pts`` (M, 3) where ``mask``,
    unit standard deviation over the masked points.  ``wavelength`` is in
    radians; each kernel's is it times exp(U(-lam_jitter, lam_jitter)), with
    the across envelope one wavelength and the along envelope ``1 +
    (elongation - 1) coherence`` of it; each kernel's strike is turned by
    N(0, ``angle_jitter_deg``), and its weight is +-(0.15 + 0.85 m) with m a
    smooth field of ``strength_wavelength`` radians (0: 1), so ranges vary
    in spacing, break en echelon, and come and go along a belt instead of
    running as one set of fringes.  ``density`` kernels per round envelope
    area."""
    from ..stubs import fbm_at

    lam = float(wavelength)
    lam_max = lam * math.exp(float(lam_jitter))
    reach = lam_max * max(float(elongation), 1.0) * math.sqrt(1.6 / math.pi)   # the cut-off of the longest envelope
    lat_n = max(2, int(2.0 / reach))
    lat_s = 2.0 / lat_n
    count = int(density * 4.0 * math.pi / (lam * lam))
    if count <= 0:
        # a planet whose whole surface is smaller than one range spacing carries no ranges
        # (a tiny test world: 80 km between crests on a 6 km circumference)
        return np.zeros(pts.shape[0], np.float64)
    v = rng.normal(size=(count, 3))
    kx = v / np.linalg.norm(v, axis=1, keepdims=True)
    f, u, w = from_sphere_v(kx)
    i = np.minimum((u * N_t).astype(np.int64), N_t - 1)
    j = np.minimum((w * N_t).astype(np.int64), N_t - 1)
    kn = n_t[f, i, j]
    ang = np.radians(rng.normal(0.0, float(angle_jitter_deg), size=count))[:, None]
    kn = kn * np.cos(ang) + np.cross(kx, kn) * np.sin(ang)
    kn = np.ascontiguousarray(kn)
    kc = np.ascontiguousarray(coh_t[f, i, j])
    klam = lam * np.exp(rng.uniform(-float(lam_jitter), float(lam_jitter), size=count))
    if strength_wavelength > 0.0:
        m = fbm_at(kx, rng, 3, 2.0 / float(strength_wavelength))
        m = (m - m.min()) / max(float(np.ptp(m)), 1e-12)
    else:
        m = np.ones(count)
    kw = rng.choice(np.array([-1.0, 1.0]), size=count) * (0.15 + 0.85 * m)
    kphi = rng.uniform(0.0, 2.0 * math.pi, size=count)
    cell = np.minimum(((kx + 1.0) / lat_s).astype(np.int64), lat_n - 1)
    bucket = (cell[:, 0] * lat_n + cell[:, 1]) * lat_n + cell[:, 2]
    order = np.argsort(bucket, kind="stable")
    kx, kn, kc, kw, kphi, klam, bucket = kx[order], kn[order], kc[order], kw[order], kphi[order], klam[order], bucket[order]
    counts = np.bincount(bucket, minlength=lat_n ** 3).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    out = np.zeros(pts.shape[0], np.float64)
    _gabor_eval(np.ascontiguousarray(pts, np.float64), np.ascontiguousarray(mask), np.ascontiguousarray(kx), kn, kc,
                np.ascontiguousarray(kw), np.ascontiguousarray(kphi), np.ascontiguousarray(klam), starts, counts, lat_n, lat_s,
                float(elongation), out)
    sd = float(out[mask].std()) if mask.any() else 0.0
    return out / sd if sd > 0 else out


def inject_ranges(bed: np.ndarray, bed_t: np.ndarray, coarse, grid_t, tp, rng: np.random.Generator, planet_radius_m: float,
                  height_scale_m: float) -> np.ndarray:
    """``bed`` (6, N, N, sea-levelled bedrock units) with ranges added over
    the belts (module docstring).  ``bed_t`` is the same surface on the tect
    grid ``grid_t``, where the strike is read.  Off (the array itself) when
    ``tectonics.ranges_amp`` is 0."""
    amp = float(getattr(tp, "ranges_amp", 0.0))
    if amp <= 0.0:
        return bed
    base = float(tp.ranges_base_m) / float(height_scale_m)
    E = np.maximum(bed - base, 0.0)
    mask = E > 0.0
    # the amplitude follows the belt up to ranges_cap_m: a 6 km swell does not
    # carry 2.5 km crests on top (peaks went to 10-11 km without it)
    E = np.minimum(E, float(tp.ranges_cap_m) / float(height_scale_m))
    if not mask.any():
        return bed
    n_t, coh_t = strike_frame(bed_t, grid_t, float(tp.ranges_orient_sigma))
    pts = coarse.interior_centers.reshape(-1, 3)
    km = 1000.0 / float(planet_radius_m)
    G = range_noise(pts, mask.reshape(-1), n_t, coh_t, grid_t.N, rng, float(tp.ranges_wavelength_km) * km,
                    float(tp.ranges_elongation), strength_wavelength=float(tp.ranges_strength_km) * km).reshape(bed.shape)
    G = np.clip(G, -2.0, 2.0) / 2.0
    # crests stand up, basins fill: the downside at ranges_basin of the upside
    G = np.where(G > 0.0, G, float(tp.ranges_basin) * G)
    return bed + amp * E * G


__all__ = ["strike_frame", "range_noise", "inject_ranges"]
