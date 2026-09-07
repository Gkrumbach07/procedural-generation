"""Numba kernels for the McDonald-2023 meandering particle model (PLAN 8.2).

Kernel contract (docs/DEVELOPING.md): the *same* kernel runs globally on the
six-face cube-sphere (``spherical=True``, particles cross faces with
:func:`~globe.cubesphere.transfer_velocity`) and inside a single-face basin
window (``spherical=False``, ``F = 1``, particles leaving the array or the
mask die).  All inputs are plain arrays shaped ``(F, NE, NE[, 2])`` with
``NE = N + 2H``; heights are in **cell units** (``metres / cell_size_m``)
so the ★ parameters apply as published.

Positions.  A particle lives at fractional *interior* cell coordinates
``(x, y)`` on face ``f``: cell ``(i, j) = (floor(x), floor(y))``,
``u = x / N``; the extended array index is ``(i + H, j + H)`` and the
bilinear sample point is ``(x - 0.5 + H, y - 0.5 + H)`` (cell centres at
integers).  Velocities are contravariant cell components ``(a, b)``; their
physical length is ``sqrt(g_ab a b)`` with the dimensionless ``metric``.
The step is always exactly one cell long (McDonald's dynamic time step):
``pos += speed / |speed|_g``.

Determinism.  Particles are processed in chunks.  Within a chunk the
terrain, discharge and momentum maps are frozen; every particle writes its
per-step effects (cell, terrain delta, volume, volume·velocity) into its own
slice of a *change list* (``cap`` entries per particle, ``cap = max_steps +
1``), and :func:`apply_changes` applies the list serially in particle order
(cell ownership is split across threads, so each cell still sees its updates
in list order).  The result is therefore independent of thread scheduling
and thread count; it depends on ``chunk`` (which is why ``erosion.chunk``
stays in the parameter hash).

Change-list memory: ``chunk * (max_steps + SPREAD)`` entries of
4 (int32 cell) + 8 (float64 delta) + 4 (float32 volume) + 8 (2 × float32
momentum) = 24 bytes; ``chunk = 2048, max_steps = 2 * 1024`` → ~100 MB.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..cubesphere import transfer_velocity

#: mask codes
MASK_OUTSIDE = 0
MASK_ACTIVE = 1
MASK_FROZEN = 2

#: a dying particle spreads its remaining load over its last SPREAD active cells
SPREAD = 8
#: minimum per-step deposition allowance (cell units) before the concurrency divisor
DEP_FLOOR = 0.02

#: 8-neighbour offsets (di, dj) used by thermal erosion (D8 order)
_D8 = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)], dtype=np.int64)


# --------------------------------------------------------------------------
# sampling helpers
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _clamp_ij(fi, fj, NE):
    i0 = int(math.floor(fi))
    j0 = int(math.floor(fj))
    if i0 < 0:
        i0 = 0
    if i0 > NE - 2:
        i0 = NE - 2
    if j0 < 0:
        j0 = 0
    if j0 > NE - 2:
        j0 = NE - 2
    wi = fi - i0
    wj = fj - j0
    if wi < 0.0:
        wi = 0.0
    if wi > 1.0:
        wi = 1.0
    if wj < 0.0:
        wj = 0.0
    if wj > 1.0:
        wj = 1.0
    return i0, j0, wi, wj


@njit(cache=True, inline="always")
def _bilin(arr, f, fi, fj):
    """Bilinear sample of the scalar extended array ``arr[f]`` at fractional
    extended index (fi, fj); clamped to the array."""
    NE = arr.shape[1]
    i0, j0, wi, wj = _clamp_ij(fi, fj, NE)
    return (
        arr[f, i0, j0] * (1.0 - wi) * (1.0 - wj)
        + arr[f, i0 + 1, j0] * wi * (1.0 - wj)
        + arr[f, i0, j0 + 1] * (1.0 - wi) * wj
        + arr[f, i0 + 1, j0 + 1] * wi * wj
    )


@njit(cache=True, inline="always")
def _bilin2(arr, f, fi, fj, c):
    """Bilinear sample of component ``c`` of the (F, NE, NE, 2) array."""
    NE = arr.shape[1]
    i0, j0, wi, wj = _clamp_ij(fi, fj, NE)
    return (
        arr[f, i0, j0, c] * (1.0 - wi) * (1.0 - wj)
        + arr[f, i0 + 1, j0, c] * wi * (1.0 - wj)
        + arr[f, i0, j0 + 1, c] * (1.0 - wi) * wj
        + arr[f, i0 + 1, j0 + 1, c] * wi * wj
    )


@njit(cache=True, inline="always")
def _bilin_grad(arr, f, fi, fj):
    """Value and covariant gradient of the bilinear interpolant of ``arr[f]``."""
    NE = arr.shape[1]
    i0, j0, wi, wj = _clamp_ij(fi, fj, NE)
    h00 = arr[f, i0, j0]
    h10 = arr[f, i0 + 1, j0]
    h01 = arr[f, i0, j0 + 1]
    h11 = arr[f, i0 + 1, j0 + 1]
    val = h00 * (1.0 - wi) * (1.0 - wj) + h10 * wi * (1.0 - wj) + h01 * (1.0 - wi) * wj + h11 * wi * wj
    gi = (h10 - h00) * (1.0 - wj) + (h11 - h01) * wj
    gj = (h01 - h00) * (1.0 - wi) + (h11 - h10) * wi
    return val, gi, gj


@njit(cache=True, inline="always")
def _surface_bilin_grad(height, sediment, f, fi, fj):
    """Value and covariant gradient (per cell along i, j) of the bilinear
    interpolant of ``height + sediment`` at fractional extended index
    (fi, fj).  The gradient is the exact gradient of the interpolant: the
    central difference at cell-square midpoints, one-sided at cell centres,
    always using the four cells the particle is currently between (the
    'upwind' stencil of PLAN 2.5 for a particle that steps one cell)."""
    NE = height.shape[1]
    i0, j0, wi, wj = _clamp_ij(fi, fj, NE)
    h00 = height[f, i0, j0] + sediment[f, i0, j0]
    h10 = height[f, i0 + 1, j0] + sediment[f, i0 + 1, j0]
    h01 = height[f, i0, j0 + 1] + sediment[f, i0, j0 + 1]
    h11 = height[f, i0 + 1, j0 + 1] + sediment[f, i0 + 1, j0 + 1]
    val = h00 * (1.0 - wi) * (1.0 - wj) + h10 * wi * (1.0 - wj) + h01 * (1.0 - wi) * wj + h11 * wi * wj
    gi = (h10 - h00) * (1.0 - wj) + (h11 - h01) * wj
    gj = (h01 - h00) * (1.0 - wi) + (h11 - h10) * wi
    return val, gi, gj


@njit(cache=True, inline="always")
def _metric_norm(metric, f, ei, ej, a, b):
    g0 = metric[f, ei, ej, 0]
    g1 = metric[f, ei, ej, 1]
    g2 = metric[f, ei, ej, 2]
    n2 = g0 * a * a + 2.0 * g1 * a * b + g2 * b * b
    if n2 < 0.0:
        n2 = 0.0
    return math.sqrt(n2)


@njit(cache=True, inline="always")
def _metric_dot(metric, f, ei, ej, a, b, c, d):
    g0 = metric[f, ei, ej, 0]
    g1 = metric[f, ei, ej, 1]
    g2 = metric[f, ei, ej, 2]
    return g0 * a * c + g1 * (a * d + b * c) + g2 * b * d


@njit(cache=True, inline="always")
def _cell_of(x, N):
    i = int(math.floor(x))
    if i < 0:
        i = 0
    if i > N - 1:
        i = N - 1
    return i


# --------------------------------------------------------------------------
# particle kernel
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def trace_particles(
    # spawn (P,) arrays: face, fractional interior x, y; one volume for all
    sp_face,
    sp_x,
    sp_y,
    volume0,
    # state (F, NE, NE[, 2]) arrays
    height,
    sediment,
    route,
    use_route,
    discharge,
    momentum,
    hardness,
    evap,
    mask,
    metric,
    metric_inv,
    N,
    H,
    spherical,
    # parameters
    dt,
    density,
    friction,
    deposition_rate,
    evap_rate,
    k_mom,
    k_disc,
    disc_saturation,
    slope_gain,
    erodibility,
    min_volume,
    max_steps,
    max_erode,
    conc_scale,
    pit_steps,
    deposit_on_exit,
    # change list: (P*cap,) arrays + (P,) counts
    cl_cell,
    cl_delta,
    cl_vol,
    cl_mom,
    cl_count,
):
    """Trace one chunk of particles on frozen maps and fill the change list.

    Per step the particle records ``(cell, terrain delta, volume, volume *
    speed)`` at the cell it *leaves* (PLAN 8.2 writes to ``prev_pos``); the
    final deposit is recorded with volume 0 (no discharge).  Frozen cells
    (mask 2) are transparent: sampled, never written, no sediment exchange.
    Returns nothing; ``cl_count[p]`` is the number of entries of particle p.
    """
    P = sp_face.shape[0]
    NE = N + 2 * H
    cap = max_steps + SPREAD
    Nf = float(N)
    inv_sat = 1.0 / disc_saturation if disc_saturation > 0.0 else 0.0
    for p in prange(P):
        f = int(sp_face[p])
        x = sp_x[p]
        y = sp_y[p]
        vol = volume0
        sed = 0.0
        sa = 0.0
        sb = 0.0
        base = p * cap
        n = 0
        uphill = 0
        # ring buffer of the last SPREAD active cells (flat indices) visited
        # (lives in the tail of this particle's change-list slice; no
        # allocation inside the parallel region)
        rbase = base + cap - SPREAD
        nring = 0
        rpos = 0
        # surface height of the previous cell (cell value); < -1e30 = none
        h_prev = -1e300
        for step in range(max_steps):
            ci = _cell_of(x, N)
            cj = _cell_of(y, N)
            ei = ci + H
            ej = cj + H
            m = mask[f, ei, ej]
            if m == MASK_OUTSIDE:
                break
            if m == MASK_ACTIVE:
                cl_cell[rbase + rpos] = (f * NE + ei) * NE + ej
                rpos = (rpos + 1) % SPREAD
                if nring < SPREAD:
                    nring += 1
            h_here = height[f, ei, ej] + sediment[f, ei, ej]
            fi = x - 0.5 + H
            fj = y - 0.5 + H
            # --- forces ---------------------------------------------------
            h0, gi, gj = _surface_bilin_grad(height, sediment, f, fi, fj)
            h0r = h0
            if use_route:
                # steer on the epsilon-filled routing surface where it is
                # above the terrain (lakes / pits); erode on the terrain
                r0, rgi, rgj = _bilin_grad(route, f, fi, fj)
                if r0 > h0:
                    h0r = r0
                    gi = rgi
                    gj = rgj
            q0 = metric_inv[f, ei, ej, 0]
            q1 = metric_inv[f, ei, ej, 1]
            q2 = metric_inv[f, ei, ej, 2]
            di = q0 * gi + q1 * gj  # contravariant gradient (cells)
            dj = q1 * gi + q2 * gj
            s2 = gi * di + gj * dj  # |grad|^2 (dimensionless slope^2)
            if s2 < 0.0:
                s2 = 0.0
            nrm = slope_gain / math.sqrt(1.0 + s2)  # tangential part of the surface normal
            na = -di * nrm
            nb = -dj * nrm
            ma = _bilin2(momentum, f, fi, fj, 0)
            mb = _bilin2(momentum, f, fi, fj, 1)
            q = _bilin(discharge, f, fi, fj)
            if q < 0.0:
                q = 0.0
            vol_rel = vol / volume0
            mlen = _metric_norm(metric, f, ei, ej, ma, mb)
            slen = _metric_norm(metric, f, ei, ej, sa, sb)
            if mlen > 1e-12 and slen > 1e-12:
                # stream momentum: m / (vol + q) is the volume-weighted mean
                # (unit) velocity of the stream, so the push is bounded by k_mom
                cosang = _metric_dot(metric, f, ei, ej, ma, mb, sa, sb) / (mlen * slen)
                k = k_mom * cosang / (vol + q)
                sa += k * ma
                sb += k * mb
            acc = dt / (vol_rel * density)
            sa += acc * na
            sb += acc * nb
            fr = 1.0 - dt * friction
            if fr < 0.0:
                fr = 0.0
            sa *= fr
            sb *= fr
            slen = _metric_norm(metric, f, ei, ej, sa, sb)
            if slen < 1e-12:
                break  # no motion: deposit here
            # dynamic time step: unit speed, the step is exactly one cell
            sa /= slen
            sb /= slen
            slen = 1.0
            # --- step (exactly one cell) -----------------------------------
            nx = x + sa
            ny = y + sb
            nf = f
            if spherical:
                if nx < 0.0 or nx >= Nf or ny < 0.0 or ny >= Nf:
                    f2, u2, v2, a2, b2 = transfer_velocity(f, nx / Nf, ny / Nf, sa, sb)
                    nf = f2
                    nx = u2 * Nf
                    ny = v2 * Nf
                    sa = a2
                    sb = b2
                    if nx >= Nf:
                        nx = Nf * 0.999999999
                    if ny >= Nf:
                        ny = Nf * 0.999999999
                    if nx < 0.0:
                        nx = 0.0
                    if ny < 0.0:
                        ny = 0.0
            else:
                if nx < -H or nx >= Nf + H or ny < -H or ny >= Nf + H:
                    break  # left the array
            nci = int(math.floor(nx))
            ncj = int(math.floor(ny))
            nei = nci + H
            nej = ncj + H
            if nei < 0 or nei >= NE or nej < 0 or nej >= NE:
                break
            nm = mask[nf, nei, nej]
            if nm == MASK_OUTSIDE:
                break
            nfi = nx - 0.5 + H
            nfj = ny - 0.5 + H
            h1, _gi, _gj = _surface_bilin_grad(height, sediment, nf, nfi, nfj)
            dh = h0 - h1
            h1r = h1
            if use_route:
                r1 = _bilin(route, nf, nfi, nfj)
                if r1 > h1:
                    h1r = r1
            dhr = h0r - h1r
            # --- mass transfer at the cell we leave -------------------------
            if m == MASK_ACTIVE:
                # equilibrium concentration (cell units of height per unit
                # spawn volume); carried mass = sed * vol_rel
                qe = math.erf(q * inv_sat) if inv_sat > 0.0 else 0.0
                c_eq = erodibility * dh * (1.0 + k_disc * qe)
                if c_eq < 0.0:
                    c_eq = 0.0
                if c_eq > sed:
                    # erosion: sediment layer fully erodible, bedrock scaled by hardness
                    if sediment[f, ei, ej] <= 0.0:
                        k_e = deposition_rate * (1.0 - hardness[f, ei, ej])
                    else:
                        k_e = deposition_rate
                else:
                    k_e = deposition_rate
                cdiff = k_e * (c_eq - sed)
                # Terrain is frozen within a chunk, so K concurrent particles
                # would each act against the same stale heights.  Cap the
                # change at (local relief) / (1 + expected visits of this cell
                # in the chunk): erosion never drops the cell below the one we
                # step to, deposition never lifts it above the one we came from.
                kexp = 1.0 + q * conc_scale
                if cdiff > 0.0:
                    ecap = dh / kexp
                    if ecap > max_erode:
                        ecap = max_erode
                    if cdiff * vol_rel > ecap:
                        cdiff = ecap / vol_rel
                elif cdiff < 0.0:
                    room = h_prev - h_here if h_prev > -1e300 else DEP_FLOOR
                    if room < DEP_FLOOR:
                        room = DEP_FLOOR
                    dcap = room / kexp
                    if -cdiff * vol_rel > dcap:
                        cdiff = -dcap / vol_rel
                sed += cdiff
                cl_cell[base + n] = (f * NE + ei) * NE + ej
                cl_delta[base + n] = -cdiff * vol_rel
                cl_vol[base + n] = vol
                cl_mom[base + n, 0] = vol * sa
                cl_mom[base + n, 1] = vol * sb
                n += 1
            # --- evaporation (mass conserving) ----------------------------
            ev = 1.0 - dt * evap_rate * evap[f, ei, ej]
            if ev < 0.01:
                ev = 0.01
            vol *= ev
            sed /= ev
            x = nx
            y = ny
            f = nf
            h_prev = h_here
            # --- stop conditions -------------------------------------------
            if height[nf, nei, nej] + sediment[nf, nei, nej] < 0.0:
                # stepped into the ocean: the whole load settles there (delta)
                if nm == MASK_ACTIVE:
                    cl_cell[rbase] = (nf * NE + nei) * NE + nej
                    nring = 1
                break
            if vol < min_volume:
                break
            if dhr < 0.0:
                uphill += 1
                if uphill > pit_steps:
                    break  # stuck climbing out of a pit
            else:
                uphill = 0
        # --- final deposit ------------------------------------------------
        # (a) natural stop inside the mask: at the last active cell visited
        # (b) left the mask/array: only with deposit_on_exit
        if sed > 0.0 and nring > 0:
            ci = _cell_of(x, N)
            cj = _cell_of(y, N)
            inside = mask[f, ci + H, cj + H] != MASK_OUTSIDE
            if inside or deposit_on_exit:
                # spread the remaining load over the last cells visited so K
                # particles dying in the same pit cannot build a spike
                share = sed * vol / volume0 / nring
                for r in range(nring):
                    cell = cl_cell[rbase + r]
                    cl_cell[base + n] = cell
                    cl_delta[base + n] = share
                    cl_vol[base + n] = 0.0
                    cl_mom[base + n, 0] = 0.0
                    cl_mom[base + n, 1] = 0.0
                    n += 1
        cl_count[p] = n


@njit(cache=True, parallel=True)
def apply_changes(cl_cell, cl_delta, cl_vol, cl_mom, cl_count, cap, height, sediment, disch_track, mom_track, mask, nthreads):
    """Apply a change list serially in particle order.

    Cells are partitioned into ``nthreads`` contiguous flat-index ranges;
    each thread scans the whole list and applies only the entries of its
    range, so every cell sees its updates in list order regardless of the
    thread count.  Erosion (delta < 0) takes sediment first, then bedrock;
    deposition adds to sediment.  Frozen / outside cells are never written.
    """
    F, NE, _ = height.shape
    total = F * NE * NE
    P = cl_count.shape[0]
    per = (total + nthreads - 1) // nthreads
    hflat = height.reshape(total)
    sflat = sediment.reshape(total)
    dflat = disch_track.reshape(total)
    mflat = mom_track.reshape(total, 2)
    kflat = mask.reshape(total)
    for t in prange(nthreads):
        lo = t * per
        hi = min(total, lo + per)
        for p in range(P):
            base = p * cap
            for k in range(cl_count[p]):
                c = cl_cell[base + k]
                if c < lo or c >= hi:
                    continue
                if kflat[c] != MASK_ACTIVE:
                    continue
                d = cl_delta[base + k]
                if d < 0.0:
                    take = -d
                    s = sflat[c]
                    if s >= take:
                        sflat[c] = s - take
                    else:
                        sflat[c] = 0.0
                        hflat[c] -= take - s
                elif d > 0.0:
                    sflat[c] += d
                v = cl_vol[base + k]
                if v > 0.0:
                    dflat[c] += v
                    mflat[c, 0] += cl_mom[base + k, 0]
                    mflat[c, 1] += cl_mom[base + k, 1]


# --------------------------------------------------------------------------
# thermal erosion
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _thermal_out(height, sediment, metric, talus, f, ei, ej, k, thermal_rate):
    """Raw outflow from (f, ei, ej) to its k-th D8 neighbour (no cap)."""
    di = _D8[k, 0]
    dj = _D8[k, 1]
    ni = ei + di
    nj = ej + dj
    d = (height[f, ei, ej] + sediment[f, ei, ej]) - (height[f, ni, nj] + sediment[f, ni, nj])
    if d <= 0.0:
        return 0.0
    g0 = metric[f, ei, ej, 0]
    g1 = metric[f, ei, ej, 1]
    g2 = metric[f, ei, ej, 2]
    dist = math.sqrt(g0 * di * di + 2.0 * g1 * di * dj + g2 * dj * dj)
    ex = d - talus[f, ei, ej] * dist
    if ex <= 0.0:
        return 0.0
    return 0.5 * thermal_rate * ex


@njit(cache=True, parallel=True)
def thermal_pass_a(height, sediment, metric, talus, mask, thermal_rate, out_total, scale):
    """Per-cell total outflow and cap scale (so a cell never drops below
    its steepest neighbour's talus line)."""
    F, NE, _ = height.shape
    for f in prange(F):
        for ei in range(1, NE - 1):
            for ej in range(1, NE - 1):
                out_total[f, ei, ej] = 0.0
                scale[f, ei, ej] = 0.0
                if mask[f, ei, ej] == MASK_OUTSIDE:
                    continue
                S = 0.0
                mx = 0.0
                for k in range(8):
                    if mask[f, ei + _D8[k, 0], ej + _D8[k, 1]] != MASK_ACTIVE:
                        continue
                    o = _thermal_out(height, sediment, metric, talus, f, ei, ej, k, thermal_rate)
                    S += o
                    if o > mx:
                        mx = o
                if S <= 0.0:
                    continue
                sc = 1.0
                if S > 2.0 * mx:
                    sc = 2.0 * mx / S
                out_total[f, ei, ej] = S * sc
                scale[f, ei, ej] = sc


@njit(cache=True, parallel=True)
def thermal_pass_b(height, sediment, metric, talus, mask, thermal_rate, out_total, scale, new_height, new_sediment):
    """Gather: every active cell loses ``out_total`` (sediment first) and
    receives the capped outflows of its neighbours as sediment."""
    F, NE, _ = height.shape
    for f in prange(F):
        for ei in range(1, NE - 1):
            for ej in range(1, NE - 1):
                h = height[f, ei, ej]
                s = sediment[f, ei, ej]
                if mask[f, ei, ej] != MASK_ACTIVE:
                    new_height[f, ei, ej] = h
                    new_sediment[f, ei, ej] = s
                    continue
                inflow = 0.0
                for k in range(8):
                    ni = ei + _D8[k, 0]
                    nj = ej + _D8[k, 1]
                    if mask[f, ni, nj] == MASK_OUTSIDE:
                        continue
                    sc = scale[f, ni, nj]
                    if sc <= 0.0:
                        continue
                    # neighbour's outflow towards us: opposite D8 direction
                    o = _thermal_out(height, sediment, metric, talus, f, ni, nj, (k + 4) % 8, thermal_rate)
                    inflow += sc * o
                out = out_total[f, ei, ej]
                if out > 0.0:
                    if s >= out:
                        s -= out
                    else:
                        h -= out - s
                        s = 0.0
                new_height[f, ei, ej] = h
                new_sediment[f, ei, ej] = s + inflow


@njit(cache=True, parallel=True)
def ema_update(discharge, momentum, disch_track, mom_track, mask, ema):
    F, NE, _ = discharge.shape
    for f in prange(F):
        for ei in range(NE):
            for ej in range(NE):
                if mask[f, ei, ej] == MASK_ACTIVE:
                    discharge[f, ei, ej] = (1.0 - ema) * discharge[f, ei, ej] + ema * disch_track[f, ei, ej]
                    momentum[f, ei, ej, 0] = (1.0 - ema) * momentum[f, ei, ej, 0] + ema * mom_track[f, ei, ej, 0]
                    momentum[f, ei, ej, 1] = (1.0 - ema) * momentum[f, ei, ej, 1] + ema * mom_track[f, ei, ej, 1]
                disch_track[f, ei, ej] = 0.0
                mom_track[f, ei, ej, 0] = 0.0
                mom_track[f, ei, ej, 1] = 0.0


@njit(cache=True)
def spawn_cells(cdf, u):
    """Inverse-CDF sampling: ``cdf`` is the cumulative spawn weight (last
    entry = total), ``u`` sorted uniforms in [0, 1).  Returns the index of
    the chosen entry for every u (monotone, so spawns are sorted by cell)."""
    P = u.shape[0]
    M = cdf.shape[0]
    total = cdf[M - 1]
    out = np.empty(P, dtype=np.int64)
    k = 0
    for p in range(P):
        target = u[p] * total
        while k < M - 1 and cdf[k] <= target:
            k += 1
        out[p] = k
    return out


__all__ = [
    "MASK_OUTSIDE",
    "MASK_ACTIVE",
    "MASK_FROZEN",
    "trace_particles",
    "apply_changes",
    "thermal_pass_a",
    "thermal_pass_b",
    "ema_update",
    "spawn_cells",
]
