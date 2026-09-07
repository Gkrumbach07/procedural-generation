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
2 * SPREAD``), and :func:`apply_changes` applies the list **serially in
particle order against the live terrain** (the only place physical limits
are enforced, see its docstring).  The result is therefore independent of
thread scheduling and thread count; it depends on ``chunk`` (which is why
``erosion.chunk`` stays in the parameter hash).

Within one iteration all terrain changes accumulate in a per-cell *net*
delta (``acc``); :func:`fold_changes` applies the net once at the end of the
iteration — erosion from sediment first, then bedrock; deposition to
sediment — so erosion/deposition pairs passing through a cell cancel
without converting bedrock into sediment.

Change-list memory: ``chunk * (max_steps + 2 * SPREAD)`` entries of
4 (int32 cell) + 8 (float64 delta) + 4 (float32 volume) + 8 (2 × float32
momentum) = 24 bytes; ``chunk = 512, max_steps = 2 * 1024`` → ~25 MB
(reused for every chunk).  The sampled fields are packed once per
iteration into a float32 ``(F, NE, NE, 16)`` array (64 B = one cache line
per cell; 384 MB at N = 1024): the kernel is memory-bound and a 2x2
stencil then costs 4 cache lines instead of ~16.
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

#: bumped whenever a kernel change alters results: part of the checkpoint
#: hash, so stale checkpoints are never resumed after a code change
KERNEL_VERSION = 2

#: a dying particle deposits its remaining load at the cell it died in; the
#: excess over that cell's caps moves back up its last SPREAD active cells
SPREAD = 8

#: why a particle stopped (``sp_death`` output of :func:`trace_particles`)
DEATH_AGE = 0  # reached max_steps
DEATH_OCEAN = 1  # stepped into a cell with surface < 0
DEATH_EXIT = 2  # left the mask / the window array
DEATH_PIT = 3  # more than pit_steps consecutive uphill steps
DEATH_EVAP = 4  # volume < min_volume
DEATH_STOP = 5  # no motion
DEATH_NAMES = ("age", "ocean", "exit", "pit", "evap", "stop")
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
def _cell_of(x, N):
    i = int(math.floor(x))
    if i < 0:
        i = 0
    if i > N - 1:
        i = N - 1
    return i


# --------------------------------------------------------------------------
# packed sample array
# --------------------------------------------------------------------------
#: channels of the packed float32 sample array ``samp[f, ei, ej, :]`` (16
#: floats = one 64-byte cache line per cell, so a bilinear 2x2 stencil
#: touches 4 lines instead of one or two per field)
S_SURF = 0  # height + sediment
S_ROUTE = 1  # routing surface (== surface when unused)
S_DISCH = 2
S_MOMA = 3
S_MOMB = 4
S_EVAP = 5
S_HARD = 6
S_SED = 7  # sediment thickness (erodibility rule)
S_G = 8  # metric g_ii, g_ij, g_jj (8..10)
S_GINV = 11  # inverse metric (11..13)
NS = 16


@njit(cache=True, parallel=True)
def pack_samples(samp, height, sediment, route, use_route, discharge, momentum, evap, hardness, metric, metric_inv):
    """Fill the packed sample array from the state arrays (once per iteration)."""
    F, NE, _ = height.shape
    for f in prange(F):
        for ei in range(NE):
            for ej in range(NE):
                h = height[f, ei, ej]
                sd = sediment[f, ei, ej]
                surf = h + sd
                r = route[f, ei, ej] if use_route else surf
                samp[f, ei, ej, S_SURF] = surf
                samp[f, ei, ej, S_ROUTE] = r
                samp[f, ei, ej, S_DISCH] = discharge[f, ei, ej]
                samp[f, ei, ej, S_MOMA] = momentum[f, ei, ej, 0]
                samp[f, ei, ej, S_MOMB] = momentum[f, ei, ej, 1]
                samp[f, ei, ej, S_EVAP] = evap[f, ei, ej]
                samp[f, ei, ej, S_HARD] = hardness[f, ei, ej]
                samp[f, ei, ej, S_SED] = sd
                for k in range(3):
                    samp[f, ei, ej, S_G + k] = metric[f, ei, ej, k]
                    samp[f, ei, ej, S_GINV + k] = metric_inv[f, ei, ej, k]
                samp[f, ei, ej, 14] = 0.0
                samp[f, ei, ej, 15] = 0.0


@njit(cache=True, inline="always")
def _sbilin(samp, f, fi, fj, c):
    """Bilinear sample of channel ``c`` at fractional extended index."""
    NE = samp.shape[1]
    i0, j0, wi, wj = _clamp_ij(fi, fj, NE)
    return (
        samp[f, i0, j0, c] * (1.0 - wi) * (1.0 - wj)
        + samp[f, i0 + 1, j0, c] * wi * (1.0 - wj)
        + samp[f, i0, j0 + 1, c] * (1.0 - wi) * wj
        + samp[f, i0 + 1, j0 + 1, c] * wi * wj
    )


@njit(cache=True, inline="always")
def _sbilin_grad(samp, f, fi, fj, c):
    """Value and covariant gradient (per cell along i, j) of the bilinear
    interpolant of channel ``c`` at fractional extended index (fi, fj).
    The gradient is the exact gradient of the interpolant: the central
    difference at cell-square midpoints, one-sided at cell centres, always
    using the four cells the particle is currently between (the 'upwind'
    stencil of PLAN 2.5 for a particle that steps one cell)."""
    NE = samp.shape[1]
    i0, j0, wi, wj = _clamp_ij(fi, fj, NE)
    h00 = samp[f, i0, j0, c]
    h10 = samp[f, i0 + 1, j0, c]
    h01 = samp[f, i0, j0 + 1, c]
    h11 = samp[f, i0 + 1, j0 + 1, c]
    val = h00 * (1.0 - wi) * (1.0 - wj) + h10 * wi * (1.0 - wj) + h01 * (1.0 - wi) * wj + h11 * wi * wj
    gi = (h10 - h00) * (1.0 - wj) + (h11 - h01) * wj
    gj = (h01 - h00) * (1.0 - wi) + (h11 - h10) * wi
    return val, gi, gj


@njit(cache=True, inline="always")
def _snorm(samp, f, ei, ej, a, b):
    g0 = samp[f, ei, ej, S_G]
    g1 = samp[f, ei, ej, S_G + 1]
    g2 = samp[f, ei, ej, S_G + 2]
    n2 = g0 * a * a + 2.0 * g1 * a * b + g2 * b * b
    if n2 < 0.0:
        n2 = 0.0
    return math.sqrt(n2)


@njit(cache=True, inline="always")
def _sdot(samp, f, ei, ej, a, b, c, d):
    g0 = samp[f, ei, ej, S_G]
    g1 = samp[f, ei, ej, S_G + 1]
    g2 = samp[f, ei, ej, S_G + 2]
    return g0 * a * c + g1 * (a * d + b * c) + g2 * b * d


# --------------------------------------------------------------------------
# particle kernel
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def trace_particles(
    # spawn (P,) arrays: face, fractional interior x, y; one volume for all
    sp_face,
    sp_x,
    sp_y,
    sp_sed,
    sp_dir,
    volume0,
    # packed samples (F, NE, NE, NS) float32 + mask (F, NE, NE) uint8
    samp,
    use_route,
    mask,
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
    slope_sat,
    erodibility,
    min_volume,
    max_steps,
    max_erode,
    conc_scale,
    pit_steps,
    deposit_on_exit,
    ocean_rate,
    ocean_steps,
    # change list: (P*cap,) arrays + (P,) counts, cell range, death cause
    cl_cell,
    cl_delta,
    cl_vol,
    cl_mom,
    cl_count,
    cl_lo,
    cl_hi,
    sp_death,
):
    """Trace one chunk of particles on frozen maps and fill the change list.

    ``sp_sed[p]`` is the load a particle starts with (cell units per unit
    spawn volume; 0 for rain, the cell's stockpile for a re-injected
    ``pending`` particle, which starts in seafloor mode when its cell is
    submerged, heading in direction ``sp_dir[p]`` (radians)).

    Per step the particle records ``(cell, terrain delta, volume, volume *
    speed)`` at the cell it *leaves* (PLAN 8.2 writes to ``prev_pos``).
    When it dies inside the mask (or leaves it with ``deposit_on_exit``) it
    appends its *final deposits*, marked with volume -1: the cell it died
    in carries the whole remaining load, followed (interior deaths only) by
    the last ``SPREAD`` active cells of its path (newest first) with delta
    0 — overflow slots that :func:`apply_changes` fills with whatever the
    death cell cannot take (a pit fills to the level of the path).  A load
    that reached the sea is offered to the ocean cell only; the excess
    waits in that cell's ``pending`` stockpile and spreads offshore.  Frozen cells (mask 2) are transparent: sampled, never
    written, no sediment exchange.  Returns nothing; ``cl_count[p]`` is the
    number of entries of particle p, ``cl_lo[p]..cl_hi[p]`` the range of
    flat cell indices they touch and ``sp_death[p]`` why it stopped
    (``DEATH_*`` codes).

    Direction update (the step is always one cell, so the velocity is a
    unit direction and ``friction`` cannot act as a terminal-speed limit):
    ``dir = normalize((1 - dt*friction) * dir_prev + momentum push +
    dt * gravity / (vol * density))`` — friction is the inertia weight of the
    previous direction (``1/dt`` = no inertia).  PLAN 8.2's ``|speed|``
    factor in ``c_eq`` is the constant 1 cell/step here, absorbed in
    ``erodibility``.

    Gravity: the tangential part of the surface normal scaled by
    ``slope_gain`` (McDonald) when ``slope_sat <= 0``; otherwise the unit
    downhill direction times ``slope_gain * s / sqrt(s² + slope_sat²)``
    (terminal-velocity flow: on gentle slopes the direction still counts
    fully, so gravity is not swamped by the momentum push, whose magnitude
    is O(1) in a stream).  The drop into an ocean cell counts only down to
    sea level (``dh = h0 - max(h1, 0)``): flow decelerates at the coast, so
    coastal cells are not planed down by the depth of the shelf.
    """
    P = sp_face.shape[0]
    NE = N + 2 * H
    cap = max_steps + 2 * SPREAD
    Nf = float(N)
    inv_sat = 1.0 / disc_saturation if disc_saturation > 0.0 else 0.0
    for p in prange(P):
        f = int(sp_face[p])
        x = sp_x[p]
        y = sp_y[p]
        vol = volume0
        sed = sp_sed[p]  # initial load (re-injected stockpile), 0 for rain
        sa = 0.0
        sb = 0.0
        if sed > 0.0:
            # a re-injected stockpile starts moving in a random direction
            # (a flat, filled platform has no slope to follow); it is
            # normalised in the metric on the first step
            sa = math.cos(sp_dir[p])
            sb = math.sin(sp_dir[p])
        base = p * cap
        n = 0
        uphill = 0
        lo = 2147483647
        hi = -1
        # ring buffer of the last SPREAD active cells (flat indices) visited
        # (lives in the tail of this particle's change-list slice; no
        # allocation inside the parallel region)
        rbase = base + cap - SPREAD
        nring = 0
        rpos = 0
        # surface height of the previous cell (cell value); < -1e30 = none
        h_prev = -1e300
        exited = False
        in_sea = False  # reached the ocean: deposit-only seafloor walk
        sea_steps = 0
        cause = DEATH_AGE
        for step in range(max_steps):
            ci = _cell_of(x, N)
            cj = _cell_of(y, N)
            ei = ci + H
            ej = cj + H
            m = mask[f, ei, ej]
            if m == MASK_OUTSIDE:
                exited = True
                cause = DEATH_EXIT
                break
            cell = (f * NE + ei) * NE + ej
            h_here = samp[f, ei, ej, S_SURF]
            if step == 0 and h_here < 0.0:
                in_sea = True  # a stockpile re-injected on the seafloor
            if in_sea:
                # a sea particle never climbs back onto land, and stops when
                # its load is gone or its seafloor walk is over
                if h_here >= 0.0 or sed * vol / volume0 < 1e-4 or sea_steps >= ocean_steps:
                    cause = DEATH_OCEAN
                    break
                sea_steps += 1
            if m == MASK_ACTIVE:
                cl_cell[rbase + rpos] = cell
                rpos = (rpos + 1) % SPREAD
                if nring < SPREAD:
                    nring += 1
            fi = x - 0.5 + H
            fj = y - 0.5 + H
            # --- forces ---------------------------------------------------
            h0, gi, gj = _sbilin_grad(samp, f, fi, fj, S_SURF)
            h0r = h0
            if use_route:
                # steer on the epsilon-filled routing surface where it is
                # above the terrain (lakes / pits); erode on the terrain
                r0, rgi, rgj = _sbilin_grad(samp, f, fi, fj, S_ROUTE)
                if r0 > h0:
                    h0r = r0
                    gi = rgi
                    gj = rgj
            q0 = samp[f, ei, ej, S_GINV]
            q1 = samp[f, ei, ej, S_GINV + 1]
            q2 = samp[f, ei, ej, S_GINV + 2]
            di = q0 * gi + q1 * gj  # contravariant gradient (cells)
            dj = q1 * gi + q2 * gj
            s2 = gi * di + gj * dj  # |grad|^2 (dimensionless slope^2)
            if s2 < 0.0:
                s2 = 0.0
            if slope_sat > 0.0:
                # unit downhill direction, magnitude saturating with slope
                nrm = slope_gain / math.sqrt(s2 + slope_sat * slope_sat)
            else:
                nrm = slope_gain / math.sqrt(1.0 + s2)  # tangential part of the surface normal
            na = -di * nrm
            nb = -dj * nrm
            ma = _sbilin(samp, f, fi, fj, S_MOMA)
            mb = _sbilin(samp, f, fi, fj, S_MOMB)
            q = _sbilin(samp, f, fi, fj, S_DISCH)
            if q < 0.0:
                q = 0.0
            vol_rel = vol / volume0
            # inertia: the previous unit direction enters with weight
            # 1 - dt*friction (applied *before* the forces; after them it
            # would cancel in the normalisation below)
            fr = 1.0 - dt * friction
            if fr < 0.0:
                fr = 0.0
            sa *= fr
            sb *= fr
            mlen = _snorm(samp, f, ei, ej, ma, mb)
            slen = _snorm(samp, f, ei, ej, sa, sb)
            if mlen > 1e-12 and slen > 1e-12:
                # stream momentum: m / (vol + q) is the volume-weighted mean
                # (unit) velocity of the stream, so the push is bounded by k_mom
                cosang = _sdot(samp, f, ei, ej, ma, mb, sa, sb) / (mlen * slen)
                k = k_mom * cosang / (vol + q)
                sa += k * ma
                sb += k * mb
            acc = dt / (vol_rel * density)
            sa += acc * na
            sb += acc * nb
            slen = _snorm(samp, f, ei, ej, sa, sb)
            if slen < 1e-12:
                cause = DEATH_STOP
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
                    exited = True
                    cause = DEATH_EXIT
                    break  # left the array
            nci = int(math.floor(nx))
            ncj = int(math.floor(ny))
            nei = nci + H
            nej = ncj + H
            if nei < 0 or nei >= NE or nej < 0 or nej >= NE:
                exited = True
                cause = DEATH_EXIT
                break
            nm = mask[nf, nei, nej]
            if nm == MASK_OUTSIDE:
                exited = True
                cause = DEATH_EXIT
                break
            nfi = nx - 0.5 + H
            nfj = ny - 0.5 + H
            h1 = _sbilin(samp, nf, nfi, nfj, S_SURF)
            dh = h0 - (h1 if h1 > 0.0 else 0.0)  # the drop into the sea counts down to sea level only
            h1r = h1
            if use_route:
                r1 = _sbilin(samp, nf, nfi, nfj, S_ROUTE)
                if r1 > h1:
                    h1r = r1
            dhr = h0r - h1r
            # --- mass transfer at the cell we leave -------------------------
            if m == MASK_ACTIVE and in_sea:
                # seafloor: settle a fixed fraction of the load, never erode
                cdiff = -ocean_rate * sed
                room = h_prev - h_here if h_prev > -1e300 else DEP_FLOOR
                if room < DEP_FLOOR:
                    room = DEP_FLOOR  # flat seafloor: fans still build (live caps bound the pile)
                lim = -DEP_FLOOR - h_here  # sea-level ceiling
                if lim < room:
                    room = lim
                if room < 0.0:
                    room = 0.0
                if -cdiff * vol_rel > room:
                    cdiff = -room / vol_rel
                sed += cdiff
                cl_cell[base + n] = cell
                cl_delta[base + n] = -cdiff * vol_rel
                cl_vol[base + n] = 0.0  # path entry without discharge
                cl_mom[base + n, 0] = 0.0
                cl_mom[base + n, 1] = 0.0
                n += 1
                if cell < lo:
                    lo = cell
                if cell > hi:
                    hi = cell
            elif m == MASK_ACTIVE:
                # equilibrium concentration (cell units of height per unit
                # spawn volume); carried mass = sed * vol_rel
                qe = math.erf(q * inv_sat) if inv_sat > 0.0 else 0.0
                c_eq = erodibility * dh * (1.0 + k_disc * qe)
                if c_eq < 0.0:
                    c_eq = 0.0
                if c_eq > sed:
                    # erosion: sediment layer fully erodible, bedrock scaled by hardness
                    if samp[f, ei, ej, S_SED] <= 0.0:
                        k_e = deposition_rate * (1.0 - samp[f, ei, ej, S_HARD])
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
                    # base level: a land cell is never eroded below sea level
                    # (a chain of cells could otherwise follow the coastal cell
                    # down towards the ocean floor)
                    if ecap > h_here:
                        ecap = h_here if h_here > 0.0 else 0.0
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
                cl_cell[base + n] = cell
                cl_delta[base + n] = -cdiff * vol_rel
                cl_vol[base + n] = vol
                cl_mom[base + n, 0] = vol * sa
                cl_mom[base + n, 1] = vol * sb
                n += 1
                if cell < lo:
                    lo = cell
                if cell > hi:
                    hi = cell
            # --- evaporation (mass conserving) ----------------------------
            ev = 1.0 - dt * evap_rate * samp[f, ei, ej, S_EVAP]
            if ev < 0.01:
                ev = 0.01
            vol *= ev
            sed /= ev
            x = nx
            y = ny
            f = nf
            h_prev = h_here
            # --- stop conditions -------------------------------------------
            if samp[nf, nei, nej, S_SURF] < 0.0 and not in_sea:
                # reached the ocean: from here on a deposit-only seafloor
                # walk (submarine fan) instead of dumping the load in one cell
                in_sea = True
                if sed <= 0.0:
                    cause = DEATH_OCEAN
                    break
            if vol < min_volume:
                cause = DEATH_OCEAN if in_sea else DEATH_EVAP
                break
            if dhr < 0.0:
                uphill += 1
                if uphill > pit_steps:
                    cause = DEATH_OCEAN if in_sea else DEATH_PIT
                    break  # stuck climbing out of a pit
            else:
                uphill = 0
        # --- final deposit ------------------------------------------------
        # (a) natural stop inside the mask: at the last active cell visited
        # (b) left the mask/array: only with deposit_on_exit
        if in_sea and cause == DEATH_AGE:
            cause = DEATH_OCEAN
        if in_sea and cause == DEATH_STOP:
            cause = DEATH_OCEAN
        sp_death[p] = cause
        if not exited:
            # the cell the particle stopped in joins the ring (the loop only
            # records cells it *left*); a natural stop re-visits the last one;
            # a sea particle never deposits on a land cell it bumped into
            ci = _cell_of(x, N)
            cj = _cell_of(y, N)
            ei = ci + H
            ej = cj + H
            if mask[f, ei, ej] == MASK_ACTIVE and not (in_sea and samp[f, ei, ej, S_SURF] >= 0.0):
                cell = (f * NE + ei) * NE + ej
                newest = (rpos + SPREAD - 1) % SPREAD
                if nring == 0 or cl_cell[rbase + newest] != cell:
                    cl_cell[rbase + rpos] = cell
                    rpos = (rpos + 1) % SPREAD
                    if nring < SPREAD:
                        nring += 1
        if sed > 0.0 and nring > 0:
            if (not exited) or deposit_on_exit:
                # final deposits, newest cell first: the death cell carries
                # the whole load, the older path cells are overflow slots
                # (delta 0) that apply_changes fills with the clamped excess.
                # A load reaching the sea never comes back on land: only the
                # ocean cell is listed, the excess goes offshore (pending).
                load = sed * vol / volume0
                nout = 1 if cause == DEATH_OCEAN else nring
                for r in range(nout):
                    cell = cl_cell[rbase + (rpos + SPREAD - 1 - r) % SPREAD]
                    cl_cell[base + n] = cell
                    cl_delta[base + n] = load if r == 0 else 0.0
                    cl_vol[base + n] = -1.0
                    cl_mom[base + n, 0] = 0.0
                    cl_mom[base + n, 1] = 0.0
                    n += 1
                    if cell < lo:
                        lo = cell
                    if cell > hi:
                        hi = cell
        cl_count[p] = n
        cl_lo[p] = lo
        cl_hi[p] = hi


@njit(cache=True)
def apply_changes(
    cl_cell, cl_delta, cl_vol, cl_mom, cl_count, cap,
    height, sediment, acc, pending, samp, disch_track, mom_track, mask,
    iter_erode, iter_deposit, fan_slope,
):
    """Apply a change list serially in particle order against the live
    terrain.  This is the single place where physical limits are enforced
    (the trace-time caps only reduce the size of the requests).

    Live terrain = ``height + sediment + acc`` where ``acc`` is the net
    change of the current iteration (folded into height/sediment once per
    iteration by :func:`fold_changes`; the packed float32 samples are kept
    in sync so later chunks see it).  Per entry, in list order:

    * erosion request ``e``: applied amount ``e' = min(e, budget, s_c -
      floor)`` with the budget ``iter_erode + acc[c]`` (net per iteration),
      ``floor = max(0 if the cell is land, live surface of the cell the
      particle stepped to next)`` — a land cell is never eroded below sea
      level nor below the cell downstream (so no pit can be dug, and the
      chain of floors ends at the ocean, which is never eroded).  The
      shortfall ``e - e'`` is carried as a *deficit* that cancels the
      particle's later deposits (it never carried that mass), so mass is
      conserved exactly; a deficit that outlives the list is mass the
      particle would have carried out of the window (window exits only).
    * deposit request ``d`` (plus any surplus from earlier clamped
      deposits, minus the deficit): ``d' = min(want, iter_deposit -
      acc[c], ceiling - s_c)``, ``ceiling`` = live surface of the previous
      path cell + ``DEP_FLOOR`` (a step deposit never lifts a cell above
      the one the particle came from; a final deposit never above the
      next-older path cell, so a pit fills to the level of its inlet and a
      river mouth aggrades backwards; a seafloor step never above the
      previous path cell minus ``fan_slope``, so fans descend away from
      their source; an ocean cell fills to ``-DEP_FLOOR`` at most — particles never turn sea into land; deltas prograde below
      the surface as submarine fans).
      The excess becomes the *surplus*
      offered to the next entries; whatever is left after the last entry
      goes to ``pending`` of the death cell — a per-cell stockpile that the
      next iteration re-injects as a particle starting with that load (in
      seafloor mode if the cell is submerged), so parked mass keeps moving
      until it finds room.

    Tracks: ``disch_track[c] += volume`` and ``mom_track[c] += volume *
    velocity`` for land step entries (volume > 0; seafloor steps carry
    volume 0, final deposits -1).
    Returns ``(clamped entries, mass sent to pending, mass of deficits
    left at the end of a list)``.
    """
    F, NE, _ = height.shape
    total = F * NE * NE
    P = cl_count.shape[0]
    hflat = height.reshape(total)
    sflat = sediment.reshape(total)
    aflat = acc.reshape(total)
    pdflat = pending.reshape(total)
    pflat = samp.reshape(total, NS)
    dflat = disch_track.reshape(total)
    mflat = mom_track.reshape(total, 2)
    kflat = mask.reshape(total)
    n_clamp = 0
    to_pending = 0.0
    lost = 0.0
    for p in range(P):
        base = p * cap
        cnt = cl_count[p]
        deficit = 0.0
        surplus = 0.0
        prev = -1  # previous step cell
        death = -1  # first final-deposit cell
        for k in range(cnt):
            c = cl_cell[base + k]
            if kflat[c] != MASK_ACTIVE:
                continue
            d = cl_delta[base + k]
            v = cl_vol[base + k]
            is_step = v >= 0.0  # path entry (v == 0: seafloor step, no discharge)
            s_c = hflat[c] + sflat[c] + aflat[c]
            if d < 0.0:
                e = -d
                e_act = e
                room = iter_erode + aflat[c]
                if e_act > room:
                    e_act = room
                floor = 0.0 if s_c >= 0.0 else s_c  # ocean cells are never eroded
                if k + 1 < cnt:
                    nc = cl_cell[base + k + 1]  # the cell the particle stepped to
                    ns = hflat[nc] + sflat[nc] + aflat[nc]
                    if ns > floor:
                        floor = ns
                if e_act > s_c - floor:
                    e_act = s_c - floor
                if e_act < 0.0:
                    e_act = 0.0
                if e_act < e:
                    n_clamp += 1
                    deficit += e - e_act
                aflat[c] -= e_act
            else:
                want = d + surplus
                surplus = 0.0
                if deficit > 0.0:
                    a = want if want < deficit else deficit
                    want -= a
                    deficit -= a
                d_act = want
                room = iter_deposit - aflat[c]
                if d_act > room:
                    d_act = room
                ref = -1
                if is_step:
                    ref = prev
                elif k + 1 < cnt:
                    ref = cl_cell[base + k + 1]  # next-older path cell
                if ref >= 0:
                    ceil = hflat[ref] + sflat[ref] + aflat[ref] + DEP_FLOOR
                    if s_c < 0.0 and is_step:
                        ceil -= DEP_FLOOR + fan_slope  # a fan descends away from its source
                    lim = ceil - s_c
                    if d_act > lim:
                        d_act = lim
                if s_c < 0.0:
                    # an ocean cell fills to just below sea level, never above
                    lim = -DEP_FLOOR - s_c
                    if d_act > lim:
                        d_act = lim
                if d_act < 0.0:
                    d_act = 0.0
                if d_act < want:
                    n_clamp += 1
                surplus = want - d_act
                aflat[c] += d_act
                if not is_step and death < 0:
                    death = c
            a2 = aflat[c]
            pflat[c, S_SURF] = hflat[c] + sflat[c] + a2
            sd = sflat[c] + a2
            pflat[c, S_SED] = sd if sd > 0.0 else 0.0
            if is_step:
                if v > 0.0:
                    dflat[c] += v
                    mflat[c, 0] += cl_mom[base + k, 0]
                    mflat[c, 1] += cl_mom[base + k, 1]
                prev = c
        if surplus > 0.0:
            if death >= 0:
                pdflat[death] += surplus
                to_pending += surplus
            # else: the particle left the window with that load
        lost += deficit
    return n_clamp, to_pending, lost


@njit(cache=True, parallel=True)
def fold_changes(height, sediment, acc, mask):
    """Fold the iteration's net change into the terrain: net erosion takes
    sediment first, then bedrock; net deposition adds sediment.  Zeroes
    ``acc``."""
    F, NE, _ = height.shape
    for f in prange(F):
        for ei in range(NE):
            for ej in range(NE):
                a = acc[f, ei, ej]
                acc[f, ei, ej] = 0.0
                if a == 0.0 or mask[f, ei, ej] != MASK_ACTIVE:
                    continue
                if a > 0.0:
                    sediment[f, ei, ej] += a
                else:
                    take = -a
                    s = sediment[f, ei, ej]
                    if s >= take:
                        sediment[f, ei, ej] = s - take
                    else:
                        height[f, ei, ej] -= take - s
                        sediment[f, ei, ej] = 0.0


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
def thermal_pass_a(height, sediment, metric, talus, mask, thermal_rate, thermal_max, out_total, scale):
    """Per-cell total outflow and cap scale (so a cell never drops below
    its steepest neighbour's talus line, nor sheds more than ``thermal_max``
    per pass)."""
    F, NE, _ = height.shape
    for f in prange(F):
        for ei in range(1, NE - 1):
            for ej in range(1, NE - 1):
                out_total[f, ei, ej] = 0.0
                scale[f, ei, ej] = 0.0
                if mask[f, ei, ej] != MASK_ACTIVE:
                    continue  # outside / frozen cells never donate
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
                if thermal_max > 0.0 and S * sc > thermal_max:
                    sc = thermal_max / S
                out_total[f, ei, ej] = S * sc
                scale[f, ei, ej] = sc


@njit(cache=True, parallel=True)
def thermal_pass_r(height, sediment, metric, talus, mask, thermal_rate, scale, rscale):
    """Receiver scale: a submerged cell (surface < 0) accepts inflow only up
    to its room below ``-DEP_FLOOR`` (mass wasting never turns sea into
    land, like particle deposition); land cells accept everything."""
    F, NE, _ = height.shape
    for f in prange(F):
        for ei in range(1, NE - 1):
            for ej in range(1, NE - 1):
                rscale[f, ei, ej] = 1.0
                surf = height[f, ei, ej] + sediment[f, ei, ej]
                if surf >= 0.0 or mask[f, ei, ej] != MASK_ACTIVE:
                    continue
                inflow = 0.0
                for k in range(8):
                    ni = ei + _D8[k, 0]
                    nj = ej + _D8[k, 1]
                    sc = scale[f, ni, nj]
                    if sc <= 0.0 or mask[f, ni, nj] == MASK_OUTSIDE:
                        continue
                    inflow += sc * _thermal_out(height, sediment, metric, talus, f, ni, nj, (k + 4) % 8, thermal_rate)
                room = -DEP_FLOOR - surf
                if room < 0.0:
                    room = 0.0
                if inflow > room:
                    rscale[f, ei, ej] = room / inflow


@njit(cache=True, parallel=True)
def thermal_pass_b(height, sediment, metric, talus, mask, thermal_rate, out_total, scale, rscale, new_height, new_sediment):
    """Gather: every active cell loses its outflows (each scaled by the
    receiver's ``rscale``; sediment first) and receives the capped outflows
    of its neighbours as sediment."""
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
                out = 0.0
                sc_self = scale[f, ei, ej]
                for k in range(8):
                    ni = ei + _D8[k, 0]
                    nj = ej + _D8[k, 1]
                    if mask[f, ni, nj] == MASK_OUTSIDE:
                        continue
                    if sc_self > 0.0 and mask[f, ni, nj] == MASK_ACTIVE:
                        out += sc_self * rscale[f, ni, nj] * _thermal_out(height, sediment, metric, talus, f, ei, ej, k, thermal_rate)
                    sc = scale[f, ni, nj]
                    if sc <= 0.0:
                        continue
                    # neighbour's outflow towards us: opposite D8 direction
                    o = _thermal_out(height, sediment, metric, talus, f, ni, nj, (k + 4) % 8, thermal_rate)
                    inflow += sc * o
                inflow *= rscale[f, ei, ej]
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
    "DEATH_NAMES",
    "trace_particles",
    "apply_changes",
    "fold_changes",
    "KERNEL_VERSION",
    "pack_samples",
    "NS",
    "thermal_pass_a",
    "thermal_pass_r",
    "thermal_pass_b",
    "ema_update",
    "spawn_cells",
]
