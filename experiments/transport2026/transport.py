"""Stochastic geomorphological transport (McDonald & Cordonnier 2026).

Prototype of the SIGGRAPH 2026 scheme, on a plain 2-D grid, so it can be
validated on its own before being wired into the bake.

The paper solves every transported quantity with one quasi-static linear
conservation law (their Eq. 13)

    div(phi v) = S - R phi

by superposing *attenuated upstream contributions* and evaluating that
integral by Monte Carlo (their Algorithm 1): seed N particles, walk each
one downstream along v in steps of dx, accumulate the probability-weighted
source into every cell it passes, and decay its weight by
exp(-dx R / |v|) as it goes.  Every cell's value then falls out of its own
outflow flux:

    phi_k = (1/N) Phi_k / sum_{c in outflow(k)} |c| v_k . n_c

Four systems, all of that shape:

  water      phi = h_f   S = p_f              R = k_e                (Eq. 10)
  sediment   phi = h_s   S = E_f              R = k_df / h_f         (Eq. 11 + 4)
  debris     phi = h_d   S = E_d + E_l        R = D_d / h_d          (Eq. 11 + 7)
  momentum   phi = v     S = -g grad z + (mu/rho) lap v
                                              R = tau / (rho h |v|)  (Eq. 12)

Momentum is the one that makes meanders: transporting v itself keeps the
inertia that the Stream Power Law throws away.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange, get_num_threads


# --------------------------------------------------------------------------
# the transport solver (their Algorithm 1)
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _bilin(f, x, y):
    """Bilinear sample of a scalar field at continuous cell coordinates."""
    n0, n1 = f.shape
    i = int(np.floor(x)); j = int(np.floor(y))
    tx = x - i; ty = y - j
    if i < 0: i, tx = 0, 0.0
    if j < 0: j, ty = 0, 0.0
    if i > n0 - 2: i, tx = n0 - 2, 1.0
    if j > n1 - 2: j, ty = n1 - 2, 1.0
    return ((1 - tx) * ((1 - ty) * f[i, j] + ty * f[i, j + 1])
            + tx * ((1 - ty) * f[i + 1, j] + ty * f[i + 1, j + 1]))


@njit(cache=True, parallel=True)
def integrate(S, R, vx, vy, mask, dx, n_per_cell, max_steps, eps, seed, step, acc):
    """Accumulate Phi into `acc` (n_threads, n0, n1) — the caller sums it.

    Per-thread accumulators rather than atomics: the paper's CUDA version
    uses atomic adds, whose ordering is nondeterministic and would break
    this project's byte-reproducibility.  Summing fixed per-thread buffers
    in thread order is deterministic and costs one reduction.
    """
    n0, n1 = S.shape
    nthreads = acc.shape[0]
    total = n0 * n1
    N = int(total * n_per_cell)
    area = total * dx * dx          # |Omega|; p(y) = 1/|Omega| uniform
    for t in prange(nthreads):
        lo = (N * t) // nthreads
        hi = (N * (t + 1)) // nthreads
        st = np.random.seed(seed + t) if False else None
        rng = np.random.RandomState(seed + t) if False else None
        # xorshift, so the stream depends only on (seed, t, index)
        for p in range(lo, hi):
            s = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(2 * p + 1)
            s ^= s >> np.uint64(33); s *= np.uint64(0xff51afd7ed558ccd)
            s ^= s >> np.uint64(33); s *= np.uint64(0xc4ceb9fe1a85ec53)
            s ^= s >> np.uint64(33)
            u0 = (s >> np.uint64(11)) * (1.0 / 9007199254740992.0)
            s ^= s >> np.uint64(31); s *= np.uint64(0x9e3779b97f4a7c15)
            u1 = (s >> np.uint64(11)) * (1.0 / 9007199254740992.0)
            x = u0 * (n0 - 1)
            y = u1 * (n1 - 1)
            if mask[int(x), int(y)] == 0:
                continue
            Sy = _bilin(S, x, y)
            if Sy == 0.0:
                continue
            Shat = Sy * area          # S(y) / p(y)
            M = 1.0
            for _ in range(max_steps):
                i = int(x); j = int(y)
                if i < 0 or j < 0 or i >= n0 or j >= n1 or mask[i, j] == 0:
                    break
                # Accumulate on EVERY step, not once per cell entered.  The
                # paper cautions that a particle must contribute to each
                # control cell once, but that pairs with a different
                # denominator: here the normaliser dx(|vx|+|vy|) already
                # carries the sqrt(2) that a diagonal crossing takes in extra
                # steps, so the two cancel.  Measured on the cone: counting
                # steps gives MSRE 0.048, counting unique cells 0.065.
                ux = _bilin(vx, x, y); uy = _bilin(vy, x, y)
                sp = np.sqrt(ux * ux + uy * uy)
                if sp < 1e-12:
                    break
                # Attenuation across this step.  The contribution of the step
                # is the MEAN of M over it, not M at its start: with strong
                # decay a particle is mostly gone before it crosses the cell,
                # and taking the entry value over-counts it by a/(1-e^-a).
                # (The cone validation has R = 0, where the factor is 1, which
                # is why it passed while the erosion loop blew up.)
                a = dx * step * _bilin(R, x, y) / sp
                if a > 1e-8:
                    dec = np.exp(-a)
                    acc[t, i, j] += Shat * M * step * (1.0 - dec) / a
                    M *= dec
                else:
                    acc[t, i, j] += Shat * M * step
                if M < eps:
                    break
                x += step * ux / sp
                y += step * uy / sp
    return N


def solve(S, R, vx, vy, mask, dx, n_per_cell=1.0, max_steps=None, eps=1e-3, seed=0, step=1.0, fallback=None):
    """phi from div(phi v) = S - R phi, by their Eqs. 19-22."""
    n0, n1 = S.shape
    if max_steps is None:
        max_steps = int(4 * max(n0, n1) / step)
    nth = get_num_threads()
    acc = np.zeros((nth, n0, n1))
    N = integrate(np.ascontiguousarray(S), np.ascontiguousarray(R),
                  np.ascontiguousarray(vx), np.ascontiguousarray(vy),
                  mask.astype(np.uint8), float(dx), float(n_per_cell),
                  int(max_steps), float(eps), int(seed), float(step), acc)
    Phi = acc.sum(axis=0) / max(N, 1)
    # divide by the cell's own outflow flux: sum |c| v.n over outflow edges,
    # which on a regular grid is dx (|vx| + |vy|)
    out = dx * (np.abs(vx) + np.abs(vy))
    # Where the flow is (near) stagnant the estimator degenerates: no particle
    # advects, so Phi is empty and the outflow flux it would be divided by is
    # zero.  The paper's control-volume construction assumes v does not vanish
    # on C.  The correct limit is the local balance of Eq. 13 with the
    # divergence dropped, phi = S/R -- without it, water on flat ground decays
    # to nothing instead of settling at p_f/k_e.
    # `fallback` is that limit. S/R is right for a LINEAR decay (water,
    # sediment, debris), but wrong for momentum, where R = tau/(rho h |v|)
    # itself depends on |v|: there S/R gives 100 m/s instead of the
    # gravity-friction root sqrt(g h slope / f_D), so the caller passes that
    # root in explicitly.
    if fallback is None:
        # The 1e-30 floor is a guard against division by zero, not a physical
        # value: with a genuinely vanishing R the balance has no finite
        # solution, and S/1e-30 overflowed to inf here before the callers
        # stopped handing in R = 0.  Take the ratio only where R is
        # meaningfully positive and leave the rest at zero.
        local = np.where(R > 1e-12, S / np.maximum(R, 1e-12), 0.0)
    else:
        local = fallback
    return np.where(out > 1e-9, Phi / np.maximum(out, 1e-12), local)


__all__ = ["solve", "integrate"]
