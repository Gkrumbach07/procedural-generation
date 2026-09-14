"""Drainage-consistent sub-cell relief (prototype for docs/zoom-windows.md
item 2).

Noise alone leaves closed pits; erosion cannot drain them in any useful
number of iterations, so they become lakes and break the river network.
This builds relief that drains by construction:

1. perturb the plain upsample with smooth fbm, so fine flow paths wander;
2. priority-flood it towards the window's drains: a pit-free spanning tree
   (each cell's ``parent`` flooded it; popping is in non-decreasing level);
3. accumulate upstream area along that tree;
4. carve every cell by a depth that grows with its upstream area -- channels
   deepest, divides untouched.  A child never has more upstream area than
   its parent and never a lower filled level, so the carved surface still
   descends along the tree: no pits;
5. remove the coarse-scale offset (a Gaussian low-pass of the change, sigma
   ~ one coarse cell) so the parent's elevations hold, then re-impose descent
   along the tree (a cell is raised to its parent + eps where the correction
   put it lower).
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

from globe.hydro.priority_flood import priority_flood_flat
from globe.refine.upsample import value_noise_2d


@njit(cache=True)
def _accumulate(pop_seq, parent, n):
    acc = np.ones(n, dtype=np.float64)
    for k in range(pop_seq.size - 1, -1, -1):
        c = pop_seq[k]
        p = parent[c]
        if p >= 0:
            acc[p] += acc[c]
    return acc


@njit(cache=True)
def _descend(z, pop_seq, parent, eps):
    for k in range(pop_seq.size):
        c = pop_seq[k]
        p = parent[c]
        if p >= 0 and z[c] < z[p] + eps:
            z[c] = z[p] + eps
    return z


def fbm(shape, base_wavelength, rng, min_wavelength=4.0, gain=0.55):
    out = np.zeros(shape, np.float64)
    w, lam, tot = 1.0, float(base_wavelength), 0.0
    while lam >= min_wavelength:
        out += w * value_noise_2d(shape, lam, rng)
        tot += w
        w *= gain
        lam *= 0.5
    out /= max(tot, 1e-12)
    return out / max(float(np.abs(out).max()), 1e-12)


def drainage_relief(surface, active, drain, R, cell_m, rng, amp_m, route_frac=0.35, theta=0.5, eps_m=0.01, stats=None):
    """``surface`` plain upsample (metres, NE x NE); ``active`` cells to
    shape; ``drain`` cells water leaves through; ``amp_m`` target valley
    depth (metres, NE x NE) for a channel draining one coarse cell.  Returns
    the new surface (metres)."""
    shape = surface.shape
    P = surface.astype(np.float64)
    amp = np.maximum(amp_m.astype(np.float64), 0.0)
    S0 = P + route_frac * amp * fbm(shape, 2.0 * R, rng)
    fr = priority_flood_flat(S0.astype(np.float32), drain, active | drain)
    active = active & ~drain
    n = P.size
    parent = fr.parent
    pop = fr.pop_seq
    acc = _accumulate(pop, parent, n).reshape(shape)
    F = fr.filled.astype(np.float64).reshape(shape)
    # valley depth: 0 at a divide, amp at one coarse cell of upstream area,
    # growing slowly beyond (stream power: depth ~ log area over this range)
    ref = float(R * R)
    t = np.log1p(acc) / np.log1p(ref)
    D = amp * np.minimum(t, 1.5) ** (1.0 / max(theta, 1e-3)) * 0.5 + amp * np.minimum(t, 1.5) * 0.5
    Z = F - D
    # hold the parent's coarse-scale elevation
    d = np.where(active, Z - P, 0.0)
    w = ndimage.gaussian_filter(active.astype(np.float64), R)
    low = ndimage.gaussian_filter(d, R) / np.maximum(w, 1e-6)
    Z = np.where(active, Z - low, P)
    Z = _descend(Z.reshape(-1).copy(), pop, parent, eps_m).reshape(shape)
    Z = np.where(active, Z, P)
    if stats is not None:
        stats.update({"acc_max": float(acc.max()), "carve_max_m": float(D[active].max()) if active.any() else 0.0,
                      "lowpass_max_m": float(np.abs(low[active]).max()) if active.any() else 0.0,
                      "reached": float((fr.order.reshape(shape)[active] >= 0).mean()) if active.any() else 0.0})
    return Z.astype(np.float32)
