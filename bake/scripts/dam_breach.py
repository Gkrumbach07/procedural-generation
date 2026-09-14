"""Breach the dams a particle pass builds (prototype).

The pit census (pit_census.py) finds 92-98 % of the depressions a window's
particle pass makes are dams: the spill cell raised by deposition, not the
bottom lowered.  This takes back, at each depression's spill cell, the part
of *this iteration's* raise that closes it -- never below the cell's surface
before the pass -- and books it as pending sediment at that cell, which the
next iteration re-injects as a loaded particle (mass kept).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from globe.hydro.priority_flood import priority_flood_flat

EIGHT = np.ones((3, 3), bool)


def breach(state, s0_m, drain, flood_active, tol_m=0.05, passes=3):
    """``state`` a window ErosionState (F = 1); ``s0_m`` the interior
    surface (metres) before the particle pass; ``drain`` / ``flood_active``
    interior bool arrays.  Returns stats."""
    H = state.H
    unit = state.height_unit_m
    inter = (0, slice(H, -H), slice(H, -H))
    hgt = state.height[inter]
    sed = state.sediment[inter]
    pend = state.pending[inter]
    act = state.mask[inter] == 1
    s0 = s0_m / unit
    tol = tol_m / unit
    moved = 0.0
    opened = 0
    left = 0
    for _ in range(passes):
        surf = hgt + sed
        fr = priority_flood_flat(surf.astype(np.float32), drain, flood_active)
        F = fr.filled.reshape(surf.shape).astype(np.float64)
        reached = fr.order.reshape(surf.shape) >= 0
        pits = reached & act & (F - surf > tol)
        lab, n = ndimage.label(pits, structure=EIGHT)
        if n == 0:
            break
        objs = ndimage.find_objects(lab)
        Hh, Ww = surf.shape
        changed = 0
        for k in range(1, n + 1):
            sl = objs[k - 1]
            i0, i1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, Hh)
            j0, j1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, Ww)
            comp = lab[i0:i1, j0:j1] == k
            sub = surf[i0:i1, j0:j1]
            ring = ndimage.binary_dilation(comp, structure=EIGHT) & ~comp & act[i0:i1, j0:j1]
            if not ring.any():
                continue
            si, sj = np.unravel_index(np.argmin(np.where(ring, sub, np.inf)), comp.shape)
            S = (i0 + si, j0 + sj)
            bottom = float(np.min(np.where(comp, sub, np.inf)))
            target = max(bottom + 1e-6, float(s0[S]))       # never below its pre-pass surface
            want = float(surf[S]) - target
            if want <= 0.0:
                left += 1
                continue
            take = min(want, float(sed[S]))
            if take <= 0.0:
                left += 1
                continue
            sed[S] -= take
            pend[S] += take
            moved += take
            changed += 1
            if take >= want - 1e-12:
                opened += 1
        if changed == 0:
            break
    return {"breached": opened, "not_breachable": left, "moved_m": moved * unit}
