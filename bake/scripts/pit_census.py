"""Where do the pits come from?  Per iteration of a window job: the surface
before the particle pass, after it and after the thermal pass; the closed
depressions of the result (local priority flood towards the job's drains);
and for every depression that is new this iteration, whether its bottom was
lowered (dug) or its spill cell raised (dammed), and by which pass.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from globe.hydro.priority_flood import priority_flood_flat

EIGHT = np.ones((3, 3), bool)


def depressions(surf, drain, flood_active, active, thr):
    fr = priority_flood_flat(surf.astype(np.float32), drain, flood_active)
    filled = fr.filled.reshape(surf.shape).astype(np.float64)
    reached = fr.order.reshape(surf.shape) >= 0
    depth = np.where(reached & active, filled - surf, 0.0)
    return depth > thr, filled, depth


def census(s0, s1, s2, drain, flood_active, active, prev_pits, thr=0.1, core=None):
    """``s0`` surface before the particle pass, ``s1`` after it, ``s2`` after
    the thermal pass (metres, window arrays).  Returns (stats, pits)."""
    pits, filled, depth = depressions(s2, drain, flood_active, active, thr)
    lab, n = ndimage.label(pits, structure=EIGHT)
    st = {"pit_cells": int(pits.sum()), "depressions": int(n)}
    if n == 0:
        return st, pits
    new_cells = pits & ~prev_pits
    ids = np.unique(lab[new_cells])
    ids = ids[ids > 0]
    dp = s1 - s0
    dt = s2 - s1
    rows = []
    objs = ndimage.find_objects(lab)
    H, W = s2.shape
    for k in ids:
        sl = objs[k - 1]
        i0, i1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, H)
        j0, j1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, W)
        comp = lab[i0:i1, j0:j1] == k
        sub = s2[i0:i1, j0:j1]
        bi, bj = np.unravel_index(np.argmin(np.where(comp, sub, np.inf)), comp.shape)
        level = float(filled[i0:i1, j0:j1][bi, bj])
        ring = ndimage.binary_dilation(comp, structure=EIGHT) & ~comp
        # the spill cell: the lowest rim cell (its surface is the level)
        rs = np.where(ring, sub, np.inf)
        si, sj = np.unravel_index(np.argmin(rs), comp.shape)
        B = (i0 + bi, j0 + bj)
        S = (i0 + si, j0 + sj)
        dig_p, dig_t = -dp[B], -dt[B]
        dam_p, dam_t = dp[S], dt[S]
        size = int(comp.sum())
        # was the bottom below the spill before this iteration?  If not, what
        # made it a pit happened this iteration
        rows.append((size, level - s2[B], dig_p, dig_t, dam_p, dam_t, s0[B] - s0[S]))
    a = np.array(rows, dtype=np.float64)
    size, dep, dig_p, dig_t, dam_p, dam_t, pre = a.T
    dig = dig_p + dig_t
    dam = dam_p + dam_t
    st.update({
        "new_depressions": int(len(rows)),
        "new_size_p50_max": [float(np.median(size)), float(size.max())],
        "new_depth_m_p50_p90": [float(np.median(dep)), float(np.percentile(dep, 90))],
        # a depression is 'dug' when its bottom went down by more than its
        # spill cell came up, 'dammed' otherwise
        "dug_share": float(np.mean(dig > dam)),
        "dug_by_particles_share": float(np.mean((dig > dam) & (dig_p >= dig_t))),
        "dammed_by_particles_share": float(np.mean((dig <= dam) & (dam_p >= dam_t))),
        "bottom_lowered_m_p50": float(np.median(dig)),
        "spill_raised_m_p50": float(np.median(dam)),
        "bottom_above_spill_before_share": float(np.mean(pre > 0.0)),
    })
    return st, pits
