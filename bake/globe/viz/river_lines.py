"""River centre lines for the viewer's final frame (docs/viewer-rivers.md,
"River lines").

The final frame's rivers are the particles' discharge, interpolated from its
texels.  At a 4.9 km or 1.2 km cell that draws every river two or three
cells wide once the view is closer than a cell per pixel -- a 15 km blue
band with a fainter ghost channel beside it.  These are the rivers as lines
instead, drawn by the viewer at a width that follows their discharge:

1. **network**: each face's surface is priority-flooded towards its water
   and its border (:func:`globe.hydro.priority_flood.priority_flood_flat`),
   and every cell drains to the cell that flooded it -- a tree, so every
   path ends in water or at the face edge.  A cell carries the largest
   particle discharge anywhere upstream of it, and it is a river where that
   is at least ``q_min``: the particles pick the rivers, the tree joins
   them, and a river whose time-averaged discharge dips for a cell or two
   does not break;
2. **spurs**: a cell beside a channel inside the same discharge band drains
   into it after a cell or two, so only cells with at least ``min_length``
   cells of river upstream of them (the longest path) are kept -- a
   downstream cell always has more, so what is kept stays connected;
3. **reaches**: chains between sources, confluences and mouths, each cell
   centre a vertex on the unit sphere, smoothed (three passes of a
   ``1/4, 1/2, 1/4`` filter with the ends fixed, so the 45-degree steps of
   a grid path are gone and confluences stay joined) and simplified
   (Douglas-Peucker at ``tolerance`` cells).

Each vertex keeps the discharge of its cell; the viewer turns it into a
width (``RIVER_WIDTH_MAX_M`` at the planet's largest discharge, as its
square root below: hydraulic geometry's ``w ∝ Q^0.5``) and fades rivers in by
it as the view zooms out, the way the discharge texture did.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..cubesphere import to_sphere_v

#: width drawn for the largest discharge on the planet, metres (the Amazon
#: is 3-5 km wide along its lower course)
RIVER_WIDTH_MAX_M = 3000.0
#: hydraulic geometry: width ∝ discharge ** RIVER_WIDTH_EXPONENT (Leopold & Maddock's b ≈ 0.5)
RIVER_WIDTH_EXPONENT = 0.5

@njit(cache=True)
def _face_network(parent, pop_seq, wet, disc):
    """One face: the discharge each cell carries -- the most of any cell
    upstream of it along the flood tree (``pop_seq`` runs downstream first,
    so walking it backwards visits every cell before the cell it drains to)."""
    M = disc.shape[0]
    carried = np.zeros(M, np.float64)
    for c in range(M):
        if not wet[c]:
            carried[c] = disc[c]
    for k in range(pop_seq.shape[0] - 1, -1, -1):
        c = pop_seq[k]
        p = parent[c]
        if p >= 0 and not wet[p] and carried[c] > carried[p]:
            carried[p] = carried[c]
    return carried


@njit(cache=True)
def _upstream_length(rec):
    n = rec.shape[0]
    indeg = np.zeros(n, np.int64)
    for c in range(n):
        if rec[c] >= 0:
            indeg[rec[c]] += 1
    length = np.ones(n, np.int64)
    queue = np.empty(n, np.int64)
    head, tail = 0, 0
    left = indeg.copy()
    for c in range(n):
        if left[c] == 0:
            queue[tail] = c
            tail += 1
    while head < tail:
        c = queue[head]
        head += 1
        r = rec[c]
        if r >= 0:
            if length[c] + 1 > length[r]:
                length[r] = length[c] + 1
            left[r] -= 1
            if left[r] == 0:
                queue[tail] = r
                tail += 1
    return length


@njit(cache=True)
def _reaches(rec, keep):
    """Chains of kept cells: ``(order, starts)`` -- cell indices reach by
    reach (a confluence ends one reach and starts the next), and each
    reach's first position in ``order``; ``starts[-1]`` is the total."""
    n = rec.shape[0]
    indeg = np.zeros(n, np.int64)
    for c in range(n):
        if keep[c] and rec[c] >= 0 and keep[rec[c]]:
            indeg[rec[c]] += 1
    order = np.empty(2 * n + 1, np.int64)
    starts = np.empty(n + 1, np.int64)
    m, k = 0, 0
    for s in range(n):
        if not keep[s] or indeg[s] == 1:
            continue
        starts[k] = m
        k += 1
        c = s
        order[m] = c
        m += 1
        while True:
            r = rec[c]
            if r < 0 or not keep[r]:
                break
            order[m] = r
            m += 1
            if indeg[r] != 1:
                break
            c = r
    starts[k] = m
    return order[:m], starts[:k + 1]


@njit(cache=True)
def _simplify(p, lo, hi, tol, keep):
    """Douglas-Peucker on ``p[lo:hi]`` (unit vectors; chord distance)."""
    stack = np.empty((hi - lo + 1, 2), np.int64)
    top = 0
    stack[0, 0] = lo
    stack[0, 1] = hi - 1
    top = 1
    keep[lo] = True
    keep[hi - 1] = True
    while top > 0:
        top -= 1
        a, b = stack[top, 0], stack[top, 1]
        if b - a < 2:
            continue
        ax, ay, az = p[a, 0], p[a, 1], p[a, 2]
        ex, ey, ez = p[b, 0] - ax, p[b, 1] - ay, p[b, 2] - az
        ee = ex * ex + ey * ey + ez * ez
        worst, at = -1.0, -1
        for c in range(a + 1, b):
            vx, vy, vz = p[c, 0] - ax, p[c, 1] - ay, p[c, 2] - az
            t = 0.0
            if ee > 0.0:
                t = (vx * ex + vy * ey + vz * ez) / ee
                t = min(max(t, 0.0), 1.0)
            dx, dy, dz = vx - t * ex, vy - t * ey, vz - t * ez
            dd = dx * dx + dy * dy + dz * dz
            if dd > worst:
                worst, at = dd, c
        if worst > tol * tol:
            keep[at] = True
            stack[top, 0], stack[top, 1] = a, at
            stack[top + 1, 0], stack[top + 1, 1] = at, b
            top += 2


def trace(discharge: np.ndarray, water: np.ndarray, surface: np.ndarray, q_min: float, min_length: int = 4, smooth_passes: int = 3,
          tolerance: float = 0.25) -> dict:
    """River lines of a ``(6, r, r)`` final frame (discharge, water code 0 =
    land, surface metres).  Returns ``{"xyz": (n, 3) float32 unit vectors,
    "discharge": (n,) float32 carried discharge, "lengths": (m,) int64
    vertices per line}`` and counts."""
    from ..hydro.priority_flood import priority_flood_flat

    res = discharge.shape[1]
    rr = res * res
    border = np.zeros((res, res), np.bool_)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    cells, recs, mouths, carried_all = [], [], [], []
    base = 0
    for f in range(6):
        wet = water[f] > 0
        fr = priority_flood_flat(surface[f].astype(np.float32), wet | border)
        carried = _face_network(fr.parent, fr.pop_seq, wet.reshape(-1), discharge[f].reshape(-1).astype(np.float64))
        chan = (carried >= q_min) & ~wet.reshape(-1)
        idx = np.flatnonzero(chan)
        par = fr.parent[idx]
        ok = par >= 0
        to_chan = np.zeros(idx.size, np.bool_)
        to_chan[ok] = chan[par[ok]]
        rec = np.full(idx.size, -1, np.int64)
        rec[to_chan] = base + np.searchsorted(idx, par[to_chan])
        mouth = np.full(idx.size, -1, np.int64)
        into_water = ok & ~to_chan
        into_water[ok] &= wet.reshape(-1)[par[ok]]
        mouth[into_water] = f * rr + par[into_water]
        cells.append(f * rr + idx)
        recs.append(rec)
        mouths.append(mouth)
        carried_all.append(carried[idx])
        base += idx.size
    flat = np.concatenate(cells)
    rec = np.concatenate(recs)
    mouth = np.concatenate(mouths)
    d = np.concatenate(carried_all)
    stats = {"channel_cells": int(flat.size)}
    if flat.size == 0:
        return {"xyz": np.zeros((0, 3), np.float32), "discharge": np.zeros(0, np.float32), "lengths": np.zeros(0, np.int64), **stats}
    f = flat // rr
    i, j = (flat % rr) // res, (flat % rr) % res
    length = _upstream_length(rec)
    keep = length >= int(min_length)
    order, starts = _reaches(rec, keep)
    centre = to_sphere_v(f, (i + 0.5) / res, (j + 0.5) / res)
    mf = np.maximum(mouth, 0) // rr
    mrem = np.maximum(mouth, 0) % rr
    wet_centre = to_sphere_v(mf, (mrem // res + 0.5) / res, (mrem % res + 0.5) / res)
    tol = float(tolerance) * (np.pi / 2) / res
    xyz, q, lengths = [], [], []
    ends_at_water = 0
    for k in range(starts.size - 1):
        run = order[starts[k]:starts[k + 1]]
        p = centre[run]
        qq = d[run]
        last = run[-1]
        if mouth[last] >= 0:
            p = np.vstack([p, wet_centre[last]])
            qq = np.append(qq, qq[-1])
            ends_at_water += 1
        if p.shape[0] < 2:
            continue
        for _ in range(int(smooth_passes)):
            if p.shape[0] < 3:
                break
            p[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
            p /= np.linalg.norm(p, axis=1, keepdims=True)
        sel = np.zeros(p.shape[0], np.bool_)
        _simplify(p, 0, p.shape[0], tol, sel)
        xyz.append(p[sel])
        q.append(qq[sel])
        lengths.append(int(sel.sum()))
    at_edge = (i == 0) | (j == 0) | (i == res - 1) | (j == res - 1)
    stats.update({"kept_cells": int(keep.sum()), "lines": len(lengths), "vertices": int(sum(lengths)), "mouths": ends_at_water,
                  "ends_at_face_edge": int((keep & (rec < 0) & (mouth < 0) & at_edge).sum()),
                  "ends_elsewhere": int((keep & (rec < 0) & (mouth < 0) & ~at_edge).sum())})
    return {"xyz": np.concatenate(xyz).astype(np.float32) if xyz else np.zeros((0, 3), np.float32),
            "discharge": np.concatenate(q).astype(np.float32) if q else np.zeros(0, np.float32),
            "lengths": np.asarray(lengths, np.int64), **stats}


__all__ = ["RIVER_WIDTH_MAX_M", "RIVER_WIDTH_EXPONENT", "trace"]
