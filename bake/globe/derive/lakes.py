"""Fine lakes (PLAN.md section 11): connected components of
``water_surface - surface > lake_min_depth`` on the fine grid, boundary
rings traced on the fine cell-corner lattice, written to
``graph/lakes.json`` (schema in docs/DEVELOPING.md).

Faces are processed one at a time (memory).  Pieces of one lake that lie
on different faces are merged when they touch across a cube edge at the
same water level (:func:`link_pieces`, through ``fine.edge_neighbors``)
or share a *coarse* lake label (built cross-face with
``grid.owner_face_ij``); merged pieces become one record with several
``rings``.  ``coarse_id`` / ``outlet`` are those of the
``graph/lakes_coarse.json`` (hydro) lake whose outlet cell carries the
coarse label under the fine lake; ``-1`` / ``null`` when there is none
(sub-coarse lakes, stub hydro).

Rings are closed (first point == last), counter-clockwise in ``(u, v)``
for outer boundaries and clockwise for holes (islands), on corner
coordinates ``u = i / N_fine``; collinear vertices are dropped.  Two lake
cells touching only at a corner are one 8-connected lake, so the ring
turns *right* at such pinch points.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

_STRUCT8 = np.ones((3, 3), dtype=bool)


# --------------------------------------------------------------------------
# masks and labels
# --------------------------------------------------------------------------
def lake_mask(surface: np.ndarray, water_surface: np.ndarray, min_depth_m: float) -> np.ndarray:
    """Lake cells: land (``surface >= 0``) with a water surface (``> 0``;
    ocean cells are 0 by contract) more than ``min_depth_m`` above it."""
    s = np.asarray(surface, dtype=np.float32)
    w = np.asarray(water_surface, dtype=np.float32)
    return (w > 0.0) & (s >= 0.0) & ((w - s) > np.float32(min_depth_m))


def label_lakes(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components: ``(labels int32 (0 = none), n)``."""
    lab, n = ndimage.label(np.asarray(mask, dtype=bool), structure=_STRUCT8)
    return lab.astype(np.int32, copy=False), int(n)


def coarse_lake_labels(lake6: np.ndarray, level6: np.ndarray, grid, tol: float = 0.01) -> tuple[np.ndarray, int]:
    """Cross-face connected components of the coarse ``(6, N, N)`` lake
    mask, joining cells across cube edges when both are lakes at the same
    ``level`` (water surface, within ``tol``).  Returns ``(labels int32,
    -1 outside, n)``; labels are dense and numbered by first occurrence in
    ``[f, i, j]`` raster order (deterministic)."""
    N, H = grid.N, grid.H
    m = np.asarray(lake6, dtype=bool)
    lv = np.asarray(level6, dtype=np.float32)
    lab = np.full((6, N, N), -1, dtype=np.int32)
    n = 0
    for f in range(6):
        l, k = ndimage.label(m[f], structure=_STRUCT8)
        on = l > 0
        lab[f][on] = l[on] - 1 + n
        n += int(k)
    if n == 0:
        return lab, 0
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ofij = grid.owner_face_ij
    rng = np.arange(N)
    sides = (
        (np.full(N, H - 1), rng + H, np.zeros(N, dtype=np.int64), rng),
        (np.full(N, H + N), rng + H, np.full(N, N - 1), rng),
        (rng + H, np.full(N, H - 1), rng, np.zeros(N, dtype=np.int64)),
        (rng + H, np.full(N, H + N), rng, np.full(N, N - 1)),
    )
    for f in range(6):
        for ei, ej, ii, jj in sides:
            o = ofij[f, ei, ej]
            a = lab[f, ii, jj]
            b = lab[o[:, 0], o[:, 1], o[:, 2]]
            ok = (a >= 0) & (b >= 0) & (np.abs(lv[f, ii, jj] - lv[o[:, 0], o[:, 1], o[:, 2]]) <= tol)
            for x, y in zip(a[ok].tolist(), b[ok].tolist()):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(x) for x in range(n)], dtype=np.int64)
    # dense ids in order of first appearance
    on = lab >= 0
    seq = roots[lab[on]]
    _, first = np.unique(seq, return_index=True)
    order = np.argsort(first)
    remap = np.empty(n, dtype=np.int32)
    remap[:] = -1
    uniq = seq[np.sort(first)]
    remap[uniq] = np.arange(uniq.size, dtype=np.int32)
    lab[on] = remap[seq]
    return lab, int(uniq.size)


# --------------------------------------------------------------------------
# ring tracing
# --------------------------------------------------------------------------
@njit(cache=True)
def _follow(nxt):
    """Follow successor pointers into rings.  Returns (ring_off, order)
    with ring r = order[ring_off[r]:ring_off[r+1]] (edge indices)."""
    K = nxt.size
    used = np.zeros(K, dtype=np.uint8)
    order = np.empty(K, dtype=np.int64)
    ring_off = np.empty(K + 1, dtype=np.int64)
    pos = 0
    nr = 0
    ring_off[0] = 0
    for s in range(K):
        if used[s]:
            continue
        e = s
        while e >= 0 and used[e] == 0:
            used[e] = 1
            order[pos] = e
            pos += 1
            e = nxt[e]
        nr += 1
        ring_off[nr] = pos
    return ring_off[: nr + 1], order[:pos]


def trace_rings(mask2d: np.ndarray) -> list[np.ndarray]:
    """Boundary rings of a 2-D bool mask ``[i, j]`` on the corner lattice.
    Each ring is a closed ``(K, 2)`` int64 array of ``(i, j)`` corners with
    collinear vertices removed; outer rings are counter-clockwise in
    ``(i, j)``, holes clockwise.  Ordered by |area| (largest first)."""
    m = np.asarray(mask2d, dtype=bool)
    if not m.any():
        return []
    Ni, Nj = m.shape
    pad = np.zeros((Ni + 2, Nj + 2), dtype=bool)
    pad[1:-1, 1:-1] = m
    segs = []
    ii, jj = np.nonzero(m & ~pad[1:-1, :-2])  # neighbour (i, j-1) absent
    segs.append(np.stack([ii, jj, ii + 1, jj], axis=1))
    ii, jj = np.nonzero(m & ~pad[2:, 1:-1])  # (i+1, j) absent
    segs.append(np.stack([ii + 1, jj, ii + 1, jj + 1], axis=1))
    ii, jj = np.nonzero(m & ~pad[1:-1, 2:])  # (i, j+1) absent
    segs.append(np.stack([ii + 1, jj + 1, ii, jj + 1], axis=1))
    ii, jj = np.nonzero(m & ~pad[:-2, 1:-1])  # (i-1, j) absent
    segs.append(np.stack([ii, jj + 1, ii, jj], axis=1))
    S = np.concatenate(segs, axis=0).astype(np.int64)
    W = Nj + 2
    start = S[:, 0] * W + S[:, 1]
    end = S[:, 2] * W + S[:, 3]
    order = np.lexsort((end, start))
    S = S[order]
    start = start[order]
    end = end[order]
    K = S.shape[0]
    lo = np.searchsorted(start, end, side="left")
    hi = np.searchsorted(start, end, side="right")
    nxt = np.full(K, -1, dtype=np.int64)
    single = hi - lo == 1
    nxt[single] = lo[single]
    multi = np.nonzero(hi - lo >= 2)[0]
    if multi.size:
        din = S[multi, 2:4] - S[multi, 0:2]
        for e, d in zip(multi.tolist(), din):
            best = -1
            best_cross = 1
            for c in range(int(lo[e]), int(hi[e])):
                dout = S[c, 2:4] - S[c, 0:2]
                cross = int(d[0] * dout[1] - d[1] * dout[0])
                if cross < best_cross:  # right turn (cross < 0) preferred
                    best_cross = cross
                    best = c
            nxt[e] = best
    ring_off, seq = _follow(nxt)
    rings = []
    for r in range(ring_off.size - 1):
        idx = seq[ring_off[r] : ring_off[r + 1]]
        pts = S[idx, 0:2]
        pts = np.vstack([pts, pts[:1]])
        pts = simplify_ring(pts)
        if pts.shape[0] >= 4:
            rings.append(pts)
    rings.sort(key=lambda p: -abs(ring_area(p)))
    return rings


def simplify_ring(pts: np.ndarray) -> np.ndarray:
    """Drop vertices on straight runs of a closed ring (first == last kept)."""
    p = np.asarray(pts, dtype=np.int64)
    if p.shape[0] < 4:
        return p
    core = p[:-1]
    prev = np.roll(core, 1, axis=0)
    nxt = np.roll(core, -1, axis=0)
    d0 = core - prev
    d1 = nxt - core
    keep = (d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]) != 0
    if not keep.any():
        return p
    core = core[keep]
    return np.vstack([core, core[:1]])


def ring_area(pts: np.ndarray) -> float:
    """Signed shoelace area of a closed ring (positive = counter-clockwise
    in (i, j))."""
    p = np.asarray(pts, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


# --------------------------------------------------------------------------
# face pieces -> records
# --------------------------------------------------------------------------
def face_lake_pieces(face: int, mask: np.ndarray, water_surface: np.ndarray, surface: np.ndarray, area_fine: np.ndarray, coarse_labels_face: np.ndarray | None, R: int, min_cells: int, piece_base: int = 0) -> tuple[list[dict], np.ndarray]:
    """Lake pieces of one fine face.  ``area_fine`` is the fine cell area
    (m², ``(Nf, Nf)`` or broadcastable); ``coarse_labels_face`` the coarse
    lake labels of this face ``(N, N)`` (or None).  Returns ``(pieces,
    frame)``: each piece ``face, cells, area_m2, surface_m, max_depth_m,
    coarse_label, rings (list of (K, 2) corner arrays), bbox``; ``frame``
    ``(4, Nf)`` int64 holds ``piece_base + index`` of the piece on the four
    edge rows (``fine.SIDES`` order: ``[0, :], [-1, :], [:, 0], [:, -1]``),
    -1 where none — the input of :func:`link_pieces`."""
    lab, n = label_lakes(mask)
    pieces = []
    Nf = mask.shape[0]
    frame = np.full((4, Nf), -1, dtype=np.int64)
    if n == 0:
        return pieces, frame
    ws = np.asarray(water_surface, dtype=np.float32)
    sf = np.asarray(surface, dtype=np.float32)
    area = np.broadcast_to(np.asarray(area_fine, dtype=np.float64), (Nf, Nf))
    objs = ndimage.find_objects(lab)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    remap = np.full(n + 1, -1, dtype=np.int64)
    for L in range(1, n + 1):
        if sizes[L] < min_cells:
            continue
        sl = objs[L - 1]
        if sl is None:
            continue
        remap[L] = piece_base + len(pieces)
        sub = lab[sl] == L
        i0, j0 = sl[0].start, sl[1].start
        cw = ws[sl][sub]
        cs = sf[sl][sub]
        rings = [r + np.array([i0, j0]) for r in trace_rings(sub)]
        coarse_label = -1
        if coarse_labels_face is not None:
            ci = np.arange(sl[0].start, sl[0].stop) // R
            cj = np.arange(sl[1].start, sl[1].stop) // R
            cl = coarse_labels_face[np.ix_(ci, cj)][sub]
            cl = cl[cl >= 0]
            if cl.size:
                u, c = np.unique(cl, return_counts=True)
                coarse_label = int(u[np.argmax(c)])
        pieces.append(
            {
                "face": int(face),
                "cells": int(sizes[L]),
                "area_m2": float(area[sl][sub].sum()),
                "surface_m": float(cw.mean()),
                "max_depth_m": float((cw - cs).max()),
                "coarse_label": coarse_label,
                "rings": rings,
                "bbox": [int(sl[0].start), int(sl[1].start), int(sl[0].stop), int(sl[1].stop)],
            }
        )
    frame[0] = remap[lab[0, :]]
    frame[1] = remap[lab[-1, :]]
    frame[2] = remap[lab[:, 0]]
    frame[3] = remap[lab[:, -1]]
    return pieces, frame


def link_pieces(pieces: list[dict], frames: list[np.ndarray], N_fine: int, tol: float = 0.01) -> np.ndarray:
    """Group id (root piece index) of every piece: pieces are joined when
    they touch across a cube edge (8-connectivity through
    ``fine.edge_neighbors``) at the same ``surface_m`` (within ``tol`` m)
    or when they carry the same coarse lake label.  ``frames`` are the six
    ``(4, N_fine)`` edge frames of :func:`face_lake_pieces`."""
    from .fine import edge_neighbors

    n = len(pieces)
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    if n == 0:
        return parent
    level = np.array([p["surface_m"] for p in pieces], dtype=np.float64)
    G = np.stack([np.asarray(fr, dtype=np.int64) for fr in frames])
    Nf = N_fine
    for f in range(6):
        for side, (f2, i2, j2) in enumerate(edge_neighbors(Nf, f, 1)):
            a = G[f, side]
            f2, i2, j2 = f2.ravel(), i2.ravel(), j2.ravel()
            side2 = np.where(i2 == 0, 0, np.where(i2 == Nf - 1, 1, np.where(j2 == 0, 2, 3)))
            along = np.where(side2 < 2, j2, i2)
            for d in (-1, 0, 1):
                b = G[f2, side2, np.clip(along + d, 0, Nf - 1)]
                ok = (a >= 0) & (b >= 0) & (a != b)
                ok[ok] &= np.abs(level[a[ok]] - level[b[ok]]) <= tol
                for x, y in zip(a[ok].tolist(), b[ok].tolist()):
                    union(x, y)
    by_label: dict[int, int] = {}
    for k, p in enumerate(pieces):
        L = int(p["coarse_label"])
        if L >= 0:
            if L in by_label:
                union(by_label[L], k)
            else:
                by_label[L] = k
    return np.array([find(x) for x in range(n)], dtype=np.int64)


def assemble_lakes(pieces: list[dict], N_fine: int, coarse_lakes: list[dict] | None, coarse_labels: np.ndarray | None, groups: np.ndarray | None = None) -> list[dict]:
    """Merge face pieces into ``graph/lakes.json`` records.  ``groups``
    (from :func:`link_pieces`; default: pieces sharing a coarse label)
    says which pieces form one lake.  ``coarse_id`` and ``outlet`` are the
    ``id`` / ``outlet`` of the ``lakes_coarse.json`` lake whose outlet
    cell carries the group's coarse label (-1 / ``null`` if none)."""
    outlet_by_label: dict[int, tuple[int, list]] = {}
    if coarse_lakes and coarse_labels is not None:
        N = coarse_labels.shape[1]
        for cl in coarse_lakes:
            o = cl.get("outlet")
            if not o:
                continue
            f, i, j = int(o[0]), int(o[1]), int(o[2])
            if not (0 <= f < 6 and 0 <= i < N and 0 <= j < N):
                continue
            L = int(coarse_labels[f, i, j])
            if L < 0:  # look around the outlet cell
                i0, i1 = max(i - 1, 0), min(i + 2, N)
                j0, j1 = max(j - 1, 0), min(j + 2, N)
                blk = coarse_labels[f, i0:i1, j0:j1]
                vals = blk[blk >= 0]
                if vals.size:
                    L = int(vals.min())
            if L >= 0 and L not in outlet_by_label:
                outlet_by_label[L] = (int(cl.get("id", -1)), [f, i, j])
    if groups is None:
        by_label: dict[int, int] = {}
        groups = np.arange(len(pieces), dtype=np.int64)
        for k, p in enumerate(pieces):
            L = int(p["coarse_label"])
            if L >= 0:
                groups[k] = by_label.setdefault(L, k)
    member_of: dict[int, list[int]] = {}
    for k in range(len(pieces)):
        member_of.setdefault(int(groups[k]), []).append(k)
    members = list(member_of.values())
    # deterministic order: by (face, bbox) of the first piece
    members.sort(key=lambda ks: (pieces[ks[0]]["face"], pieces[ks[0]]["bbox"]))
    lakes = []
    for ks in members:
        ps = [pieces[k] for k in ks]
        rings = []
        for p in ps:
            for r in p["rings"]:
                rings.append((abs(ring_area(r)), [[p["face"], float(a) / N_fine, float(b) / N_fine] for a, b in r.tolist()]))
        rings.sort(key=lambda t: -t[0])
        cells = sum(p["cells"] for p in ps)
        labels = sorted({int(p["coarse_label"]) for p in ps if p["coarse_label"] >= 0})
        hydro_id, outlet = -1, None
        for L in labels:
            if L in outlet_by_label:
                hydro_id, outlet = outlet_by_label[L]
                break
        lakes.append(
            {
                "id": len(lakes),
                "coarse_id": int(hydro_id),
                "surface_m": float(sum(p["surface_m"] * p["cells"] for p in ps) / max(cells, 1)),
                "area_m2": float(sum(p["area_m2"] for p in ps)),
                "area_cells": int(cells),
                "max_depth_m": float(max(p["max_depth_m"] for p in ps)),
                "outlet": outlet,
                "faces": sorted({p["face"] for p in ps}),
                "polygon": rings[0][1] if rings else [],
                "rings": [r[1] for r in rings],
            }
        )
    return lakes


__all__ = [
    "lake_mask", "label_lakes", "coarse_lake_labels", "trace_rings", "simplify_ring", "ring_area",
    "face_lake_pieces", "link_pieces", "assemble_lakes",
]
