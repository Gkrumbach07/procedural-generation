"""Coarse lakes (PLAN.md section 9 step 6): connected components of
``water_surface - surface > lake_min_depth`` across faces, one record per
lake with surface elevation, area, outlet and a coarse polygon ring.

Polygons are traced per face on the cell-corner lattice (``u = i/N``,
``v = j/N``), oriented counter-clockwise in (u, v).  A lake that crosses a
face edge has one ring per face piece; ``polygon`` is the ring of its
largest piece and ``rings`` lists all of them.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .d8 import cid_fij, neighbor_cid


@njit(cache=True)
def label_components(mask, level, owner, N, H):
    """Connected components (8-connected, cross-face) of ``mask`` (M,) bool,
    joining only cells with identical ``level`` (M,) values.  Returns
    (labels int32 (M,) with -1 outside the mask, n_labels).  Labels are
    numbered in order of their lowest cell id (deterministic)."""
    M = mask.size
    labels = np.full(M, -1, dtype=np.int32)
    stack = np.empty(M, dtype=np.int64)
    n = 0
    for s in range(M):
        if not mask[s] or labels[s] >= 0:
            continue
        labels[s] = n
        sp = 0
        stack[sp] = s
        sp += 1
        while sp > 0:
            sp -= 1
            c = stack[sp]
            f, i, j = cid_fij(c, N)
            for k in range(8):
                q = neighbor_cid(owner, N, H, f, i, j, k)
                if mask[q] and labels[q] < 0 and level[q] == level[c]:
                    labels[q] = n
                    stack[sp] = q
                    sp += 1
        n += 1
    return labels, n


def trace_rings(mask2d: np.ndarray) -> list[np.ndarray]:
    """Boundary rings of a 2-D boolean mask indexed [i, j] on the corner
    lattice.  Each ring is an (K, 2) int array of (i, j) corner coordinates,
    closed (first == last), counter-clockwise in (i, j) (interior on the
    left).  Rings are ordered by length, longest first; holes come out as
    clockwise rings."""
    m = np.asarray(mask2d, dtype=bool)
    if not m.any():
        return []
    Ni, Nj = m.shape
    pad = np.zeros((Ni + 2, Nj + 2), dtype=bool)
    pad[1:-1, 1:-1] = m
    segs = []
    # bottom (j side): neighbour (i, j-1) absent -> (i, j) -> (i+1, j)
    ii, jj = np.nonzero(m & ~pad[1:-1, :-2])
    segs.append(np.stack([ii, jj, ii + 1, jj], axis=1))
    # right: neighbour (i+1, j) absent -> (i+1, j) -> (i+1, j+1)
    ii, jj = np.nonzero(m & ~pad[2:, 1:-1])
    segs.append(np.stack([ii + 1, jj, ii + 1, jj + 1], axis=1))
    # top: neighbour (i, j+1) absent -> (i+1, j+1) -> (i, j+1)
    ii, jj = np.nonzero(m & ~pad[1:-1, 2:])
    segs.append(np.stack([ii + 1, jj + 1, ii, jj + 1], axis=1))
    # left: neighbour (i-1, j) absent -> (i, j+1) -> (i, j)
    ii, jj = np.nonzero(m & ~pad[:-2, 1:-1])
    segs.append(np.stack([ii, jj + 1, ii, jj], axis=1))
    S = np.concatenate(segs, axis=0)
    # sort for determinism, then chain start -> end
    order = np.lexsort((S[:, 3], S[:, 2], S[:, 1], S[:, 0]))
    S = S[order]
    W = Nj + 2
    start_key = S[:, 0] * W + S[:, 1]
    end_key = S[:, 2] * W + S[:, 3]
    nxt: dict[int, list[int]] = {}
    for idx in range(S.shape[0]):
        nxt.setdefault(int(start_key[idx]), []).append(idx)
    used = np.zeros(S.shape[0], dtype=bool)
    rings = []
    for s0 in range(S.shape[0]):
        if used[s0]:
            continue
        ring = [(int(S[s0, 0]), int(S[s0, 1]))]
        cur = s0
        while True:
            used[cur] = True
            ring.append((int(S[cur, 2]), int(S[cur, 3])))
            cands = nxt.get(int(end_key[cur]), [])
            nxt_idx = -1
            for c in cands:
                if not used[c]:
                    nxt_idx = c
                    break
            if nxt_idx < 0:
                break
            cur = nxt_idx
        rings.append(np.array(ring, dtype=np.int64))
    rings.sort(key=lambda r: -r.shape[0])
    return rings


def ring_area(ring) -> float:
    """Signed shoelace area of a closed (i, j) ring in cell units: positive
    for counter-clockwise rings (outer boundaries from :func:`trace_rings`),
    negative for holes."""
    r = np.asarray(ring, dtype=np.float64)
    if r.shape[0] < 3:
        return 0.0
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def extract_lakes(lake_mask, water_surface, order, down, grid, cell_area=None) -> tuple[list[dict], np.ndarray]:
    """Lake records for ``graph/lakes_coarse.json``.

    ``lake_mask`` (6, N, N) bool, ``water_surface`` (6, N, N) float32
    (constant over each depression), ``order`` (M,) flood pop rank,
    ``down`` (M,) downstream cell ids.  Returns ``(lakes, labels)`` with
    ``labels`` (6, N, N) int32 (-1 = not a lake).

    Per lake: ``surface_m``, ``area_m2``, ``area_cells``, ``outlet``
    ``[f, i, j]`` — the lake cell at the spill point (first cell of the lake
    popped by the flood; its downstream cell is outside the lake) —
    ``outlet_downstream`` (the cell it drains to), ``faces``, ``polygon``
    (outer ring enclosing the largest area, counter-clockwise in (u, v))
    and ``rings`` (every outer ring of every face piece, largest first) in
    ``[f, u, v]``.  Holes (islands in the lake) are not emitted.
    """
    N, H = grid.N, grid.H
    M = 6 * N * N
    lk = np.ascontiguousarray(lake_mask, dtype=np.bool_).reshape(-1)
    ws = np.ascontiguousarray(water_surface, dtype=np.float32).reshape(-1)
    labels, n = label_components(lk, ws, grid.owner, N, H)
    area = grid.interior_cell_area.reshape(-1) if cell_area is None else np.asarray(cell_area, dtype=np.float32).reshape(-1)
    lakes = []
    if n == 0:
        return lakes, labels.reshape(6, N, N)
    cells_of = np.nonzero(labels >= 0)[0]
    lab_sorted = np.argsort(labels[cells_of], kind="stable")
    cells_of = cells_of[lab_sorted]
    bounds = np.searchsorted(labels[cells_of], np.arange(n + 1))
    lab3 = labels.reshape(6, N, N)
    for L in range(n):
        cc = cells_of[bounds[L] : bounds[L + 1]]
        first = cc[np.argmin(order[cc])]
        f0, i0, j0 = cid_fij(int(first), N)
        d = int(down[first])
        if d >= 0:
            fd_, id_, jd_ = cid_fij(d, N)
            outlet_down = [int(fd_), int(id_), int(jd_)]
        else:
            outlet_down = None
        faces = np.unique(cc // (N * N))
        rings = []
        for f in faces:
            # trace inside the lake's bbox on this face only (O(lake) not O(N²))
            fc = cc[cc // (N * N) == f]
            li = (fc % (N * N)) // N
            lj = fc % N
            bi0, bi1 = int(li.min()), int(li.max()) + 1
            bj0, bj1 = int(lj.min()), int(lj.max()) + 1
            m2 = lab3[f, bi0:bi1, bj0:bj1] == L
            for r in trace_rings(m2):
                a_r = ring_area(r)
                if a_r <= 0:  # hole (clockwise): an island inside the lake, not the shore
                    continue
                rings.append((a_r, [[int(f), float(a + bi0) / N, float(b + bj0) / N] for a, b in r]))
        # largest enclosed area first (a face piece may consist of several
        # 4-disconnected parts, each with its own outer ring)
        rings.sort(key=lambda t: (-t[0], t[1][0]))
        lakes.append(
            {
                "id": L,
                "surface_m": float(ws[first]),
                "area_m2": float(area[cc].sum()),
                "area_cells": int(cc.size),
                "outlet": [int(f0), int(i0), int(j0)],
                "outlet_downstream": outlet_down,
                "faces": [int(f) for f in faces],
                "polygon": rings[0][1] if rings else [],
                "rings": [r[1] for r in rings],
            }
        )
    return lakes, lab3


__all__ = ["label_components", "trace_rings", "ring_area", "extract_lakes"]
