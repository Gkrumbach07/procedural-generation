"""Priority-flood depression filling (Barnes, Lehman & Mulla 2014), numba.

Two front ends share one binary-heap implementation:

* :func:`priority_flood_flat` — a single 2-D window with arbitrary drain
  cells and an active mask (used by the refinement stage: drain = basin
  outlet, active = basin mask).
* :func:`priority_flood_sphere` — the six cube faces with cross-face
  8-neighbours through ``grid.owner`` (drain = ocean).

Both use the "priority-flood + plain queue" variant: a cell whose surface
is not above the current flood level is appended to a FIFO queue (it is
flooded to that level) instead of the heap, so flats and depressions cost
O(1) per cell.  Cells are popped in non-decreasing filled elevation, which
makes the pop order a topological order of the drainage: every cell's
*flood parent* (the cell that discovered it) was popped before it and has
``filled <= own filled``.  ``routing.flow_directions`` drains flat cells to
their flood parent, so ``(filled, pop order)`` strictly decreases along
every flow path and cycles are impossible.

Determinism: neighbours are visited in D8 code order and equal heap keys
are broken by insertion sequence, so the result is bit-identical run to
run (single-threaded by construction).

Across a face edge a cell is only discovered through a link that exists in
both directions (``d8.is_neighbor``), so the flood parent is always one of
the child's own D8 neighbours and a D8 code towards it exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .d8 import cid_fij, is_neighbor, neighbor_cid


@dataclass
class FloodResult:
    """Output of a priority flood.

    ``filled``  float32, same shape as the input surface: the filled DEM /
        water surface (``>= surface``; equal outside depressions; equal to
        the surface on drain and inactive cells).
    ``parent``  int64 flat index (cell id for the sphere, ``i*W + j`` for a
        window) of the cell that flooded each cell; -1 for drains and cells
        never reached.
    ``order``   int64 pop rank of every cell (-1 if never popped); a
        topological key: ``order[parent[c]] < order[c]`` and popping is in
        non-decreasing ``filled``.
    ``pop_seq`` int64 (n_popped,) flat indices in pop order (drains first).
    """

    filled: np.ndarray
    parent: np.ndarray
    order: np.ndarray
    pop_seq: np.ndarray

    @property
    def n_popped(self) -> int:
        return int(self.pop_seq.size)


# --------------------------------------------------------------------------
# binary heap on parallel arrays; key = (float, seq) lexicographic
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _less(k1, s1, k2, s2):
    return k1 < k2 or (k1 == k2 and s1 < s2)


@njit(cache=True)
def heap_push(hk, hs, hc, hn, key, seq, cell):
    """Push (key, seq, cell); ``hn`` is a 1-element int64 array holding the
    heap size."""
    n = hn[0]
    hk[n] = key
    hs[n] = seq
    hc[n] = cell
    hn[0] = n + 1
    # sift up
    while n > 0:
        p = (n - 1) >> 1
        if _less(hk[n], hs[n], hk[p], hs[p]):
            hk[n], hk[p] = hk[p], hk[n]
            hs[n], hs[p] = hs[p], hs[n]
            hc[n], hc[p] = hc[p], hc[n]
            n = p
        else:
            break


@njit(cache=True)
def heap_pop(hk, hs, hc, hn):
    """Pop the smallest (key, seq); returns (key, cell)."""
    key = hk[0]
    cell = hc[0]
    n = hn[0] - 1
    hn[0] = n
    if n > 0:
        hk[0] = hk[n]
        hs[0] = hs[n]
        hc[0] = hc[n]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= n:
                break
            r = l + 1
            m = l
            if r < n and _less(hk[r], hs[r], hk[l], hs[l]):
                m = r
            if _less(hk[m], hs[m], hk[i], hs[i]):
                hk[i], hk[m] = hk[m], hk[i]
                hs[i], hs[m] = hs[m], hs[i]
                hc[i], hc[m] = hc[m], hc[i]
                i = m
            else:
                break
    return key, cell


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@njit(cache=True)
def _flood_sphere_kernel(surface, drain, active, owner, N, H):
    """surface/drain/active: flat (6*N*N,) arrays indexed by cell id."""
    M = surface.size
    filled = surface.copy()
    visited = np.zeros(M, dtype=np.bool_)
    parent = np.full(M, -1, dtype=np.int64)
    order = np.full(M, -1, dtype=np.int64)
    pop_seq = np.empty(M, dtype=np.int64)
    hk = np.empty(M, dtype=np.float64)
    hs = np.empty(M, dtype=np.int64)
    hc = np.empty(M, dtype=np.int64)
    hn = np.zeros(1, dtype=np.int64)
    queue = np.empty(M, dtype=np.int64)
    qh = 0
    qt = 0
    seq = 0
    npop = 0
    # seeds: drain cells with at least one active non-drain neighbour
    for c in range(M):
        if not active[c]:
            visited[c] = True
        elif drain[c]:
            visited[c] = True
            f, i, j = cid_fij(c, N)
            push = False
            for k in range(8):
                n = neighbor_cid(owner, N, H, f, i, j, k)
                if active[n] and not drain[n]:
                    push = True
                    break
            if push:
                heap_push(hk, hs, hc, hn, surface[c], seq, c)
                seq += 1
    while qh < qt or hn[0] > 0:
        if qh < qt:
            c = queue[qh]
            qh += 1
        else:
            _, c = heap_pop(hk, hs, hc, hn)
        lvl = filled[c]
        order[c] = npop
        pop_seq[npop] = c
        npop += 1
        f, i, j = cid_fij(c, N)
        for k in range(8):
            n = neighbor_cid(owner, N, H, f, i, j, k)
            if visited[n]:
                continue
            if n // (N * N) != f:
                fn, inn, jn = cid_fij(n, N)
                if not is_neighbor(owner, N, H, fn, inn, jn, c):
                    continue
            visited[n] = True
            parent[n] = c
            if surface[n] <= lvl:
                filled[n] = lvl
                queue[qt] = n
                qt += 1
            else:
                filled[n] = surface[n]
                heap_push(hk, hs, hc, hn, surface[n], seq, n)
                seq += 1
    return filled, parent, order, pop_seq[:npop]


@njit(cache=True)
def _flood_flat_kernel(surface, drain, active):
    """2-D window (Hh, W); flat index i*W + j; 8-neighbours inside bounds."""
    Hh, W = surface.shape
    M = Hh * W
    surf = surface.reshape(M)
    dr = drain.reshape(M)
    act = active.reshape(M)
    filled = surf.copy()
    visited = np.zeros(M, dtype=np.bool_)
    parent = np.full(M, -1, dtype=np.int64)
    order = np.full(M, -1, dtype=np.int64)
    pop_seq = np.empty(M, dtype=np.int64)
    hk = np.empty(M, dtype=np.float64)
    hs = np.empty(M, dtype=np.int64)
    hc = np.empty(M, dtype=np.int64)
    hn = np.zeros(1, dtype=np.int64)
    queue = np.empty(M, dtype=np.int64)
    qh = 0
    qt = 0
    seq = 0
    npop = 0
    di8 = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    dj8 = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    for c in range(M):
        if not act[c]:
            visited[c] = True
        elif dr[c]:
            visited[c] = True
            i = c // W
            j = c - i * W
            push = False
            for k in range(8):
                i2 = i + di8[k]
                j2 = j + dj8[k]
                if i2 < 0 or i2 >= Hh or j2 < 0 or j2 >= W:
                    continue
                n = i2 * W + j2
                if act[n] and not dr[n]:
                    push = True
                    break
            if push:
                heap_push(hk, hs, hc, hn, surf[c], seq, c)
                seq += 1
    while qh < qt or hn[0] > 0:
        if qh < qt:
            c = queue[qh]
            qh += 1
        else:
            _, c = heap_pop(hk, hs, hc, hn)
        lvl = filled[c]
        order[c] = npop
        pop_seq[npop] = c
        npop += 1
        i = c // W
        j = c - i * W
        for k in range(8):
            i2 = i + di8[k]
            j2 = j + dj8[k]
            if i2 < 0 or i2 >= Hh or j2 < 0 or j2 >= W:
                continue
            n = i2 * W + j2
            if visited[n]:
                continue
            visited[n] = True
            parent[n] = c
            if surf[n] <= lvl:
                filled[n] = lvl
                queue[qt] = n
                qt += 1
            else:
                filled[n] = surf[n]
                heap_push(hk, hs, hc, hn, surf[n], seq, n)
                seq += 1
    return filled.reshape(Hh, W), parent, order, pop_seq[:npop]


# --------------------------------------------------------------------------
# front ends
# --------------------------------------------------------------------------
def priority_flood_flat(surface2d: np.ndarray, drain_mask: np.ndarray, active_mask: np.ndarray | None = None) -> FloodResult:
    """Fill depressions of one 2-D window.

    ``surface2d`` float32 (Hh, W) indexed [i, j]; ``drain_mask`` bool: cells
    that water can leave through (seeded at their own elevation; e.g. the
    basin outlet, or a border ring); ``active_mask`` bool (default all):
    cells outside it are ignored (never flooded, ``filled == surface``,
    ``order == -1``).  Active cells not connected to a drain are never
    reached either (``order == -1``).

    Returns :class:`FloodResult` with flat indices ``i*W + j``.
    """
    s = np.ascontiguousarray(surface2d, dtype=np.float32)
    d = np.ascontiguousarray(drain_mask, dtype=np.bool_)
    a = np.ones_like(d) if active_mask is None else np.ascontiguousarray(active_mask, dtype=np.bool_)
    if s.shape != d.shape or s.shape != a.shape:
        raise ValueError("surface, drain and active masks must have the same 2-D shape")
    filled, parent, order, pop_seq = _flood_flat_kernel(s, d, a)
    return FloodResult(filled, parent, order, pop_seq)


def priority_flood_sphere(surface, ocean, grid=None, active=None) -> FloodResult:
    """Fill depressions on the whole cube-sphere.

    ``surface`` a FaceField or interior (6, N, N) float32 array; ``ocean``
    bool (6, N, N) drain cells (seeded at their own elevation, so pass the
    real ocean surface: all land is above it).  ``grid`` is required when
    ``surface`` is a plain array.  If there is no ocean at all the lowest
    cell is used as the single drain (so the output is still cycle-free).

    Returns :class:`FloodResult` with (6, N, N) ``filled``, flat cell-id
    arrays for ``parent``/``order``/``pop_seq``.
    """
    if grid is None:
        grid = surface.grid
    N, H = grid.N, grid.H
    s = np.ascontiguousarray(getattr(surface, "interior", surface), dtype=np.float32).reshape(-1)
    oc = np.ascontiguousarray(ocean, dtype=np.bool_).reshape(-1)
    act = np.ones_like(oc) if active is None else np.ascontiguousarray(active, dtype=np.bool_).reshape(-1)
    if s.size != 6 * N * N or oc.size != s.size:
        raise ValueError("surface/ocean must be (6, N, N) interior arrays")
    if not (oc & act).any():
        oc = oc.copy()
        oc[int(np.argmin(np.where(act, s, np.inf)))] = True
    filled, parent, order, pop_seq = _flood_sphere_kernel(s, oc, act, grid.owner, N, H)
    return FloodResult(filled.reshape(6, N, N), parent, order, pop_seq)


__all__ = ["FloodResult", "priority_flood_flat", "priority_flood_sphere", "heap_push", "heap_pop"]
