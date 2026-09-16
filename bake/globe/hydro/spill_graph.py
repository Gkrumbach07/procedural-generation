"""Depression levels that agree across tiles (Barnes, Lehman & Mulla 2016,
"Parallel priority-flood depression filling for trillion cell digital
elevation models on desktops or clusters", Computers & Geosciences 96).

A tile flooded on its own drains every depression that reaches its edge
there, at whatever height the edge happens to have: two tiles sharing a
basin disagree about where it spills, and water handed between them can
circle for ever.  The fix floods each tile once with every edge cell a seed
of its own label (:func:`label_flood`): each cell gets the label of the
seed whose flood reached it and the tile's filled level, and every place two
labels touch records the level at which water passes between them.  Joined
with the links across tile edges, the labels form a small graph that a
priority flood from the sea label resolves (:func:`spill_levels`): a
cell's level on the whole domain is ``max(filled, level[label])``.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .priority_flood import heap_pop, heap_push


@njit(cache=True)
def _label_flood_kernel(surface, seed_label, cap):
    Hh, W = surface.shape
    M = Hh * W
    surf = surface.reshape(M)
    filled = surf.copy()
    label = seed_label.copy()
    visited = seed_label >= 0
    hk = np.empty(M, dtype=np.float64)
    hs = np.empty(M, dtype=np.int64)
    hc = np.empty(M, dtype=np.int64)
    hn = np.zeros(1, dtype=np.int64)
    queue = np.empty(M, dtype=np.int32)
    qh = 0
    qt = 0
    seq = 0
    ea = np.empty(cap, np.int32)
    eb = np.empty(cap, np.int32)
    ew = np.empty(cap, np.float32)
    ne = 0
    di8 = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    dj8 = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    for c in range(M):
        if not visited[c]:
            continue
        i = c // W
        j = c - i * W
        for k in range(8):
            i2 = i + di8[k]
            j2 = j + dj8[k]
            if i2 < 0 or i2 >= Hh or j2 < 0 or j2 >= W:
                continue
            if not visited[i2 * W + j2] or label[i2 * W + j2] != label[c]:
                heap_push(hk, hs, hc, hn, surf[c], seq, c)
                seq += 1
                break
    while qh < qt or hn[0] > 0:
        if qh < qt:
            c = np.int64(queue[qh])
            qh += 1
        else:
            _, c = heap_pop(hk, hs, hc, hn)
        lvl = filled[c]
        a = label[c]
        i = c // W
        j = c - i * W
        for k in range(8):
            i2 = i + di8[k]
            j2 = j + dj8[k]
            if i2 < 0 or i2 >= Hh or j2 < 0 or j2 >= W:
                continue
            n = i2 * W + j2
            if visited[n]:
                b = label[n]
                if b != a:
                    if ne < cap:
                        ea[ne] = min(a, b)
                        eb[ne] = max(a, b)
                        ew[ne] = max(lvl, filled[n])
                    ne += 1
                continue
            visited[n] = True
            label[n] = a
            if surf[n] <= lvl:
                filled[n] = lvl
                queue[qt] = n
                qt += 1
            else:
                heap_push(hk, hs, hc, hn, surf[n], seq, n)
                seq += 1
    return filled.reshape(Hh, W), label.reshape(Hh, W), ea, eb, ew, ne


def reduce_edges(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected label pairs ``a < b`` with the lowest level of each pair."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    w = np.asarray(w, np.float32)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    lo, hi, w = lo[keep], hi[keep], w[keep]
    if lo.size == 0:
        return lo, hi, w
    key = lo * (int(hi.max()) + 1) + hi
    o = np.lexsort((w, key))
    key, lo, hi, w = key[o], lo[o], hi[o], w[o]
    first = np.ones(key.size, bool)
    first[1:] = key[1:] != key[:-1]
    return lo[first], hi[first], w[first]


def label_flood(surface2d: np.ndarray, seed_label: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Priority flood of one tile from the seeds ``seed_label >= 0`` (int32,
    same shape; each seed floods with its label, a seed's cell keeps its
    height).  Returns ``filled`` (float32), ``label`` (int32; -1 where no
    seed reaches) and the touching label pairs ``(a, b, level)`` reduced to
    the lowest level of each pair (:func:`reduce_edges`)."""
    s = np.ascontiguousarray(surface2d, dtype=np.float32)
    lab = np.ascontiguousarray(seed_label, dtype=np.int32)
    if s.shape != lab.shape or s.ndim != 2:
        raise ValueError("surface and seed labels must have the same 2-D shape")
    cap = max(1 << 16, s.size // 32)
    while True:
        filled, label, ea, eb, ew, ne = _label_flood_kernel(s, lab.reshape(-1), cap)
        if ne <= cap:
            return filled, label, reduce_edges(ea[:ne], eb[:ne], ew[:ne])
        cap = int(ne)


@njit(cache=True)
def _spill_kernel(n_labels, indptr, nbr, wt, sources):
    level = np.full(n_labels, np.inf)
    done = np.zeros(n_labels, np.bool_)
    m = nbr.size + sources.size + 1
    hk = np.empty(m, dtype=np.float64)
    hs = np.empty(m, dtype=np.int64)
    hc = np.empty(m, dtype=np.int64)
    hn = np.zeros(1, dtype=np.int64)
    seq = 0
    for s in sources:
        level[s] = -np.inf
        heap_push(hk, hs, hc, hn, -np.inf, seq, s)
        seq += 1
    while hn[0] > 0:
        key, a = heap_pop(hk, hs, hc, hn)
        if done[a]:
            continue
        done[a] = True
        for e in range(indptr[a], indptr[a + 1]):
            b = nbr[e]
            lv = max(key, np.float64(wt[e]))
            if lv < level[b]:
                level[b] = lv
                heap_push(hk, hs, hc, hn, lv, seq, b)
                seq += 1
    return level


def spill_levels(n_labels: int, a: np.ndarray, b: np.ndarray, w: np.ndarray, sources) -> np.ndarray:
    """The level at which each label's water reaches a source label (the
    sea): the lowest, over paths of touching labels, of the highest link on
    the path (float64, ``-inf`` at the sources, ``inf`` where unreachable)."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    w = np.asarray(w, np.float32)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    ww = np.concatenate([w, w])
    o = np.argsort(src, kind="stable")
    src, dst, ww = src[o], dst[o], ww[o]
    indptr = np.zeros(int(n_labels) + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    return _spill_kernel(int(n_labels), indptr, dst, ww, np.asarray(sources, np.int64).reshape(-1))


__all__ = ["label_flood", "reduce_edges", "spill_levels"]
