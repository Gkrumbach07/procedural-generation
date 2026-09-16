"""A priority flood whose drainage tree runs along the drowned bed.

:func:`priority_flood.priority_flood_flat` floods a depression through a
plain FIFO queue: every cell below the spill level is discovered in
breadth-first order from the spill cell and drains to its discoverer, so
the tree -- and a river accumulated down it -- crosses a filled
depression or an exact flat in straight 45-degree and axis rays that
ignore the bed underneath.

:func:`priority_flood_bed` fills the same surface (``filled`` is identical)
but keeps every cell on the heap, keyed on ``(filled level, bed elevation,
insertion sequence)``, where the bed is the unfilled surface:

* at one fill level the heap pops the drowned cells deepest first, so the
  flood runs from the spill point down into the bed and then fills it from
  the bottom up -- the path it advanced along is the bed's thalweg, and it
  is the path a pit's water takes back to the outlet;
* a cell's receiver is chosen when it pops, among its neighbours already
  popped (so the tree is acyclic and ``pop_seq`` is a topological order):
  the lowest filled level, then the steepest descent of the bed per unit
  distance (the least climb when there is no descent), then the earliest
  popped.  Outside depressions the bed is the filled level, so the lowest
  neighbour wins as before; inside one, water runs down the bed to the
  deepest cells and along them to the spill;
* equal keys pop in insertion order, so an exact flat is crossed by a
  breadth-first front from all of its lower edge at once (each cell drains
  toward its nearest lower edge) rather than a ray from one cell of it.

Cost: every cell passes through the heap (the queue variant skips it for
drowned cells); keys pack into two ``uint64`` words (filled and bed as
order-preserving ``uint32`` codes; sequence and cell), 33 bytes a cell at
most (int32 cell ids), ~2.5 GB at 8192^2.  Deterministic (single-threaded,
total order on keys).

On dry ground a receiver picked from the eight neighbours runs in 45-degree
and axis rays down any smooth slope (the lowest neighbour is nearly always
a diagonal).  :func:`accumulate_ltd` re-picks the receivers of dry cells
with D8-LTD (Orlandini et al. 2003, "Path-based methods for the
determination of nondispersive drainage directions in grid-based digital
elevation models", WRR 39(6)): the steepest triangular facet (Tarboton's
D-infinity) gives the true direction, the cell steps to the facet's
cardinal or diagonal neighbour, whichever keeps the lateral deviation
accumulated down the path (from the donor carrying the most water) the
smaller, so a path follows the aspect on average instead of the nearest
of eight directions.  Cells are visited upstream-first (reverse pop order)
and a receiver is always a strictly lower cell popped earlier, so the tree
stays acyclic and every cell drains; the rain is accumulated in the same
pass.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .priority_flood import FloodResult

_SQRT2 = np.float32(np.sqrt(2.0))


@njit(cache=True, inline="always")
def _code(bits):
    """An order-preserving uint32 code of a float32's bits (-0 == +0)."""
    mag = bits & np.uint32(0x7FFFFFFF)
    if bits >> np.uint32(31):
        return np.uint32(0x80000000) - mag
    return np.uint32(0x80000000) + mag


@njit(cache=True, inline="always")
def _push(hk, hv, n, key, val):
    hk[n] = key
    hv[n] = val
    while n > 0:
        p = (n - 1) >> 1
        if hk[n] < hk[p] or (hk[n] == hk[p] and hv[n] < hv[p]):
            hk[n], hk[p] = hk[p], hk[n]
            hv[n], hv[p] = hv[p], hv[n]
            n = p
        else:
            break
    return


@njit(cache=True, inline="always")
def _pop(hk, hv, n):
    """Remove the root of a heap of size ``n`` (the caller read it)."""
    n -= 1
    if n > 0:
        hk[0] = hk[n]
        hv[0] = hv[n]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= n:
                break
            r = l + 1
            m = l
            if r < n and (hk[r] < hk[l] or (hk[r] == hk[l] and hv[r] < hv[l])):
                m = r
            if hk[m] < hk[i] or (hk[m] == hk[i] and hv[m] < hv[i]):
                hk[i], hk[m] = hk[m], hk[i]
                hv[i], hv[m] = hv[m], hv[i]
                i = m
            else:
                break
    return


@njit(cache=True)
def _bed_flood_kernel(surface, drain, late_border):
    Hh, W = surface.shape
    M = Hh * W
    surf = surface.reshape(M)
    bits = surf.view(np.uint32)
    dr = drain.reshape(M)
    filled = surf.copy()
    visited = dr.copy()
    order = np.full(M, -1, dtype=np.int32)
    parent = np.full(M, -1, dtype=np.int32)
    pop_seq = np.empty(M, dtype=np.int32)
    hk = np.empty(M, dtype=np.uint64)
    hv = np.empty(M, dtype=np.uint64)
    hn = 0
    seq = 0
    npop = 0
    di8 = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    dj8 = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dist8 = np.array([1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2], dtype=np.float32)
    s32 = np.uint64(32)
    for c in range(M):
        if not dr[c]:
            continue
        i = c // W
        j = c - i * W
        for k in range(8):
            i2 = i + di8[k]
            j2 = j + dj8[k]
            if i2 < 0 or i2 >= Hh or j2 < 0 or j2 >= W:
                continue
            if not dr[i2 * W + j2]:
                sc = np.uint64(_code(bits[c]))
                if late_border and (i == 0 or i == Hh - 1 or j == 0 or j == W - 1):
                    key = (sc << s32) | np.uint64(0xFFFFFFFF)       # after every real cell of its level
                else:
                    key = (sc << s32) | sc
                _push(hk, hv, hn, key, (np.uint64(seq) << s32) | np.uint64(c))
                hn += 1
                seq += 1
                break
    while hn > 0:
        fcode = hk[0] >> s32
        c = np.int64(hv[0] & np.uint64(0xFFFFFFFF))
        _pop(hk, hv, hn)
        hn -= 1
        order[c] = npop
        pop_seq[npop] = c
        npop += 1
        i = c // W
        j = c - i * W
        if not dr[c]:
            # receiver: a popped neighbour -- lowest filled, steepest bed
            # descent per unit distance (least climb), earliest popped
            zc = surf[c]
            best = -1
            bf = np.float32(0.0)
            bs = np.float32(0.0)
            bo = 0
            for k in range(8):
                i2 = i + di8[k]
                j2 = j + dj8[k]
                if i2 < 0 or i2 >= Hh or j2 < 0 or j2 >= W:
                    continue
                n = i2 * W + j2
                o = order[n]
                if o < 0:
                    continue
                f = filled[n]
                dz = zc - surf[n]
                s = dz / dist8[k] if dz > 0 else dz
                if best < 0 or f < bf or (f == bf and (s > bs or (s == bs and o < bo))):
                    best = n
                    bf = f
                    bs = s
                    bo = o
            parent[c] = best
        for k in range(8):
            i2 = i + di8[k]
            j2 = j + dj8[k]
            if i2 < 0 or i2 >= Hh or j2 < 0 or j2 >= W:
                continue
            n = i2 * W + j2
            if visited[n]:
                continue
            visited[n] = True
            sc = np.uint64(_code(bits[n]))
            if sc <= fcode:
                filled[n] = filled[c]
                key = (fcode << s32) | sc
            else:
                key = (sc << s32) | sc
            _push(hk, hv, hn, key, (np.uint64(seq) << s32) | np.uint64(n))
            hn += 1
            seq += 1
    return filled.reshape(Hh, W), parent, order, pop_seq[:npop]


def priority_flood_bed(surface2d: np.ndarray, drain_mask: np.ndarray, late_border: bool = False) -> FloodResult:
    """Fill the depressions of a 2-D window draining to ``drain_mask`` (as
    :func:`priority_flood.priority_flood_flat` with every cell active), with
    a drainage tree that crosses filled depressions along their bed and exact
    flats toward their nearest lower edge (module docstring).

    ``late_border``: drains on the window's border ring are stand-ins for
    ground beyond it (seeded at that ground's level): at their level they
    pop after every real cell, so water at that level leaves by a real
    outlet in the window where there is one.

    Returns :class:`FloodResult`: ``filled`` float32 (identical to the plain
    flood's), ``parent`` int32 flat receiver (-1 on drains and cells never
    reached), ``order`` int32 pop rank, ``pop_seq`` int32 flat indices in
    pop order (every receiver before its donors)."""
    s = np.ascontiguousarray(surface2d, dtype=np.float32)
    d = np.ascontiguousarray(drain_mask, dtype=np.bool_)
    if s.ndim != 2 or s.shape != d.shape:
        raise ValueError("surface and drain mask must have the same 2-D shape")
    if s.size >= 2 ** 31:
        raise ValueError(f"priority_flood_bed packs cell ids in 32 bits: {s.size} cells is too many")
    filled, parent, order, pop_seq = _bed_flood_kernel(s, d, bool(late_border))
    return FloodResult(filled, parent, order, pop_seq)


# D8 codes k -> (di, dj), as the flood kernels; the eight triangular facets
# of a cell as (cardinal code, diagonal code, sign of cross(cardinal, diagonal))
_DI8 = np.array([1, 1, 0, -1, -1, -1, 0, 1])
_DJ8 = np.array([0, 1, 1, 1, 0, -1, -1, -1])
_FACET_CARD = np.array([0, 0, 2, 2, 4, 4, 6, 6])
_FACET_DIAG = np.array([7, 1, 1, 3, 3, 5, 5, 7])
_FACET_SIGN = np.array([np.sign(_DI8[a] * _DJ8[b] - _DJ8[a] * _DI8[b]) for a, b in zip(_FACET_CARD, _FACET_DIAG)], dtype=np.float32)


@njit(cache=True)
def _ltd_kernel(surface, filled, order, pop_seq, parent, w, lam):
    Hh, W = surface.shape
    M = Hh * W
    z = filled.reshape(M)
    bed = surface.reshape(M)
    acc = w.copy()
    dev = np.zeros(M, np.float32)        # a cell's incoming deviation until it is visited
    best = np.zeros(M, np.float32)       # the water of the donor that set it
    off = np.empty(8, np.int64)
    for k in range(8):
        off[k] = _DI8[k] * W + _DJ8[k]
    half = np.float32(np.sqrt(0.5))
    for q in range(pop_seq.size - 1, -1, -1):
        c = np.int64(pop_seq[q])
        p = np.int64(parent[c])
        if p < 0:
            continue
        dout = np.float32(0.0)
        i = c // W
        j = c - i * W
        zc = z[c]
        if zc == bed[c] and i > 0 and i < Hh - 1 and j > 0 and j < W - 1:
            # steepest facet: s1 along the cardinal, s2 from it toward the
            # diagonal; the direction is clamped to the facet's two edges
            bf = -1
            bkey = np.float32(0.0)
            bs1 = np.float32(0.0)
            bs2 = np.float32(0.0)
            btype = 0
            for f in range(8):
                z1 = z[c + off[_FACET_CARD[f]]]
                s1 = zc - z1
                s2 = z1 - z[c + off[_FACET_DIAG[f]]]
                if s2 <= 0.0:
                    if s1 <= 0.0:
                        continue
                    key = s1 * s1
                    t = 0
                elif s2 >= s1:
                    e = s1 + s2
                    if e <= 0.0:
                        continue
                    key = np.float32(0.5) * e * e
                    t = 2
                else:
                    key = s1 * s1 + s2 * s2
                    t = 1
                if key > bkey:
                    bkey = key
                    bf = f
                    bs1 = s1
                    bs2 = s2
                    btype = t
            if bf >= 0:
                if btype == 0:
                    sn = np.float32(0.0)
                    cs = np.float32(1.0)
                elif btype == 2:
                    sn = half
                    cs = half
                else:
                    h = np.sqrt(bkey)
                    sn = bs2 / h
                    cs = bs1 / h
                sg = _FACET_SIGN[bf]
                din = lam * dev[c]
                d1 = din - sg * sn                 # after a cardinal step
                d2 = din + sg * (cs - sn)          # after a diagonal step
                n1 = c + off[_FACET_CARD[bf]]
                n2 = c + off[_FACET_DIAG[bf]]
                oc = order[c]
                ok1 = z[n1] < zc and order[n1] >= 0 and order[n1] < oc
                ok2 = z[n2] < zc and order[n2] >= 0 and order[n2] < oc
                if ok1 and (not ok2 or abs(d1) <= abs(d2)):
                    p = n1
                    dout = d1
                elif ok2:
                    p = n2
                    dout = d2
                parent[c] = p
        acc[p] += acc[c]
        if acc[c] > best[p]:
            best[p] = acc[c]
            dev[p] = dout
    return acc


def accumulate_ltd(surface2d: np.ndarray, fr: FloodResult, weight: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """``weight`` (flat, float64) accumulated down ``fr``'s tree (from
    :func:`priority_flood_bed` of ``surface2d``) with the receivers of dry
    interior cells re-picked by D8-LTD (module docstring); ``fr.parent`` is
    updated in place.  Drowned cells, exact flats and the window's border
    ring keep their flood receivers (a deviation stops there).  ``lam`` is
    the share of the incoming deviation a cell carries on (1: the full path,
    0: plain D-infinity rounding).  Returns the accumulation, flat float64:
    at the drains it sums to ``weight.sum()``."""
    s = np.ascontiguousarray(surface2d, dtype=np.float32)
    w = np.ascontiguousarray(weight, dtype=np.float64).reshape(-1)
    if w.size != s.size or fr.parent.size != s.size:
        raise ValueError("weight and flood tree must cover the surface")
    return _ltd_kernel(s, fr.filled, fr.order, fr.pop_seq, fr.parent, w, np.float32(lam))


__all__ = ["priority_flood_bed", "accumulate_ltd"]
