"""Rivers (PLAN.md section 11) from the *fine* discharge.

Per face:

1. threshold the (lightly smoothed) fine discharge — the threshold is
   global, chosen so that a given fraction of the land is river
   (:func:`discharge_threshold`; by default density-matched to the coarse
   D8 channel network, which makes it independent of the discharge units);
2. clean the mask (drop small blobs, fill small holes);
3. thin it to a 1-pixel skeleton (Zhang-Suen, numba, on the active pixel
   list so every pass costs O(mask)), remove redundant corner pixels and
   prune short spurs;
4. trace the skeleton into segments between nodes (endpoints /
   junctions), orient each downstream (higher surface -> lower), assign a
   Strahler order (from the coarse drainage graph when a coarse channel
   lies within ``graph_match_cells``, else from the fine segment graph),
   width ``w = a (Q / Q_thr)^b`` fine cells, and Catmull-Rom smooth the
   polyline;
5. burn the river mask: 255 within ``w / 2`` of every centreline pixel.

Everything is deterministic (fixed raster order; the only parallel loops
compute per-pixel flags from a frozen state or paint a constant).  The
hysteresis keep decision and the minimum blob size are *global*
(:class:`RiverConnectivity` links the low-threshold blobs across cube
edges), so a river whose discharge crosses the threshold only after a
cube edge is kept on the upstream face too; thinning, tracing and the
mask discs are then per face, so a river crossing an edge is two
polylines meeting at the edge (each ends within one cell of it).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy import ndimage

#: 8-ring around a pixel, clockwise from "north" (-i): index k -> (di, dj).
#: Zhang-Suen's P2..P9 are k = 0..7; P2 = k0, P4 = k2, P6 = k4, P8 = k6.
RING_DI = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
RING_DJ = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_STRUCT8 = np.ones((3, 3), dtype=bool)
_STRUCT4 = ndimage.generate_binary_structure(2, 1)
LOG_HIST_MAX = 30.0
LOG_HIST_BINS = 6000


# --------------------------------------------------------------------------
# pixel helpers (njit)
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _px(img, i, j):
    if i < 0 or j < 0 or i >= img.shape[0] or j >= img.shape[1]:
        return 0
    return 1 if img[i, j] != 0 else 0


@njit(cache=True, inline="always")
def _ring(img, i, j):
    """8 neighbours of (i, j) as a bit mask (bit k = ring position k)."""
    bits = 0
    for k in range(8):
        if _px(img, i + RING_DI[k], j + RING_DJ[k]):
            bits |= 1 << k
    return bits


@njit(cache=True, inline="always")
def _popcount8(bits):
    n = 0
    for k in range(8):
        n += (bits >> k) & 1
    return n


@njit(cache=True, inline="always")
def _transitions(bits):
    """Number of 0 -> 1 transitions around the ring."""
    a = 0
    for k in range(8):
        if ((bits >> k) & 1) == 0 and ((bits >> ((k + 1) & 7)) & 1) == 1:
            a += 1
    return a


@njit(cache=True, inline="always")
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _ring_components(bits):
    """Number of 8-connected components formed by the *set* ring positions
    among themselves (positions k, k+1 are always adjacent; k, k+2 are
    adjacent for even k — two 4-neighbours around a corner)."""
    parent = np.arange(8)
    for k in range(8):
        if ((bits >> k) & 1) == 0:
            continue
        m = (k + 1) & 7
        if (bits >> m) & 1:
            a = _find(parent, k)
            b = _find(parent, m)
            if a != b:
                parent[a] = b
        if (k & 1) == 0:
            m = (k + 2) & 7
            if (bits >> m) & 1:
                a = _find(parent, k)
                b = _find(parent, m)
                if a != b:
                    parent[a] = b
    n = 0
    for k in range(8):
        if ((bits >> k) & 1) and _find(parent, k) == k:
            n += 1
    return n


# --------------------------------------------------------------------------
# thinning
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _zs_flags(img, ii, jj, step, flags):
    """Zhang-Suen deletion flags of the active pixels for sub-iteration
    ``step`` (0 or 1), computed from the frozen ``img``."""
    for n in prange(ii.size):
        i = ii[n]
        j = jj[n]
        bits = _ring(img, i, j)
        b = _popcount8(bits)
        ok = False
        if b >= 2 and b <= 6 and _transitions(bits) == 1:
            p2 = bits & 1
            p4 = (bits >> 2) & 1
            p6 = (bits >> 4) & 1
            p8 = (bits >> 6) & 1
            if step == 0:
                ok = (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0)
            else:
                ok = (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0)
        flags[n] = ok


def thin(mask: np.ndarray, max_iter: int = 100000) -> np.ndarray:
    """Zhang-Suen thinning of a 2-D bool mask -> uint8 skeleton (1 = on).
    Works on the list of remaining mask pixels, so each pass is O(mask)."""
    img = np.ascontiguousarray(np.asarray(mask, dtype=bool)).astype(np.uint8)
    ii, jj = np.nonzero(img)
    ii = ii.astype(np.int64)
    jj = jj.astype(np.int64)
    flags = np.zeros(ii.size, dtype=np.bool_)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            if ii.size == 0:
                break
            flags = np.zeros(ii.size, dtype=np.bool_)
            _zs_flags(img, ii, jj, step, flags)
            if flags.any():
                img[ii[flags], jj[flags]] = 0
                keep = ~flags
                ii = ii[keep]
                jj = jj[keep]
                changed = True
        if not changed:
            break
    return img


@njit(cache=True)
def remove_redundant(img):
    """Delete (in raster order, sequentially) skeleton pixels whose
    neighbours stay 8-connected without them and that are not endpoints:
    staircase corners and 2x2 blocks left by Zhang-Suen.  In place; returns
    the number removed."""
    N0, N1 = img.shape
    removed = 0
    for i in range(N0):
        for j in range(N1):
            if img[i, j] == 0:
                continue
            bits = _ring(img, i, j)
            if _popcount8(bits) < 2:
                continue
            if _ring_components(bits) == 1:
                img[i, j] = 0
                removed += 1
    return removed


@njit(cache=True)
def prune_spurs(img, max_len, guard):
    """Delete skeleton branches from an endpoint to a junction that are
    shorter than ``max_len`` pixels.  Endpoints within ``guard`` cells of
    the face border are kept (the river continues on the next face), as
    are isolated segments (both ends free).  In place; returns the number
    of pixels removed."""
    N0, N1 = img.shape
    path_i = np.empty(max_len + 2, dtype=np.int64)
    path_j = np.empty(max_len + 2, dtype=np.int64)
    removed = 0
    for i in range(N0):
        for j in range(N1):
            if img[i, j] == 0:
                continue
            if i < guard or j < guard or i >= N0 - guard or j >= N1 - guard:
                continue
            bits = _ring(img, i, j)
            if _popcount8(bits) != 1:
                continue
            k0 = 0
            for k in range(8):
                if (bits >> k) & 1:
                    k0 = k
            path_i[0] = i
            path_j[0] = j
            n = 1
            pi = i
            pj = j
            ci = i + RING_DI[k0]
            cj = j + RING_DJ[k0]
            hit = False
            while n <= max_len:
                b = _ring(img, ci, cj)
                deg = _popcount8(b)
                if deg >= 3:
                    hit = True
                    break
                if deg <= 1:
                    break
                if ci < guard or cj < guard or ci >= N0 - guard or cj >= N1 - guard:
                    break
                path_i[n] = ci
                path_j[n] = cj
                n += 1
                found = False
                ni = ci
                nj = cj
                for k in range(8):
                    if (b >> k) & 1:
                        ti = ci + RING_DI[k]
                        tj = cj + RING_DJ[k]
                        if ti == pi and tj == pj:
                            continue
                        ni = ti
                        nj = tj
                        found = True
                        break
                if not found:
                    break
                pi = ci
                pj = cj
                ci = ni
                cj = nj
            if hit:
                for m in range(n):
                    img[path_i[m], path_j[m]] = 0
                removed += n
    return removed


# --------------------------------------------------------------------------
# tracing
# --------------------------------------------------------------------------
@njit(cache=True)
def _walk(img, deg, used, i, j, k, pts_i, pts_j, pos):
    """Follow the skeleton from node (i, j) through exit k until the next
    node (``deg != 2``) or a used exit.  Appends the pixels (inclusive)."""
    used[i, j] |= 1 << k
    pts_i[pos] = i
    pts_j[pos] = j
    pos += 1
    pi = i
    pj = j
    ci = i + RING_DI[k]
    cj = j + RING_DJ[k]
    used[ci, cj] |= 1 << ((k + 4) & 7)
    while True:
        pts_i[pos] = ci
        pts_j[pos] = cj
        pos += 1
        if deg[ci, cj] != 2:
            break
        found = False
        ni = ci
        nj = cj
        for m in range(8):
            if used[ci, cj] & (1 << m):
                continue
            ti = ci + RING_DI[m]
            tj = cj + RING_DJ[m]
            if not _px(img, ti, tj):
                continue
            if ti == pi and tj == pj:
                continue
            used[ci, cj] |= 1 << m
            used[ti, tj] |= 1 << ((m + 4) & 7)
            ni = ti
            nj = tj
            found = True
            break
        if not found:
            break
        pi = ci
        pj = cj
        ci = ni
        cj = nj
    return pos


@njit(cache=True)
def trace_segments(img):
    """Split a skeleton into segments between nodes (pixels with a number
    of neighbours != 2).  Returns ``(seg_off, pts_i, pts_j)``: segment s is
    ``pts[seg_off[s]:seg_off[s+1]]`` (first and last pixel are nodes; node
    pixels belong to every incident segment).  Node-free loops become one
    segment starting at their first pixel in raster order."""
    N0, N1 = img.shape
    deg = np.zeros((N0, N1), dtype=np.int8)
    n_sk = 0
    for i in range(N0):
        for j in range(N1):
            if img[i, j] != 0:
                deg[i, j] = _popcount8(_ring(img, i, j))
                n_sk += 1
    cap_pts = 11 * n_sk + 32
    cap_seg = 5 * n_sk + 16
    pts_i = np.empty(cap_pts, dtype=np.int32)
    pts_j = np.empty(cap_pts, dtype=np.int32)
    seg_off = np.empty(cap_seg, dtype=np.int64)
    used = np.zeros((N0, N1), dtype=np.uint8)
    n_seg = 0
    pos = 0
    seg_off[0] = 0
    for i in range(N0):
        for j in range(N1):
            if img[i, j] == 0 or deg[i, j] == 2:
                continue
            for k in range(8):
                if used[i, j] & (1 << k):
                    continue
                if not _px(img, i + RING_DI[k], j + RING_DJ[k]):
                    continue
                pos = _walk(img, deg, used, i, j, k, pts_i, pts_j, pos)
                n_seg += 1
                seg_off[n_seg] = pos
    for i in range(N0):
        for j in range(N1):
            if img[i, j] == 0 or deg[i, j] != 2 or used[i, j] != 0:
                continue
            for k in range(8):
                if _px(img, i + RING_DI[k], j + RING_DJ[k]):
                    pos = _walk(img, deg, used, i, j, k, pts_i, pts_j, pos)
                    n_seg += 1
                    seg_off[n_seg] = pos
                    break
    return seg_off[: n_seg + 1], pts_i[:pos], pts_j[:pos], deg


@njit(cache=True, parallel=True)
def paint_mask(out, pi, pj, half):
    """Set ``out = 255`` within radius ``half[n]`` (cells, Euclidean) of
    every pixel ``(pi[n], pj[n])``.  Idempotent writes, so parallel-safe."""
    N0, N1 = out.shape
    for n in prange(pi.size):
        r = half[n]
        ri = int(np.ceil(r))
        r2 = r * r
        for di in range(-ri, ri + 1):
            i = pi[n] + di
            if i < 0 or i >= N0:
                continue
            for dj in range(-ri, ri + 1):
                j = pj[n] + dj
                if j < 0 or j >= N1:
                    continue
                if di * di + dj * dj <= r2:
                    out[i, j] = 255


# --------------------------------------------------------------------------
# threshold
# --------------------------------------------------------------------------
def smooth_discharge(q: np.ndarray, sigma: float) -> np.ndarray:
    """Float32 copy of ``q`` Gaussian-smoothed by ``sigma`` cells (0 = as is)."""
    q32 = np.array(q, dtype=np.float32, copy=True)
    if sigma > 0:
        q32 = ndimage.gaussian_filter(q32, sigma=float(sigma), mode="nearest")
    return q32


def log_hist_edges() -> np.ndarray:
    return np.linspace(0.0, LOG_HIST_MAX, LOG_HIST_BINS + 1)


def log_histogram(q: np.ndarray, land: np.ndarray) -> tuple[np.ndarray, int]:
    """Histogram of ``log1p(q)`` over land cells on the fixed
    :func:`log_hist_edges` bins (values above the range go to the last
    bin).  Returns ``(counts int64, n_land)``."""
    x = np.log1p(np.maximum(np.asarray(q, dtype=np.float32)[np.asarray(land, dtype=bool)], 0.0)).astype(np.float64)
    x = np.minimum(x, LOG_HIST_MAX - 1e-9)
    counts, _ = np.histogram(x, bins=log_hist_edges())
    return counts.astype(np.int64), int(x.size)


def threshold_from_histogram(counts: np.ndarray, n_land: int, fraction: float) -> float:
    """Discharge value ``t`` such that about ``fraction`` of the land cells
    have ``q > t`` (linear interpolation inside the histogram bin)."""
    if n_land <= 0 or fraction <= 0:
        return float("inf")
    edges = log_hist_edges()
    want = fraction * n_land
    tail = np.cumsum(counts[::-1])[::-1]  # tail[b] = cells in bins >= b
    if tail[0] <= want:
        return 0.0
    b = int(np.searchsorted(-tail, -want, side="left"))  # first bin with tail <= want
    b = min(max(b, 1), counts.size)
    # bins >= b hold tail[b] (<= want) cells; bin b-1 supplies the rest
    need = want - (tail[b] if b < counts.size else 0)
    c = counts[b - 1]
    frac = 1.0 - (need / c if c > 0 else 0.0)
    x = edges[b - 1] + frac * (edges[b] - edges[b - 1])
    return float(np.expm1(x))


def river_fraction(dp, coarse_channel_fraction: float) -> float:
    """Fraction of fine land cells above the discharge threshold
    (``derive.river_mask_fraction``, or auto from the coarse channel
    fraction of land)."""
    if dp.river_mask_fraction > 0:
        return float(dp.river_mask_fraction)
    if coarse_channel_fraction > 0:
        return float(dp.river_fraction_scale * coarse_channel_fraction)
    return float(dp.river_fallback_fraction)


# --------------------------------------------------------------------------
# mask cleaning + skeleton
# --------------------------------------------------------------------------
def hysteresis_mask(q_s: np.ndarray, land: np.ndarray, q_high: float, q_low: float) -> np.ndarray:
    """Cells with ``q_s > q_low`` (on land) whose 8-connected blob contains
    a cell with ``q_s > q_high``: the low threshold keeps rivers connected,
    the high one rejects blobs that are only ever weak."""
    land = np.asarray(land, dtype=bool)
    high = (q_s > q_high) & land
    if q_low >= q_high or not high.any():
        return high
    low = (q_s > q_low) & land
    lab, n = ndimage.label(low, structure=_STRUCT8)
    if n == 0:
        return high
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(lab[high])] = True
    keep[0] = False
    return keep[lab]


class RiverConnectivity:
    """Global hysteresis / minimum-size decision for the river mask.

    Pass 1 (``add_face`` for every face, any order): label the 8-connected
    blobs of ``low = q_s > q_low & land`` on the face, record each blob's
    size, whether it contains a ``high`` (``q_s > q_thr``) cell, and the
    blob ids along the four edge rows.  ``finalize`` joins blobs across
    cube edges (``fine.edge_neighbors``: a blob cell on an edge row is
    joined to the neighbouring face's owner cell and its two along-edge
    neighbours, i.e. 8-connectivity across the edge) with a union-find and
    keeps a component iff it has a high cell somewhere and its total size
    is at least ``min_cells``.  Pass 2 (``mask``) relabels the face
    (``ndimage.label`` is deterministic) and applies the decision.
    """

    def __init__(self, Nf: int, min_cells: int):
        self.Nf = int(Nf)
        self.min_cells = int(min_cells)
        self.offset = np.zeros(7, dtype=np.int64)
        self.sizes: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * 6
        self.has_high: list[np.ndarray] = [np.zeros(0, dtype=bool)] * 6
        self.frames: list[np.ndarray] = [np.full((4, self.Nf), -1, dtype=np.int64)] * 6
        self.keep: np.ndarray | None = None
        self.n_labels = 0

    @staticmethod
    def label(low: np.ndarray) -> tuple[np.ndarray, int]:
        lab, n = ndimage.label(np.asarray(low, dtype=bool), structure=_STRUCT8)
        return lab, int(n)

    def add_face(self, face: int, low: np.ndarray, high: np.ndarray) -> None:
        lab, n = self.label(low)
        self.sizes[face] = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.int64)
        hh = np.zeros(n + 1, dtype=bool)
        hh[np.unique(lab[np.asarray(high, dtype=bool) & (lab > 0)])] = True
        self.has_high[face] = hh[1:]
        fr = np.stack([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]).astype(np.int64) - 1  # -1 = background
        self.frames[face] = fr
        self.offset[face + 1] = n  # offsets are prefix-summed in finalize

    def _global_frame(self, face: int) -> np.ndarray:
        fr = self.frames[face]
        return np.where(fr >= 0, fr + self.offset[face], -1)

    def finalize(self) -> np.ndarray:
        from .fine import edge_neighbors

        self.offset = np.concatenate([[0], np.cumsum(self.offset[1:])])
        n = int(self.offset[6])
        self.n_labels = n
        if n == 0:
            self.keep = np.zeros(0, dtype=bool)
            return self.keep
        sizes = np.concatenate(self.sizes)
        has_high = np.concatenate(self.has_high)
        parent = np.arange(n, dtype=np.int64)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        Nf = self.Nf
        G = np.stack([self._global_frame(f) for f in range(6)])  # (6, 4, Nf) global blob ids on the edge rows
        for f in range(6):
            for side, (f2, i2, j2) in enumerate(edge_neighbors(Nf, f, 1)):
                a = G[f, side]
                f2, i2, j2 = f2.ravel(), i2.ravel(), j2.ravel()
                side2 = np.where(i2 == 0, 0, np.where(i2 == Nf - 1, 1, np.where(j2 == 0, 2, 3)))
                along = np.where(side2 < 2, j2, i2)
                for d in (-1, 0, 1):
                    b = G[f2, side2, np.clip(along + d, 0, Nf - 1)]
                    ok = (a >= 0) & (b >= 0) & (a != b)
                    for x, y in zip(a[ok].tolist(), b[ok].tolist()):
                        rx, ry = find(x), find(y)
                        if rx != ry:
                            parent[max(rx, ry)] = min(rx, ry)
        roots = np.array([find(x) for x in range(n)], dtype=np.int64)
        root_size = np.bincount(roots, weights=sizes, minlength=n)
        root_high = np.zeros(n, dtype=bool)
        root_high[roots[has_high]] = True
        keep_root = root_high & (root_size >= self.min_cells)
        self.keep = keep_root[roots]
        return self.keep

    def mask(self, face: int, low: np.ndarray) -> np.ndarray:
        """Kept river cells of ``face`` (bool) from the same ``low`` mask
        that was given to ``add_face``."""
        if self.keep is None:
            raise RuntimeError("RiverConnectivity.finalize() first")
        lab, n = self.label(low)
        if n == 0:
            return np.zeros(lab.shape, dtype=bool)
        if n != self.sizes[face].size:
            raise ValueError("low mask differs from the one given to add_face")
        k = np.concatenate([[False], self.keep[self.offset[face] : self.offset[face] + n]])
        return k[lab]


def clean_mask(mask: np.ndarray, min_cells: int, drop_small: bool = True) -> np.ndarray:
    """Drop 8-connected blobs smaller than ``min_cells`` (unless
    ``drop_small`` is False: the size decision was already taken globally
    by :class:`RiverConnectivity`) and fill (4-connected) holes smaller
    than ``min_cells`` that do not touch the face border."""
    m = np.array(mask, dtype=bool, copy=True)
    if not m.any() or min_cells <= 1:
        return m
    lab, n = ndimage.label(m, structure=_STRUCT8) if drop_small else (None, 0)
    if n:
        sizes = np.bincount(lab.ravel())
        small = sizes < min_cells
        small[0] = False
        m[small[lab]] = False
    holes, nh = ndimage.label(~m, structure=_STRUCT4)
    if nh:
        sizes = np.bincount(holes.ravel())
        fill = sizes < min_cells
        fill[0] = False
        border = np.unique(np.concatenate([holes[0, :], holes[-1, :], holes[:, 0], holes[:, -1]]))
        fill[border] = False
        m[fill[holes]] = True
    return m


def skeletonize(mask: np.ndarray, spur_len: int, guard: int = 1) -> np.ndarray:
    """Cleaned mask -> pruned 1-pixel skeleton (uint8)."""
    img = thin(mask)
    if not img.any():
        return img
    remove_redundant(img)
    if spur_len > 0:
        prune_spurs(img, int(spur_len), int(guard))
        remove_redundant(img)
    return img


# --------------------------------------------------------------------------
# segments -> rivers
# --------------------------------------------------------------------------
def running_mean(x: np.ndarray, half: int) -> np.ndarray:
    """Mean over the clamped window ``[k - half, k + half]``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if half <= 0 or n <= 1:
        return x.copy()
    c = np.concatenate([[0.0], np.cumsum(x)])
    k = np.arange(n)
    lo = np.maximum(k - half, 0)
    hi = np.minimum(k + half + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def catmull_rom(points: np.ndarray, widths: np.ndarray, control_step: float, out_step: float, values: np.ndarray | None = None, clamp: float = 0.5):
    """Smooth a pixel chain ``points`` (n, 2) with a centripetal-free
    (uniform) Catmull-Rom spline through control points every
    ``control_step`` chain pixels, sampled every ``out_step`` pixels.  The
    first and last output points are exactly the chain's endpoints (so
    segments still meet at junctions), and every output point is clamped
    to within ``clamp`` cells (per axis) of the chain pixel it belongs to,
    so the smoothed line never leaves the channel cells.  ``widths`` (n,)
    and the optional per-pixel ``values`` (n,) are interpolated linearly
    (a monotone ``values`` stays monotone).  Returns ``(pts (m, 2)
    float64, w (m,))``, plus ``vals (m,)`` when ``values`` is given."""
    P = np.asarray(points, dtype=np.float64)
    W = np.asarray(widths, dtype=np.float64)
    Vv = None if values is None else np.asarray(values, dtype=np.float64)
    n = P.shape[0]
    L = n - 1
    if n < 3 or L < 2 * control_step:
        return (P.copy(), W.copy()) if Vv is None else (P.copy(), W.copy(), Vv.copy())
    s = np.arange(n, dtype=np.float64)
    nc = max(3, int(round(L / control_step)) + 1)
    sc = np.linspace(0.0, L, nc)
    C = np.stack([np.interp(sc, s, P[:, 0]), np.interp(sc, s, P[:, 1])], axis=1)
    Cp = np.vstack([2 * C[0] - C[1], C, 2 * C[-1] - C[-2]])
    n_out = max(2, int(np.ceil(L / out_step)) + 1)
    so = np.linspace(0.0, L, n_out)
    h = L / (nc - 1)
    seg = np.clip(np.floor(so / h).astype(np.int64), 0, nc - 2)
    t = (so - seg * h) / h
    t = np.clip(t, 0.0, 1.0)[:, None]
    p0, p1, p2, p3 = Cp[seg], Cp[seg + 1], Cp[seg + 2], Cp[seg + 3]
    out = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
    if clamp is not None and clamp >= 0:
        anchor = P[np.clip(np.rint(so).astype(np.int64), 0, n - 1)]
        out = np.clip(out, anchor - clamp, anchor + clamp)
    out[0] = P[0]
    out[-1] = P[-1]
    w = np.interp(so, s, W)
    if Vv is None:
        return out, w
    return out, w, np.interp(so, s, Vv)


def strahler_orders(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Strahler order of oriented segments given their start / end node
    keys (segments flow start -> end).  Kahn's algorithm over the node
    graph; segments on cycles (never released) get order 1."""
    n = starts.size
    order = np.zeros(n, dtype=np.int32)
    if n == 0:
        return order
    node_in: dict[int, list[int]] = {}
    node_out: dict[int, list[int]] = {}
    for s in range(n):
        node_in.setdefault(int(ends[s]), []).append(s)
        node_out.setdefault(int(starts[s]), []).append(s)
        node_in.setdefault(int(starts[s]), [])
        node_out.setdefault(int(ends[s]), [])
    remaining = {k: len(v) for k, v in node_in.items()}
    ready = sorted(k for k, r in remaining.items() if r == 0)
    while ready:
        k = ready.pop()
        ins = node_in[k]
        if ins:
            m = max(int(order[s]) for s in ins)
            o = m + 1 if sum(1 for s in ins if order[s] == m) >= 2 else m
        else:
            o = 1
        for s in node_out[k]:
            order[s] = o
            t = int(ends[s])
            remaining[t] -= 1
            if remaining[t] == 0:
                ready.append(t)
    order[order == 0] = 1
    return order


def extend_to_border(ci: np.ndarray, cj: np.ndarray, Nf: int, free0: bool, free1: bool, reach: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Thinning pulls a skeleton's free ends about half a river width back
    from the face border; prepend / append the cells straight out to the
    border when a free end lies within ``reach`` cells of it and (with
    ``mask`` given) the straight path stays inside the river mask, so
    polylines (and the mask) of a river crossing a cube edge meet at the
    edge."""
    if reach <= 0 or ci.size == 0:
        return ci, cj

    def steps(i, j):
        di = dj = 0
        if 0 < i <= reach:
            di = -1
        elif Nf - 1 - reach <= i < Nf - 1:
            di = 1
        if 0 < j <= reach:
            dj = -1
        elif Nf - 1 - reach <= j < Nf - 1:
            dj = 1
        if di == 0 and dj == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        n = 0
        if di != 0:
            n = i if di < 0 else Nf - 1 - i
        if dj != 0:
            m = j if dj < 0 else Nf - 1 - j
            n = m if n == 0 else min(n, m)
        k = np.arange(1, n + 1)
        ei, ej = i + di * k, j + dj * k
        if mask is not None and not mask[ei, ej].all():
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return ei, ej

    if free0:
        ei, ej = steps(int(ci[0]), int(cj[0]))
        ci, cj = np.concatenate([ei[::-1], ci]), np.concatenate([ej[::-1], cj])
    if free1:
        ei, ej = steps(int(ci[-1]), int(cj[-1]))
        ci, cj = np.concatenate([ci, ei]), np.concatenate([cj, ej])
    return ci, cj


def mode_nonneg(values: np.ndarray) -> int:
    """Most frequent value >= 0 (ties -> smallest); -1 if none."""
    v = np.asarray(values)
    v = v[v >= 0]
    if v.size == 0:
        return -1
    u, c = np.unique(v, return_counts=True)
    return int(u[np.argmax(c)])


class CoarseGraphIndex:
    """Nearest coarse drainage edge for every coarse cell (per face), from
    ``graph/drainage.json``.  ``edge_id[f, i, j]`` / ``order[f, i, j]`` are
    the nearest channel cell's edge (id) and Strahler order for cells
    within ``match_cells`` of a channel, else -1 / 0."""

    def __init__(self, drainage: dict | None, N: int, match_cells: int):
        self.N = N
        self.edge_id = np.full((6, N, N), -1, dtype=np.int32)
        self.order = np.zeros((6, N, N), dtype=np.int16)
        self.n_edges = 0
        if not drainage or not drainage.get("edges"):
            return
        eid = np.full((6, N, N), -1, dtype=np.int32)
        order = np.zeros((6, N, N), dtype=np.int16)
        for e in sorted(drainage["edges"], key=lambda e: (int(e.get("order", 1)), int(e["id"]))):
            cells = np.asarray(e.get("cells", ()), dtype=np.int64)
            if cells.size == 0:
                continue
            cells = cells.reshape(-1, 3)
            ok = (cells[:, 0] >= 0) & (cells[:, 0] < 6) & (cells[:, 1] >= 0) & (cells[:, 1] < N) & (cells[:, 2] >= 0) & (cells[:, 2] < N)
            cells = cells[ok]
            eid[cells[:, 0], cells[:, 1], cells[:, 2]] = int(e["id"])
            order[cells[:, 0], cells[:, 1], cells[:, 2]] = int(e.get("order", 1))
            self.n_edges += 1
        chan = eid >= 0
        for f in range(6):
            if not chan[f].any():
                continue
            if chan[f].all():
                self.edge_id[f] = eid[f]
                self.order[f] = order[f]
                continue
            d, (ni, nj) = ndimage.distance_transform_edt(~chan[f], return_distances=True, return_indices=True)
            near = d <= match_cells
            self.edge_id[f][near] = eid[f][ni[near], nj[near]]
            self.order[f][near] = order[f][ni[near], nj[near]]

    def match(self, face: int, ci: np.ndarray, cj: np.ndarray) -> tuple[int, int]:
        """(edge_id, order) for a chain of coarse cells (mode of the matched
        edges; (-1, 0) if none)."""
        ids = self.edge_id[face, ci, cj]
        e = mode_nonneg(ids)
        if e < 0:
            return -1, 0
        o = int(self.order[face, ci, cj][ids == e].max())
        return e, o


def extract_face_rivers(face: int, q_s: np.ndarray, surface: np.ndarray, land: np.ndarray, q_thr: float, dp, R: int, fine_cell_size_m: float, graph: CoarseGraphIndex | None = None, q_low: float | None = None, mask: np.ndarray | None = None):
    """Rivers of one fine face.

    ``q_s`` smoothed fine discharge, ``surface`` fine surface (m), ``land``
    bool, ``q_thr`` the global discharge threshold (``q_low`` the lower
    hysteresis threshold, default = ``q_thr``), ``dp`` = ``params.derive``.
    ``mask`` (bool, optional) is the river cell mask decided globally by
    :class:`RiverConnectivity`; without it the hysteresis and the minimum
    blob size are applied to this face alone.  Returns ``(river_mask uint8
    (Nf, Nf), rivers, info)`` where ``rivers`` is a list of
    ``graph/rivers.json`` records (ids are per-face; the driver renumbers
    them) and ``info`` counts."""
    Nf = q_s.shape[0]
    min_cells = int(round(dp.min_river_cells * R * R))
    spur_len = int(round(dp.spur_cells * R))
    if mask is None:
        mask = clean_mask(hysteresis_mask(q_s, land, q_thr, q_thr if q_low is None else q_low), min_cells)
    else:
        mask = clean_mask(np.asarray(mask, dtype=bool) & np.asarray(land, dtype=bool), min_cells, drop_small=False)
    river_mask = np.zeros((Nf, Nf), dtype=np.uint8)
    info = {"mask_cells": int(mask.sum()), "skeleton_cells": 0, "segments": 0, "rivers": 0, "graph_matched": 0}
    if not mask.any():
        return river_mask, [], info
    skel = skeletonize(mask, spur_len)
    info["skeleton_cells"] = int(skel.sum())
    if not skel.any():
        return river_mask, [], info
    seg_off, pts_i, pts_j, deg = trace_segments(skel)
    n_seg = seg_off.size - 1
    info["segments"] = int(n_seg)
    if n_seg == 0:
        return river_mask, [], info

    surf = np.asarray(surface, dtype=np.float32)
    chains = []
    for s in range(n_seg):
        a, b = int(seg_off[s]), int(seg_off[s + 1])
        if b - a < 2:
            continue
        ci = pts_i[a:b].astype(np.int64)
        cj = pts_j[a:b].astype(np.int64)
        k = max(1, min(3, ci.size // 2))  # disjoint head / tail windows
        h0 = float(surf[ci[:k], cj[:k]].mean())
        h1 = float(surf[ci[-k:], cj[-k:]].mean())
        q0 = float(q_s[ci[0], cj[0]])
        q1 = float(q_s[ci[-1], cj[-1]])
        if h1 > h0 or (h1 == h0 and q1 < q0):
            ci = ci[::-1]
            cj = cj[::-1]
        free0 = deg[ci[0], cj[0]] <= 1
        free1 = deg[ci[-1], cj[-1]] <= 1
        if free0 and free1 and ci.size < max(spur_len, 2):
            continue
        ci, cj = extend_to_border(ci, cj, Nf, free0, free1, int(np.ceil(dp.max_width_cells)), mask)
        chains.append((ci, cj))
    if not chains:
        return river_mask, [], info

    # Strahler from the fine segment graph (fallback / no coarse graph)
    starts = np.array([int(c[0][0]) * Nf + int(c[1][0]) for c in chains], dtype=np.int64)
    ends = np.array([int(c[0][-1]) * Nf + int(c[1][-1]) for c in chains], dtype=np.int64)
    fine_order = strahler_orders(starts, ends)

    a_w = float(dp.river_width_a)
    b_w = float(dp.river_width_b)
    max_w = float(dp.max_width_cells)
    q_ref = max(float(q_thr), 1e-9)
    rivers = []
    all_i, all_j, all_half = [], [], []
    for s, (ci, cj) in enumerate(chains):
        edge_id, order = -1, int(fine_order[s])
        if graph is not None and graph.n_edges:
            e, o = graph.match(face, np.minimum(ci // R, graph.N - 1), np.minimum(cj // R, graph.N - 1))
            if e >= 0:
                edge_id, order = e, o
                info["graph_matched"] += 1
        if order < dp.min_river_order:
            continue
        q = running_mean(q_s[ci, cj], R)
        w = np.clip(a_w * np.power(np.maximum(q, 0.0) / q_ref, b_w), 1.0, max_w)
        all_i.append(ci)
        all_j.append(cj)
        all_half.append(np.maximum(w * 0.5, 0.5))
        hz = np.minimum.accumulate(surf[ci, cj].astype(np.float64))  # water surface: never rises downstream
        pts, wq, hq = catmull_rom(np.stack([ci, cj], axis=1).astype(np.float64), w, float(R), float(dp.river_point_step), values=hz)
        u = np.clip((pts[:, 0] + 0.5) / Nf, 0.0, (Nf - 0.5) / Nf)
        v = np.clip((pts[:, 1] + 0.5) / Nf, 0.0, (Nf - 0.5) / Nf)
        wm = wq * fine_cell_size_m
        rivers.append(
            {
                "id": len(rivers),
                "edge_id": int(edge_id),
                "order": int(order),
                "face": int(face),
                "cells": int(ci.size),
                "length_m": float(np.sum(np.hypot(np.diff(ci), np.diff(cj))) * fine_cell_size_m),
                "discharge": float(np.mean(q_s[ci, cj])),
                "points": [[int(face), round(float(a), 6), round(float(b), 6), round(float(c), 1), round(float(d), 2)] for a, b, c, d in zip(u, v, wm, hq)],
            }
        )
    if all_i:
        pi = np.concatenate(all_i).astype(np.int32)
        pj = np.concatenate(all_j).astype(np.int32)
        half = np.concatenate(all_half).astype(np.float32)
        paint_mask(river_mask, pi, pj, half)
        river_mask[~np.asarray(land, dtype=bool)] = 0
    info["rivers"] = len(rivers)
    return river_mask, rivers, info


__all__ = [
    "RING_DI", "RING_DJ", "thin", "remove_redundant", "prune_spurs", "trace_segments", "paint_mask",
    "smooth_discharge", "log_histogram", "threshold_from_histogram", "river_fraction", "hysteresis_mask", "RiverConnectivity", "clean_mask", "skeletonize",
    "running_mean", "catmull_rom", "extend_to_border", "strahler_orders", "CoarseGraphIndex", "extract_face_rivers",
]
