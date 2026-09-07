"""Tile vertex lattices, their cross-face topology and the LOD pyramid
(PLAN.md section 10.3; tile sample convention in ``globe.io.tiles``).

Lattice
-------
Face ``f`` at LOD ``l`` holds ``(n_l + 1)²`` vertex samples, ``n_l = N_fine
>> l``; vertex ``(i, j)`` sits at ``u = i / n_l, v = j / n_l`` (fine-cell
corner ``(i << l, j << l)``).  A tile ``(lod, f, x, y)`` is the window
``[x*T : x*T + T + 1, y*T : y*T + T + 1]`` of that lattice, so the last
column of a tile *is* the first column of the next tile.  Vertices on a
face edge are shared with the neighbouring face and the 8 cube corners are
shared by three faces; every value is computed from the same set of inputs
with an order-independent reduction on every face that owns it, so shared
vertices are bit-identical (no interpolation across the seam).

Sides are numbered ``0 = +i (i = n, u = 1), 1 = -i (i = 0), 2 = +j (j = n,
v = 1), 3 = -j (j = 0)``; the *along* index of a vertex on side 0/1 is
``j``, on side 2/3 ``i``.  Fine-cell corners along a cube edge coincide on
both faces (EAC parametrises an edge by angle on either face), so vertex
``k`` of one side is vertex ``k`` or ``n - k`` (``flip``) of the
neighbouring side — :func:`edge_links` derives this from ``cubesphere``.

Reductions
----------
* LOD 0 from fine cells: vertex ``(i, j)`` reduces the 2×2 cells around the
  corner (2 + 2 across an edge, 3 at a cube corner).
* LOD ``l + 1`` from LOD ``l``: vertex ``(k, m)`` reduces the 3×3 LOD-``l``
  vertices centred on ``(2k, 2m)`` — on the sphere, i.e. 6 own + 3
  neighbour vertices on an edge, the 7 distinct vertices at a cube corner.

Ops: ``STRIDE`` (centre value; heights: LOD ``l + 1`` vertex ``(k, m)`` ==
LOD ``l`` vertex ``(2k, 2m)`` exactly), ``MAX`` (water surface, river mask),
``MODE`` (ids, biome; ties -> largest value), ``MEAN`` (floats: sum in sorted
order; integers: exact, round half up).  All are permutation invariant.

Everything here works on one face (plus 1-deep rings from the neighbours)
at a time; callers hand in a ``getter(face, side=None, depth=0)`` that
returns the face array or one edge row of it, so faces can be memmaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit, prange

from ..cubesphere import BASES, HALF_PI, face_of_v, from_sphere_v, project_to_face_v, to_sphere_v

STRIDE, MAX, MODE, MEAN = 0, 1, 2, 3
OPS = {"stride": STRIDE, "max": MAX, "mode": MODE, "mean": MEAN}
SIDES = ("+i", "-i", "+j", "-j")


# --------------------------------------------------------------------------
# topology (derived from cubesphere, never hand-coded)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeLink:
    """Side ``side`` of ``face`` meets side ``nb_side`` of ``nb_face``;
    along-index ``k`` on this side is ``n - k`` there if ``flip`` else ``k``."""

    face: int
    side: int
    nb_face: int
    nb_side: int
    flip: bool

    def map_along(self, k, n: int):
        return n - np.asarray(k) if self.flip else np.asarray(k)


def edge_points(face: int, side: int, N: int) -> np.ndarray:
    """Exact unit vectors of the ``N + 1`` lattice vertices along ``side`` of
    ``face`` (built from ``BASES`` so the points lie exactly on the cube
    edge; ``to_sphere(face, 1.0, v)`` is off by an ulp)."""
    r, up, n = BASES[face]
    t = np.tan((np.arange(N + 1) / N - 0.5) * HALF_PI)[:, None]
    if side == 0:
        p = n + r + t * up
    elif side == 1:
        p = n - r + t * up
    elif side == 2:
        p = n + t * r + up
    else:
        p = n + t * r - up
    return p / np.linalg.norm(p, axis=1, keepdims=True)


@lru_cache(maxsize=None)
def edge_links(N: int = 64) -> tuple[EdgeLink, ...]:
    """The 24 (face, side) links, indexed ``face * 4 + side``.  Independent
    of ``N`` (pass the lattice size to have the along-index identity checked
    at that resolution)."""
    links = []
    for f in range(6):
        for s in range(4):
            # neighbouring face: a point half a cell beyond the middle of the side
            uv = ((1 + 0.5 / N, 0.5), (-0.5 / N, 0.5), (0.5, 1 + 0.5 / N), (0.5, -0.5 / N))[s]
            g = int(face_of_v(to_sphere_v(np.array([f]), np.array([uv[0]]), np.array([uv[1]])))[0])
            if g == f:
                raise RuntimeError("edge probe landed on its own face")
            u2, v2 = project_to_face_v(np.full(N + 1, g), edge_points(f, s, N))
            iu, iv = u2 * N, v2 * N
            k = np.arange(N + 1)
            if np.all(np.abs(iu - iu[0]) < 1e-6) and (abs(iu[0]) < 1e-6 or abs(iu[0] - N) < 1e-6):
                nb_side, along = (0 if iu[0] > N / 2 else 1), iv
            elif np.all(np.abs(iv - iv[0]) < 1e-6) and (abs(iv[0]) < 1e-6 or abs(iv[0] - N) < 1e-6):
                nb_side, along = (2 if iv[0] > N / 2 else 3), iu
            else:
                raise RuntimeError(f"side {s} of face {f} is not an edge of face {g}")
            ka = np.rint(along)
            if np.abs(along - ka).max() > 1e-6 * N:
                raise RuntimeError("edge vertices are not lattice vertices on the neighbouring face")
            if np.array_equal(ka, k):
                flip = False
            elif np.array_equal(ka, N - k):
                flip = True
            else:
                raise RuntimeError("edge vertex order is neither identity nor reversal")
            links.append(EdgeLink(f, s, g, nb_side, flip))
    links = tuple(links)
    for ln in links:  # symmetry
        back = links[ln.nb_face * 4 + ln.nb_side]
        if (back.nb_face, back.nb_side, back.flip) != (ln.face, ln.side, ln.flip):
            raise RuntimeError("edge links are not symmetric")
    return links


def side_row(arr: np.ndarray, side: int, depth: int = 0) -> np.ndarray:
    """The 1-D row of a square face array at ``depth`` cells/vertices from
    ``side`` (depth 0 = the row touching / lying on the side), in that
    face's along order.  Works for cell arrays and lattices."""
    n = arr.shape[0]
    if side == 0:
        return np.ascontiguousarray(arr[n - 1 - depth, :])
    if side == 1:
        return np.ascontiguousarray(arr[depth, :])
    if side == 2:
        return np.ascontiguousarray(arr[:, n - 1 - depth])
    return np.ascontiguousarray(arr[:, depth])


def neighbour_ring(getter, links, face: int, side: int, depth: int) -> np.ndarray:
    """Row at ``depth`` from the shared edge on the neighbouring face,
    re-ordered into ``face``'s along order."""
    ln = links[face * 4 + side]
    row = np.asarray(getter(ln.nb_face, ln.nb_side, depth))
    return row[::-1].copy() if ln.flip else row


def extend(own: np.ndarray, rings, lattice: bool) -> tuple[np.ndarray, np.ndarray]:
    """Surround a face array with a 1-deep ring of neighbour values and
    return ``(E, valid)`` (``E`` is ``(n + 2)²``).  Ring corners (no fourth
    face exists there) are invalid; for lattices the *j*-side ring ends are
    invalid too, because they duplicate the *i*-side ring ends (both are the
    vertex one step along the third edge at that cube corner)."""
    n = own.shape[0]
    E = np.empty((n + 2, n + 2), dtype=own.dtype)
    E[1:-1, 1:-1] = own
    E[-1, 1:-1] = rings[0]
    E[0, 1:-1] = rings[1]
    E[1:-1, -1] = rings[2]
    E[1:-1, 0] = rings[3]
    valid = np.ones((n + 2, n + 2), dtype=np.uint8)
    valid[0, 0] = valid[0, -1] = valid[-1, 0] = valid[-1, -1] = 0
    E[0, 0] = E[0, -1] = E[-1, 0] = E[-1, -1] = own[0, 0]  # defined bytes, never read
    if lattice:
        valid[1, 0] = valid[-2, 0] = valid[1, -1] = valid[-2, -1] = 0
    return E, valid


def build_extended(getter, links, face: int, lattice: bool) -> tuple[np.ndarray, np.ndarray]:
    """``extend`` for ``face`` with rings pulled from the neighbours through
    ``getter(face, side, depth)``: depth 0 (edge-adjacent cells) for cell
    arrays, depth 1 (the row inside the shared edge column) for lattices."""
    own = np.asarray(getter(face))
    depth = 1 if lattice else 0
    rings = [neighbour_ring(getter, links, face, s, depth) for s in range(4)]
    return extend(own, rings, lattice)


# --------------------------------------------------------------------------
# pooling kernels
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _pool_float(E, valid, stride, win, m, op):
    out = np.empty((m, m), dtype=E.dtype)
    c2 = win // 2
    for k in prange(m):
        vals = np.empty(win * win, dtype=np.float64)
        for l in range(m):
            i0 = k * stride
            j0 = l * stride
            if op == 0:
                out[k, l] = E[i0 + c2, j0 + c2]
                continue
            c = 0
            for a in range(win):
                for b in range(win):
                    if valid[i0 + a, j0 + b]:
                        vals[c] = E[i0 + a, j0 + b]
                        c += 1
            if c == 0:
                out[k, l] = 0
            elif op == 1:
                mx = vals[0]
                for t in range(1, c):
                    if vals[t] > mx:
                        mx = vals[t]
                out[k, l] = mx
            else:  # mean: sum in sorted order (permutation invariant)
                for t in range(1, c):
                    x = vals[t]
                    q = t - 1
                    while q >= 0 and vals[q] > x:
                        vals[q + 1] = vals[q]
                        q -= 1
                    vals[q + 1] = x
                s = 0.0
                for t in range(c):
                    s += vals[t]
                out[k, l] = s / c
    return out


@njit(cache=True, parallel=True)
def _pool_int(E, valid, stride, win, m, op):
    out = np.empty((m, m), dtype=E.dtype)
    c2 = win // 2
    for k in prange(m):
        vals = np.empty(win * win, dtype=np.int64)
        for l in range(m):
            i0 = k * stride
            j0 = l * stride
            if op == 0:
                out[k, l] = E[i0 + c2, j0 + c2]
                continue
            c = 0
            for a in range(win):
                for b in range(win):
                    if valid[i0 + a, j0 + b]:
                        vals[c] = E[i0 + a, j0 + b]
                        c += 1
            if c == 0:
                out[k, l] = 0
            elif op == 1:
                mx = vals[0]
                for t in range(1, c):
                    if vals[t] > mx:
                        mx = vals[t]
                out[k, l] = mx
            elif op == 2:  # mode, ties -> largest value
                best = vals[0]
                bestc = 0
                for t in range(c):
                    cnt = 0
                    for q in range(c):
                        if vals[q] == vals[t]:
                            cnt += 1
                    if cnt > bestc or (cnt == bestc and vals[t] > best):
                        best = vals[t]
                        bestc = cnt
                out[k, l] = best
            else:  # mean, round half up
                s = 0
                for t in range(c):
                    s += vals[t]
                out[k, l] = (2 * s + c) // (2 * c)
    return out


def pool(E: np.ndarray, valid: np.ndarray, stride: int, win: int, m: int, op) -> np.ndarray:
    """``out[k, l] = op(E[k*stride : k*stride+win, l*stride : l*stride+win])``
    over valid entries, ``out`` is ``(m, m)`` of ``E``'s dtype."""
    op = OPS[op] if isinstance(op, str) else int(op)
    if np.issubdtype(E.dtype, np.floating):
        if op == MODE:
            raise ValueError("mode is for integer arrays")
        return _pool_float(np.ascontiguousarray(E), valid, stride, win, m, op)
    return _pool_int(np.ascontiguousarray(E), valid, stride, win, m, op)


# --------------------------------------------------------------------------
# lattice construction
# --------------------------------------------------------------------------
def lod0_lattice(getter, face: int, op, links=None) -> np.ndarray:
    """LOD-0 vertex lattice ``(N + 1)²`` of ``face`` from fine cells.
    ``getter(face, side=None, depth=0)`` returns the ``(N, N)`` cell array of
    a face, or its edge-adjacent row on ``side`` (depth 0)."""
    links = links or edge_links()
    E, valid = build_extended(getter, links, face, lattice=False)
    N = E.shape[0] - 2
    return pool(E, valid, 1, 2, N + 1, op)


def next_lod(getter, face: int, op, links=None) -> np.ndarray:
    """LOD ``l + 1`` lattice ``(n/2 + 1)²`` of ``face`` from the LOD-``l``
    lattices.  ``getter(face, side=None, depth=0)`` returns the ``(n + 1)²``
    lattice of a face or its row at ``depth`` from ``side``."""
    links = links or edge_links()
    E, valid = build_extended(getter, links, face, lattice=True)
    n = E.shape[0] - 3
    if n % 2:
        raise ValueError("lattice size must be even to halve")
    return pool(E, valid, 2, 3, n // 2 + 1, op)


def pyramid(faces: list[np.ndarray], op, levels: int, links=None) -> list[list[np.ndarray]]:
    """In-memory convenience: ``faces`` = six LOD-0 lattices ->
    ``[[lod0 f0..f5], [lod1 ...], ...]`` with ``levels + 1`` entries."""
    links = links or edge_links()
    out = [list(faces)]
    for _ in range(levels):
        cur = out[-1]

        def getter(f, side=None, depth=0, cur=cur):
            return cur[f] if side is None else side_row(cur[f], side, depth)

        out.append([next_lod(getter, f, op, links) for f in range(6)])
    return out


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------
def tile_neighbors(lod: int, face: int, x: int, y: int, n: int, links=None) -> list[list[int]]:
    """``[lod, face, x, y]`` of the edge-adjacent tiles across sides
    ``+i, -i, +j, -j`` (crossing cube edges via :func:`edge_links`)."""
    links = links or edge_links()
    out = []
    for s in range(4):
        di, dj = ((1, 0), (-1, 0), (0, 1), (0, -1))[s]
        x2, y2 = x + di, y + dj
        if 0 <= x2 < n and 0 <= y2 < n:
            out.append([lod, face, x2, y2])
            continue
        ln = links[face * 4 + s]
        a = y if s < 2 else x
        a2 = n - 1 - a if ln.flip else a
        if ln.nb_side == 0:
            g = (n - 1, a2)
        elif ln.nb_side == 1:
            g = (0, a2)
        elif ln.nb_side == 2:
            g = (a2, n - 1)
        else:
            g = (a2, 0)
        out.append([lod, ln.nb_face, int(g[0]), int(g[1])])
    return out


def tile_of_point(p, lod: int, N_fine: int, T: int):
    """(face, x, y) of the tile containing unit vector(s) ``p`` at ``lod``
    (``from_sphere`` + clamp, the C++ ``tile_of`` convention)."""
    f, u, v = from_sphere_v(p)
    n = max(1, (N_fine // T) >> lod)
    x = np.minimum(np.floor(u * n).astype(np.int64), n - 1)
    y = np.minimum(np.floor(v * n).astype(np.int64), n - 1)
    return f, x, y


__all__ = [
    "STRIDE", "MAX", "MODE", "MEAN", "OPS", "SIDES", "EdgeLink", "edge_links", "edge_points", "side_row",
    "neighbour_ring", "extend", "build_extended", "pool", "lod0_lattice", "next_lod", "pyramid",
    "tile_neighbors", "tile_of_point",
]
