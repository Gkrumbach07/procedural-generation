"""D8 flow-direction codes and cross-face neighbour lookups (docs/DEVELOPING.md).

Codes 0..7 index :data:`D8_OFFSETS`; ``OCEAN = 255`` marks ocean / no
outflow.  The downstream cell of interior cell ``(f, i, j)`` with code ``k``
is the *extended* cell ``(f, i+H+di, j+H+dj)`` mapped through ``grid.owner``
(the interior cell owning that extended cell, possibly on another face).
Nothing here knows how faces are oriented; everything goes through
``owner``.

Two index conventions are used by the hydro package:

* **extended flat** — index into a ``(6, NE, NE)`` array (what ``owner``
  holds); ``grid.unflat_index`` decodes it.
* **cell id** ``cid = (f*N + i)*N + j`` — index into a flattened interior
  ``(6, N, N)`` array.  All hydro kernels work on cell ids; :func:`cid_fij`
  decodes one.

All scalar helpers are ``numba.njit`` so kernels can call them.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

#: (di, dj) for code 0..7 — exactly as in docs/DEVELOPING.md.
D8_OFFSETS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
D8_DI = np.array([o[0] for o in D8_OFFSETS], dtype=np.int64)
D8_DJ = np.array([o[1] for o in D8_OFFSETS], dtype=np.int64)
#: flow_dir value for ocean / no outflow
OCEAN = 255
#: flow_dir value the routing kernel writes when it cannot find a valid
#: direction (never expected; ``routing.flow_directions`` asserts on it).
INVALID = 254


@njit(cache=True, inline="always")
def ext_to_fij(flat, NE, H):
    """Extended flat index -> interior (f, i, j) (no halo offset)."""
    f = flat // (NE * NE)
    rem = flat - f * NE * NE
    ei = rem // NE
    ej = rem - ei * NE
    return f, ei - H, ej - H


@njit(cache=True, inline="always")
def cid_fij(cid, N):
    """Cell id -> (f, i, j)."""
    f = cid // (N * N)
    rem = cid - f * N * N
    i = rem // N
    return f, i, rem - i * N


@njit(cache=True, inline="always")
def fij_cid(f, i, j, N):
    return (f * N + i) * N + j


@njit(cache=True)
def downstream_flat(owner, N, H, f, i, j, code):
    """Contract helper (docs/DEVELOPING.md): extended flat index (into a
    ``(6, NE, NE)`` array) of the interior cell downstream of interior cell
    ``(f, i, j)`` with D8 ``code``; ``-1`` for ``OCEAN``."""
    if code >= 8:
        return -1
    return owner[f, i + H + D8_DI[code], j + H + D8_DJ[code]]


@njit(cache=True, inline="always")
def neighbor_cid(owner, N, H, f, i, j, k):
    """Cell id of the k-th D8 neighbour of interior cell (f, i, j) (may be on
    another face)."""
    NE = owner.shape[1]
    o = owner[f, i + H + D8_DI[k], j + H + D8_DJ[k]]
    f2, i2, j2 = ext_to_fij(o, NE, H)
    return (f2 * N + i2) * N + j2


@njit(cache=True)
def neighbor_fij(owner, N, H, f, i, j, k):
    """(f2, i2, j2) of the k-th D8 neighbour of interior cell (f, i, j)."""
    NE = owner.shape[1]
    o = owner[f, i + H + D8_DI[k], j + H + D8_DJ[k]]
    return ext_to_fij(o, NE, H)


@njit(cache=True)
def is_neighbor(owner, N, H, f, i, j, cid):
    """True if cell id ``cid`` is among the 8 D8 neighbours of (f, i, j).
    Same-face neighbourhoods are always symmetric; across a face edge the
    nearest-cell mapping can (in principle) be asymmetric, so kernels that
    need a reversible link check this."""
    for k in range(8):
        if neighbor_cid(owner, N, H, f, i, j, k) == cid:
            return True
    return False


@njit(cache=True)
def code_towards(owner, N, H, f, i, j, cid):
    """D8 code k such that the k-th neighbour of (f, i, j) is ``cid``; -1 if
    ``cid`` is not a neighbour."""
    for k in range(8):
        if neighbor_cid(owner, N, H, f, i, j, k) == cid:
            return k
    return -1


@njit(cache=True, inline="always")
def step_length_m(metric, cell_size_m, H, f, i, j, k):
    """Physical length (m) of a D8 step of code ``k`` from interior cell
    (f, i, j), from the local metric tensor (``grid.metric``)."""
    di = D8_DI[k]
    dj = D8_DJ[k]
    g0 = metric[f, i + H, j + H, 0]
    g1 = metric[f, i + H, j + H, 1]
    g2 = metric[f, i + H, j + H, 2]
    d2 = g0 * di * di + 2.0 * g1 * di * dj + g2 * dj * dj
    return np.sqrt(d2) * cell_size_m


@njit(cache=True, parallel=True)
def downstream_table(flow_dir, owner, H):
    """``flow_dir`` interior (6, N, N) uint8 -> (6*N*N,) int64 cell id of the
    downstream cell, -1 where ``flow_dir >= 8`` (ocean)."""
    F, N, _ = flow_dir.shape
    out = np.full(F * N * N, -1, dtype=np.int64)
    for f in prange(F):
        for i in range(N):
            for j in range(N):
                k = flow_dir[f, i, j]
                if k < 8:
                    out[(f * N + i) * N + j] = neighbor_cid(owner, N, H, f, i, j, k)
    return out


@njit(cache=True, parallel=True)
def neighbor_table(N, H, owner):
    """(6*N*N, 8) int64 cell ids of every cell's D8 neighbours (48 MB per
    million cells; prefer :func:`neighbor_cid` inside kernels)."""
    out = np.empty((6 * N * N, 8), dtype=np.int64)
    for f in prange(6):
        for i in range(N):
            for j in range(N):
                c = (f * N + i) * N + j
                for k in range(8):
                    out[c, k] = neighbor_cid(owner, N, H, f, i, j, k)
    return out


def downstream_fij(flow_dir_fij, grid):
    """Python convenience: (6, N, N, 3) int32 (f, i, j) of the downstream
    cell for every interior cell, -1 where ocean."""
    N, H = grid.N, grid.H
    down = downstream_table(np.ascontiguousarray(flow_dir_fij, dtype=np.uint8), grid.owner, H)
    out = np.full((6 * N * N, 3), -1, dtype=np.int32)
    ok = down >= 0
    d = down[ok]
    out[ok, 0] = d // (N * N)
    rem = d - out[ok, 0] * N * N
    out[ok, 1] = rem // N
    out[ok, 2] = rem % N
    return out.reshape(6, N, N, 3)


__all__ = [
    "D8_OFFSETS", "D8_DI", "D8_DJ", "OCEAN", "INVALID",
    "ext_to_fij", "cid_fij", "fij_cid", "downstream_flat", "neighbor_cid", "neighbor_fij",
    "is_neighbor", "code_towards", "step_length_m", "downstream_table", "neighbor_table", "downstream_fij",
]
