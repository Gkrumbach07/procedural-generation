"""Operations on a drainage tree over a 2-D window (flat index ``i*W + j``;
``parent`` -1 on drains; receivers are 8-neighbours).

* :func:`receiver_codes` packs a tree into one byte a cell (D8 code, 255 on a
  drain) so it can be kept while other windows are routed;
* :func:`accumulate_codes` accumulates a weight down a packed tree (Kahn's
  order: no pop sequence needed);
* :func:`window_exits` follows water across the border of a sub-window: for
  every cell, where its water finally leaves the sub-window (or -1 if it
  ends at a drain inside), with re-entries resolved.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_DI8 = np.array([1, 1, 0, -1, -1, -1, 0, 1])
_DJ8 = np.array([0, 1, 1, 1, 0, -1, -1, -1])


@njit(cache=True)
def _codes_kernel(parent, W):
    M = parent.size
    code = np.full(M, 255, np.uint8)
    for c in range(M):
        p = np.int64(parent[c])
        if p < 0:
            continue
        di = p // W - c // W
        dj = p % W - c % W
        for k in range(8):
            if _DI8[k] == di and _DJ8[k] == dj:
                code[c] = k
                break
    return code


def receiver_codes(parent: np.ndarray, W: int) -> np.ndarray:
    """uint8 D8 code of each cell's receiver (255: none)."""
    return _codes_kernel(np.ascontiguousarray(parent).reshape(-1), int(W))


@njit(cache=True)
def _accumulate_codes_kernel(code, W, w):
    M = code.size
    off = np.empty(8, np.int64)
    for k in range(8):
        off[k] = _DI8[k] * W + _DJ8[k]
    indeg = np.zeros(M, np.uint8)
    for c in range(M):
        if code[c] < 8:
            indeg[c + off[code[c]]] += 1
    acc = w.copy()
    stack = np.empty(M, np.int32)
    top = 0
    for c in range(M):
        if indeg[c] == 0:
            stack[top] = c
            top += 1
    done = 0
    while top > 0:
        top -= 1
        c = np.int64(stack[top])
        done += 1
        k = code[c]
        if k < 8:
            p = c + off[k]
            acc[p] += acc[c]
            indeg[p] -= 1
            if indeg[p] == 0:
                stack[top] = p
                top += 1
    return acc, done


def accumulate_codes(code: np.ndarray, W: int, weight: np.ndarray) -> np.ndarray:
    """``weight`` (flat float64) accumulated down the packed tree ``code``
    (:func:`receiver_codes`); raises if the codes hold a cycle."""
    c = np.ascontiguousarray(code, dtype=np.uint8).reshape(-1)
    w = np.ascontiguousarray(weight, dtype=np.float64).reshape(-1)
    acc, done = _accumulate_codes_kernel(c, int(W), w)
    if done != c.size:
        raise ValueError(f"receiver codes hold a cycle: {c.size - done} cells never drained")
    return acc


@njit(cache=True)
def _exits_kernel(parent, pop_seq, W, i0, i1, j0, j1):
    M = parent.size
    ex = np.full(M, -1, np.int32)
    for q in range(pop_seq.size):
        c = np.int64(pop_seq[q])
        i = c // W
        j = c - i * W
        inside = i >= i0 and i < i1 and j >= j0 and j < j1
        p = np.int64(parent[c])
        if p < 0:
            ex[c] = -1 if inside else c
            continue
        pi = p // W
        pj = p - pi * W
        pin = pi >= i0 and pi < i1 and pj >= j0 and pj < j1
        if inside or pin or ex[p] != p:
            ex[c] = ex[p]
        else:
            ex[c] = c                 # outside, and its water never comes back in
    return ex


def window_exits(parent: np.ndarray, pop_seq: np.ndarray, W: int, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
    """int32 per cell (flat): -1 where the cell's water ends at a drain inside
    ``[i0, i1) x [j0, j1)``; else the cell outside that sub-window through
    which it leaves it for the last time (a cell outside whose own water
    never re-enters is its own exit).  ``pop_seq`` must list every receiver
    before its donors."""
    return _exits_kernel(np.ascontiguousarray(parent).reshape(-1), np.ascontiguousarray(pop_seq), int(W), int(i0), int(i1), int(j0), int(j1))


__all__ = ["receiver_codes", "accumulate_codes", "window_exits"]
