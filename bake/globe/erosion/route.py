"""Routing surface for particles: priority-flood + epsilon (Barnes et al.
2014) of the terrain surface so that every land cell drains monotonically
to the ocean.

PLAN 8.2 keeps lakes out of the global pass, but tectonic bedrock is full of
closed depressions and particles that die in pits never build the river
network the acceptance criteria ask for.  The kernel therefore takes an
optional ``route`` array: particles steer on ``max(route, surface)`` (a
lake is crossed on its flat-with-epsilon water surface towards the outlet)
while erosion/deposition still use the true terrain, so lakes silt up
naturally.  The flood is recomputed every ``erosion.flood_every``
iterations (the terrain changes slowly; between refreshes the true surface
wins wherever it rose above the stale route).

Cross-face neighbours come from ``grid.owner`` (interior cell owning each
extended cell), so the flood is seamless.  Window mode (``F = 1``) treats
mask-0 cells as outside and flat mask-0 halo cells as outlets only if the
caller marks them ocean (surface < 0).
"""
from __future__ import annotations

import heapq

import numpy as np
from numba import njit

from .particle import MASK_OUTSIDE, _D8


@njit(cache=True)
def priority_flood_eps(surface, mask, owner, H, N, eps):
    """Epsilon-filled surface (extended array, same shape as ``surface``).

    ``surface`` float64 (F, NE, NE); ``mask`` uint8 (0 = outside);
    ``owner`` int64 (F, NE, NE) flat index of the owning interior cell for
    every extended cell (identity on the interior, cross-face for halos —
    or identity everywhere for a single non-spherical window);
    ``eps`` the minimum drop per cell along the filled surface.

    Seeds are the interior cells with ``surface < 0`` (ocean) and, in
    window mode, interior cells whose neighbour is outside the array or
    outside the mask (they drain out of the window).  Halo cells keep their
    surface value; refresh them with a halo exchange afterwards.
    """
    F, NE, _ = surface.shape
    total = F * NE * NE
    sflat = surface.reshape(total)
    mflat = mask.reshape(total)
    oflat = owner.reshape(total)
    filled = surface.copy()
    fflat = filled.reshape(total)
    visited = np.zeros(total, dtype=np.uint8)
    heap = [(0.0, 0)]
    heap.pop()
    # Never expand into mask-0 cells or past the array border: in window
    # mode ``owner`` is the identity, so a flooded halo cell would otherwise
    # look at neighbours outside the array (no bounds checks under numba).
    for f in range(F):
        for ei in range(NE):
            for ej in range(NE):
                c = (f * NE + ei) * NE + ej
                if mflat[c] == MASK_OUTSIDE or ei == 0 or ej == 0 or ei == NE - 1 or ej == NE - 1:
                    visited[c] = 1
    for f in range(F):
        for ei in range(H, H + N):
            for ej in range(H, H + N):
                c = (f * NE + ei) * NE + ej
                if visited[c]:
                    continue
                seed = sflat[c] < 0.0
                if not seed:
                    for k in range(8):
                        ni = ei + _D8[k, 0]
                        nj = ej + _D8[k, 1]
                        n = oflat[(f * NE + ni) * NE + nj]
                        if mflat[n] == MASK_OUTSIDE:
                            seed = True
                            break
                        nf = n // (NE * NE)
                        rem = n - nf * NE * NE
                        nei = rem // NE
                        nej = rem - nei * NE
                        if nei < H or nei >= H + N or nej < H or nej >= H + N:
                            seed = True  # owner is a halo cell: no real neighbour (window edge)
                            break
                if seed:
                    visited[c] = 1
                    heapq.heappush(heap, (sflat[c], c))
    while len(heap) > 0:
        h, c = heapq.heappop(heap)
        f = c // (NE * NE)
        rem = c - f * NE * NE
        ei = rem // NE
        ej = rem - ei * NE
        for k in range(8):
            ni = ei + _D8[k, 0]
            nj = ej + _D8[k, 1]
            n = oflat[(f * NE + ni) * NE + nj]
            if visited[n]:
                continue
            visited[n] = 1
            v = sflat[n]
            if v < h + eps:
                v = h + eps
            fflat[n] = v
            heapq.heappush(heap, (v, n))
    return filled


def window_owner(F: int, NE: int) -> np.ndarray:
    """Identity owner table for a non-spherical window."""
    return np.arange(F * NE * NE, dtype=np.int64).reshape(F, NE, NE)


__all__ = ["priority_flood_eps", "window_owner"]
