"""Glacial erosion: the process that makes lakes.

Fluvial erosion cannot leave a lake behind.  A river incises its outlet
sill from above and fills the basin with sediment from below, so a
landscape run toward fluvial steady state has no closed depressions at
all — measured on this world before this pass existed, 1 depression in
456,693 land cells, and injecting fractal roughness into the bedrock did
not help (75 seeded depressions were down to 9 within 60 iterations).

Earth is not lake-poor for the same reason it is not fluvially mature:
almost every one of its lakes was made by ice, and recently.  Ice deforms
under its own weight and flows *uphill* out of a basin, so it can excavate
a closed hollow no river could cut, and the spoil it carries is dumped at
the margin as a moraine that dams the valley behind it.  Both mechanisms
are here:

* **Overdeepening** — the bed under ice is lowered with no base-level
  constraint whatsoever.  That single omission is what creates closed
  basins; every other erosional process in this tool is bounded below by
  where the water can get out.
* **Moraine** — the excavated rock is not deleted.  It is deposited on the
  ice margin (active land cells adjacent to ice), in proportion to the ice
  arriving there, so valleys leaving an ice field get dammed.

Ice is where the mean annual temperature is at or below freezing.  The
kernel already carries that: ``evap`` is ``k_evap * max(T, 0)``
(docs/DEVELOPING.md), so ``evap <= 0`` is exactly ``T <= 0`` and no extra
field has to be threaded through the erosion contract.  Because the
temperature field already includes the lapse rate, this picks out both
high latitudes and high ground, which is where Earth's lake districts are.

Ice flux is taken from ``discharge``.  Under ice the same precipitation
falls and the same catchment feeds the same point — it simply travels as
ice rather than water — and ``discharge`` is already an accumulation of
precipitation that is correct across face seams, so it needs no separate
routing pass.

The pass is mass-conserving between the ice and its margin (the tests
check this), so it composes with ``hold_datum``: it moves rock around,
it does not add or remove any.
"""
from __future__ import annotations

from ..config import cell_units

import numpy as np

from . import particle as pk


def ice_mask(state, ice_evap: float = 0.0) -> np.ndarray:
    """Extended-array bool: active land cold enough to hold ice.

    ``evap`` is ``k_evap * max(T, 0)``, so ``ice_evap = 0`` is exactly the
    freezing line and a positive threshold is a warmer equilibrium-line
    altitude.  It is worth having as a knob rather than a constant: whether
    a world can have glacial lakes at all is decided here, and a world whose
    continents all land in warm latitudes has no land below freezing (seed
    22 of the small preset bottoms out at +1.64 C), so no setting of the
    other glacial parameters can give it a single lake.
    """
    return (state.mask == pk.MASK_ACTIVE) & (state.evap <= float(ice_evap)) & (state.surface() > 0.0)


def ice_depth(state, ice: np.ndarray, ramp: int) -> np.ndarray:
    """How far inside the ice each cell is, in cells, capped at ``ramp`` and
    normalised to [0, 1] — 0 on the margin ring, 1 at ``ramp`` cells in.

    This is what makes a *basin* rather than a deeper valley.  Erosion that
    simply scales with ice flux grows downstream, so it cuts a trough that
    still drains: deepest at the outlet.  Real glacial overdeepening tapers
    to nothing at the snout, where the ice thins and stops sliding, and
    leaves a rock lip across the valley — the deepest point sits *inside*
    the glacier, which is precisely a closed depression.  Shrinking the ice
    mask one ring at a time and counting how long a cell survives gives that
    taper directly.

    One halo exchange per ring, so a glacier spanning a face edge is
    measured correctly rather than being cut at the seam.
    """
    if ramp <= 0:
        return ice.astype(np.float64)
    inner = ice.copy()
    depth = ice.astype(np.float64)
    keep = state.mask.copy()
    for _ in range(int(ramp)):
        shrunk = inner.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                shrunk &= np.roll(np.roll(inner, di, axis=1), dj, axis=2)
        state.mask[...] = np.where(shrunk, np.uint8(200), np.uint8(0))
        state.exchange_halos()
        inner = state.mask == 200
        depth += inner
    state.mask[...] = keep
    state.exchange_halos()
    return np.clip(depth / float(ramp + 1), 0.0, 1.0)


def _margin(ice: np.ndarray, state) -> np.ndarray:
    """Active land cells that are not ice but touch it, on the extended
    array.  The halo is exchanged before this is called, so a margin that
    lies across a face edge is found correctly."""
    n = np.zeros_like(ice)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            n |= np.roll(np.roll(ice, di, axis=1), dj, axis=2)
    return n & ~ice & (state.mask == pk.MASK_ACTIVE) & (state.surface() > 0.0)


def carve(state, params) -> dict:
    """One glacial cycle.  Lowers ``height`` under ice and puts the spoil on
    the margin as sediment.  Returns per-pass statistics.

    The bed is lowered by ``glacial_rate * sqrt(q / disc_saturation)``,
    scaled by ``1 - 0.5 * hardness`` (hard rock resists the ice too) and
    capped at ``glacial_max`` per pass — and, unlike every other erosional
    term in this tool, *not* limited by the elevation of anywhere
    downstream.  That is the whole point: it is what leaves a hole.
    """
    ep = params.erosion
    rate = cell_units(ep, "glacial_rate", state.height_unit_m)
    cap = cell_units(ep, "glacial_max", state.height_unit_m)
    stats = {"ice_cells": 0, "carved": 0.0, "moraine_cells": 0, "max_carve": 0.0, "outwash": 0.0}
    if rate <= 0.0:
        return stats

    ice = ice_mask(state, float(ep.ice_evap))
    sticky = bool(getattr(ep, "glacial_sticky", False))
    prev = getattr(state, "ice_prev", None) if sticky else None
    if prev is not None:
        # A glacier does not stop being one because it has cut its bed below
        # the sea.  Without this, `ice_mask`'s "surface > 0" makes the ice
        # margin *the coastline* in cold lowlands; every pass carves the
        # outer steps below sea level, they drop out of the mask, and the
        # margin -- and the next pass's taper -- jumps inland, printing a
        # land/sea band per pass (docs/streaks-and-flats.md).  Sticky: a
        # cell glaciated before stays glaciated while it is still cold.
        cold = (state.mask == pk.MASK_ACTIVE) & (state.evap <= float(ep.ice_evap))
        ice = ice | (prev & cold)
    if not ice.any():
        return stats
    # halo exchange so a margin across a face edge is seen; the mask is
    # integer-valued so a nearest-neighbour exchange is exact
    tmp = state.mask.copy()
    state.mask[...] = np.where(ice, np.uint8(200), state.mask)
    state.exchange_halos()
    ice = state.mask == 200
    state.mask[...] = tmp
    if sticky:
        state.ice_prev = ice.copy()

    sat = max(float(ep.disc_saturation), 1e-9)
    q = np.maximum(state.discharge, 0.0) / sat
    taper = ice_depth(state, ice, int(ep.glacial_ramp))
    dz = rate * np.sqrt(q) * (1.0 - 0.5 * state.hardness) * taper
    np.clip(dz, 0.0, cap, out=dz)
    dz = np.where(ice, dz, 0.0)
    # never carve a cell below sea level: a fjord is as deep as this gets,
    # and dropping land into the sea would fight `hold_datum` every pass
    room = np.maximum(state.surface() + pk.DEP_FLOOR, 0.0)
    dz = np.minimum(dz, room)

    interior = state.interior
    carved = float(dz[interior].sum())
    if carved <= 0.0:
        return stats
    state.height -= dz

    # Spoil -> moraine on the margin, weighted by the ice arriving there.
    # Capped per cell at `iter_deposit`, the same limit the fluvial pass
    # obeys: a whole ice field's excavation shared between a handful of
    # margin cells would otherwise land as a single kilometre-high spike
    # (a stub world put 177 cell units on one cell before this cap).  What
    # the margin cannot take waits in `pending`, the per-cell stockpile the
    # next iteration re-injects as particle load — which is what glacial
    # outwash is: debris the ice dropped and the meltwater then spread.
    margin = _margin(ice, state)
    frac = float(ep.moraine_frac)
    parked = 0.0
    if margin.any() and frac > 0.0:
        w = np.where(margin, np.maximum(state.discharge, 0.0) + 1e-6, 0.0)
        wsum = float(w[interior].sum())
        if wsum > 0.0:
            add = (w / wsum) * (carved * frac)
            placed = np.minimum(add, cell_units(ep, "iter_deposit", state.height_unit_m))
            state.sediment += placed
            state.pending += add - placed
            parked = float((add - placed)[interior].sum())
    state.exchange_halos()

    stats.update(
        ice_cells=int(ice[interior].sum()),
        carved=carved,
        moraine_cells=int(margin[interior].sum()),
        max_carve=float(dz[interior].max()),
        outwash=parked,
    )
    return stats


__all__ = ["ice_mask", "carve"]
