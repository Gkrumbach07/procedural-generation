"""Evaporation-limited lake levels (the endorheic balance).

The priority flood fills every closed depression to its **spill point**,
which is the level a lake reaches when nothing removes water from it.  Real
closed basins do not get there: they settle where the inflow they receive
equals what evaporates off the water surface, and that level can be a long
way below the rim.  The Caspian is the standard case -- it is the largest
lake on Earth and its surface stands 28 m *below* sea level, inside a basin
whose rim is well above it.  Filling to the spill point instead is why the
lakes in docs/plates-rifts-and-water.md came out so large (coarse lake cells
51,419 -> 204,806 when closed basins stopped being called ocean).

The balance, per depression, with ``L`` the trial water level:

    inflow  =  the flow accumulation arriving at the depression
    loss(L) =  hydro.lake_evap * sum over the cells under L of
               evap[c] * area[c] / cell_size_m**2

``precip`` is a volume per cell normalised to a land mean of 1 and ``evap``
is the dimensionless ``k_evap * max(T, 0)`` (~1 at a warm sea-level cell),
so the two are commensurable and ``lake_evap`` is one number: **open-water
evaporation at 28 C in units of the land-mean precipitation depth**.  Rain
falling on the lake is already part of ``inflow`` (the lake's own cells are
in its catchment), so the balance is against *gross* evaporation.

``loss`` is non-decreasing in ``L`` -- raising the water only ever wets more
cells -- so the balance level is found by sorting the depression's cells by
elevation and walking the cumulative curve, which is the basin's hypsometric
curve.  Three outcomes:

* ``loss(spill) < inflow``: even at full pool the basin cannot evaporate
  what arrives, so it overflows.  The level stays at the spill point and the
  surplus goes downstream -- **this is the current behaviour and the common
  case**, and it must not regress.
* ``loss(L*) == inflow`` for some ``L* < spill``: a closed lake at ``L*``,
  no outflow.
* ``inflow <= 0``: a dry basin.

Cascades come out right because the whole thing rides on *one* accumulation
pass in topological order (upstream first, :func:`globe.hydro.routing.accumulate`
walks the same order).  A depression's outlet is the first of its cells the
flood popped and therefore the last in the upstream-first order, so when the
pass reaches it every drop the basin receives has arrived; what it writes
back is the outflow, which is zero for a closed lake.  A lake that stops
overflowing therefore stops supplying everything below it in the same pass,
with no iteration.

What this does not do: ``flow_dir`` still points out of a closed lake's
outlet, because it is computed on the filled DEM and has to stay acyclic.
``flow_acc`` is what says the river is not there -- it is 0 below the outlet
of a closed lake.  The same applies inside a basin that has shrunk: the
exposed bed keeps the flow directions the fill gave it, which run to the
lake that is left, and that is where the water goes.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .lakes import label_components


@njit(cache=True)
def _accumulate_endorheic(w, down, topo, outlet_lake, lake_ptr, lake_z, lake_cum, spill, lake_evap, level, outflow):
    """Flow accumulation that solves each depression's balance as it passes
    its outlet.  Fills ``level``/``outflow`` per lake; returns ``acc``."""
    acc = w.copy()
    for k in range(topo.size):
        c = topo[k]
        L = outlet_lake[c]
        if L >= 0:
            inflow = acc[c]
            lo = lake_ptr[L]
            hi = lake_ptr[L + 1]
            if inflow <= 0.0:
                level[L] = lake_z[lo]      # dry: the floor, no water
                outflow[L] = 0.0
            else:
                need = inflow / lake_evap
                i = np.searchsorted(lake_cum[lo:hi], need)
                if i >= hi - lo:
                    # even the full pool cannot evaporate the inflow
                    level[L] = spill[L]
                    outflow[L] = inflow - lake_evap * lake_cum[hi - 1]
                else:
                    level[L] = lake_z[lo + i]
                    outflow[L] = 0.0
            acc[c] = outflow[L]
        d = down[c]
        if d >= 0:
            acc[d] += acc[c]
    return acc


def balance_lakes(surface, filled, ocean, order, down, topo, precip, evap, grid, lake_evap):
    """Lower every closed depression to the level where inflow = evaporation.

    ``surface``/``filled``/``ocean``/``precip``/``evap`` are ``(6, N, N)``;
    ``order``/``down``/``topo`` flat ``(M,)`` from the flood and the D8
    routing.  Returns ``(water_surface, acc, info)`` with ``water_surface``
    ``(6, N, N)`` (``>= surface``, equal to it where the basin dried out),
    ``acc`` the flow accumulation of the same pass and ``info`` a dict of
    counts.  ``lake_evap <= 0`` keeps the spill-point fill and just runs the
    ordinary accumulation.
    """
    N, H = grid.N, grid.H
    M = 6 * N * N
    surf = np.ascontiguousarray(surface, dtype=np.float32).reshape(-1)
    fill = np.ascontiguousarray(filled, dtype=np.float32).reshape(-1)
    oc = np.ascontiguousarray(ocean, dtype=np.bool_).reshape(-1)
    w = np.ascontiguousarray(precip, dtype=np.float64).reshape(-1)
    down = np.ascontiguousarray(down, dtype=np.int64)
    topo = np.ascontiguousarray(topo, dtype=np.int64)
    dep = (fill > surf) & ~oc
    # every key is always present: the stage logs and `manifest.json` read
    # this dict, and a world with no depression at all is a normal case
    info = {"lake_evap": float(lake_evap), "depressions": 0, "overflowing": 0, "closed": 0, "dry": 0,
            "drawdown_m_median": 0.0, "drawdown_m_max": 0.0,
            "cells_spill": int(np.count_nonzero(dep)), "cells_balanced": int(np.count_nonzero(dep))}
    if float(lake_evap) <= 0.0 or not dep.any():
        from .routing import accumulate
        return fill.reshape(6, N, N).copy(), accumulate(w, down, topo), info

    # one label per depression: `label_components` joins only cells with an
    # identical level, and a depression is flat at its spill point, so this
    # is the same partition `lakes.extract_lakes` reports
    lab, n = label_components(np.ascontiguousarray(dep), fill, grid.owner, N, H)
    cells = np.flatnonzero(lab >= 0)
    cells = cells[np.lexsort((surf[cells], lab[cells]))]   # by lake, then by depth
    lab_s = lab[cells].astype(np.int64)
    lake_ptr = np.searchsorted(lab_s, np.arange(n + 1)).astype(np.int64)
    lake_z = surf[cells].astype(np.float64)
    area = grid.interior_cell_area.reshape(-1).astype(np.float64)
    ev = np.ascontiguousarray(evap, dtype=np.float64).reshape(-1)
    # evaporative demand of one cell, in the same volume units as `precip`
    # (which climate normalises as depth x cell_area / cell_size_m**2)
    per = ev[cells] * area[cells] / (float(grid.cell_size_m) ** 2)
    csum = np.cumsum(per)
    before = np.zeros(n, dtype=np.float64)
    before[1:] = csum[lake_ptr[1:n] - 1]
    lake_cum = np.ascontiguousarray(csum - before[lab_s])

    # the outlet is the first cell of the lake the flood popped; `order` is a
    # unique rank, so the minimum picks exactly one cell
    ordv = np.asarray(order, dtype=np.int64)[cells]
    best = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(best, lab_s, ordv)
    sel = ordv == best[lab_s]
    outlet = np.zeros(n, dtype=np.int64)
    outlet[lab_s[sel]] = cells[sel]
    outlet_lake = np.full(M, -1, dtype=np.int64)
    outlet_lake[outlet] = np.arange(n, dtype=np.int64)
    spill = fill[outlet].astype(np.float64)

    level = np.zeros(n, dtype=np.float64)
    outflow = np.zeros(n, dtype=np.float64)
    acc = _accumulate_endorheic(w, down, topo, outlet_lake, lake_ptr, lake_z, lake_cum,
                                spill, float(lake_evap), level, outflow)

    ws = surf.astype(np.float32).copy()
    lv = level[lab_s].astype(np.float32)
    np.maximum.at(ws, cells, lv)          # a cell above the balance level dries out
    ws[oc] = fill[oc]
    closed = outflow <= 0.0
    wet = level > lake_z[lake_ptr[:n]]    # holds more than its single lowest cell
    info.update(
        depressions=int(n),
        overflowing=int(np.count_nonzero(~closed)),
        closed=int(np.count_nonzero(closed & wet)),
        dry=int(np.count_nonzero(closed & ~wet)),
        drawdown_m_median=float(np.median(spill - level)) if n else 0.0,
        drawdown_m_max=float((spill - level).max()) if n else 0.0,
        cells_spill=int(np.count_nonzero(fill[cells] > surf[cells])),
        cells_balanced=int(np.count_nonzero(lv > surf[cells])),
    )
    return ws.reshape(6, N, N), acc, info


__all__ = ["balance_lakes"]
