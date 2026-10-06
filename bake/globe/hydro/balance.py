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
def _accumulate_endorheic(w, down, topo, outlet_lake, lake_ptr, lake_z, lake_cum, spill, lake_evap, level, outflow, wr, runoff):
    """Flow accumulation that solves each depression's balance as it passes
    its outlet.  Fills ``level``/``outflow`` per lake; returns ``acc``.

    ``wr`` is what of ``w`` runs off (``w`` itself without land
    evaporation): the balance is solved on the accumulation of ``wr``, and
    ``acc`` -- the rain upstream, which the rivers are drawn from -- passes a
    lake in the share of its inflow the lake lets through."""
    acc = w.copy()
    run = wr.copy()
    for k in range(topo.size):
        c = topo[k]
        L = outlet_lake[c]
        if L >= 0:
            inflow = run[c]
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
            if runoff:
                acc[c] = acc[c] * (outflow[L] / inflow) if inflow > 0.0 else 0.0
            else:
                acc[c] = outflow[L]            # wr is w: the outflow itself, to the bit
            run[c] = outflow[L]
        d = down[c]
        if d >= 0:
            acc[d] += acc[c]
            run[d] += run[c]
    return acc


#: second Legendre coefficient of the annual-mean insolation by latitude at Earth's obliquity
#: (North 1975): Q(lat) = Q0 (1 + s2 P2(sin lat)), the poles at 0.42 of the equator
INSOLATION_S2 = -0.477


def potential_evaporation(temperature_c: np.ndarray, latitude: np.ndarray, t_eq: float, t0: float) -> np.ndarray:
    """What open water can evaporate in a year, as a multiple of what it does
    at a sea-level cell on the equator (``hydro.pet_t0``): Hargreaves' law of
    temperature under the latitude's own sun,

        (T + t0) / (t_eq + t0) * Q(lat) / Q(0),   never below 0

    with ``t0`` = 17.8 C his constant and ``Q`` the annual-mean insolation.
    Same units as the climate's ``evap`` (1 at ``t_eq`` on the equator), which
    it replaces in the water balance; float32, the shape of ``temperature_c``
    (``latitude`` in radians, the same shape).

    ``evap`` is ``k_evap max(T, 0)``: nothing evaporates at freezing and a
    3 C cell loses a ninth of what a 28 C one does.  That is the kernel's
    particle decay and its ice line, and it is not what lakes do.  Open-water
    evaporation in mm a year, roughly as measured / this law / ``evap``'s,
    the scale set by Lake Chad (13 N, 28 C, ~2200):

    * the Caspian (42 N, 12 C): ~1000 / 1090 / 940;
    * Titicaca (16 S, 8 C, 3800 m): ~1700 / 1220 / 630;
    * Nam Co, Tibet (31 N, 0 C, 4700 m): ~900 / 750 / 0;
    * Lake Superior (47 N, 4 C): ~600 / 750 / 310;
    * Baikal (53 N, -1 C): ~400 / 530 / 0;
    * Great Bear Lake (66 N, -7 C): ~300 / 280 / 0.

    Cold comes two ways.  Ground cold for its latitude has little sun and
    loses little: the lake country of the shields keeps its water.  Ground
    cold for its height has the sun of its latitude: a dry plateau's basins
    are salt pans.  With ``evap`` both kept every drop: earth-v24 had lakes
    on 1.83 % of its land with under a quarter of the mean rain, against
    1.55 % of the rest, and on 13 % of a 1650 m plateau at 44 S that gets a
    tenth of it."""
    T = np.asarray(temperature_c, dtype=np.float64)
    x = np.sin(np.asarray(latitude, dtype=np.float64))
    q = (1.0 + INSOLATION_S2 * 0.5 * (3.0 * x * x - 1.0)) / (1.0 - 0.5 * INSOLATION_S2)
    return (np.maximum(T + float(t0), 0.0) / (float(t_eq) + float(t0)) * q).astype(np.float32)


def budyko_evaporation(aridity: np.ndarray) -> np.ndarray:
    """Share of the rain a land surface evaporates, by its aridity index
    (potential evaporation over rain): Budyko's (1974) curve, ``sqrt(a
    tanh(1 / a) (1 - exp(-a)))``.  0 where nothing can evaporate, 0.57 at an
    index of 0.7 (a rainforest), 0.93 at 2.5, 1 in a desert."""
    a = np.maximum(np.asarray(aridity, np.float64), 1e-9)
    return np.sqrt(a * np.tanh(1.0 / a) * (1.0 - np.exp(-a)))


def balance_lakes(surface, filled, ocean, order, down, topo, precip, evap, grid, lake_evap, land_evap=0.0):
    """Lower every closed depression to the level where inflow = evaporation.

    ``surface``/``filled``/``ocean``/``precip``/``evap`` are ``(6, N, N)``;
    ``order``/``down``/``topo`` flat ``(M,)`` from the flood and the D8
    routing.  Returns ``(water_surface, acc, info)`` with ``water_surface``
    ``(6, N, N)`` (``>= surface``, equal to it where the basin dried out),
    ``acc`` the flow accumulation of the same pass and ``info`` a dict of
    counts.  ``lake_evap <= 0`` keeps the spill-point fill and just runs the
    ordinary accumulation.

    ``land_evap`` > 0 (``hydro.land_evap``): the land evaporates too.  Until
    it did, all the rain a catchment received arrived at its lake, and the
    only water ever lost was off a lake's surface at ``lake_evap`` -- so a
    lake that silted up handed its whole inflow to the next basin down, and a
    plateau with a twentieth of the land's mean rain kept a lake of
    30,000 km2.  With it the potential evaporation is ``land_evap * evap``
    (the land-mean rain's depth at ``evap`` 1, a 28 C cell), a cell's rain
    runs off in the share Budyko's curve leaves (:func:`budyko_evaporation`),
    and open water loses the potential evaporation less what the land under
    it would have evaporated anyway.  ``lake_evap`` is not read then.  The
    returned ``acc`` stays the rain upstream, which the rivers are scaled to,
    less the share each lake keeps.
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
    if (float(lake_evap) <= 0.0 and float(land_evap) <= 0.0) or not dep.any():
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
    af = area / (float(grid.cell_size_m) ** 2)
    wr = w
    k_lake = float(lake_evap)
    if float(land_evap) > 0.0:
        rain = w / af                                  # depth, in the land-mean rain's
        pet = float(land_evap) * ev
        et = rain * budyko_evaporation(pet / np.maximum(rain, 1e-9))
        wr = np.ascontiguousarray((rain - et) * af)    # what runs off
        per = (pet[cells] - et[cells]) * af[cells]     # open water over what the land would have lost
        k_lake = 1.0
    else:
        per = ev[cells] * af[cells]
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
                                spill, k_lake, level, outflow, wr, float(land_evap) > 0.0)

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


__all__ = ["balance_lakes", "budyko_evaporation"]
