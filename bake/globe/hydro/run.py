"""Hydro stage driver (PLAN.md section 9): sea level, priority flood, D8
routing, flow accumulation, drainage tree, lakes.

The sea is the *connected* body of water below sea level
(:func:`open_ocean`), so a closed basin under the waterline is land with a
lake in it rather than ocean; ``flow_dir == OCEAN`` is the mask every later
stage should read.

Inputs (coarse): ``height``, ``sediment``, ``precip``.
Outputs: ``water_surface`` (f32, 0 on ocean), ``flow_dir`` (u8, 255 ocean),
``flow_acc`` (f32 accumulated precip volume), ``graph/drainage.json``,
``graph/lakes_coarse.json`` (``graph/lakes.json`` is derive's fine
version; consumers that run before derive read the coarse file).  With ``hydro.requantile_land_fraction`` the stage
also shifts ``height`` (not ``sediment``) so that exactly the land fraction
erosion held (:func:`globe.erosion.maps.datum_land_fraction`: ``world.land_fraction``,
or in shelf mode the bedrock's own) of the coarse cells have ``surface >= 0`` and
writes ``height`` back (not listed in ``OUTPUTS`` — it is an upstream
field that must survive ``clear_outputs``).
"""
from __future__ import annotations

import time

import numpy as np

from ..field import FaceField
from .d8 import OCEAN, downstream_table
from .balance import balance_lakes
from .lakes import extract_lakes, label_components
from .priority_flood import priority_flood_sphere
from .routing import channel_network, flow_directions

OUTPUTS = ["water_surface", "flow_dir", "flow_acc", "marsh", "graph/drainage.json", "graph/lakes_coarse.json"]


def open_ocean(surface: np.ndarray, grid, min_fraction: float = 0.02) -> np.ndarray:
    """The sea: the cells below sea level that are *connected* to it,
    ``(6, N, N)`` bool.

    Being below sea level is not what makes a cell ocean; being joined to the
    ocean is.  Earth carries several million km² of dry land and inland sea
    under the waterline -- the Caspian depression, Qattara, the Dead Sea,
    Turpan, Death Valley -- and none of it is ocean, because a rim of higher
    ground stands between it and the water.  The same basins arise here
    whenever plate motion traps a piece of ocean floor inside a continent or
    a rift drops one below the waterline, and taking ``surface < 0`` as the
    sea called all of them ocean: measured on the first Earth-scale bake,
    **504 enclosed basins covering 6.9 M km², 4.5 % of the land**, the
    largest of them 5.8 M km² of trapped oceanic crust.

    A below-sea body is open ocean when it covers at least ``min_fraction``
    of the globe; the largest always counts, so a world whose sea is small
    still has one.  The rest are closed basins on land, which the priority
    flood below fills to their spill points and reports as lakes.  That is
    the Caspian: a landlocked sea in a basin the ocean cannot reach, with a
    lake surface rather than a sea surface.
    """
    below, labels, keep = below_sea_components(surface, grid, min_fraction)
    if labels is None:
        return np.zeros((6, grid.N, grid.N), dtype=bool)
    return (below & keep[np.maximum(labels, 0)]).reshape(6, grid.N, grid.N)


def below_sea_components(surface: np.ndarray, grid, min_fraction: float = 0.02):
    """The pieces :func:`open_ocean` decides between, kept as one function so
    that erosion and hydro cannot drift apart in what they call the sea.

    Returns ``(below, labels, keep)`` flat over the interior: ``below`` the
    cells under the waterline, ``labels`` their connected component (-1
    elsewhere, ``None`` when nothing is below), ``keep[label]`` True for the
    components that are open ocean.  Everything below with ``keep`` False is
    a closed basin -- land with standing water in it, not sea.
    """
    N, H = grid.N, grid.H
    below = np.ascontiguousarray(np.asarray(surface) < 0.0).reshape(-1)
    if not below.any():
        return below, None, None
    labels, n = label_components(below, np.zeros(below.size, dtype=np.float32), grid.owner, N, H)
    area = grid.interior_cell_area.reshape(-1).astype(np.float64)
    per = np.bincount(labels[below], weights=area[below], minlength=max(n, 1))
    keep = per >= float(min_fraction) * float(area.sum())
    keep[int(np.argmax(per))] = True
    return below, labels, keep


def requantile_height(height: np.ndarray, sediment: np.ndarray, land_fraction: float) -> tuple[np.ndarray, float]:
    """Shift ``height`` so that exactly ``round(land_fraction * M)`` of the
    ``M`` cells have ``height + sediment >= 0`` (cell count, like the
    tectonics contract; ties/float rounding can move it by a cell or two).
    Returns (new height float32, shift applied in metres)."""
    surface = (height.astype(np.float32) + sediment.astype(np.float32)).reshape(-1)
    M = surface.size
    n_land = int(round(float(land_fraction) * M))
    n_land = min(max(n_land, 0), M)
    if n_land == 0:
        q = float(np.max(surface)) + 1.0
    elif n_land == M:
        q = float(np.min(surface))
    else:
        k = M - n_land  # index of the lowest land cell in sorted order
        q = float(np.partition(surface, k)[k])
    new_surface = surface.reshape(height.shape) - np.float32(q)
    return (new_surface - sediment.astype(np.float32)).astype(np.float32), q


def river_threshold_volume(precip_interior: np.ndarray, land: np.ndarray, river_threshold: float) -> float:
    """``hydro.river_threshold`` is in cells of accumulation; convert to a
    ``flow_acc`` volume using the mean precip volume over land cells."""
    p = precip_interior[land]
    mean_p = float(p.mean()) if p.size else 1.0
    return float(river_threshold) * mean_p


def run(store, params, log=print) -> dict:
    grid = params.coarse_grid()
    N, H = grid.N, grid.H
    M = 6 * N * N
    hp = params.hydro
    t0 = time.time()
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    precip = store.load_field("precip", grid)
    evap = store.load_field("evap", grid)
    info: dict = {}

    # 1. sea level
    if hp.requantile_land_fraction:
        # the fraction erosion held (erosion.maps.datum_land_fraction)
        from ..erosion.maps import datum_land_fraction
        bed = store.load_field("bedrock", grid).interior if params.tectonics.shelf_fraction > 0 else None
        target = datum_land_fraction(params, bed)
        new_h, shift = requantile_height(h.interior, sed.interior, target)
        h.interior[...] = new_h
        h.exchange_halos()
        store.save_field(h)
        info["height_shift_m"] = float(shift)
        info["land_fraction_target"] = target
        log(f"[hydro] re-quantiled height: shift {shift:+.2f} m so land fraction = {target:.4f}")
    surface = (h.interior + sed.interior).astype(np.float32)
    # the sea is what the sea is connected to; a closed basin below sea level
    # is land with a lake in it (see `open_ocean`)
    ocean = open_ocean(surface, grid, hp.ocean_min_fraction)
    closed = int(np.count_nonzero((surface < 0.0) & ~ocean))
    n_land = int(np.count_nonzero(~ocean))
    info["land_fraction"] = n_land / M
    info["closed_basin_cells"] = closed
    log(f"[hydro] land cells {n_land:,} / {M:,} ({n_land / M:.3%}); "
        f"{closed:,} of them closed basins below sea level")

    # 2. priority flood
    t = time.time()
    flood = priority_flood_sphere(surface, ocean, grid)
    info["t_flood_s"] = time.time() - t
    log(f"[hydro] priority flood: {flood.n_popped:,} cells in {info['t_flood_s']:.2f}s")
    filled = flood.filled

    # 3. D8 on the filled DEM
    t = time.time()
    fd = flow_directions(filled, ocean, flood.parent, grid)
    down = downstream_table(fd, grid.owner, H)
    info["t_route_s"] = time.time() - t

    # 4. accumulation (reverse pop order = upstream first), which is also
    # where each closed basin settles on the level its inflow can sustain
    # against evaporation instead of filling to its rim (hydro/balance.py)
    t = time.time()
    topo = flood.pop_seq[::-1]
    topo = topo[~ocean.reshape(-1)[topo]]
    water, acc, bal = balance_lakes(surface, filled, ocean, flood.order, down, topo,
                                    precip.interior, evap.interior, grid, hp.lake_evap)
    info["t_balance_s"] = time.time() - t
    info["lake_balance"] = bal
    depth = water - surface
    depth[ocean] = 0.0
    # standing water too shallow to be a lake is marsh (hydro.marsh_depth): dry in the water
    # surface every later stage reads, and marked for derive's wetland.  The routing above is
    # the flood's and does not change: its rivers cross a marsh as they cross a plain
    marsh = (depth > hp.lake_min_depth) & (depth <= float(getattr(hp, "marsh_depth", 0.0))) & ~ocean
    if marsh.any():
        water = np.where(marsh, surface.astype(water.dtype), water)
        depth[marsh] = 0.0
    info["marsh_cells"] = int(np.count_nonzero(marsh))
    acc[ocean.reshape(-1)] = 0.0
    acc = acc.astype(np.float32).reshape(6, N, N)
    if hp.lake_evap > 0:
        log(f"[hydro] endorheic balance: {bal['depressions']:,} depressions -> "
            f"{bal['overflowing']:,} overflowing, {bal['closed']:,} closed, {bal['dry']:,} dry; "
            f"cells under water {bal['cells_spill']:,} -> {bal['cells_balanced']:,}; "
            f"drawdown median {bal['drawdown_m_median']:.0f} m max {bal['drawdown_m_max']:.0f} m "
            f"({info['t_balance_s']:.2f}s)")

    # 5. channels + drainage tree
    thr = river_threshold_volume(precip.interior, ~ocean, hp.river_threshold)
    lake = depth > hp.lake_min_depth
    t = time.time()
    nodes, edges, cell_order = channel_network(fd, acc, thr, grid, lake=lake, down=down)
    info["t_graph_s"] = time.time() - t
    info["river_threshold_volume"] = thr
    info["n_channel_cells"] = int(np.count_nonzero(cell_order > 0))
    info["n_nodes"] = len(nodes)
    info["n_edges"] = len(edges)
    info["max_strahler"] = int(cell_order.max()) if len(edges) else 0
    store.write_json("graph/drainage.json", {"nodes": nodes, "edges": edges, "river_threshold_volume": thr})
    log(f"[hydro] channels: {info['n_channel_cells']:,} cells, {len(nodes)} nodes, {len(edges)} reaches, max order {info['max_strahler']}")

    # 6. lakes
    t = time.time()
    lakes, lake_labels = extract_lakes(lake, water, flood.order, down, grid)
    info["t_lakes_s"] = time.time() - t
    info["n_lakes"] = len(lakes)
    info["lake_cells"] = int(np.count_nonzero(lake))
    lakes_json = {"lakes": lakes, "lake_min_depth": hp.lake_min_depth}
    store.write_json("graph/lakes_coarse.json", lakes_json)
    log(f"[hydro] lakes: {len(lakes)} ({info['lake_cells']:,} cells); marsh, water no deeper than {float(getattr(hp, 'marsh_depth', 0.0)):g} m: {info['marsh_cells']:,} cells")

    # outputs
    ws = water.copy()
    ws[ocean] = 0.0
    store.save_field(FaceField.from_interior(grid, ws, name="water_surface"))
    store.save_field(FaceField.from_interior(grid, fd, name="flow_dir"))
    store.save_field(FaceField.from_interior(grid, acc, name="flow_acc"))
    store.save_field(FaceField.from_interior(grid, marsh.astype(np.uint8), name="marsh", exchange=False))
    info["t_total_s"] = time.time() - t0
    return info


# --------------------------------------------------------------------------
# quicklook
# --------------------------------------------------------------------------
def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r cells within faces (drawing only)."""
    out = mask.copy()
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            sh = np.roll(np.roll(mask, di, axis=1), dj, axis=2)
            if di > 0:
                sh[:, :di, :] = False
            elif di < 0:
                sh[:, di:, :] = False
            if dj > 0:
                sh[:, :, :dj] = False
            elif dj < 0:
                sh[:, :, dj:] = False
            out |= sh
    return out


def render_hydro(surface, water_surface, cell_order, cell_size_m, lake_min_depth=0.5, minor=None, ocean=None):
    """Hillshade + lakes + channels drawn with width by Strahler order.
    ``surface``/``water_surface`` (6, N, N) float, ``cell_order`` (6, N, N)
    int; ``minor`` optional (6, N, N) bool mask of sub-threshold channels
    drawn faintly underneath (drawing aid only); ``ocean`` the sea mask
    (default ``surface < 0``).  Returns (6, N, N, 3) uint8 [f, i, j]."""
    from ..viz import quicklook as ql

    img = ql.render_height(surface, 0.0, cell_size_m)
    lake = (water_surface - surface) > lake_min_depth
    lake &= (surface >= 0) if ocean is None else ~np.asarray(ocean, dtype=bool)
    img = ql.overlay(img, lake, (70, 130, 230), 0.9)
    if minor is not None:
        img = ql.overlay(img, minor & (surface >= 0), (90, 140, 230), 0.45)
    omax = int(cell_order.max()) if cell_order.size else 0
    for o in range(1, omax + 1):
        m = cell_order >= o
        r = 0 if o <= 2 else (1 if o <= 4 else 2)
        if r:
            m = _dilate(m, r)
        t = (o - 1) / max(omax - 1, 1)
        col = (int(60 - 50 * t), int(120 - 90 * t), 255)
        img = ql.overlay(img, m, col, 0.75 + 0.25 * t)
    return img


def quicklook(store, params, path):
    from ..viz import quicklook as ql

    grid = params.coarse_grid()
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    ws = store.load_field("water_surface", grid)
    fd = store.load_field("flow_dir", grid)
    acc = store.load_field("flow_acc", grid)
    precip = store.load_field("precip", grid)
    surface = (h.interior + sed.interior).astype(np.float32)
    land = fd.interior != OCEAN
    thr = river_threshold_volume(precip.interior, land, params.hydro.river_threshold)
    lake = (ws.interior - surface) > params.hydro.lake_min_depth
    lake &= land
    _, _, cell_order = channel_network(fd.interior, acc.interior, thr, grid, lake=lake)
    minor = (acc.interior > 0.25 * thr) & land  # faint sub-threshold channels (drawing only)
    img = render_hydro(surface, ws.interior, cell_order, grid.cell_size_m, params.hydro.lake_min_depth,
                       minor=minor, ocean=~land)
    return ql.save_image(path, img)
