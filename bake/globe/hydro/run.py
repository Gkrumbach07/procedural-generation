"""Hydro stage driver (PLAN.md section 9): sea level, priority flood, D8
routing, flow accumulation, drainage tree, lakes.

Inputs (coarse): ``height``, ``sediment``, ``precip``.
Outputs: ``water_surface`` (f32, 0 on ocean), ``flow_dir`` (u8, 255 ocean),
``flow_acc`` (f32 accumulated precip volume), ``graph/drainage.json``,
``graph/lakes_coarse.json`` (``graph/lakes.json`` is derive's fine
version; consumers that run before derive read the coarse file).  With ``hydro.requantile_land_fraction`` the stage
also shifts ``height`` (not ``sediment``) so that exactly
``world.land_fraction`` of the coarse cells have ``surface >= 0`` and
writes ``height`` back (not listed in ``OUTPUTS`` — it is an upstream
field that must survive ``clear_outputs``).
"""
from __future__ import annotations

import time

import numpy as np

from ..field import FaceField
from .d8 import OCEAN, downstream_table
from .lakes import extract_lakes
from .priority_flood import priority_flood_sphere
from .routing import accumulate, channel_network, flow_directions

OUTPUTS = ["water_surface", "flow_dir", "flow_acc", "graph/drainage.json", "graph/lakes_coarse.json"]


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
    info: dict = {}

    # 1. sea level
    if hp.requantile_land_fraction:
        new_h, shift = requantile_height(h.interior, sed.interior, params.world.land_fraction)
        h.interior[...] = new_h
        h.exchange_halos()
        store.save_field(h)
        info["height_shift_m"] = float(shift)
        log(f"[hydro] re-quantiled height: shift {shift:+.2f} m so land fraction = {params.world.land_fraction}")
    surface = (h.interior + sed.interior).astype(np.float32)
    ocean = surface < 0.0
    n_land = int(np.count_nonzero(~ocean))
    info["land_fraction"] = n_land / M
    log(f"[hydro] land cells {n_land:,} / {M:,} ({n_land / M:.3%})")

    # 2. priority flood
    t = time.time()
    flood = priority_flood_sphere(surface, ocean, grid)
    info["t_flood_s"] = time.time() - t
    log(f"[hydro] priority flood: {flood.n_popped:,} cells in {info['t_flood_s']:.2f}s")
    filled = flood.filled
    depth = filled - surface
    depth[ocean] = 0.0

    # 3. D8 on the filled DEM
    t = time.time()
    fd = flow_directions(filled, ocean, flood.parent, grid)
    down = downstream_table(fd, grid.owner, H)
    info["t_route_s"] = time.time() - t

    # 4. accumulation (reverse pop order = upstream first)
    topo = flood.pop_seq[::-1]
    topo = topo[~ocean.reshape(-1)[topo]]
    acc = accumulate(precip.interior, down, topo)
    acc[ocean.reshape(-1)] = 0.0
    acc = acc.astype(np.float32).reshape(6, N, N)

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
    lakes, lake_labels = extract_lakes(lake, filled, flood.order, down, grid)
    info["t_lakes_s"] = time.time() - t
    info["n_lakes"] = len(lakes)
    info["lake_cells"] = int(np.count_nonzero(lake))
    lakes_json = {"lakes": lakes, "lake_min_depth": hp.lake_min_depth}
    store.write_json("graph/lakes_coarse.json", lakes_json)
    log(f"[hydro] lakes: {len(lakes)} ({info['lake_cells']:,} cells)")

    # outputs
    ws = filled.copy()
    ws[ocean] = 0.0
    store.save_field(FaceField.from_interior(grid, ws, name="water_surface"))
    store.save_field(FaceField.from_interior(grid, fd, name="flow_dir"))
    store.save_field(FaceField.from_interior(grid, acc, name="flow_acc"))
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


def render_hydro(surface, water_surface, cell_order, cell_size_m, lake_min_depth=0.5, minor=None):
    """Hillshade + lakes + channels drawn with width by Strahler order.
    ``surface``/``water_surface`` (6, N, N) float, ``cell_order`` (6, N, N)
    int; ``minor`` optional (6, N, N) bool mask of sub-threshold channels
    drawn faintly underneath (drawing aid only).  Returns (6, N, N, 3)
    uint8 [f, i, j]."""
    from ..viz import quicklook as ql

    img = ql.render_height(surface, 0.0, cell_size_m)
    lake = (water_surface - surface) > lake_min_depth
    lake &= surface >= 0
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
    img = render_hydro(surface, ws.interior, cell_order, grid.cell_size_m, params.hydro.lake_min_depth, minor=minor)
    return ql.save_image(path, img)
