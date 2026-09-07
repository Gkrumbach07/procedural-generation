"""D8 routing on the filled DEM, flow accumulation, channel network and
Strahler orders (PLAN.md section 9 steps 3-5).

Cell ids ``cid = (f*N + i)*N + j`` index flattened interior arrays; see
``globe.hydro.d8``.  All hot loops are numba; the channel graph assembly
(tens of thousands of nodes at N_c = 1024) is plain Python.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .d8 import (
    D8_DI,
    D8_DJ,
    INVALID,
    OCEAN,
    cid_fij,
    code_towards,
    downstream_table,
    neighbor_cid,
    step_length_m,
)

# --------------------------------------------------------------------------
# flow directions
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _flow_dir_kernel(filled, ocean, parent, owner, metric, N, H):
    M = filled.size
    out = np.full(M, OCEAN, dtype=np.uint8)
    for c in prange(M):
        if ocean[c]:
            continue
        f, i, j = cid_fij(c, N)
        hc = filled[c]
        g0 = metric[f, i + H, j + H, 0]
        g1 = metric[f, i + H, j + H, 1]
        g2 = metric[f, i + H, j + H, 2]
        best = -1
        best_s = 0.0
        for k in range(8):
            n = neighbor_cid(owner, N, H, f, i, j, k)
            hn = filled[n]
            if hn < hc:
                di = D8_DI[k]
                dj = D8_DJ[k]
                d2 = g0 * di * di + 2.0 * g1 * di * dj + g2 * dj * dj
                s = (hc - hn) / np.sqrt(d2)
                if s > best_s:
                    best_s = s
                    best = k
        if best < 0:
            # flat: drain towards the flood parent (popped earlier, same level)
            p = parent[c]
            if p >= 0:
                best = code_towards(owner, N, H, f, i, j, p)
        out[c] = best if best >= 0 else INVALID
    return out


def flow_directions(filled, ocean, parent, grid) -> np.ndarray:
    """D8 codes on the filled DEM: steepest descent (physical slope via
    ``grid.metric``) among strictly lower 8-neighbours; a cell with no lower
    neighbour (lake surface, flat) drains to its flood parent, which was
    popped earlier at the same level — so ``(filled, pop order)`` strictly
    decreases along every path and there are no cycles.

    ``filled`` (6, N, N) float32, ``ocean`` (6, N, N) bool, ``parent``
    (6*N*N,) int64 from :func:`priority_flood_sphere`.  Returns (6, N, N)
    uint8 (255 = ocean).  Raises if a land cell got no direction (only
    possible for a land cell the flood never reached).
    """
    N, H = grid.N, grid.H
    fd = _flow_dir_kernel(
        np.ascontiguousarray(filled, dtype=np.float32).reshape(-1),
        np.ascontiguousarray(ocean, dtype=np.bool_).reshape(-1),
        np.ascontiguousarray(parent, dtype=np.int64),
        grid.owner,
        grid.metric,
        N,
        H,
    )
    bad = int(np.count_nonzero(fd == INVALID))
    if bad:
        raise RuntimeError(f"{bad} land cells have no D8 direction (unreached by the flood)")
    return fd.reshape(6, N, N)


# --------------------------------------------------------------------------
# accumulation / topological order
# --------------------------------------------------------------------------
@njit(cache=True)
def _accumulate_kernel(weight, down, topo):
    """topo: cell ids upstream-first.  acc[c] = weight[c] + sum(acc upstream)."""
    acc = np.zeros(weight.size, dtype=np.float64)
    for t in range(topo.size):
        c = topo[t]
        acc[c] += weight[c]
        d = down[c]
        if d >= 0:
            acc[d] += acc[c]
    return acc


def accumulate(weight, down, topo) -> np.ndarray:
    """Flow accumulation of ``weight`` (flat (M,) or (6, N, N)) along
    ``down`` (M,) in the topological order ``topo`` (upstream first, e.g.
    ``FloodResult.pop_seq[::-1]`` or :func:`topological_order`).  Ocean
    cells (down == -1) accumulate nothing beyond what flows into them; the
    caller decides whether to keep those values.  Returns (M,) float64."""
    w = np.ascontiguousarray(weight, dtype=np.float64).reshape(-1)
    return _accumulate_kernel(w, np.ascontiguousarray(down, dtype=np.int64), np.ascontiguousarray(topo, dtype=np.int64))


@njit(cache=True)
def _topo_kernel(down, land):
    M = down.size
    indeg = np.zeros(M, dtype=np.int64)
    for c in range(M):
        if land[c]:
            d = down[c]
            if d >= 0:
                indeg[d] += 1
    queue = np.empty(M, dtype=np.int64)
    qt = 0
    for c in range(M):
        if land[c] and indeg[c] == 0:
            queue[qt] = c
            qt += 1
    qh = 0
    while qh < qt:
        c = queue[qh]
        qh += 1
        d = down[c]
        if d >= 0 and land[d]:
            indeg[d] -= 1
            if indeg[d] == 0:
                queue[qt] = d
                qt += 1
    return queue[:qt]


def topological_order(down, land) -> np.ndarray:
    """Kahn topological order (upstream first) of the land cells of a
    ``flow_dir`` graph given as ``down`` (M,) cell ids (-1 = ocean) and a
    ``land`` (M,) bool mask.  Deterministic (FIFO, index order).  Raises if
    the graph has a cycle (some land cell never reaches in-degree 0)."""
    d = np.ascontiguousarray(down, dtype=np.int64)
    l = np.ascontiguousarray(land, dtype=np.bool_).reshape(-1)
    topo = _topo_kernel(d, l)
    n_land = int(np.count_nonzero(l))
    if topo.size != n_land:
        raise RuntimeError(f"flow graph has a cycle: {n_land - topo.size} land cells unreachable in topological order")
    return topo


def upstream_counts(down, topo) -> np.ndarray:
    """Number of cells draining through each cell (including itself), int64
    (M,)."""
    return accumulate(np.ones(down.size, dtype=np.float64), down, topo).astype(np.int64)


@njit(cache=True)
def walk_to_sink(down, start, max_steps):
    """Follow ``down`` from ``start`` until a cell with ``down < 0`` (an
    ocean cell, or the last land cell if ocean cells are not in ``down``);
    returns (that cell, steps) — steps == max_steps means the walk did not
    terminate (cycle)."""
    c = start
    for s in range(max_steps):
        d = down[c]
        if d < 0:
            return c, s
        c = d
    return c, max_steps


# --------------------------------------------------------------------------
# channel network
# --------------------------------------------------------------------------
@njit(cache=True)
def _inflow_counts(chan, down):
    M = chan.size
    n_in = np.zeros(M, dtype=np.int32)
    for c in range(M):
        if chan[c]:
            d = down[c]
            if d >= 0:
                n_in[d] += 1
    return n_in


@njit(cache=True)
def _node_kinds(chan, down, n_in, lake, N):
    """0 none, 1 source, 2 junction, 3 outlet, 4 lake_in, 5 lake_out."""
    M = chan.size
    kind = np.zeros(M, dtype=np.int8)
    # upstream channel cell of single-input cells (for lake transitions)
    up1 = np.full(M, -1, dtype=np.int64)
    for c in range(M):
        if chan[c]:
            d = down[c]
            if d >= 0 and n_in[d] == 1:
                up1[d] = c
    for c in range(M):
        if not chan[c]:
            continue
        if down[c] < 0 or not chan[down[c]]:
            kind[c] = 3  # drains to the ocean (channels are downstream-closed on land)
        elif n_in[c] >= 2:
            kind[c] = 2
        elif n_in[c] == 0:
            kind[c] = 1
        else:
            u = up1[c]
            if lake[c] and not lake[u]:
                kind[c] = 4
            elif lake[u] and not lake[c]:
                kind[c] = 5
    return kind


@njit(cache=True)
def _walk_edges(node_cells, kind, down, cap):
    """For every non-outlet node walk downstream to the next node.  Returns
    (edge_from_node_cell, edge_to_node_cell, cell_offsets, cells)."""
    n_nodes = node_cells.size
    e_from = np.empty(n_nodes, dtype=np.int64)
    e_to = np.empty(n_nodes, dtype=np.int64)
    offs = np.zeros(n_nodes + 1, dtype=np.int64)
    cells = np.empty(cap, dtype=np.int64)
    ne = 0
    pos = 0
    for a in range(n_nodes):
        u = node_cells[a]
        if kind[u] == 3:
            continue
        e_from[ne] = u
        cells[pos] = u
        pos += 1
        x = down[u]
        while kind[x] == 0:
            cells[pos] = x
            pos += 1
            x = down[x]
        cells[pos] = x
        pos += 1
        e_to[ne] = x
        ne += 1
        offs[ne] = pos
    return e_from[:ne], e_to[:ne], offs[: ne + 1], cells[:pos]


@njit(cache=True)
def _edge_lengths(cells, offs, flow_dir_flat, metric, cell_size_m, N, H):
    ne = offs.size - 1
    out = np.zeros(ne, dtype=np.float64)
    for e in range(ne):
        s = 0.0
        for p in range(offs[e], offs[e + 1] - 1):
            c = cells[p]
            f, i, j = cid_fij(c, N)
            k = flow_dir_flat[c]
            if k < 8:
                s += step_length_m(metric, cell_size_m, H, f, i, j, k)
        out[e] = s
    return out


@njit(cache=True)
def _paint_orders(cells, offs, edge_order, out):
    ne = offs.size - 1
    for e in range(ne):
        o = edge_order[e]
        for p in range(offs[e], offs[e + 1]):
            c = cells[p]
            if o > out[c]:
                out[c] = o
    return out


KIND_NAMES = {1: "source", 2: "junction", 3: "outlet", 4: "lake_in", 5: "lake_out"}


def channel_network(flow_dir, flow_acc, threshold: float, grid, lake=None, down=None) -> tuple[list, list, np.ndarray]:
    """Extract the channel network (cells with ``flow_acc > threshold``),
    its junction/reach graph and Strahler orders.

    Returns ``(nodes, edges, cell_order)`` where ``nodes``/``edges`` are the
    ``graph/drainage.json`` records (docs/DEVELOPING.md) and ``cell_order``
    is a (6, N, N) int16 Strahler order per cell (0 = not a channel).

    * A node is a channel cell that is a source (no channel inflow), a
      junction (>= 2 channel inflows), an outlet (drains to the ocean) or a
      lake entry / exit (``lake`` mask given: the first lake cell on a
      channel / first non-lake cell after one).
    * An edge runs from a node downstream to the next node; ``cells`` lists
      every cell from the from-node to the to-node inclusive (junction
      cells therefore belong to all incident edges).  ``length_m`` sums the
      physical D8 step lengths, ``mean_discharge`` is the mean ``flow_acc``
      over the edge's cells.
    * Strahler order: a source edge has order 1; the edge leaving a node
      has order ``max`` of the inflowing orders, +1 if that maximum occurs
      at least twice.  Node ids are dense (0..), edges likewise; the graph is
      a forest rooted at the outlet nodes.

    Deterministic: nodes are numbered in cell-id order.
    """
    N, H = grid.N, grid.H
    fd = np.ascontiguousarray(flow_dir, dtype=np.uint8).reshape(-1)
    acc = np.ascontiguousarray(flow_acc, dtype=np.float32).reshape(-1)
    if down is None:
        down = downstream_table(fd.reshape(6, N, N), grid.owner, H)
    chan = (acc > threshold) & (fd != OCEAN)
    lk = np.zeros(chan.size, dtype=np.bool_) if lake is None else np.ascontiguousarray(lake, dtype=np.bool_).reshape(-1)
    n_in = _inflow_counts(chan, down)
    kind = _node_kinds(chan, down, n_in, lk, N)
    node_cells = np.nonzero(kind)[0].astype(np.int64)
    n_chan = int(np.count_nonzero(chan))
    e_from, e_to, offs, cells = _walk_edges(node_cells, kind, down, n_chan + 2 * node_cells.size + 1)
    lengths = _edge_lengths(cells, offs, fd, grid.metric, grid.cell_size_m, N, H)
    node_index = {int(c): k for k, c in enumerate(node_cells)}
    ne = e_from.size
    # Strahler by Kahn over the node graph
    in_edges: list[list[int]] = [[] for _ in node_cells]
    for e in range(ne):
        in_edges[node_index[int(e_to[e])]].append(e)
    out_edge = {node_index[int(e_from[e])]: e for e in range(ne)}
    edge_order = np.zeros(ne, dtype=np.int64)
    remaining = [len(l) for l in in_edges]
    ready = [k for k in range(len(node_cells)) if remaining[k] == 0]
    done = 0
    while ready:
        k = ready.pop()
        done += 1
        ins = in_edges[k]
        if ins:
            m = max(int(edge_order[e]) for e in ins)
            o = m + 1 if sum(1 for e in ins if edge_order[e] == m) >= 2 else m
        else:
            o = 1
        e = out_edge.get(k)
        if e is not None:
            edge_order[e] = o
            t = node_index[int(e_to[e])]
            remaining[t] -= 1
            if remaining[t] == 0:
                ready.append(t)
    if done != len(node_cells):
        raise RuntimeError("channel graph is not acyclic")
    cell_order = _paint_orders(cells, offs, edge_order, np.zeros(chan.size, dtype=np.int16))
    nodes = []
    for k, c in enumerate(node_cells):
        f, i, j = cid_fij(int(c), N)
        nodes.append({"id": k, "cell": [int(f), int(i), int(j)], "kind": KIND_NAMES[int(kind[c])], "acc": float(acc[c])})
    edges = []
    for e in range(ne):
        cc = cells[offs[e] : offs[e + 1]]
        fs = cc // (N * N)
        rem = cc - fs * N * N
        ii = rem // N
        jj = rem - ii * N
        edges.append(
            {
                "id": e,
                "from": node_index[int(e_from[e])],
                "to": node_index[int(e_to[e])],
                "order": int(edge_order[e]),
                "length_m": float(lengths[e]),
                "mean_discharge": float(acc[cc].mean()),
                "cells": np.stack([fs, ii, jj], axis=1).astype(int).tolist(),
            }
        )
    return nodes, edges, cell_order.reshape(6, N, N)


__all__ = [
    "flow_directions", "accumulate", "topological_order", "upstream_counts", "walk_to_sink",
    "channel_network", "KIND_NAMES",
]
