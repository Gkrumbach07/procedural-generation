"""Watershed partition stage (PLAN.md section 10.1, docs/DEVELOPING.md).

Pipeline:

1. **Outlets**: a land cell is a *cut* cell if its downstream cell is
   ocean.  ``basin = cells draining to the same cut cell`` on the
   cross-face flow graph (``downstream_table`` follows ``flow_dir`` across
   cube edges), so a basin is a drainage basin whatever faces its cells lie
   on; a channel crossing an edge stays one basin (docs/cross-face-basins.md).
2. **Split** every basin larger than ``basin_max_cells`` at the cell whose
   upstream sub-tree (inside the basin) is closest to half the basin's
   area; the sub-tree becomes a new basin with that cell as outlet
   (Pfafstetter-style), the remainder keeps the id.  Lake cells are avoided
   as split points when any other candidate exists.
3. **Merge** every basin smaller than ``basin_min_cells`` into the
   D8-neighbouring basin sharing the longest boundary (same-face and
   cross-face contacts count alike), smallest basins first, provided the
   union stays ``<= basin_max_cells`` and no drainage cycle between basins
   results.  Micro-basins along a coast therefore agglomerate into coastal
   strips or join the basin behind them.  Basins that cannot be merged
   (islands smaller than the minimum, or every neighbour too large) stay
   undersized and carry an ``undersized_reason``.
4. Ids are renumbered densely in (outlet face, outlet i, outlet j) order.

Hierarchy: ``parent`` is the basin a basin drains into (``downstream_basin``;
-1 for coastal roots), so the hierarchy is the drainage tree.  Endorheic
basins do not exist after priority flood (every lake spills), so every
root drains to the ocean.

Faces: a basin's cells on one face are a *piece* (``pieces`` in the
record: ``face``, ``bbox``, ``area_cells``, ``tiles`` per face); refine
runs one job per piece, and ``face`` / ``bbox`` / ``tiles`` at the top
level are the outlet face's piece.  A cell whose downstream cell lies on
another face is an ``exit`` of kind ``"face"`` of its piece (still inside
the basin): refine's per-face flood drains there.

Outputs: ``basin_id`` (i32, -1 ocean), ``graph/basins.json`` (DEVELOPING.md
schema plus ``outlet_downstream``, ``exits``, ``exit_kinds``, ``pieces``,
``faces``, ``undersized_reason``, ``basin_max_cells`` / ``basin_min_cells``).
"""
from __future__ import annotations

import heapq
import time

import numpy as np
from numba import njit

from ..field import FaceField
from .d8 import D8_DI, D8_DJ, OCEAN, cid_fij, downstream_table, is_neighbor, neighbor_cid
from .routing import channel_network, topological_order
from .run import river_threshold_volume

OUTPUTS = ["basin_id", "graph/basins.json"]


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@njit(cache=True)
def _label_outlets(down, topo, cut):
    """label[c] = cell id of the cut cell reached by following ``down`` from c
    (stopping at the first cut cell).  ``topo`` upstream-first."""
    M = down.size
    label = np.full(M, -1, dtype=np.int64)
    for t in range(topo.size - 1, -1, -1):  # downstream first
        c = topo[t]
        if cut[c]:
            label[c] = c
        else:
            label[c] = label[down[c]]
    return label


@njit(cache=True)
def _subtree_counts(cells, down, label, B):
    """cells: basin B's cells sorted upstream-first.  Returns up-counts
    (per position) restricted to the basin."""
    n = cells.size
    up = np.ones(n, dtype=np.int64)
    pos = {}
    for p in range(n):
        pos[cells[p]] = p
    for p in range(n):
        d = down[cells[p]]
        if d >= 0 and label[d] == B:
            up[pos[d]] += up[p]
    return up


@njit(cache=True)
def _mark_subtree(cells, down, label, B, split_cell):
    """Boolean per position: cell drains through ``split_cell`` inside B
    (including split_cell itself).  cells sorted upstream-first."""
    n = cells.size
    pos = {}
    for p in range(n):
        pos[cells[p]] = p
    insub = np.zeros(n, dtype=np.bool_)
    for p in range(n - 1, -1, -1):  # downstream first
        c = cells[p]
        if c == split_cell:
            insub[p] = True
        else:
            d = down[c]
            if d >= 0 and label[d] == B and insub[pos[d]]:
                insub[p] = True
    return insub


@njit(cache=True)
def _boundary_pairs(label3):
    """Same-face neighbouring cell pairs with different labels (>= 0):
    returns (a, b, w) with a < b and weight 2 for a shared edge (4-neighbour)
    and 1 for a diagonal contact, so the summed weight per basin pair ranks
    'longest shared boundary' with diagonal touches as tie-breakers."""
    F, N, _ = label3.shape
    cap = F * N * N * 4
    a = np.empty(cap, dtype=np.int64)
    b = np.empty(cap, dtype=np.int64)
    w = np.empty(cap, dtype=np.int64)
    n = 0
    for f in range(F):
        for i in range(N):
            for j in range(N):
                x = label3[f, i, j]
                if x < 0:
                    continue
                for di, dj, wt in ((1, 0, 2), (0, 1, 2), (1, 1, 1), (1, -1, 1)):
                    i2 = i + di
                    j2 = j + dj
                    if i2 < 0 or i2 >= N or j2 < 0 or j2 >= N:
                        continue
                    y = label3[f, i2, j2]
                    if y >= 0 and y != x:
                        a[n] = min(x, y)
                        b[n] = max(x, y)
                        w[n] = wt
                        n += 1
    return a[:n], b[:n], w[:n]


@njit(cache=True)
def _cross_face_boundary_pairs(label3, owner, N, H):
    """Cross-face counterpart of :func:`_boundary_pairs`: label pairs of
    D8-adjacent land cells on *different* faces (only face-border cells
    have such neighbours), weight 2 for a shared edge and 1 for a diagonal
    contact, so a sliver on one face ranks its neighbour across the edge
    like one beside it.  A contact is counted once, from its lower cell id
    (the cross-face neighbour map is symmetric on the cubed sphere; a
    one-way link is counted from the side that sees it)."""
    NN = N * N
    cap = 6 * 4 * N * 8
    a = np.empty(cap, dtype=np.int64)
    b = np.empty(cap, dtype=np.int64)
    w = np.empty(cap, dtype=np.int64)
    n = 0
    for f in range(6):
        for i in range(N):
            for j in range(N):
                if i != 0 and i != N - 1 and j != 0 and j != N - 1:
                    continue
                x = label3[f, i, j]
                if x < 0:
                    continue
                c = (f * N + i) * N + j
                for k in range(8):
                    q = neighbor_cid(owner, N, H, f, i, j, k)
                    f2 = q // NN
                    if f2 == f:
                        continue
                    r = q - f2 * NN
                    i2 = r // N
                    j2 = r - i2 * N
                    y = label3[f2, i2, j2]
                    if y < 0 or y == x:
                        continue
                    if q < c and is_neighbor(owner, N, H, f2, i2, j2, c):
                        continue  # counted from the other side
                    a[n] = min(x, y)
                    b[n] = max(x, y)
                    w[n] = 2 if (D8_DI[k] == 0 or D8_DJ[k] == 0) else 1
                    n += 1
    return a[:n], b[:n], w[:n]


# --------------------------------------------------------------------------
# partition
# --------------------------------------------------------------------------
class Partition:
    """Result of :func:`partition`: ``basin_id`` (6, N, N) int32 and the
    per-basin records (list of dicts, ``graph/basins.json`` schema)."""

    def __init__(self, basin_id: np.ndarray, basins: list[dict], info: dict):
        self.basin_id = basin_id
        self.basins = basins
        self.info = info


def partition(flow_dir, grid, basin_max_cells: int, basin_min_cells: int, lake=None, cell_order=None, R: int = 1, T: int = 1, log=print) -> Partition:
    """Watershed partition of a ``flow_dir`` field (see module docstring).

    ``flow_dir`` (6, N, N) uint8; ``lake`` (6, N, N) bool (cells avoided as
    split points); ``cell_order`` (6, N, N) Strahler order per cell (for the
    ``order`` at each outlet; 0 if None); ``R``, ``T`` for the LOD-0 tile
    lists (fine grid ``N*R``, tiles of ``T`` fine cells).
    """
    N, H = grid.N, grid.H
    NN = N * N
    M = 6 * NN
    fd = np.ascontiguousarray(flow_dir, dtype=np.uint8)
    down = downstream_table(fd, grid.owner, H)
    land = fd.reshape(-1) != OCEAN
    topo = topological_order(down, land)  # upstream first; raises on cycles
    topo_pos = np.full(M, -1, dtype=np.int64)
    topo_pos[topo] = np.arange(topo.size)
    dn = np.maximum(down, 0)
    cut = land & ((down < 0) | ~land[dn])  # outlets: the last land cell before the sea, on whatever face
    label = _label_outlets(down, topo, cut)  # cut-cell id per cell, -1 ocean
    lk = np.zeros(M, dtype=bool) if lake is None else np.ascontiguousarray(lake, dtype=bool).reshape(-1)
    info = {"n_initial": int(cut.sum())}

    # ---- split ------------------------------------------------------------
    # basin bookkeeping: outlet cell id per basin id (basin ids are cell ids
    # of their outlet for the initial basins; new ids are M + k)
    outlet_of: dict[int, int] = {int(c): int(c) for c in np.nonzero(cut)[0]}
    next_id = M
    n_splits = 0
    areas = _areas(label, land)
    todo = sorted((b for b, a in areas.items() if a > basin_max_cells), key=lambda b: (-areas[b], b))
    while todo:
        B = todo.pop(0)
        cells = np.nonzero(label == B)[0]
        cells = cells[np.argsort(topo_pos[cells], kind="stable")]
        area = cells.size
        if area <= basin_max_cells:
            continue
        up = _subtree_counts(cells, down, label, B)
        cand = (cells != outlet_of[B]) & (up < area)
        score = np.abs(up - area / 2.0)
        cand_nl = cand & ~lk[cells]
        pick = cand_nl if cand_nl.any() else cand
        if not pick.any():
            info.setdefault("unsplittable", []).append(int(B))
            continue
        sc = np.where(pick, score, np.inf)
        p = int(np.argmin(sc))  # first minimum -> smallest topo position
        split_cell = int(cells[p])
        insub = _mark_subtree(cells, down, label, B, split_cell)
        new = next_id
        next_id += 1
        label[cells[insub]] = new
        outlet_of[new] = split_cell
        n_splits += 1
        a_new = int(insub.sum())
        a_rem = area - a_new
        areas[new] = a_new
        areas[B] = a_rem
        if a_new > basin_max_cells:
            todo.append(new)
        if a_rem > basin_max_cells:
            todo.append(B)
        todo.sort(key=lambda b: (-areas[b], b))
    info["n_splits"] = n_splits

    # ---- merge --------------------------------------------------------------
    # dense temporary ids
    ids = sorted(outlet_of)
    dense = {b: k for k, b in enumerate(ids)}
    lut = np.full(next_id, -1, dtype=np.int64)
    for b, k in dense.items():
        lut[b] = k
    lab = np.where(land, lut[np.maximum(label, 0)], -1)
    K = len(ids)
    area_k = np.bincount(lab[land], minlength=K).astype(np.int64)
    outlet_k = np.array([outlet_of[b] for b in ids], dtype=np.int64)
    lab3 = lab.reshape(6, N, N)
    pa, pb, pw = _boundary_pairs(lab3)
    xa, xb, xw = _cross_face_boundary_pairs(lab3, grid.owner, N, H)
    pa, pb, pw = np.concatenate([pa, xa]), np.concatenate([pb, xb]), np.concatenate([pw, xw])
    key = pa * K + pb
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=pw).astype(np.int64)
    nb: list[dict[int, int]] = [dict() for _ in range(K)]
    for kk, c in zip(uk.tolist(), cnt.tolist()):
        x, y = divmod(kk, K)
        nb[x][y] = c
        nb[y][x] = c
    # union-find alias
    alias = np.arange(K, dtype=np.int64)

    def find(x):
        while alias[x] != x:
            alias[x] = alias[alias[x]]
            x = alias[x]
        return int(x)

    def downstream_of(k):
        d = int(down[outlet_k[k]])
        return -1 if d < 0 or lab[d] < 0 else find(int(lab[d]))

    def on_chain(x, y):
        """True if basin y lies on x's downstream chain (x excluded)."""
        z = downstream_of(x)
        steps = 0
        while z >= 0 and steps <= K:
            if z == y:
                return True
            z = downstream_of(z)
            steps += 1
        return False

    reason: dict[int, str] = {}
    n_merges = 0
    heap = [(int(area_k[k]), k) for k in range(K) if area_k[k] < basin_min_cells]
    heapq.heapify(heap)
    while heap:
        a_s, S = heapq.heappop(heap)
        if find(S) != S or area_k[S] != a_s or a_s >= basin_min_cells:
            continue  # stale entry
        cands = sorted(((c, n) for n, c in nb[S].items() if n != S), key=lambda t: (-t[0], t[1]))
        target = -1
        for c, A in cands:
            if area_k[S] + area_k[A] > basin_max_cells:
                continue
            target = A
            break
        if target < 0:
            reason[S] = "island" if not cands else "max"
            continue
        A = target
        # the union keeps the outlet that still exits it: if A drains
        # (through any chain) into S, S's outlet is the exit
        if on_chain(A, S):
            outlet_k[A] = outlet_k[S]
        alias[S] = A
        area_k[A] += area_k[S]
        area_k[S] = 0
        # move S's boundaries to A
        for n, c in nb[S].items():
            if n == A:
                continue
            nb[A][n] = nb[A].get(n, 0) + c
            dn_ = nb[n]
            dn_.pop(S, None)
            dn_[A] = dn_.get(A, 0) + c
        nb[A].pop(S, None)
        nb[S] = {}
        n_merges += 1
        reason.pop(A, None)
        if area_k[A] < basin_min_cells:
            heapq.heappush(heap, (int(area_k[A]), A))
    info["n_merges"] = n_merges
    root = np.array([find(k) for k in range(K)], dtype=np.int64)
    lab = np.where(land, root[np.maximum(lab, 0)], -1)

    # ---- final ids ----------------------------------------------------------
    roots = np.unique(root)
    # order by (outlet face, outlet i, outlet j)
    ofij = np.array([cid_fij(int(outlet_k[r]), N) for r in roots], dtype=np.int64).reshape(-1, 3)
    order_idx = np.lexsort((ofij[:, 2], ofij[:, 1], ofij[:, 0]))
    roots = roots[order_idx]
    final = np.full(K, -1, dtype=np.int64)
    final[roots] = np.arange(roots.size)
    basin_id = np.where(land, final[np.maximum(lab, 0)], -1).astype(np.int32)
    nb_final = roots.size
    # per-basin stats
    bid = basin_id
    land_idx = np.nonzero(land)[0]
    ii = (land_idx % NN) // N
    jj = land_idx % N
    ff = land_idx // NN
    b_of = bid[land_idx]
    order_cells = np.argsort(b_of, kind="stable")
    bounds = np.searchsorted(b_of[order_cells], np.arange(nb_final + 1))
    co = np.zeros(M, dtype=np.int16) if cell_order is None else np.ascontiguousarray(cell_order, dtype=np.int16).reshape(-1)
    # exit cells: downstream is ocean, another basin, or another face (the
    # last cell of a face piece: still inside the basin, but refine's
    # per-face job drains and sinks there)
    d_l = down[land_idx]
    d_ok = np.maximum(d_l, 0)
    is_exit = (d_l < 0) | (bid[d_ok] != b_of) | (d_ok // NN != ff)
    # ocean > basin > face: a split outlet whose downstream cell lies on
    # another face drains into another basin, and is a basin exit, not a
    # face exit (a face exit is inside its basin by definition)
    exit_kind = np.where((d_l < 0) | (bid[d_ok] < 0), 0, np.where(bid[d_ok] != b_of, 2, 1))  # ocean / face / basin
    KINDS = ("ocean", "face", "basin")
    basins = []
    for b in range(nb_final):
        sel = order_cells[bounds[b] : bounds[b + 1]]
        r = roots[b]
        oc = int(outlet_k[r])
        of, oi, oj = cid_fij(oc, N)
        d = int(down[oc])
        if d >= 0:
            df, di_, dj_ = cid_fij(d, N)
            downstream_cell = [int(df), int(di_), int(dj_)]
            downstream_basin = int(bid[d])
        else:
            downstream_cell = None
            downstream_basin = -1
        # per-face pieces (sel is in cell-id order, so already grouped by face)
        pieces = []
        for f in np.unique(ff[sel]).tolist():
            ps = sel[ff[sel] == f]
            bi = ii[ps]
            bj = jj[ps]
            pieces.append(
                {
                    "face": int(f),
                    "bbox": [int(bi.min()), int(bj.min()), int(bi.max()) + 1, int(bj.max()) + 1],
                    "area_cells": int(ps.size),
                    "tiles": _tiles(bi, bj, R, T),
                }
            )
        outlet_piece = next(p for p in pieces if p["face"] == int(of))
        rec = {
            "id": b,
            "parent": downstream_basin,
            "face": int(of),
            "faces": [p["face"] for p in pieces],
            "outlet": [int(of), int(oi), int(oj)],
            "outlet_downstream": downstream_cell,
            "downstream_basin": downstream_basin,
            "area_cells": int(sel.size),
            "bbox": outlet_piece["bbox"],
            "order": int(co[oc]),
            "tiles": outlet_piece["tiles"],
            "pieces": pieces,
        }
        ex = sel[is_exit[sel]]  # cell-id order = (f, i, j) order
        ex_c = land_idx[ex]
        o_pos = np.nonzero(ex_c == oc)[0]
        assert o_pos.size == 1, "outlet is not an exit of its basin"
        ex = np.concatenate([ex[o_pos], np.delete(ex, o_pos[0])])
        rec["exits"] = [[int(ff[e]), int(ii[e]), int(jj[e])] for e in ex]
        rec["exit_kinds"] = [KINDS[int(exit_kind[e])] for e in ex]
        if sel.size < basin_min_cells:
            rec["undersized_reason"] = reason.get(int(r), "max")
        basins.append(rec)
    info["n_basins"] = nb_final
    info["n_undersized"] = sum(1 for b in basins if "undersized_reason" in b)
    info["undersized_reasons"] = {k: sum(1 for b in basins if b.get("undersized_reason") == k) for k in ("island", "max")}
    info["n_multi_exit"] = sum(1 for b in basins if len(b["exits"]) > 1)
    info["n_multi_face"] = sum(1 for b in basins if len(b["pieces"]) > 1)
    info["n_pieces"] = sum(len(b["pieces"]) for b in basins)
    info["cells_in_multi_face_basins"] = sum(b["area_cells"] for b in basins if len(b["pieces"]) > 1)
    info["max_area"] = max((b["area_cells"] for b in basins), default=0)
    return Partition(basin_id.reshape(6, N, N), basins, info)


def _tiles(bi, bj, R, T) -> list[list[int]]:
    """Sorted LOD-0 tile coordinates ``[x, y]`` (tiles of ``T`` fine cells,
    face refined ``R x``) touched by the coarse cells ``(bi, bj)``."""
    tx0 = (bi * R) // T
    tx1 = ((bi + 1) * R - 1) // T
    ty0 = (bj * R) // T
    ty1 = ((bj + 1) * R - 1) // T
    tiles = set()
    for x0, x1, y0, y1 in zip(tx0.tolist(), tx1.tolist(), ty0.tolist(), ty1.tolist()):
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                tiles.add((x, y))
    return sorted(list(t) for t in tiles)


def _areas(label, land):
    b, c = np.unique(label[land], return_counts=True)
    return {int(x): int(y) for x, y in zip(b, c)}


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------
def run(store, params, log=print) -> dict:
    grid = params.coarse_grid()
    t0 = time.time()
    fd = store.load_field("flow_dir", grid)
    acc = store.load_field("flow_acc", grid)
    ws = store.load_field("water_surface", grid)
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    precip = store.load_field("precip", grid)
    land = fd.interior != OCEAN
    surface = (h.interior + sed.interior).astype(np.float32)
    lake = ((ws.interior - surface) > params.hydro.lake_min_depth) & land
    thr = river_threshold_volume(precip.interior, land, params.hydro.river_threshold)
    _, _, cell_order = channel_network(fd.interior, acc.interior, thr, grid, lake=lake)
    wp = params.watersheds
    part = partition(fd.interior, grid, wp.basin_max_cells, wp.basin_min_cells, lake=lake, cell_order=cell_order, R=params.world.R, T=params.world.T, log=log)
    store.save_field(FaceField.from_interior(grid, part.basin_id, name="basin_id"))
    store.write_json("graph/basins.json", {"basins": part.basins, "basin_max_cells": wp.basin_max_cells, "basin_min_cells": wp.basin_min_cells})
    info = dict(part.info)
    info["t_total_s"] = time.time() - t0
    log(
        f"[watersheds] {info['n_basins']} basins from {info['n_initial']} outlets "
        f"({info['n_splits']} splits, {info['n_merges']} merges, {info['n_undersized']} undersized {info['undersized_reasons']}, "
        f"{info['n_multi_exit']} multi-exit, {info['n_multi_face']} on more than one face ({info['n_pieces']} face pieces, "
        f"{info['cells_in_multi_face_basins']} cells), max {info['max_area']} cells) in {info['t_total_s']:.1f}s"
    )
    return info


def quicklook(store, params, path):
    from ..viz import quicklook as ql

    grid = params.coarse_grid()
    bid = store.load_field("basin_id", grid)
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    surface = (h.interior + sed.interior).astype(np.float64)
    labels = ql.render_labels(bid.interior, seed=params.seed)
    hs = ql.hillshade(surface, grid.cell_size_m, z_factor=0.5 * grid.N * grid.cell_size_m / max(float(np.ptp(surface)), 1.0))
    img = (labels.astype(np.float32) * (0.55 + 0.45 * hs[..., None])).astype(np.uint8)
    ocean = bid.interior < 0
    img[ocean] = (25, 45, 80)
    img = ql.contour_lines(img, bid.interior, (255, 255, 255))
    marks = np.zeros(bid.interior.shape, dtype=bool)
    if store.has("graph/basins.json"):
        for b in store.read_json("graph/basins.json")["basins"]:
            f, i, j = b["outlet"]
            marks[f, max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2] = True
    img = ql.overlay(img, marks, (255, 40, 40), 1.0)
    return ql.save_image(path, img)
