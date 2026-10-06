"""A planet level's finish in bounded memory (docs/zoom-windows.md, "The
planet at 305 m"): the same outputs as the in-memory finish of the 1.2 km
level -- height, sediment and discharge from the work raster, seams blended,
refine's coast rule, a water surface with the planet's lakes -- for a
face of any size.  At R = 32 a face is 32768^2 cells, 4.3 GB a float32
field, and a priority flood of it would need ~50 GB:

* the outputs are memmaps written a strip of coarse rows at a time;
* the seams read the neighbour face's work raster through its memmap;
* the water surface is a flood of the whole face while it is at most
  ``FLOOD_WHOLE`` cells a side (so the 1.2 km level is byte-identical to
  the in-memory finish), else of ``FLOOD_BLOCK`` blocks overlapping by
  ``FLOOD_OVERLAP``, each draining to its border as well -- at a lake's
  level where the border lies over a lake of the planet's
  (:func:`parent_lakes.level_lakes`), so a lake that crosses a block edge keeps
  its water.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import WorldParams
from ..io.world_store import WorldStore
from ..refine.zoom import zoom_params
from .parent_lakes import lakes_of_the_planet, level_lakes, over_cells  # noqa: F401  (lakes_of_the_planet: the name the finish has had)

FLOOD_WHOLE = 8192
FLOOD_BLOCK = 8192
FLOOD_OVERLAP = 512


def strip_rows(R: int) -> int:
    """Coarse rows per strip: ~512 fine rows (an upsample of a strip holds a
    dozen fields of it)."""
    return max(1, 512 // int(R))


def finish_face(root: Path, out: Path, level, face: int) -> dict:
    from ..refine import rasterize as rz
    from ..refine.upsample import upsample_face
    from . import planet as zp

    t0 = time.time()
    root, out = Path(root), Path(out)
    params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    lp = zoom_params(params, level.R)
    grid, fields, derived = zp.planet_inputs(root, lp, out)[:3]
    N, R = grid.N, level.R
    n = N * R
    g = level.guard * R
    NF = n + 2 * g
    strip = strip_rows(R)
    wk = {k: zp.open_work(out, face, k, NF, mode="r") for k in zp.WORK_FIELDS}
    arrays = {k: np.lib.format.open_memmap(zp.out_path(out, R, face, k), mode="w+", dtype=np.float32, shape=(n, n)) for k in zp.OUT_FIELDS}
    for i0 in range(0, N, strip):
        i1 = min(N, i0 + strip)
        up = upsample_face(fields, derived, face, R, i0, i1)
        rs = slice(i0 * R, i1 * R)
        wr = slice(g + i0 * R, g + i1 * R)
        done = np.asarray(wk["done"][wr, g:g + n])
        for k in ("height", "sediment", "discharge"):
            arrays[k][rs] = np.where(done, wk[k][wr, g:g + n], up[k])
        del up, done
    seams = zp.blend_face_seams(out, level, N, face, arrays)
    ocean_all = np.stack([np.load(root / "coarse" / f"flow_dir.f{k}.npy") == 255 for k in range(6)])
    frac = rz.coast_fraction(grid, ocean_all)
    zone = rz.coast_zone(grid, ocean_all)
    M = np.float32(rz.COAST_MARGIN_M)
    coast_lowered = coast_raised = 0
    for i0 in range(0, N, strip):
        i1 = min(N, i0 + strip)
        near = np.repeat(np.repeat(zone[face, i0:i1], R, axis=0), R, axis=1)
        if not near.any():
            continue
        rs = slice(i0 * R, i1 * R)
        sea = (frac.sample_window(face, i0, i1, 0, N, R, order=1) > 0.5) & near
        h, sd = np.array(arrays["height"][rs]), np.array(arrays["sediment"][rs])
        sf = h + sd
        drop = np.where(sea & (sf > -M), sf + M, np.float32(0.0)).astype(np.float32)
        take = np.minimum(np.maximum(sd, 0.0), drop)
        sd -= take
        h -= drop - take
        low = near & ~sea & (sf < M)
        sd[low] += M - sf[low]
        arrays["height"][rs] = h
        arrays["sediment"][rs] = sd
        coast_lowered += int((drop > 0).sum())
        coast_raised += int(low.sum())
    sea_near_c = ndimage.binary_dilation(ocean_all[face], structure=np.ones((3, 3), bool))

    def rows(a0, a1, b0, b1):
        surf = np.asarray(arrays["height"][a0:a1, b0:b1]) + np.asarray(arrays["sediment"][a0:a1, b0:b1])
        ci0, cj0 = a0 // R, b0 // R
        near = np.repeat(np.repeat(sea_near_c[ci0:(a1 + R - 1) // R, cj0:(b1 + R - 1) // R], R, 0), R, 1)
        near = near[a0 - ci0 * R:a1 - ci0 * R, b0 - cj0 * R:b1 - cj0 * R]
        return surf, (surf < 0.0) & near

    ws_out = arrays["water_surface"]
    lake_cells = land_cells = 0
    # the planet's own lakes: its coarse water deeper than the marsh line, off the sea
    lake_depth = max(float(params.hydro.lake_min_depth), float(getattr(params.hydro, "marsh_depth", 0.0)))
    coarse_level = fields["water_surface"].interior[face].astype(np.float32)
    coarse_lake = ((coarse_level - derived["surface"].interior[face].astype(np.float32)) > lake_depth) & ~ocean_all[face]
    if n <= FLOOD_WHOLE:
        surf, ocean = rows(0, n, 0, n)
        ws = level_lakes(surf, ocean, over_cells(coarse_lake, R, surf.shape), over_cells(coarse_level, R, surf.shape))
        ws_out[:] = ws
        lake_cells = int(((ws - surf > float(params.hydro.lake_min_depth)) & ~ocean).sum())
        land_cells = int((~ocean).sum())
        del surf, ocean, ws
    else:
        B, O = FLOOD_BLOCK, FLOOD_OVERLAP
        for a0 in range(0, n, B):
            for b0 in range(0, n, B):
                a1, b1 = min(n, a0 + B), min(n, b0 + B)
                e0, e1 = max(0, a0 - O), min(n, a1 + O)
                f0, f1 = max(0, b0 - O), min(n, b1 + O)
                surf, ocean = rows(e0, e1, f0, f1)
                cs = (slice(e0 // R, (e1 + R - 1) // R), slice(f0 // R, (f1 + R - 1) // R))
                cut = (slice(e0 - cs[0].start * R, e1 - cs[0].start * R), slice(f0 - cs[1].start * R, f1 - cs[1].start * R))
                big = ((cs[0].stop - cs[0].start) * R, (cs[1].stop - cs[1].start) * R)
                ws = level_lakes(surf, ocean, over_cells(coarse_lake[cs], R, big)[cut], over_cells(coarse_level[cs], R, big)[cut])
                core = (slice(a0 - e0, a1 - e0), slice(b0 - f0, b1 - f0))
                s, oc, w = surf[core], ocean[core], ws[core]
                ws_out[a0:a1, b0:b1] = w
                lake_cells += int(((w - s > float(params.hydro.lake_min_depth)) & ~oc).sum())
                land_cells += int((~oc).sum())
                del surf, ocean, ws
    for a in arrays.values():
        a.flush()
    del arrays
    return {"face": face, "seam_cells": seams, "coast_lowered_cells": coast_lowered, "coast_raised_cells": coast_raised,
            "land_cells": land_cells, "lake_cells": lake_cells, "seconds": round(time.time() - t0, 1)}


def flow_face(root: Path, out: Path, R: int, face: int) -> dict:
    """The river map of one finished face, ``L{R}.f{k}.flow.npy`` (float32):
    the planet's rain accumulated down the drainage tree of the face's final
    surface, in the coarse ``flow_acc``'s volume units.  :func:`flow_faces`
    routes the six faces together and hands the water over at the cube edges
    where the fine rivers cross them; this is the one-face form, where the
    face border drains and the planet's coarse drainage carries water in.

    The erosion's ``discharge`` is an average of the particles' tracks, not a
    sum over a catchment.  Measured on earth-v9 at R = 8 (face 2), it falls on
    74 % of the downstream steps from its brightest cells, only 4 % of those
    cells lie on the surface's own drainage, and below a lake it is 3-9 % of
    what flows in.  A river drawn from it starts, runs a way and stops.
    Accumulation never falls downstream, carries a lake's inflow on from its
    spill point, and follows the valleys the surface shows.  The sea and the
    face border drain; the planet's drainage carries water in across the
    edges (:func:`planet.planet_inflow`), entering inside the border (an
    inflow spawned on it was a drain's: on earth-v9 15-63 % of the cross-face
    inflow of faces 0, 3, 4, 5).

    The tree (:mod:`hydro.bed_routing`) crosses a filled depression along its
    drowned bed and an exact flat toward its nearest lower edge, not in the
    breadth-first rays of the plain flood's queue, and runs down dry ground
    by D8-LTD, following the slope's aspect rather than the nearest of eight
    directions.  Measured on earth-v9 (faces 0-5), the drawn river cells
    (off lakes) in straight runs of 6 steps: 29-33 % with the plain flood's
    tree, 19-21 % with this one; 1.4-1.5x its time.  One flood of the whole
    face, ~3 GB peak at 8192^2; a larger face raises."""
    from ..hydro.bed_routing import accumulate_ltd, priority_flood_bed
    from . import planet as zp

    t0 = time.time()
    root, out, R = Path(root), Path(out), int(R)
    precip = np.maximum(np.load(root / "coarse" / f"precip.f{face}.npy").astype(np.float64), 0.0)
    N = precip.shape[0]
    n = N * R
    if n > FLOOD_WHOLE:
        raise NotImplementedError(f"flow_face floods a face whole: {n}^2 cells is past FLOOD_WHOLE = {FLOOD_WHOLE}")
    surf = np.asarray(np.load(zp.out_path(out, R, face, "height"), mmap_mode="r"), np.float32) + \
        np.asarray(np.load(zp.out_path(out, R, face, "sediment"), mmap_mode="r"), np.float32)
    sea_near = ndimage.binary_dilation(np.load(root / "coarse" / f"flow_dir.f{face}.npy") == 255, structure=np.ones((3, 3), bool))
    drain = (surf < 0.0) & np.repeat(np.repeat(sea_near, R, axis=0), R, axis=1)
    drain[0, :] = drain[-1, :] = drain[:, 0] = drain[:, -1] = True
    land = int((~drain).sum())
    weight = np.repeat(np.repeat(precip / float(R * R), R, axis=0), R, axis=1)
    fd, fa = zp.planet_flow(root)
    ring = np.pad(surf, 1, mode="edge")
    ring[1, :] = ring[-2, :] = ring[:, 1] = ring[:, -2] = np.inf        # the face's border drains: an inflow enters inside them
    weight += zp.planet_inflow(fd, fa, face, 0, N, 0, N, R, ring)[1:-1, 1:-1]
    del ring
    fr = priority_flood_bed(surf, drain)
    del drain
    acc = accumulate_ltd(surf, fr, weight.ravel()).reshape(n, n)
    del fr, surf, weight
    np.save(zp.out_path(out, R, face, "flow"), acc.astype(np.float32))
    return {"face": face, "land_cells": land, "flow_max": float(acc.max()), "seconds": round(time.time() - t0, 1)}


#: fine cells of the neighbouring faces around a face's flood in :func:`flow_faces`
FLOW_MARGIN = 64
#: rows along each cube edge whose water's exits are kept for the hand-over
FLOW_BAND = 4
#: face-to-face hand-overs a parcel of water is followed through
FLOW_HOPS = 32


def _beyond(face: int, i: np.ndarray, j: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells ``(i, j)`` of ``face``'s fine grid, extended past its edges
    (negative or ``>= n``): the face and cell that own that ground."""
    from ..cubesphere import from_sphere_v, to_sphere_v

    i = np.asarray(i, np.float64)
    j = np.asarray(j, np.float64)
    f2, u2, v2 = from_sphere_v(to_sphere_v(np.full(i.shape, int(face), np.int64), (i + 0.5) / n, (j + 0.5) / n))
    return f2, np.clip((u2 * n).astype(np.int64), 0, n - 1), np.clip((v2 * n).astype(np.int64), 0, n - 1)


def route_faces(n: int, own, rain, sample, write, margin: int = FLOW_MARGIN, hops: int = FLOW_HOPS, log=None) -> list[dict]:
    """The six faces' drainage (``n x n`` cells each), handed over at the cube
    edges where the fine rivers cross them.

    ``own(f) -> (surface float32, sea bool)``, ``rain(f) -> float64`` (the
    face's cells), ``sample(f, i, j) -> (surface, sea)`` at arrays of cells,
    ``write(f, flow float32)``.

    1. Depression levels for the whole planet (:mod:`hydro.spill_graph`):
       each face is flooded alone with every edge cell a labelled seed, and
       the labels, linked across the cube edges, are flooded from the sea.
       Without it a basin reaching past a face's flood spills wherever that
       flood happens to end, a different place for each face, and the water
       handed between them circles.
    2. Each face is flooded (:func:`hydro.bed_routing.priority_flood_bed`,
       :func:`accumulate_ltd`) with ``margin`` cells of its neighbours' ground
       around it, so a river reaching an edge runs on across it (or along
       it); the sea drains, and so does the margin's outer ring, seeded at
       the planet's level for that ground and after the real cells of its
       level.  Only the face's own cells rain.  Water leaving the face's cells
       for good (:func:`hydro.tree.window_exits`) is handed to the
       neighbouring face at the cell it crossed into, and for the
       ``FLOW_BAND`` rows along each edge the face keeps where water arriving
       there would leave it.
    3. The hand-overs are followed from face to face through those tables
       (the trees do not depend on the water); a parcel that comes back to a
       cell it entered by stops there (``cycled``), as does one still moving
       after ``hops`` hand-overs.
    4. Each face's tree (one byte a cell, :func:`hydro.tree.receiver_codes`)
       accumulates its rain plus every parcel handed to it.

    A parcel enters a face once per crossing and leaves it by its exit, so
    nothing is counted twice.  Returns per face: ``rain``, ``sea`` (reaching
    the face's sea), ``inflow`` / ``outflow`` across its edges, and
    ``unresolved`` (handed over by it and stopped); the planet's rain is its
    sea's plus the unresolved.

    Measured on earth-v9 at R = 8 (``margin`` 64): the sea takes 0.99974 of
    the rain (the rest circled at a seam), 2.2 % of it crosses a cube edge;
    of the river cells that end on a face's edge, 81 % go on across it
    within 3 cells (10 % one face at a time, with the coarse inflow).  With
    the margin's ring seeded at its own height instead of the planet's
    level, 98 % of the handed-over water circled (a basin on the +Y/+Z seam
    spilled at 114.7 m in one face's flood and 124.3 m in the other's).
    108 s for the six faces (58 s a face at a time), 3.7 GB peak."""
    from ..hydro.bed_routing import accumulate_ltd, priority_flood_bed
    from ..hydro.spill_graph import label_flood, reduce_edges, spill_levels
    from ..hydro.tree import accumulate_codes, receiver_codes, window_exits

    K = int(margin)
    NE = n + 2 * K
    nn = n * n
    t_all = time.time()
    # geometry shared by the faces
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    edge = np.minimum(np.minimum(I, J), np.minimum(n - 1 - I, n - 1 - J))
    band = np.flatnonzero(edge < FLOW_BAND)                  # own flat indices, sorted
    ring = np.flatnonzero(edge == 0)
    del I, J, edge
    P = ring.size
    ri, rj = ring // n, ring % n
    ext = lambda c: (c // n + K) * NE + (c % n + K)
    inner = np.zeros((NE, NE), bool)
    inner[K:K + n, K:K + n] = True
    m_idx = np.flatnonzero(~inner)
    del inner
    m_i, m_j = m_idx // NE - K, m_idx % NE - K
    o_sel = (m_i == -K) | (m_i == n + K - 1) | (m_j == -K) | (m_j == n + K - 1)
    o_idx, o_i, o_j = m_idx[o_sel], m_i[o_sel], m_j[o_sel]    # the margin's outer ring
    del o_sel
    # what each face's first flood must report: the ground under the others'
    # outer rings, and across its edges
    need = [[] for _ in range(6)]
    oring, links = [], []
    for f in range(6):
        g, gi, gj = _beyond(f, o_i, o_j, n)
        oring.append((g, gi * n + gj))
        for fg in range(6):
            need[fg].append(gi[g == fg] * n + gj[g == fg])
        lk = []
        for di, dj in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            a_, b_ = ri + di, rj + dj
            out_ = (a_ < 0) | (a_ >= n) | (b_ < 0) | (b_ >= n)
            k_ = np.flatnonzero(out_)
            g2, gi2, gj2 = _beyond(f, a_[out_], b_[out_], n)
            lk.append((k_, g2, gi2 * n + gj2))
            for fg in range(6):
                need[fg].append(gi2[g2 == fg] * n + gj2[g2 == fg])
        links.append(lk)
    need = [np.unique(np.concatenate(x)) for x in need]
    # 1. the planet's depression levels
    fA, lA, rsurf, rlab, EA, EB, EW = [], [], [], [], [], [], []
    stats = []
    for f in range(6):
        t0 = time.time()
        surf, sea = own(f)
        seed = np.full((n, n), -1, np.int32)
        seed[sea] = 0
        lab_ring = np.where(sea.reshape(-1)[ring], 0, 1 + f * P + np.arange(P)).astype(np.int32)
        seed.reshape(-1)[ring] = lab_ring
        filled, label, (ea, eb, ew) = label_flood(surf, seed)
        del seed
        fA.append(filled.reshape(-1)[need[f]])
        lA.append(label.reshape(-1)[need[f]])
        rsurf.append(surf.reshape(-1)[ring])
        rlab.append(lab_ring)
        EA.append(ea), EB.append(eb), EW.append(ew)
        stats.append({"face": f, "land_cells": int((~sea).sum()), "seconds_levels": round(time.time() - t0, 1)})
        del surf, sea, filled, label
    for f in range(6):
        for k_, g2, c2 in links[f]:
            for fg in range(6):
                sel = g2 == fg
                pos = np.searchsorted(need[fg], c2[sel])
                EA.append(rlab[f][k_[sel]]), EB.append(lA[fg][pos])
                EW.append(np.maximum(rsurf[f][k_[sel]], fA[fg][pos]))
    ea, eb, ew = reduce_edges(np.concatenate(EA), np.concatenate(EB), np.concatenate(EW))
    del EA, EB, EW
    level = spill_levels(1 + 6 * P, ea, eb, ew, [0])
    del ea, eb, ew
    t_levels = time.time() - t_all
    # 2. route each face with its margin
    codes, tables, seas = [], [], []
    ex_face, ex_cell, ex_amt, ex_src = [], [], [], []
    for f in range(6):
        t0 = time.time()
        surf, sea = own(f)
        E = np.empty((NE, NE), np.float32)
        D = np.zeros((NE, NE), bool)
        E[K:K + n, K:K + n] = surf
        D[K:K + n, K:K + n] = sea
        seas.append(np.packbits(sea))
        del surf, sea
        f2, i2, j2 = _beyond(f, m_i, m_j, n)
        for g in np.unique(f2):
            sel = np.flatnonzero(f2 == g)
            sv, dv = sample(int(g), i2[sel], j2[sel])
            E.reshape(-1)[m_idx[sel]] = sv
            D.reshape(-1)[m_idx[sel]] = dv
        del f2, i2, j2
        g, c2 = oring[f]
        lv = np.empty(o_idx.size, np.float32)
        for fg in range(6):
            sel = g == fg
            pos = np.searchsorted(need[fg], c2[sel])
            lab = lA[fg][pos]
            lvl = np.where(lab >= 0, level[np.maximum(lab, 0)], -np.inf)
            lv[sel] = np.maximum(fA[fg][pos], np.where(np.isfinite(lvl), lvl, -np.inf))
        E.reshape(-1)[o_idx] = lv
        D.reshape(-1)[o_idx] = True
        w = np.zeros((NE, NE))
        w[K:K + n, K:K + n] = rain(f)
        stats[f]["rain"] = float(w.sum())
        fr = priority_flood_bed(E, D, late_border=True)
        del D
        acc = accumulate_ltd(E, fr, w.reshape(-1))
        del E, w
        fr.filled = fr.order = None
        ex = window_exits(fr.parent, fr.pop_seq, NE, K, K + n, K, K + n)
        fr.pop_seq = None
        # hand-overs: a face cell draining out of the face for good
        rc = ext(ring)
        p = fr.parent[rc].astype(np.int64)
        pi, pj = p // NE - K, p % NE - K
        out_ = (p >= 0) & ((pi < 0) | (pi >= n) | (pj < 0) | (pj >= n))
        out_[out_] = ex[p[out_]] == p[out_]
        at, inv = np.unique(p[out_], return_inverse=True)
        amt = np.bincount(inv.reshape(-1), acc[rc[out_]], minlength=at.size)
        g2, gi, gj = _beyond(f, at // NE - K, at % NE - K, n)
        ex_face.append(g2), ex_cell.append(gi * n + gj), ex_amt.append(amt), ex_src.append(np.full(at.size, f))
        # where water arriving on the edge band leaves the face
        e = ex[ext(band)].astype(np.int64)
        dest = np.full(band.size, -1, np.int64)
        has = e >= 0
        g3, gi3, gj3 = _beyond(f, e[has] // NE - K, e[has] % NE - K, n)
        dest[has] = g3 * nn + gi3 * n + gj3
        tables.append(dest)
        del ex, acc, e, rc, p
        codes.append(receiver_codes(fr.parent, NE))
        del fr
        stats[f]["seconds_route"] = round(time.time() - t0, 1)
        if log is not None:
            log(f"flow face {f}: levels {stats[f]['seconds_levels']}s, routed with a {K}-cell margin {stats[f]['seconds_route']}s")
    del fA, lA, level
    # 3. follow the hand-overs
    t1 = time.time()
    inj = [[] for _ in range(6)]
    pf_, pc_, pa_, ps_ = (np.concatenate(x) for x in (ex_face, ex_cell, ex_amt, ex_src))
    pid = np.arange(pa_.size)
    seen = np.full((int(hops), pa_.size), -1, np.int64)
    unresolved = np.zeros(6)
    cycled = 0.0
    for h in range(int(hops) + 1):
        if pa_.size == 0:
            break
        land = pf_ * nn + pc_
        if h == int(hops):
            np.add.at(unresolved, ps_, pa_)
            break
        again = (seen[:h, pid] == land[None, :]).any(axis=0) if h else np.zeros(pa_.size, bool)
        if again.any():
            np.add.at(unresolved, ps_[again], pa_[again])
            cycled += float(pa_[again].sum())
            keep = ~again
            pf_, pc_, pa_, ps_, pid, land = pf_[keep], pc_[keep], pa_[keep], ps_[keep], pid[keep], land[keep]
        seen[h, pid] = land
        dest = np.full(pa_.size, -1, np.int64)
        for g in range(6):
            sel = np.flatnonzero(pf_ == g)
            if sel.size == 0:
                continue
            inj[g].append((pc_[sel], pa_[sel]))
            pos = np.minimum(np.searchsorted(band, pc_[sel]), band.size - 1)
            dest[sel] = np.where(band[pos] == pc_[sel], tables[g][pos], -1)
        go = dest >= 0
        ps_ = pf_[go]
        pf_, pc_, pa_, pid = dest[go] // nn, dest[go] % nn, pa_[go], pid[go]
    t_hand = time.time() - t1
    # 4. accumulate
    for f in range(6):
        t0 = time.time()
        w = np.zeros((NE, NE))
        w[K:K + n, K:K + n] = rain(f)
        wf = w.reshape(-1)
        inflow = 0.0
        for cells, amts in inj[f]:
            np.add.at(wf, ext(cells), amts)
            inflow += float(amts.sum())
        total = float(wf.sum())
        acc = accumulate_codes(codes[f], NE, wf)
        del w, wf
        codes[f] = None
        q = acc.reshape(NE, NE)[K:K + n, K:K + n]
        sea = np.unpackbits(seas[f], count=nn).reshape(n, n).astype(bool)
        sea_sum = float(q[sea].sum())
        write(f, q.astype(np.float32))
        stats[f].update(flow_max=float(q.max()), sea=sea_sum, inflow=inflow, outflow=total - sea_sum, unresolved=float(unresolved[f]),
                        seconds=round(stats[f]["seconds_levels"] + stats[f]["seconds_route"] + time.time() - t0 + (t_hand + t_levels
                                      - sum(s_["seconds_levels"] for s_ in stats)) / 6.0, 1))
        del acc, q, sea
    stats[0]["cycled_total"] = cycled
    if log is not None:
        rain_t, sea_t = sum(s_["rain"] for s_ in stats), sum(s_["sea"] for s_ in stats)
        log(f"flow: {time.time() - t_all:.0f}s; the sea takes {sea_t / max(rain_t, 1e-30):.6f} of the rain, unresolved {unresolved.sum() / max(rain_t, 1e-30):.2e} "
            f"(cycled {cycled / max(rain_t, 1e-30):.2e})")
    return stats


def flow_faces(root: Path, out: Path, R: int, margin: int = FLOW_MARGIN, hops: int = FLOW_HOPS, log=None) -> list[dict]:
    """The river maps of all six finished faces, ``L{R}.f{k}.flow.npy``
    (float32), in the coarse ``flow_acc``'s volume units: each face's rain
    and the water the other faces hand it at the cube edges, where their
    fine rivers cross (:func:`route_faces`), accumulated down the tree of
    :func:`flow_face` (a flood of the whole face plus ``margin`` cells of its
    neighbours).  No coarse inflow: the six faces are the whole planet.
    Returns the per-face stats of :func:`route_faces` (``seconds`` each)."""
    from . import planet as zp

    root, out, R = Path(root), Path(out), int(R)
    precip = [np.maximum(np.load(root / "coarse" / f"precip.f{f}.npy").astype(np.float64), 0.0) for f in range(6)]
    N = precip[0].shape[0]
    n = N * R
    if n > FLOOD_WHOLE:
        raise NotImplementedError(f"flow_faces floods a face whole: {n}^2 cells is past FLOOD_WHOLE = {FLOOD_WHOLE}")
    near = [ndimage.binary_dilation(np.load(root / "coarse" / f"flow_dir.f{f}.npy") == 255, structure=np.ones((3, 3), bool)) for f in range(6)]
    # a face's files are opened per call, so their pages are not held across faces
    load = lambda f, k: np.load(zp.out_path(out, R, f, k), mmap_mode="r")

    def own(f):
        surf = np.asarray(load(f, "height"), np.float32) + np.asarray(load(f, "sediment"), np.float32)
        return surf, (surf < 0.0) & np.repeat(np.repeat(near[f], R, axis=0), R, axis=1)

    def rain(f):
        return np.repeat(np.repeat(precip[f] / float(R * R), R, axis=0), R, axis=1)

    def sample(f, i, j):
        s_ = np.asarray(load(f, "height")[i, j], np.float32) + np.asarray(load(f, "sediment")[i, j], np.float32)
        return s_, (s_ < 0.0) & near[f][i // R, j // R]

    def write(f, q):
        np.save(zp.out_path(out, R, f, "flow"), q)

    return route_faces(n, own, rain, sample, write, margin, hops, log)


def quicklook_rows(out: Path, R: int, face: int, name: str, k: int, m: int, how: str = "mean") -> np.ndarray:
    """``name`` of a face reduced by ``k x k`` blocks, a block row at a time."""
    from . import planet as zp

    a = np.load(zp.out_path(out, R, face, name), mmap_mode="r")
    res = np.empty((m // k, m // k), np.float32)
    for r in range(m // k):
        blk = np.asarray(a[r * k:(r + 1) * k, :m], np.float32).reshape(k, m // k, k)
        res[r] = blk.max(axis=(0, 2)) if how == "max" else blk.mean(axis=(0, 2))
    return res


__all__ = ["FLOOD_WHOLE", "FLOOD_BLOCK", "FLOOD_OVERLAP", "FLOW_MARGIN", "FLOW_BAND", "FLOW_HOPS", "strip_rows", "finish_face", "flow_face",
           "route_faces", "flow_faces", "quicklook_rows"]
