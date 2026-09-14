"""Cube-face seams of a baked world: the watershed partition against the
cross-face flow graph, and the fine raster across the 12 cube edges
(docs/cross-face-basins.md).

    python scripts/face_seams.py <world_root> [--no-fine]

Coarse (watersheds): basins, basins on more than one face, land cells whose
downstream cell is on another face and how many of those links lie inside
one basin, exits of kind 'face', undersized reasons.

Fine (refine raster): mean |step| of the surface between the two land
cells straddling an edge against the step between the first two cells
inside the face (a crease shows as a ratio well above 1), the refined
minus plain detail |d| by distance from the seam (the frozen / feathered
strip shows as a detail deficit), and the fine discharge ratio across the
face-crossing coarse links (the river continues or it does not).  Reads
only; prints JSON.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from globe.config import WorldParams  # noqa: E402
from globe.cubesphere import Grid  # noqa: E402
from globe.hydro.d8 import OCEAN, downstream_table  # noqa: E402
from globe.refine import basin_job as bj  # noqa: E402
from globe.refine import rasterize as rz  # noqa: E402
from globe.refine.lod import edge_links, neighbour_ring, side_row  # noqa: E402
from globe.refine.upsample import upsample_face  # noqa: E402


def main(root: str, fine: bool = True):
    root = Path(root)
    m = json.load(open(root / "manifest.json"))
    params = WorldParams.from_dict(m["params"])
    N, H, R = m["N_c"], m["halo"], m["R"]
    NN = N * N
    g = Grid(N, H, m["cell_size_m"])
    fd = np.stack([np.load(root / f"coarse/flow_dir.f{f}.npy") for f in range(6)]).astype(np.uint8)
    bid = np.stack([np.load(root / f"coarse/basin_id.f{f}.npy") for f in range(6)])
    down = downstream_table(fd, g.owner, H)
    land = fd.reshape(-1) != OCEAN
    face = np.arange(6 * NN) // NN
    dn = np.maximum(down, 0)
    cross = land & (down >= 0) & land[dn] & (face[dn] != face)
    b = json.load(open(root / "graph/basins.json"))["basins"]
    nf = np.zeros((len(b), 6), bool)
    for f in range(6):
        nf[np.unique(bid[f][bid[f] >= 0]), f] = True
    multi = nf.sum(1) > 1
    wi = m["stages"]["watersheds"]["info"]
    out = {
        "n_basins": len(b),
        "basins_on_more_than_one_face": int(multi.sum()),
        "on_three_or_more": int((nf.sum(1) > 2).sum()),
        "cells_in_multi_face_basins": int(sum(x["area_cells"] for k, x in enumerate(b) if multi[k])),
        "largest_basin_cells": max(x["area_cells"] for x in b),
        "land_cells": int(land.sum()),
        "cells_draining_across_a_face": int(cross.sum()),
        "cells_draining_across_a_face_within_one_basin": int((cross & (bid.reshape(-1)[dn] == bid.reshape(-1))).sum()),
        "exits_kind_face": sum(k == "face" for x in b for k in x["exit_kinds"]),
        "basins_with_face_exits": sum("face" in x["exit_kinds"] for x in b),
        "undersized": wi.get("undersized_reasons"),
        "watersheds_info": {k: wi.get(k) for k in ("n_initial", "n_splits", "n_merges", "n_pieces", "n_multi_face")},
        "watersheds_seconds": m["stages"]["watersheds"]["seconds"],
    }
    if not fine or not rz.fine_exists(root, params):
        print(json.dumps(out, indent=1))
        return
    grid, fields, derived = bj.coarse_inputs(root, params)
    surf = [np.asarray(rz.open_fine(root, "height", f)) + np.asarray(rz.open_fine(root, "sediment", f)) for f in range(6)]
    fbid = [np.asarray(rz.open_fine(root, "basin_id", f)) for f in range(6)]
    q = [np.asarray(rz.open_fine(root, "discharge", f)) for f in range(6)]
    plain = [sum(upsample_face(fields, derived, f, R, 0, N)[k] for k in ("height", "sediment")) for f in range(6)]
    detail = [np.abs(s - p) for s, p in zip(surf, plain)]
    links = edge_links(N * R)
    D = 4 * R + 2
    seam_step, in_step, ref_step = [], [], []
    det_by_depth = np.zeros(D)
    det_cnt = np.zeros(D)
    det_interior, det_interior_n = 0.0, 0
    for f in range(6):
        for s in range(4):
            own0 = side_row(surf[f], s, 0)
            own1 = side_row(surf[f], s, 1)
            nb0 = neighbour_ring(lambda F, S, d: side_row(surf[F], S, d), links, f, s, 0)
            l0 = side_row(fbid[f], s, 0) >= 0
            l1 = side_row(fbid[f], s, 1) >= 0
            ln0 = neighbour_ring(lambda F, S, d: side_row(fbid[F], S, d), links, f, s, 0) >= 0
            seam_step.append(np.abs(own0 - nb0)[l0 & ln0])
            in_step.append(np.abs(own0 - own1)[l0 & l1])
            r8, r9 = side_row(surf[f], s, 8), side_row(surf[f], s, 9)
            k8 = (side_row(fbid[f], s, 8) >= 0) & (side_row(fbid[f], s, 9) >= 0)
            ref_step.append(np.abs(r8 - r9)[k8])
            for d in range(D):
                row = side_row(detail[f], s, d)
                lk = side_row(fbid[f], s, d) >= 0
                det_by_depth[d] += row[lk].sum()
                det_cnt[d] += lk.sum()
        inner = detail[f][D:-D, D:-D]
        li = fbid[f][D:-D, D:-D] >= 0
        det_interior += inner[li].sum()
        det_interior_n += li.sum()
    seam_step = np.concatenate(seam_step)
    in_step = np.concatenate(in_step)
    ref_step = np.concatenate(ref_step)
    out["fine_seam"] = {
        "land_pairs_across_seams": int(seam_step.size),
        "mean_abs_step_across_seam_m": float(seam_step.mean()),
        "p99_abs_step_across_seam_m": float(np.percentile(seam_step, 99)),
        "mean_abs_step_first_cell_inside_m": float(in_step.mean()),
        "mean_abs_step_8_cells_inside_m": float(ref_step.mean()),
        "seam_over_inside_ratio": float(seam_step.mean() / max(in_step.mean(), 1e-9)),
    }
    out["detail_by_depth_from_seam_m"] = [round(float(x), 3) for x in (det_by_depth / np.maximum(det_cnt, 1))]
    out["detail_interior_m"] = float(det_interior / max(det_interior_n, 1))
    ratios = []
    for c in np.nonzero(cross)[0]:
        d = int(down[c])
        f0, i0, j0 = c // NN, (c % NN) // N, c % N
        f1, i1, j1 = d // NN, (d % NN) // N, d % N
        q0 = float(q[f0][i0 * R : (i0 + 1) * R, j0 * R : (j0 + 1) * R].max())
        q1 = float(q[f1][i1 * R : (i1 + 1) * R, j1 * R : (j1 + 1) * R].max())
        ratios.append(min(q0, q1) / max(max(q0, q1), 1e-9))
    ratios = np.array(ratios)
    out["discharge_ratio_across_face_links"] = {
        "n": int(ratios.size),
        "median": float(np.median(ratios)) if ratios.size else None,
        "frac_below_0.3": float((ratios < 0.3).mean()) if ratios.size else None,
    }
    ri = m["stages"].get("refine", {}).get("info", {})
    out["refine"] = {k: ri.get(k) for k in ("n_basins", "n_pieces", "cells_written", "deaths", "seconds_total")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], fine="--no-fine" not in sys.argv)
