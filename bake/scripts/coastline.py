"""Coastline and shelf-edge roughness of a baked world's tectonic bedrock.

    python3 scripts/coastline.py <world-dir> [<world-dir> ...]

For the isolines at 0 m (the coastline), at -500 / -1000 / -2500 m and at
the *shelf edge* -- the midpoint of the median drowned-continental cell and
the median oceanic-crust cell, which is the isoline the viewer shades as the
outer edge of the light band -- plus the crust-type boundary itself
(``diagnostics/crust_kind``), report:

* ``L/sqrt(A)``: land/sea 4-neighbour edges within faces over the square
  root of the cells above the level.  Scale-free; a disc is 2 sqrt(pi) =
  3.54 and higher is more convoluted.
* ``fingers``: the share of the mask removed by a per-face binary opening
  with a disc of radius half a segment spacing -- the area sitting in spits,
  fingers and scallops narrower than one segment.
* ``inlets``: the same on the complement.
* ``necks``: coast cells whose 8-neighbour ring changes land/sea more than
  twice.

Why.  The viewer showed a spiky, fingered light band along every coast
before erosion had run (docs/lakes-in-erosion.md section 4): it is the
tectonic splat resolving the continental/oceanic step one segment at a
time.  ``tectonics.margin_sigma_factor`` re-positions that step with a wide
kernel (``globe/tectonics/run.py`` ``margin_ramp``) and this is the
measurement it was tuned against (docs/coast-fringe.md).  The crust-boundary
row is the control: the ramp must not move it, and the coastline row should
not move either -- the ramp is raise-only and works below sea level.

``--equal-area`` adds a row per world after the first: the level at which
that world has the *first* world's land cell count (``bed > level``), and
the metrics there.  L/sqrt(A) is not area-free for a fixed level -- a
change that shallows the sea grows the 0 m mask and shortens the isoline
whatever its shape -- so a before/after pair is read at equal area, and
the physical coast (0 m) is still printed beside it (docs/coast-fringe.md
sections 3 and 5).

The segment spacing comes from the manifest (``stages.tectonics.info
.spacing_rad``); the metric itself only needs ``coarse/bedrock`` and
``diagnostics/crust_kind``.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

#: (label, level in metres); None = the shelf edge of that world
LEVELS = (("coast", 0.0), ("-500 m", -500.0), ("-1000 m", -1000.0), ("shelf edge", None), ("-2500 m", -2500.0))


def _disc(r: float) -> np.ndarray:
    k = int(math.ceil(r))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-9


def isoline_metrics(field: np.ndarray, level: float, r_cells: float) -> dict:
    """Roughness of the isoline ``field == level`` on a (6, N, N) interior
    array: ``ratio`` = L/sqrt(A), ``fingers``, ``inlets``, ``necks`` (see
    the module docstring) and ``coast``, the number of coast cells."""
    m = field > level
    L = int((m[:, :-1, :] != m[:, 1:, :]).sum() + (m[:, :, :-1] != m[:, :, 1:]).sum())
    A = int(m.sum())
    se = _disc(r_cells)
    op = np.stack([ndimage.binary_opening(m[f], structure=se) for f in range(6)])
    opc = np.stack([ndimage.binary_opening(~m[f], structure=se) for f in range(6)])
    fingers = 1.0 - op.sum() / max(A, 1)
    inlets = 1.0 - opc.sum() / max((~m).sum(), 1)
    p = np.pad(m, ((0, 0), (1, 1), (1, 1)), mode="edge")
    ring = [p[:, :-2, 1:-1], p[:, :-2, 2:], p[:, 1:-1, 2:], p[:, 2:, 2:],
            p[:, 2:, 1:-1], p[:, 2:, :-2], p[:, 1:-1, :-2], p[:, :-2, :-2]]
    t = np.zeros(m.shape, dtype=np.int32)
    for k in range(8):
        t += ring[k] != ring[(k + 1) % 8]
    coast = m & ((~ring[0]) | (~ring[2]) | (~ring[4]) | (~ring[6]))
    necks = float((t[coast] > 2).mean()) if coast.any() else float("nan")
    return dict(ratio=L / math.sqrt(max(A, 1)), fingers=float(fingers), inlets=float(inlets),
                necks=necks, coast=int(coast.sum()))


def shelf_edge_level(bed_m: np.ndarray, crust: np.ndarray) -> float:
    """Midpoint between the median drowned continental cell and the median
    oceanic-crust cell: the isoline the viewer shades as the shelf edge."""
    dc = bed_m[crust & (bed_m < 0)]
    oc = bed_m[~crust]
    if dc.size == 0 or oc.size == 0:
        return float("nan")
    return 0.5 * (float(np.median(dc)) + float(np.median(oc)))


def load_world(root: str):
    """(bedrock m, continental mask, segment spacing in coarse cells)."""
    bed = np.stack([np.load(os.path.join(root, "coarse", f"bedrock.f{f}.npy")) for f in range(6)]).astype(np.float64)
    crust = np.stack([np.load(os.path.join(root, "diagnostics", f"crust_kind.f{f}.npy")) for f in range(6)]).astype(bool)
    with open(os.path.join(root, "manifest.json")) as fh:
        m = json.load(fh)
    spacing = float(m["stages"]["tectonics"]["info"]["spacing_rad"]) * int(m["N_c"]) / (math.pi / 2)
    return bed, crust, spacing


def equal_area_level(bed_m: np.ndarray, n_land: int) -> float:
    """The level at which exactly ``n_land`` cells of ``bed_m`` lie above it
    (the ``n_land``-th highest cell; ``bed_m > level`` has that count unless
    ties sit at the level)."""
    flat = np.sort(bed_m.ravel())[::-1]
    n = int(min(max(n_land, 1), flat.size))
    return float(flat[n - 1]) - 1e-9


def report(name: str, bed_m: np.ndarray, crust: np.ndarray, spacing_cells: float, shelf_edge: float,
           equal_area: int | None = None) -> None:
    r = 0.5 * spacing_cells
    own = shelf_edge_level(bed_m, crust)
    print(f"--- {name}")
    print(f"    land {float((bed_m > 0).mean()) * 100:5.1f} %   continental {float(crust.mean()) * 100:5.1f} %   "
          f"cont drowned {float((bed_m[crust] < 0).mean()) * 100:5.1f} %   own shelf edge {own:7.1f} m   "
          f"(spacing {spacing_cells:.2f} coarse cells, opening disc r = {r:.1f})")
    for lab, lv in LEVELS:
        lv = shelf_edge if lv is None else lv
        if not np.isfinite(lv):
            continue
        d = isoline_metrics(bed_m, lv, r)
        print(f"    {lab:>14s} @ {lv:8.1f} m : L/sqrt(A) {d['ratio']:6.3f}  fingers {d['fingers'] * 100:5.2f} %  "
              f"inlets {d['inlets'] * 100:5.2f} %  necks {d['necks'] * 100:4.1f} %  coast cells {d['coast']}")
    if equal_area is not None:
        lv = equal_area_level(bed_m, equal_area)
        d = isoline_metrics(bed_m, lv, r)
        print(f"    {'equal-area':>14s} @ {lv:8.1f} m : L/sqrt(A) {d['ratio']:6.3f}  fingers {d['fingers'] * 100:5.2f} %  "
              f"inlets {d['inlets'] * 100:5.2f} %  necks {d['necks'] * 100:4.1f} %  coast cells {d['coast']}   "
              f"(land cells {int((bed_m > lv).sum())}, the first world's {equal_area})")
    d = isoline_metrics(crust.astype(np.float64), 0.5, r)
    print(f"    {'crust boundary':>14s}            : L/sqrt(A) {d['ratio']:6.3f}  fingers {d['fingers'] * 100:5.2f} %  "
          f"inlets {d['inlets'] * 100:5.2f} %  necks {d['necks'] * 100:4.1f} %  boundary cells {d['coast']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--shelf-edge", type=float, default=None,
                    help="metres: measure this isoline as the 'shelf edge' in every world (default: the first "
                         "world's own shelf edge, so a before/after pair is read at one level -- filling the "
                         "oceanic side raises the level a world derives for itself)")
    ap.add_argument("--equal-area", action="store_true",
                    help="for every world after the first, also measure the isoline at which it has the first "
                         "world's land cell count (a change that shallows the sea grows the 0 m mask and "
                         "shortens the coast whatever its shape)")
    args = ap.parse_args()
    shelf_edge = args.shelf_edge
    n_land = None
    for w in args.worlds:
        try:
            bed, crust, spacing = load_world(w)
        except FileNotFoundError as e:
            print(f"! {w}: {e.filename} missing -- skipping", file=sys.stderr)
            continue
        if shelf_edge is None:
            shelf_edge = shelf_edge_level(bed, crust)
        report(os.path.basename(w.rstrip("/")) or w, bed, crust, spacing, shelf_edge,
               equal_area=n_land if args.equal_area else None)
        if n_land is None:
            n_land = int((bed > 0).sum())


if __name__ == "__main__":
    main()
