#!/usr/bin/env python3
"""Why the ocean is too shallow: split the sea floor by the crust under it.

    python3 scripts/ocean_depth.py <world> [<world> ...]

Our ocean median runs around -2900 m after tectonics against Earth's
-3700, and the obvious reading -- "the abyssal plain is not deep enough" --
conflates two different populations.  ``tectonics.shelf_fraction`` places
sea level by drowning a fixed share of the *continental* crust, so a large
part of what the hypsometry calls ocean is submerged continent: shelf, not
abyss.  Earth has that too (its shelves are ~8 % of the surface), but if
ours holds far more, the median is being pulled up by crust that is not
sea floor at all and the abyssal plain may be fine.

Measured on the shipped `earth` preset, the answer is that the shelf is not
the problem: drowned continent is 15 % of the ocean by area and lifts the
median by +124 m of an 840 m gap.  The sea floor itself sits high, and it
cannot do otherwise -- fully subsided ocean floor is pinned at
``(h_ocean - sea_level) * height_scale_m`` = -4018 m, which is where Earth's
abyssal plain *starts*.  See docs/crust-audit.md.

Needs ``diagnostics/crust_kind`` (written by the tectonics stage; 1 =
continental).  Worlds baked before that field existed are reported without
the split.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from globe.config import PRESETS, WorldParams  # noqa: E402

#: Earth, for the two populations separately.  The abyssal sea floor runs
#: -3000 to -6000 m with a median near -4300; the shelves are -200 m to 0
#: and are ~8 % of the surface.
EARTH = {"ocean median m": -3700, "abyssal median m": -4300, "shelf % of globe": 8.0}


def load(world: str, params: WorldParams):
    grid = params.coarse_grid()
    bed = np.stack([np.load(os.path.join(world, "coarse", f"bedrock.f{f}.npy")) for f in range(6)])
    kind = None
    p0 = os.path.join(world, "diagnostics", "crust_kind.f0.npy")
    if os.path.exists(p0):
        kind = np.stack([np.load(os.path.join(world, "diagnostics", f"crust_kind.f{f}.npy"))
                         for f in range(6)]).astype(bool)
    return bed, kind, grid.interior_cell_area.astype(np.float64)


def wq(v, w, q):
    o = np.argsort(v)
    v, w = v[o], w[o]
    return float(np.interp(q, np.cumsum(w) / w.sum(), v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--preset", default="earth", choices=sorted(PRESETS))
    ap.add_argument("--params", default=None)
    args = ap.parse_args()
    params = WorldParams.from_yaml(args.params) if args.params else PRESETS[args.preset]()

    for w in args.worlds:
        bed, kind, area = load(w, params)
        A = area.sum()
        sea = bed <= 0
        print(f"\n{w}")
        print(f"  ocean            {area[sea].sum() / A * 100:5.1f} % of globe   "
              f"median {wq(bed[sea], area[sea], 0.5):7.0f} m   (Earth {EARTH['ocean median m']})")
        if kind is None:
            print("  no diagnostics/crust_kind -- re-bake the tectonics stage to split this")
            continue
        drowned = sea & kind
        abyss = sea & ~kind
        for name, m, ref in (("drowned continent", drowned, None), ("oceanic crust", abyss, EARTH["abyssal median m"])):
            if not m.any():
                continue
            s = f"  {name:17s} {area[m].sum() / A * 100:5.1f} % of globe   median {wq(bed[m], area[m], 0.5):7.0f} m"
            print(s + (f"   (Earth's abyssal plain {ref})" if ref else ""))
            print(f"  {'':17s} {'':5s}                p10 {wq(bed[m], area[m], 0.10):7.0f}   "
                  f"p90 {wq(bed[m], area[m], 0.90):7.0f}   deepest {bed[m].min():7.0f}")
        # how much of the ocean median is the drowned continent's doing
        if abyss.any() and drowned.any():
            med_all = wq(bed[sea], area[sea], 0.5)
            med_ab = wq(bed[abyss], area[abyss], 0.5)
            print(f"  drowned continent is {area[drowned].sum() / area[sea].sum() * 100:.0f} % of the ocean by area "
                  f"and lifts the ocean median by {med_all - med_ab:+.0f} m")
        land = bed > 0
        shelf = drowned & (bed > -200)
        print(f"  land             {area[land].sum() / A * 100:5.1f} % of globe; "
              f"continental crust {area[kind].sum() / A * 100:5.1f} % of globe "
              f"(Earth ~40 %), of which {area[drowned].sum() / max(area[kind].sum(), 1e-9) * 100:.0f} % is drowned")
        print(f"  shelf (drowned continent above -200 m) {area[shelf].sum() / A * 100:5.1f} % of globe "
              f"(Earth ~{EARTH['shelf % of globe']:.0f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
