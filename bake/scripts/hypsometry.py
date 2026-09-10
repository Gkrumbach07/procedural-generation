"""Hypsometry: how a baked world's elevation distribution compares to Earth.

    python3 scripts/hypsometry.py <world-dir> [<world-dir> ...]
    python3 scripts/hypsometry.py <world-dir> --checkpoints

Reports the elevation bands of the classic hypsographic curve as a share of
land area, plus land fraction, mean/median/max, ocean median, the fraction
of the globe within +-50 m of sea level, and -- when an erosion checkpoint
is read -- where the eroded sediment ended up.

Why these numbers.  `terrain_stats.py` measures how *dissected* a surface
is (spectral slope, drainage density, hillslope length); this measures how
its mass is distributed vertically, which is the axis that crust types, sea
level placement, orogen decay and glacial carving all move.  A world can be
perfectly dissected and still have no mountains, or the right peaks and no
coastal plain, and only this view shows it.

The Earth column is the classic hypsographic curve, given there as a
percentage of *total* surface and converted here to a share of land.

The sediment split is the one to watch offshore: on Earth shelves and
slopes trap the great majority of terrigenous sediment.  A run that puts
most of it below -200 m is bypassing its own continental margins, and
material that leaves the margin can never backfill a valley or build a
coastal plain -- so base level is never locally raised and incision never
slows.  See docs/earth-bake.md.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from globe.config import PRESETS, WorldParams  # noqa: E402

#: classic hypsographic curve, % of *total* Earth surface per band
EARTH_BANDS = ((0, 1000, 20.9), (1000, 2000, 4.5), (2000, 3000, 2.2),
               (3000, 4000, 1.1), (4000, 5000, 0.5), (5000, np.inf, 0.1))
EARTH_LAND = 29.2
EARTH_SCALARS = (("land % of globe", "29.2"), ("land mean m", "840"),
                 ("land median m", "~350"), ("max m", "8849"),
                 ("ocean median m", "-3700"), ("+-50 m of sea %", "1-2"))


def measure(bed_m, area, halo_stripped=True):
    """Scalars and bands for a (6, N, N) bedrock field in metres."""
    A = area.sum()
    land = bed_m > 0
    if not land.any():
        raise SystemExit("no land: is this field in metres and sea-levelled?")
    la = area[land].sum()
    bands = [area[(bed_m > lo) & (bed_m <= hi)].sum() / la * 100 for lo, hi, _ in EARTH_BANDS]
    return bands, {
        "land % of globe": la / A * 100,
        "land mean m": float((bed_m[land] * area[land]).sum() / la),
        "land median m": float(np.percentile(bed_m[land], 50)),
        "max m": float(bed_m.max()),
        "ocean median m": float(np.percentile(bed_m[~land], 50)),
        "+-50 m of sea %": area[np.abs(bed_m) < 50].sum() / A * 100,
    }


def sediment_split(sed_m, bed_m, area):
    tot = float((sed_m * area).sum())
    if tot <= 0:
        return None
    sh = lambda m: float((sed_m[m] * area[m]).sum()) / tot * 100  # noqa: E731
    return {"deep (< -200 m)": sh(bed_m <= -200), "shelf (-200..0)": sh((bed_m > -200) & (bed_m <= 0)),
            "coast (0..50)": sh((bed_m > 0) & (bed_m <= 50)), "land (> 50)": sh(bed_m > 50)}


def load_world(world, params):
    grid = params.coarse_grid()
    bed = np.stack([np.load(os.path.join(world, "coarse", f"bedrock.f{f}.npy")) for f in range(6)])
    return bed, grid.interior_cell_area.astype(np.float64), None


def load_checkpoint(path, params):
    grid = params.coarse_grid()
    H = grid.H
    z = np.load(path)
    hu = params.world.cell_size_m
    return (z["height"][:, H:-H, H:-H] * hu, grid.interior_cell_area.astype(np.float64),
            z["sediment"][:, H:-H, H:-H] * hu)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--preset", default="earth", choices=sorted(PRESETS))
    ap.add_argument("--params", default=None, help="YAML parameter file (overrides --preset)")
    ap.add_argument("--checkpoints", action="store_true",
                    help="also read every erosion checkpoint in each world, oldest first")
    args = ap.parse_args()

    params = WorldParams.from_yaml(args.params) if args.params else PRESETS[args.preset]()
    cols = []
    for w in args.worlds:
        try:
            bed, area, _ = load_world(w, params)
            cols.append((os.path.basename(w.rstrip("/")) or w, bed, area, None))
        except FileNotFoundError:
            print(f"! {w}: no coarse/bedrock — skipping", file=sys.stderr)
        if args.checkpoints:
            for p in sorted(glob.glob(os.path.join(w, "checkpoints", "erosion_iter*.npz"))):
                it = int(re.search(r"(\d{4})", os.path.basename(p)).group(1))
                b, a, s = load_checkpoint(p, params)
                cols.append((f"iter{it}", b, a, s))
    if not cols:
        raise SystemExit("nothing to measure")

    names = [c[0] for c in cols]
    res = [measure(c[1], c[2]) for c in cols]
    w = max(9, max(len(n) for n in names) + 1)
    print(f"{'band':>16} {'Earth':>7} " + " ".join(f"{n:>{w}}" for n in names))
    for i, (lo, hi, e) in enumerate(EARTH_BANDS):
        label = f"{lo//1000}-{'' if hi == np.inf else hi//1000} km" if hi != np.inf else ">5 km"
        print(f"{label:>16} {e / EARTH_LAND * 100:7.1f} " + " ".join(f"{r[0][i]:{w}.1f}" for r in res))
    print()
    for key, ref in EARTH_SCALARS:
        print(f"{key:>16} {ref:>7} " + " ".join(f"{r[1][key]:{w}.0f}" for r in res))

    for name, (_, bed, area, sed) in zip(names, cols):
        if sed is None:
            continue
        split = sediment_split(sed, bed, area)
        if split:
            print(f"\nsediment, {name}: " + "  ".join(f"{k} {v:.1f} %" for k, v in split.items()))


if __name__ == "__main__":
    main()
