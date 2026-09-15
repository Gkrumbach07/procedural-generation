#!/usr/bin/env python3
"""Erode the whole planet at a zoom level's resolution (globe/zoom/planet.py):
R = 8 is 1.2 km on the earth preset, over every tile with land.

    python scripts/planet_bake.py --world worlds/earth-v9
    python scripts/planet_bake.py --world worlds/earth-v9 --faces 5 --no-finish      # one face, to time it

Output: ``<world>/zoom/planet_R{R}/``: ``L{R}.f{k}.{height,sediment,discharge,
water_surface}.npy`` per face, ``L{R}.png``, ``planet.json``, and the work
rasters and ``progress.json`` it resumes from.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("BAKE") or os.path.join(os.path.dirname(__file__), ".."))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--margin", type=int, default=64)
    ap.add_argument("--faces", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=0, help="tile processes per pass (default one per tile, up to one per core)")
    ap.add_argument("--no-finish", action="store_true", help="tiles only: no seam blend, water surface or outputs")
    a = ap.parse_args(argv)

    from globe.zoom.planet import PlanetLevel, run_planet

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    level = PlanetLevel(R=a.R, iterations=a.iterations, tile=a.tile, margin=a.margin)
    out = run_planet(Path(a.world), level, faces=a.faces, workers=a.workers, log=log, finish=not a.no_finish)
    log(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
