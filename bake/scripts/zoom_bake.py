#!/usr/bin/env python3
"""Bake a zoom of a world: a square around a spot refined to 1.2 km, 305 m
and 76 m on the earth preset, each level tiled and chained from the one
above (globe/zoom/bake.py, docs/zoom-windows.md), then the 3-D pages and
the globe viewer's zoom list.

    python scripts/zoom_bake.py --world worlds/earth-v9 --lat -12.5 --lon 131.2
    python scripts/zoom_bake.py --world worlds/earth-v9 --cell 5 239 913 --shot
    python scripts/zoom_bake.py --world worlds/earth-v9 --cell 5 239 913 --levels 8:48:200 32:16:400 128:16:150:1024

``--levels R:cells:iterations[:tile[:margin]]`` replaces the default
levels.  Output: ``<world>/zoom/<name>/`` (``L{R}.npz`` / ``.json`` /
``.html`` per level, ``view.html`` = the finest, ``zoom.json``), and
``<world>/viewer/zooms.js`` so the globe viewer marks it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("BAKE") or os.path.join(os.path.dirname(__file__), ".."))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--cell", type=int, nargs=3, metavar=("FACE", "I", "J"), help="coarse cell of the spot")
    where.add_argument("--lat", type=float, help="latitude of the spot (with --lon)")
    ap.add_argument("--lon", type=float)
    ap.add_argument("--name", default=None, help="directory under <world>/zoom (default f<face>_<i>_<j>)")
    ap.add_argument("--levels", nargs="*", default=None, help="R:cells:iterations[:tile[:margin]] per level, coarse to fine")
    ap.add_argument("--erosion", nargs="*", default=[], help="erosion overrides k=v on top of the zoom profile")
    ap.add_argument("--threads", type=int, default=0, help="numba threads of this process (default all)")
    ap.add_argument("--workers", type=int, default=0, help="tile worker processes per pass (default one per 4 cores)")
    ap.add_argument("--no-resume", action="store_true", help="re-bake levels whose files exist")
    ap.add_argument("--shot", action="store_true", help="also screenshot view.html (Playwright)")
    a = ap.parse_args(argv)

    import numba

    from globe.config import WorldParams
    from globe.io.world_store import WorldStore
    from globe.zoom import index
    from globe.zoom.bake import DEFAULT_LEVELS, ZoomLevel, run_zoom, spot_of_lonlat

    if a.threads > 0:
        numba.set_num_threads(min(a.threads, numba.config.NUMBA_NUM_THREADS))
    root = Path(a.world)
    params = WorldParams.from_dict(WorldStore(root).manifest["params"])
    if a.cell:
        spot = tuple(a.cell)
    else:
        if a.lon is None:
            ap.error("--lat needs --lon")
        spot = spot_of_lonlat(params, a.lat, a.lon)
    levels = DEFAULT_LEVELS
    if a.levels:
        levels = []
        for s in a.levels:
            v = [int(x) for x in s.split(":")]
            levels.append(ZoomLevel(*v))
        levels = tuple(levels)
    erosion = {}
    for kv in a.erosion:
        k, _, v = kv.partition("=")
        erosion[k] = float(v)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    log(f"zoom of {root} at cell {spot}")
    out = run_zoom(root, spot, levels, name=a.name, log=log, erosion=erosion or None, resume=not a.no_resume, workers=a.workers)
    zooms = index.write(root)
    info = json.loads((out / "zoom.json").read_text())
    log(f"done: {out / 'view.html'} ({info['seconds']:.0f}s; {len(zooms)} zoom(s) listed for the globe viewer)")
    if a.shot:
        from globe.zoom.view import screenshot

        errors = screenshot(out / "view.html", out / "view.png", yaw=30, pitch=38, dist=0.9)
        log(f"shot {out / 'view.png'}" + (f"  page errors: {errors[:3]}" if errors else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
