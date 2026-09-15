#!/usr/bin/env python3
"""Scout a world's finished planet level for places to refine into a game map
(globe/zoom/scout.py): ranked candidates with thumbnails in
``<world>/zoom/scout/``, and markers in the globe viewer.

    python scripts/scout.py --world worlds/earth-v9
    python scripts/scout.py --world worlds/earth-v9 --window-km 16 --top 40
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
    ap.add_argument("--planet", default=None, help="planet level directory (default: the finest finished zoom/planet_R*)")
    ap.add_argument("--window-km", type=float, default=24.0)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--separation-km", type=float, default=150.0)
    a = ap.parse_args(argv)

    from globe.zoom.scout import ScoutSpec, scout

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    rec = scout(Path(a.world), a.planet, ScoutSpec(window_km=a.window_km, top=a.top, min_separation_km=a.separation_km), log=log)
    for c in rec["candidates"]:
        print(f"#{c['rank']:2d} score {c['score']:.2f}  lat {c['lat']:7.2f} lon {c['lon']:8.2f}  cell {tuple(c['spot'])}  "
              f"relief {c['measures']['relief_m']:.0f} m  creeks {100 * c['measures']['creek_share']:.1f} %  lakes {100 * c['measures']['lake_share']:.1f} %  "
              f"{c['measures']['temperature_c']:.1f} C  forest {100 * c['measures']['forest_share']:.0f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
