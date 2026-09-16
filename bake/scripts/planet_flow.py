#!/usr/bin/env python3
"""Write the river map of a finished planet level (globe/zoom/planet_finish.py
``flow_faces``): rain accumulated down each face's final surface, rivers
handed over where they cross a cube edge -- the field the viewer draws rivers
from.  ``planet_bake.py`` writes it when it finishes; this adds or refreshes
it on a level finished before that.

    python scripts/planet_flow.py --world worlds/earth-v9 --level planet_R8

All six faces together (~2 min, ~4 GB at R = 8); ``--faces`` routes faces
alone, with the planet's coarse inflow at their edges.  Output:
``L{R}.f{k}.flow.npy`` in the level's directory, and ``flow`` in its
``planet.json``.
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
    ap.add_argument("--level", default="planet_R8", help="the level's directory under <world>/zoom")
    ap.add_argument("--faces", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)

    from globe.zoom import planet_finish as pf

    root = Path(a.world)
    out = root / "zoom" / a.level
    info_path = out / "planet.json"
    info = json.loads(info_path.read_text())
    R = int(info["level"]["R"])
    t0 = time.time()
    if a.faces is None:
        # all six: rivers handed over at the cube edges
        stats = pf.flow_faces(root, out, R, log=lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True))
        info["flow"] = stats
    else:
        # some faces alone (the planet's coarse inflow at their edges)
        flow = {int(s["face"]): s for s in info.get("flow", []) if "face" in s}
        for f in a.faces:
            flow[f] = pf.flow_face(root, out, R, f)
            print(f"[{time.time() - t0:7.1f}s] face {f}: {flow[f]}", flush=True)
        info["flow"] = [flow[f] for f in sorted(flow)]
    info_path.write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
