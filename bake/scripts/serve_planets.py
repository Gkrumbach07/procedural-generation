#!/usr/bin/env python3
"""Serve every baked world, and bake new ones.

    python scripts/serve_planets.py                       # http://localhost:8700/ , worlds/
    python scripts/serve_planets.py --worlds-dir worlds --host 100.114.208.82

The page at ``/`` lists the planets: what each has baked, how long it took,
its quicklooks, its zoom windows.  From it you can make a planet (a preset
and a few knobs), run its stages one at a time or in a run, stop one, carry
on where it stopped, bake the planet-wide 1.2 km level, and export a viewer
for a world that is only half baked -- the stages write a picture as they go,
so there is something to look at while it runs.

Each world is served whole under ``/<name>/``: its viewer is
``/<name>/viewer/index.html`` and the zoom bakes that page asks for come back
here (:mod:`globe.serve.hub`).  One job runs at a time, whatever kind it is:
a bake uses every core.

Listens on localhost unless ``--host`` says otherwise (a Tailscale address
serves the tailnet): this runs bakes on the machine for whoever reaches it.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds-dir", default="worlds", help="the directory the worlds live in")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--bake-args", nargs=argparse.REMAINDER, default=[], help="extra arguments for every bake")
    a = ap.parse_args(argv)

    from globe.serve import hub
    from globe.serve import jobs as jb

    root = Path(a.worlds_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    queue = jb.JobQueue(root / "jobs.json")
    handler = partial(hub.make_handler(root, queue, sys.executable, a.bake_args), directory=str(root))
    srv = ThreadingHTTPServer((a.host, a.port), handler)
    print(f"planets at http://{a.host}:{a.port}/  ({root})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
