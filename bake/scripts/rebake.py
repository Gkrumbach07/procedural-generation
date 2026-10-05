"""Bake a world again from a stage, on the parameters in its own manifest.

    python scripts/rebake.py --world worlds/earth-v20 --from erosion
    python scripts/rebake.py --world worlds/lk-trap --from erosion --to hydro --no-viewer
    python scripts/rebake.py --world worlds/x --from hydro --set hydro.marsh_depth=0

``scripts/bake.py`` builds its parameters from a preset and ``--set``
overrides, and a world baked with overrides has to be given every one of
them again or the bake finds its parameters changed and starts over from the
first stage they touch.  This reads the parameters the world was baked with
instead -- a parameter added to the code since takes its default, which is
the point of baking again -- and reruns from ``--from``.  To try a change on
a copy, copy the world's ``manifest.json``, ``coarse``, ``diagnostics`` and
``frames/tectonics`` to a new directory and rebake that from ``erosion``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe.config import WorldParams  # noqa: E402
from globe.io.world_store import WorldStore  # noqa: E402
from globe.pipeline import STAGES, bake  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    ap.add_argument("--from", dest="from_stage", choices=STAGES, required=True)
    ap.add_argument("--to", dest="to_stage", choices=STAGES, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE", help="override a parameter (JSON value)")
    ap.add_argument("--no-viewer", action="store_true", help="skip the viewer export at the end")
    a = ap.parse_args(argv)
    params = WorldParams.from_dict(WorldStore(a.world).manifest["params"])
    over: dict = {}
    for kv in a.set:
        key, value = kv.split("=", 1)
        group, name = key.split(".", 1)
        over.setdefault(group, {})[name] = json.loads(value)
    if over:
        params = params.with_overrides(**over)
    if a.no_viewer:
        params.render.viewer = False
    t0 = time.time()
    bake(a.world, params, from_stage=a.from_stage, to_stage=a.to_stage, force=True,
         logger=lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
