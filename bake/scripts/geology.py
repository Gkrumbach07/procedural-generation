"""Draw the geologic map of a world baked before there was one.

    python scripts/tect_diagnostics.py --world worlds/earth-v19      # once: crust thickness and provinces
    python scripts/geology.py --world worlds/earth-v19

Runs the derive stage again on the world's own parameters -- seconds, and
its hashed outputs come out as they were -- which now also writes the coarse
``rock`` and ``basement`` fields (derive/geology.py), then exports the
viewer with its Geology layer and cross-sections.  ``--no-viewer`` stops
after the fields.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe.config import WorldParams  # noqa: E402
from globe.derive import run as derive  # noqa: E402
from globe.io.world_store import WorldStore  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    ap.add_argument("--no-viewer", action="store_true")
    a = ap.parse_args(argv)
    store = WorldStore(a.world)
    params = WorldParams.from_dict(store.manifest["params"])
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    info = derive.run(store, params, log)
    geo = info.get("geology")
    if geo is None:
        return 1
    for where in ("land", "sea"):
        log(f"{where}: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in sorted(geo[where].items(), key=lambda kv: -kv[1])))
    if not a.no_viewer and params.render.viewer:
        from globe.viz.viewer import export_viewer

        export_viewer(store.root, formats=params.render.viewer_formats, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
