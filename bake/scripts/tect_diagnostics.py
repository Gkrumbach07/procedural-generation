"""Write the tectonics diagnostics a world was baked without.

    python scripts/tect_diagnostics.py --world worlds/earth-v19

Tectonics is deterministic, so its end state can be had again from the
world's own parameters (about four minutes on the earth preset).  This
reruns it, checks that the bedrock it arrives at is the bedrock the world
was baked from -- bit for bit, or the code has moved on and the diagnostics
would describe another planet -- and saves the diagnostics under
``<world>/diagnostics``: ``crust_thickness`` and ``crust_province`` (what
the geology map and the viewer's cross-sections read, derive/geology.py)
beside the ones every bake writes.  Nothing a stage hashes is touched;
``bake.py --from derive`` then draws the map.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe.config import WorldParams  # noqa: E402
from globe.io.world_store import WorldStore  # noqa: E402
from globe.tectonics import run as tect  # noqa: E402

DIAGNOSTICS = ("heat", "collision_zone", "crust_kind", "crust_age", "crust_thickness", "crust_province", "volcano_cone")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    ap.add_argument("--force", action="store_true", help="save them even where the bedrock differs from the world's")
    a = ap.parse_args(argv)
    store = WorldStore(a.world)
    params = WorldParams.from_dict(store.manifest["params"])
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    sim = tect.simulate(params, log)
    out = tect.finalise(sim)
    grid = params.coarse_grid()
    was = store.load_field("bedrock", grid).interior
    now = out["bedrock"].interior
    worst = float(np.abs(now.astype(np.float64) - was).max())
    log(f"bedrock against the world's: largest difference {worst:.6g} m")
    if worst > 0.0 and not a.force:
        log("the tectonics code no longer arrives at this world's bedrock: nothing saved (--force to save anyway)")
        return 1
    diag = store.root / "diagnostics"
    for name in DIAGNOSTICS:
        if name in out:
            out[name].save(diag)
    log(f"diagnostics -> {diag}: " + ", ".join(n for n in DIAGNOSTICS if n in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
