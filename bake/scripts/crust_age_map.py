#!/usr/bin/env python3
"""Where the crust under a planet came from, and whether the run made it.

    python3 scripts/crust_age_map.py --preset small --steps 9000 \
        --set tectonics.variable_extent=true --set tectonics.slab_pull=1000

Runs the plate simulation (no bake: tectonics only) and maps
``diagnostics/crust_age`` -- steps since the crust under a cell formed --
with a ramp for each kind of crust, the way the viewer draws it: the sea
floor hot at the ridges through to the cold blue of old floor, the
continents pale where they last grew to the dark red of a craton.

The point of the map is the *continents*.  A run that starts with its
continental crust already assembled and never breaks it up gives every
continental cell the same age, and the only structure a crust-age map can
then show is invented (``tectonics.run.inherited_age``, which gives the
crust that was there from the start a past the run did not simulate).  A
run that cycles -- assembling, rifting and re-assembling, which needs
``variable_extent`` and ``slab_pull`` (docs/plate-forces.md section 4c) --
makes its own: belts of crust welded at each collision, with the oldest
cores between them.  ``--prehistory 0`` turns the invented past off so the
map shows only what the run itself produced, which is what the printed
spread measures.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globe.cli import load_params  # noqa: E402
from globe.tectonics import run as tect  # noqa: E402
from globe.viz import quicklook as ql  # noqa: E402

#: the two ramps, as globe/viz/viewer.html draws them (keep in sync)
OCEAN_RAMP = np.array([[.80, .12, .16], [.96, .58, .20], [.88, .90, .38], [.24, .62, .56], [.06, .16, .42]])
CONT_RAMP = np.array([[.93, .93, .72], [.80, .74, .40], [.68, .50, .26], [.52, .26, .20], [.30, .09, .13]])


def ramp(t: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Piecewise-linear colour ramp over ``stops`` (n, 3), t in [0, 1]."""
    x = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(x.astype(np.int64), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def colour(age: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """Age (steps) -> RGB, a log scale per kind of crust with the tails clipped,
    as :func:`globe.viz.viewer.channel_specs` does it."""
    img = np.zeros(age.shape + (3,), np.float64)
    for sel, stops in ((~cont, OCEAN_RAMP), (cont, CONT_RAMP)):
        if not sel.any():
            continue
        v = age[sel]
        lo = max(float(np.percentile(v, 1)), 1.0)
        hi = max(float(np.percentile(v, 99)), lo * 2.0)
        img[sel] = ramp(np.log(np.maximum(v, lo) / lo) / math.log(hi / lo), stops)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--params", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--prehistory", type=float, default=None,
                    help="runs' worth of invented continental past (default: the code's PREHISTORY_RUNS; 0 = only what this run made)")
    ap.add_argument("--out", default="crust_age.png")
    a = ap.parse_args(argv)

    params = load_params(a)
    steps = int(a.steps if a.steps is not None else params.tectonics.steps)
    if a.prehistory is not None:
        tect.PREHISTORY_RUNS = float(a.prehistory)

    sim = tect.simulate(params, log=print, steps=steps)
    out = tect.finalise(sim)
    age = np.asarray(out["crust_age"].interior, np.float64)
    cont = np.asarray(out["crust_kind"].interior).astype(bool)
    bed = np.asarray(out["bedrock"].interior, np.float64)

    def spread(v):
        q = np.percentile(v, [5, 25, 50, 75, 95])
        return f"p5 {q[0]:8.0f}  p25 {q[1]:8.0f}  p50 {q[2]:8.0f}  p75 {q[3]:8.0f}  p95 {q[4]:8.0f}"

    print(f"\ncrust age after {steps} steps (steps since the crust formed)")
    print(f"  ocean      {spread(age[~cont])}")
    print(f"  continent  {spread(age[cont])}")
    c = age[cont]
    if c.size:
        # what a crust-age map can actually show on land: the spread of the continental ages
        # against the run, and how much of the continent is *not* as old as the run itself
        younger = float((c < 0.95 * steps).mean())
        iqr = float(np.percentile(c, 75) - np.percentile(c, 25))
        print(f"  continental spread: IQR {iqr:.0f} steps ({iqr / max(steps, 1):.2f} runs), "
              f"{younger * 100:.0f} % of it younger than the run")
        print("  -> " + ("the run made its own age structure" if younger > 0.25 and iqr > 0.1 * steps
                         else "flat: every continent is as old as the run, so only an invented past can colour it"))
    print(f"  land {float((bed >= 0).mean()) * 100:.1f} % of cells, continental crust {float(cont.mean()) * 100:.1f} %")

    ql.save_image(a.out, colour(age, cont), net=True)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
