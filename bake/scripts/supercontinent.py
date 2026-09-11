#!/usr/bin/env python3
"""Does the continental crust break up *and* reassemble?

    python3 scripts/supercontinent.py --steps 1500 --samples 60

The supercontinent cycle is the one piece of long-run plate behaviour the
world is built to have and nobody has measured: `rift` splits a plate, and
the animation shows the Atlantic opening, but "Pangaea re-forms" is a claim
about the *second* half of the cycle and has never been checked.  Breakup
is easy to produce and easy to mistake for the cycle -- crust that
fragments and then stays fragmented looks identical for the first few
hundred steps.

What is measured, on the segment cloud rather than on the rendered grid so
that sea level and the splat kernel cannot colour the answer: continental
segments within ``link`` spacings of each other are one landmass
(``scipy.sparse.csgraph.connected_components`` over a radius graph), and

``biggest``
    the largest landmass as a share of all continental *area*.  1.0 is one
    supercontinent; Earth today is ~0.37 (Afro-Eurasia of all continental
    crust).  This is the cycle's order parameter: assembly drives it up,
    breakup drives it down.
``masses``
    how many landmasses hold at least 1 % of the continental area, which
    separates a genuine split from a few segments calving off.

A dip is not a cycle.  Crust that splits and re-merges within a sample or
two is a collision healing, so the verdict is based on how long the crust
stays apart, and ``--links`` reports several link radii so that neither
answer can be an artifact of one threshold.

The run is sampled through :meth:`TectonicSim.run`'s ``on_frame`` hook, so
the trajectory comes from the same simulation that would have been baked --
not a re-implementation, and not a second run.  Two things that come free
with having the simulation in hand are printed at the end: the belt census
(:func:`globe.tectonics.orogeny.classify`'s verdict on every collision, and
the age of every subducting slab) and the age distribution of the surviving
sea floor, which is what the ocean's depth curve is a function of.

Measured on the shipped `earth` preset: no breakup at all -- see
docs/crust-audit.md.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from globe.cli import load_params  # noqa: E402
from globe.tectonics.run import initialise  # noqa: E402
from globe.tectonics.segments import CONTINENTAL, OCEANIC  # noqa: E402


def landmasses(seg, spacing_rad: float, link: float = 1.6):
    """(biggest share of continental area, count of masses >= 1 %, n components)."""
    m = seg.kind == CONTINENTAL
    if not m.any():
        return 0.0, 0, 0
    pos, area = seg.pos[m], np.maximum(seg.area[m], 0.0)
    # chord length of the angular link radius: the positions are unit vectors
    r = 2.0 * np.sin(0.5 * min(link * spacing_rad, np.pi))
    pairs = cKDTree(pos).query_pairs(r, output_type="ndarray")
    n = pos.shape[0]
    if pairs.size:
        g = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    else:
        g = coo_matrix((n, n))
    ncomp, lab = connected_components(g, directed=False)
    tot = float(area.sum())
    if tot <= 0:
        return 0.0, 0, ncomp
    per = np.bincount(lab, weights=area, minlength=ncomp)
    return float(per.max() / tot), int((per / tot >= 0.01).sum()), int(ncomp)


def seafloor_age(seg, ridge_age: float) -> dict:
    """Age of the surviving ocean floor, weighted by segment area.

    The seafloor's depth is a function of its age and nothing else
    (``ridge_buoyancy``: ``max(0, 1 - sqrt(age / ridge_age))``), so this is
    the ocean's hypsometry stated in the variable that actually drives it.
    A conveyor that spawns crust at a gap and subducts it a few steps later
    never lets any of it finish subsiding, and the abyssal plain the depth
    curve needs is simply never built.  Earth's seafloor has a median age
    around a third of the ~80 My at which subsidence is done.
    """
    m = seg.kind == OCEANIC
    if not m.any():
        return {}
    age, area = seg.age[m], np.maximum(seg.area[m], 1e-12)
    o = np.argsort(age)
    age, area = age[o], area[o]
    c = np.cumsum(area) / area.sum()
    q = {f"p{p}": float(np.interp(p / 100.0, c, age)) for p in (10, 25, 50, 75, 90)}
    q["subsided_pct"] = float(area[age >= ridge_age].sum() / area.sum() * 100)
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="earth")
    ap.add_argument("--params", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--link", type=float, default=1.6, help="landmass link radius, in segment spacings")
    ap.add_argument("--links", type=float, nargs="*", default=(1.1, 1.6, 2.5),
                    help="extra link radii to report alongside --link, so the answer is not an "
                         "artifact of one threshold")
    ap.add_argument("--out", default=None, help="write the trajectory as JSON here")
    args = ap.parse_args()

    params = load_params(args)
    steps = int(args.steps if args.steps is not None else params.tectonics.steps)
    sim = initialise(params, log=print)
    rows: list[dict] = []

    links = sorted({round(float(x), 3) for x in (*args.links, args.link)})

    def sample(s, i):
        big, n1, ncomp = landmasses(s.seg, s.spacing, args.link)
        alt = {str(L): landmasses(s.seg, s.spacing, L)[0] for L in links}
        cont = float((s.seg.kind == CONTINENTAL).mean())
        sf = seafloor_age(s.seg, float(params.tectonics.ridge_age))
        rows.append({"step": int(i), "biggest": big, "masses": n1, "components": ncomp,
                     "continental_fraction": cont, "plates": int(s.plates.n_alive()),
                     "biggest_by_link": alt, "seafloor_age": sf})
        print(f"  step {i:5d}: biggest {big:.3f}  masses>=1% {n1:2d}  components {ncomp:4d}  "
              f"cont {cont:.3f}  plates {s.plates.n_alive()}  "
              f"link " + " ".join(f"{L}:{alt[str(L)]:.2f}" for L in links)
              + (f"  seafloor age p50 {sf['p50']:.0f} subsided {sf['subsided_pct']:.1f} %" if sf else ""),
              flush=True)

    sim.run(steps, log=print, on_frame=sample, frames=int(args.samples))

    big = np.array([r["biggest"] for r in rows])
    st = np.array([r["step"] for r in rows])
    # a cycle needs a fall and then a rise: the deepest point after the start,
    # and the best recovery after it
    i_min = int(np.argmin(big))
    after = big[i_min:]
    summary = {
        "steps": steps, "start_biggest": float(big[0]), "min_biggest": float(big[i_min]),
        "min_at_step": int(st[i_min]), "recovery_biggest": float(after.max()),
        "recovery_at_step": int(st[i_min + int(np.argmax(after))]),
        "end_biggest": float(big[-1]), "max_masses": int(max(r["masses"] for r in rows)),
    }
    print("\nsupercontinent cycle:")
    print(f"  start          {summary['start_biggest']:.3f}")
    print(f"  minimum        {summary['min_biggest']:.3f} at step {summary['min_at_step']}  (breakup)")
    print(f"  best recovery  {summary['recovery_biggest']:.3f} at step {summary['recovery_at_step']}  (reassembly)")
    print(f"  end            {summary['end_biggest']:.3f};  most landmasses >=1 %: {summary['max_masses']}")
    sf = rows[-1].get("seafloor_age") or {}
    if sf:
        ra = float(params.tectonics.ridge_age)
        print(f"\nsurviving ocean floor at step {steps}, age in tectonic steps "
              f"(ridge_age = {ra:.0f}, the age at which subsidence is done):")
        print("  " + "  ".join(f"{k} {sf[k]:.0f}" for k in ("p10", "p25", "p50", "p75", "p90")))
        print(f"  fully subsided (age >= ridge_age): {sf['subsided_pct']:.1f} % of ocean floor by area")
    if sim.belt_census:
        # free with this run: the same simulation classified every collision
        print("\nbelt census (collisions classified, whole run):")
        hist = sim.belt_census.get("slab_age_hist")
        for k, v in sorted(sim.belt_census.items()):
            if k != "slab_age_hist":
                print(f"  {k:12s} {v:8d}")
        if hist:
            from globe.tectonics.orogeny import SLAB_AGE_BINS
            tot = max(sum(hist), 1)
            edges = list(SLAB_AGE_BINS) + ["inf"]
            print("  subducting slab age (steps), share of oceanic slabs:")
            for b, n in enumerate(hist):
                print(f"    {edges[b]:>4}-{edges[b + 1]:<4} {n:8d}  {n / tot * 100:5.1f} %")
        summary["belts"] = {k: v for k, v in sim.belt_census.items()}
    # A cycle is not a dip: crust that splits and re-merges within one or two
    # samples is a collision healing, not an ocean opening.  Ask instead how
    # long the crust stayed apart -- Pangaea has been broken for ~180 My of a
    # ~400 My cycle, so a real breakup occupies a large fraction of the run.
    apart = 0.7
    below = big < apart
    longest = held = 0
    for v in below:
        held = held + 1 if v else 0
        longest = max(longest, held)
    span = steps / max(len(big) - 1, 1)
    summary["apart_threshold"] = apart
    summary["samples_below"] = int(below.sum())
    summary["longest_spell_steps"] = int(longest * span)
    rec = summary["recovery_biggest"] - summary["min_biggest"]
    print(f"  recovery after the minimum: {rec:+.3f} of continental area")
    print(f"  steps with the crust apart (biggest < {apart}): "
          f"{int(below.sum() * span)} of {steps}; longest unbroken spell {summary['longest_spell_steps']} steps")
    if below.sum() == 0:
        print("  VERDICT: no breakup. The continental crust is one mass for the whole run; "
              "a dip in `biggest` that heals within a sample or two is a collision, not an ocean.")
    elif longest * span < 0.1 * steps:
        print("  VERDICT: breakup is transient -- it never holds for a tenth of the run, "
              "so there is no cycle to reassemble from.")
    else:
        print("  VERDICT: the crust breaks up and stays apart; reassembly is the number above.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "history": rows}, fh, indent=1)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
