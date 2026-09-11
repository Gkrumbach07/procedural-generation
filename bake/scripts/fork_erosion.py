#!/usr/bin/env python3
"""Fork a baked world's erosion state at a checkpoint and run on from it.

    python3 scripts/fork_erosion.py <world> --at 600 --iterations 200 \
        --set erosion.glacial_rate=25 --label rate25m

Why this exists.  Erosion at Earth scale is ~2 hours, and most of what is
worth asking about it -- how hard the glacial pass carves, where the
offshore sediment goes -- concerns the last quarter of the run.  Re-baking
from bedrock to ask each question is the wrong instrument.  A checkpoint
carries the whole state (height, sediment, discharge, momentum, pending,
route), so a fork is exact: the arm and its baseline share every iteration
up to the fork point and differ only in what is being varied.

It drives :func:`globe.erosion.maps.step` -- the same function
``erosion/run.py`` drives -- so a fork is not a re-implementation of the
iteration and cannot drift from it.

The result is written as an ordinary checkpoint under
``<world>/forks/<label>/checkpoints/``, so

    python3 scripts/hypsometry.py <world>/forks/<label> --checkpoints

measures it against Earth with no special case.  Per-iteration diagnostics
(death causes, offshore loss, glacial carve) go to ``diagnostics.json``
beside it.

A fork changes parameters that are part of the checkpoint hash, so the
normal ``bake --from erosion`` resume would *reject* the checkpoint and
start from bedrock.  Loading it explicitly is the point.

``--deaths`` adds the particle census: where every particle made its last
deposit, how far it walked on the sea floor to get there, and how much
sediment each depth zone gained.  ``stats["deaths"]`` counts *causes* and
attaches no position to them, so without this there is nothing to check the
offshore-sediment distribution against.

Verified exact: a fork with no overrides reproduces the baseline's next
checkpoint **bit for bit** in every array, which is the property the whole
method rests on.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from globe.cli import load_params  # noqa: E402
from globe.erosion import maps, run as erun  # noqa: E402
from globe.io.world_store import WorldStore  # noqa: E402


def write_checkpoint(out: Path, state, label: str, forked_from: str, overrides) -> Path:
    """An ordinary checkpoint under ``<out>/checkpoints``, readable by
    ``hypsometry.py --checkpoints``.  The hash is deliberately not the
    baseline's: this state is a fork and must never be picked up by
    ``bake --from erosion`` as a resume point."""
    p = out / "checkpoints" / f"erosion_iter{state.iteration:04d}.npz"
    arrays = dict(height=state.height, sediment=state.sediment, discharge=state.discharge,
                  momentum=state.momentum, pending=state.pending)
    if state.route is not None:
        arrays["route"] = state.route
    if getattr(state, "iso_acc", None) is not None:
        arrays["iso_acc"] = state.iso_acc
    if getattr(state, "ice_prev", None) is not None:
        arrays["ice_prev"] = state.ice_prev.astype(np.uint8)
    with open(p, "wb") as fh:
        np.savez(fh, **arrays)
    p.with_suffix(".json").write_text(json.dumps(
        {"iteration": state.iteration, "params_hash": f"fork:{label}",
         "height_unit_m": state.height_unit_m, "forked_from": forked_from,
         "overrides": overrides}, indent=1))
    return p


def pick_checkpoint(store: WorldStore, at: int | None) -> Path:
    """The checkpoint to fork from: exactly ``at`` if given, else the newest.

    Chosen by *filename iteration* and not by hash, because the whole point
    of a fork is that the parameters no longer match.
    """
    d = store.checkpoint_dir
    found = sorted(d.glob("erosion_iter*.npz")) if d.exists() else []
    if not found:
        raise SystemExit(f"no erosion checkpoints in {d}")
    if at is None:
        return found[-1]
    want = d / f"erosion_iter{int(at):04d}.npz"
    if not want.exists():
        have = ", ".join(p.name for p in found)
        raise SystemExit(f"no checkpoint at iteration {at}; have: {have}")
    return want


#: Depth bands the death census and the mass census both use, as (label,
#: lower bound in metres).  The shelf/deep boundary is -200 m, the same one
#: `hypsometry.py` splits the sediment on, so the two measurements line up.
ZONES = (("land", 0.0), ("coast", -50.0), ("shelf", -200.0), ("slope", -1000.0), ("deep", -1e9))


def zone_of(depth_m: np.ndarray) -> np.ndarray:
    """Index into :data:`ZONES` for each depth in metres."""
    z = np.full(depth_m.shape, len(ZONES) - 1, dtype=np.int8)
    for i in range(len(ZONES) - 2, -1, -1):
        z[depth_m >= ZONES[i][1]] = i
    return z


class DeathCensus:
    """Where particles stop, and how far they walked to get there.

    A particle that reaches the sea keeps walking the seafloor for up to
    ``ocean_steps`` steps, depositing as it goes.  Whether the offshore
    sediment lands on the shelf or in the abyss is therefore a question
    about *that walk* -- how far it runs, and how deep the water is where it
    ends -- and nothing in the run reports it: ``stats["deaths"]`` counts
    causes with no position attached.

    Reads the raw change list through :func:`maps.run_iteration`'s ``diag``
    hook rather than re-tracing anything, so the census is of the particles
    the run actually used.  A particle's entries are
    ``cl_cell[p*cap : p*cap + cl_count[p]]``; a seafloor step carries
    ``cl_vol == 0`` and a final deposit ``cl_vol < 0``, so the death cell is
    the first entry with ``cl_vol < 0``.
    """

    def __init__(self):
        self.deaths = np.zeros((len(ZONES),), dtype=np.int64)          # death cell by zone
        self.by_cause = {}                                             # cause -> per-zone counts
        self.sea_steps = np.zeros(len(ZONES), dtype=np.int64)          # seafloor steps taken, by death zone
        self.no_final = 0                                              # left the window with its load

    def __call__(self, state, cl_cell, cl_vol, cl_count, cap, sp_death):
        m = cl_count.shape[0]
        n = cl_count.sum()
        if m == 0 or n == 0:
            self.no_final += m
            return
        # Gather the *used* entries only.  A (chunk, cap) mask would be
        # 2048 x 2064 at N=1024 while a particle writes ~70 entries, and the
        # hook runs 768 times an iteration, so the dense form costs more than
        # the iteration it is measuring.  `flat` is the ragged-range trick:
        # entry k of particle p lives at p*cap + k.
        starts = np.arange(m, dtype=np.int64) * cap
        part = np.repeat(np.arange(m, dtype=np.int64), cl_count)
        flat = (np.arange(n, dtype=np.int64)
                - np.repeat(np.cumsum(cl_count) - cl_count, cl_count)
                + np.repeat(starts, cl_count))
        vals = cl_vol[flat]

        # the death cell is the particle's *first* final deposit; writing the
        # hits in reverse leaves the earliest one standing
        first = np.full(m, -1, dtype=np.int64)
        hit = np.flatnonzero(vals < 0.0)
        first[part[hit][::-1]] = hit[::-1]
        has = first >= 0
        self.no_final += int((~has).sum())
        if not has.any():
            return
        cells = cl_cell[flat[first[has]]]
        surf = (state.height + state.sediment).reshape(-1)
        z = zone_of(surf[cells] * state.height_unit_m)
        self.deaths += np.bincount(z, minlength=len(ZONES))
        walked = np.bincount(part[vals == 0.0], minlength=m)[has]
        self.sea_steps += np.bincount(z, weights=walked, minlength=len(ZONES)).astype(np.int64)
        cause = sp_death[has]
        for c in np.unique(cause):
            row = self.by_cause.setdefault(int(c), np.zeros(len(ZONES), dtype=np.int64))
            row += np.bincount(z[cause == c], minlength=len(ZONES))

    def report(self) -> dict:
        from globe.erosion.particle import DEATH_NAMES
        tot = max(int(self.deaths.sum()), 1)
        out = {
            "by_zone": {ZONES[i][0]: int(self.deaths[i]) for i in range(len(ZONES))},
            "by_zone_pct": {ZONES[i][0]: round(self.deaths[i] / tot * 100, 2) for i in range(len(ZONES))},
            "mean_seafloor_steps": {ZONES[i][0]: round(self.sea_steps[i] / max(int(self.deaths[i]), 1), 1)
                                    for i in range(len(ZONES))},
            "left_window": self.no_final,
            "by_cause": {},
        }
        for c, row in sorted(self.by_cause.items()):
            name = DEATH_NAMES[c] if c < len(DEATH_NAMES) else str(c)
            out["by_cause"][name] = {ZONES[i][0]: int(row[i]) for i in range(len(ZONES))}
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world")
    ap.add_argument("--at", type=int, default=None, help="fork from this iteration (default: the newest checkpoint)")
    ap.add_argument("--fresh", action="store_true",
                    help="start from iteration 0 (the world's own bedrock, no checkpoint): run an erosion "
                         "variant from scratch on the same tectonics and climate, for changes that act from "
                         "the first iteration and so cannot be forked mid-run")
    ap.add_argument("--checkpoint", default=None,
                    help="fork from this .npz explicitly, wherever it lives.  `erosion/run.py` keeps only "
                         "the two newest checkpoints per parameter hash, so a fork point well before the "
                         "end of a run has to be copied aside before it is pruned")
    ap.add_argument("--iterations", type=int, required=True, help="how many iterations to run *past* the fork point")
    ap.add_argument("--label", required=True, help="name of this arm; output goes to <world>/forks/<label>")
    ap.add_argument("--preset", default="earth")
    ap.add_argument("--params", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE")
    ap.add_argument("--every", type=int, default=25, help="log every N iterations")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also write a checkpoint every N iterations (default: only the last)")
    ap.add_argument("--deaths", action="store_true",
                    help="also record where particles die and where the iteration's mass moves "
                         "(costs a few seconds an iteration; see `death_zones` in diagnostics.json)")
    args = ap.parse_args()

    params = load_params(args)
    store = WorldStore(args.world)
    state = erun.build_state(store, params)
    if args.fresh:
        ck = Path("fresh")            # iteration 0: bedrock as tectonics left it
    else:
        ck = Path(args.checkpoint) if args.checkpoint else pick_checkpoint(store, args.at)
        if not ck.exists():
            raise SystemExit(f"no such checkpoint: {ck}")
        erun.load_checkpoint(state, ck, json.loads(ck.with_suffix(".json").read_text()))
    start = state.iteration
    end = start + int(args.iterations)
    # `glacial_from` is a *fraction of erosion.iterations*, so the gate in
    # maps.step only lands where the baseline put it if `iterations` still
    # says what the baseline said.  A fork that ran 200 iterations with
    # iterations=200 would glaciate from iteration 150 of the fork instead.
    if int(params.erosion.iterations) < end:
        raise SystemExit(
            f"erosion.iterations={params.erosion.iterations} is below the fork end {end}: "
            "set it to the baseline's value so glacial_from gates identically")

    out = Path(args.world) / "forks" / args.label
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"fork {args.label}: {ck.name} (iteration {start}) -> {end}, "
          f"glacial_from={params.erosion.glacial_from} rate={params.erosion.glacial_rate} "
          f"max={params.erosion.glacial_max} of {params.erosion.iterations} iterations", flush=True)

    hist = []
    deaths_tot: dict[str, int] = {}
    lost_offshore = 0.0
    census = DeathCensus() if args.deaths else None
    # Change in *sediment* per depth zone, binned on the depth before each
    # iteration.  Sediment rather than surface because surface also carries
    # uplift (mean-free: it lifts land and drops the sea floor by
    # construction) and the `hold_datum` shift, either of which is larger
    # than the signal.  Below sea level the particle kernel is deposit-only,
    # so offshore this is exactly the deposition -- which is the question.
    moved = np.zeros(len(ZONES)) if args.deaths else None
    t0 = time.time()
    while state.iteration < end:
        it = state.iteration
        if args.deaths:
            before = state.sediment.copy()
            zpre = zone_of((state.height + state.sediment)[state.interior] * state.height_unit_m).ravel()
        st = maps.step(state, params, it, diag=census)
        if args.deaths:
            d = (state.sediment - before)[state.interior] * state.height_unit_m
            moved += np.bincount(zpre, weights=d.ravel(), minlength=len(ZONES))
        for k, v in st.get("deaths", {}).items():
            deaths_tot[k] = deaths_tot.get(k, 0) + int(v)
        lost_offshore += float(st.get("lost_offshore", 0.0))
        surf = (state.height + state.sediment)[state.interior]
        land = surf > 0
        row = {
            "iteration": state.iteration,
            "max_m": float(state.height[state.interior].max() * state.height_unit_m),
            "land_pct": float(np.mean(land) * 100),
            "land_median_m": float(np.median(surf[land]) * state.height_unit_m) if land.any() else 0.0,
            "above2km_pct": float(np.mean(surf[land] * state.height_unit_m > 2000) * 100) if land.any() else 0.0,
            "glacial": st.get("glacial"),
            "seconds": st.get("seconds_total"),
        }
        hist.append(row)
        if args.save_every and state.iteration % args.save_every == 0 and state.iteration != end:
            write_checkpoint(out, state, args.label, ck.name, args.set)
        if args.every and (state.iteration % args.every == 0 or state.iteration == end):
            g = row["glacial"] or {}
            print(f"  iter {state.iteration}: max {row['max_m']:.0f} m  land {row['land_pct']:.1f} %  "
                  f"median {row['land_median_m']:.0f} m  >2 km {row['above2km_pct']:.1f} %  "
                  + (f"ice {g['ice_cells']} cells carved {g['carved'] * state.height_unit_m:.3g} m "
                     f"deepest {g['max_carve'] * state.height_unit_m:.0f} m  " if g.get("ice_cells") else "")
                  + f"({time.time() - t0:.0f}s)", flush=True)

    p = write_checkpoint(out, state, args.label, ck.name, args.set)
    diagnostics = {"label": args.label, "forked_from": str(ck), "start": start, "end": end,
                   "overrides": args.set, "deaths": deaths_tot,
                   "lost_offshore_m": lost_offshore * state.height_unit_m,
                   "seconds": time.time() - t0, "history": hist}
    if census is not None:
        diagnostics["death_zones"] = census.report()
        diagnostics["sediment_change_m_by_zone"] = {ZONES[i][0]: float(moved[i]) for i in range(len(ZONES))}
        r = diagnostics["death_zones"]
        print("\nwhere particles die (share of all particles that made a final deposit):")
        for name, _ in ZONES:
            print(f"  {name:6s} {r['by_zone_pct'][name]:6.2f} %   "
                  f"mean seafloor steps walked {r['mean_seafloor_steps'][name]:6.1f}")
        print(f"  left the window without depositing: {r['left_window']}")
        print("sediment gained over the fork, metres summed per zone "
              "(offshore this is exactly the deposition: the kernel does not erode below sea level):")
        for i, (name, _) in enumerate(ZONES):
            print(f"  {name:6s} {moved[i]:+14.0f} m")
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=1))
    print(f"fork {args.label}: wrote {p} and {out / 'diagnostics.json'} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
