#!/usr/bin/env python3
"""The tectonics scorecard: one table every tectonics change is judged with.

    python3 scripts/tect_scorecard.py --preset earth --seeds 0 1 2 3 --steps 4000 \\
        --every 250 --out runs/base --jobs 2 [--set tectonics.key=value ...]
    python3 scripts/tect_scorecard.py --compare runs/base/summary.json runs/fix/summary.json

Each seed runs the stage's own simulation under
:class:`globe.tectonics.diagnostics.Observer` -- a pure observer: the run is
bit-identical to one without it -- and is sampled every ``--every`` steps:
crust books and the rendered continental / land share, landmasses, plates and
their speeds, boundary lengths by kind, collisions by crust pairing, the
ocean floor's age, arcs, land components on the tect grid, rifts, and the
hypsometry in metres.  ``DIR/seed_<s>.json`` holds every sample of a seed,
``DIR/summary.json`` the mean / min / max over seeds at steps
1000 / 2000 / 4000 / 8000 (those the run reached, and its last), and the
printed table sets them beside Earth's numbers (``EARTH_TARGETS``).

Seeds run as parallel subprocesses (``--jobs``), each through ``--launcher``:
by default the repository's memory-capped wrapper ``scratch/artifact/pyc``
(4 GB, so one runaway seed is killed alone) when it is found above this
script, else this interpreter.  An Earth seed peaks well under 2 GB.
``--jobs 0`` runs the seeds in this process, one after another.

Time is reported in My through ``--myr-per-step`` (default
``tectonics.myr_per_step``, 0.15: plate speeds, ocean-floor age and
ridge_age all put the Earth preset's step at 0.09-0.25 My).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from globe.cli import load_params  # noqa: E402
from globe.tectonics import diagnostics as D  # noqa: E402
from globe.tectonics.run import initialise  # noqa: E402

#: the rows of the printed table: (flattened key, label, format); every other metric is in the JSON
TABLE = (
    ("books", None, None),
    ("books.cont_extent_share", "continental extent share", "{:.3f}"),
    ("books.rendered_cont_share", "rendered continental share", "{:.3f}"),
    ("books.land_share", "land share", "{:.3f}"),
    ("books.land_of_rendered_cont", "land / rendered continent", "{:.3f}"),
    ("books.n_continental", "continental segments", "{:.0f}"),
    ("books.cont_volume_km3", "continental volume km3", "{:.3g}"),
    ("books.crust_mass", "crust_mass (conserved)", "{:.1f}"),
    ("books.total_mass", "total_mass (progress line)", "{:.1f}"),
    ("books.ledger.subducted", "ledger subducted", "{:.2f}"),
    ("books.ledger.delaminated", "ledger delaminated", "{:.2f}"),
    ("books.ledger.orogen_decayed", "ledger orogen_decayed", "{:.2f}"),
    ("books.ledger.arc_mantle", "ledger arc_mantle", "{:.2f}"),
    ("books.ledger.extent_closed", "ledger extent_closed", "{:.2f}"),
    ("continents", None, None),
    ("continents.largest_share", "largest landmass / continental", "{:.3f}"),
    ("continents.n_ge_1pct", "landmasses >= 1 %", "{:.1f}"),
    ("plates", None, None),
    ("plates.alive", "plates alive", "{:.1f}"),
    ("plates.n_ge_1pct", "plates >= 1 %", "{:.1f}"),
    ("plates.largest_share", "largest plate share", "{:.3f}"),
    ("plates.top7_share", "top-7 share", "{:.3f}"),
    ("plates.capped_share", "plates at the speed cap", "{:.3f}"),
    ("window.births_rift", "births (window): rift", "{:.1f}"),
    ("window.births_split", "births (window): split", "{:.1f}"),
    ("window.births_other", "births (window): other", "{:.1f}"),
    ("window.deaths", "deaths (window)", "{:.1f}"),
    ("kinematics", None, None),
    ("kin.v_mean_cmyr", "mean speed cm/yr", "{:.2f}"),
    ("kin.v_median_cmyr", "median speed cm/yr", "{:.2f}"),
    ("kin.v_ocean_median_cmyr", "oceanic plates (<20 % cont) cm/yr", "{:.2f}"),
    ("kin.v_cont_median_cmyr", "continental plates (>50 %) cm/yr", "{:.2f}"),
    ("kin.v_max_cmyr", "fastest plate (rms) cm/yr", "{:.2f}"),
    ("kin.net_rotation_ratio", "net rotation / rms speed", "{:.3f}"),
    ("kin.spearman_speed_trench", "Spearman(speed, trench share)", "{:.2f}"),
    ("kin.spearman_speed_trench_ocean", "  ... oceanic plates only", "{:.2f}"),
    ("kin.spearman_speed_cont", "Spearman(speed, continental share)", "{:.2f}"),
    ("boundaries", None, None),
    ("bnd.convergent_km", "convergent km", "{:.0f}"),
    ("bnd.subduction_km", "  of which subduction (not C-C) km", "{:.0f}"),
    ("bnd.collision_cc_km", "  of which C-C km", "{:.0f}"),
    ("bnd.divergent_km", "divergent km", "{:.0f}"),
    ("bnd.transform_km", "transform km", "{:.0f}"),
    ("window.coll_per_step", "collisions / step", "{:.1f}"),
    ("window.cc_share", "  C-C share", "{:.3f}"),
    ("window.oc_share", "  O-C share", "{:.3f}"),
    ("window.oo_share", "  O-O share", "{:.3f}"),
    ("coast.passive_share", "coast: passive", "{:.3f}"),
    ("coast.active_conv_share", "coast: active converging", "{:.3f}"),
    ("coast.active_other_share", "coast: active other", "{:.3f}"),
    ("ocean floor", None, None),
    ("ocean.mean_age_myr", "mean age My", "{:.1f}"),
    ("ocean.median_over_mean", "median / mean", "{:.2f}"),
    ("ocean.p90_over_mean", "p90 / mean", "{:.2f}"),
    ("ocean.max_over_mean", "max / mean", "{:.2f}"),
    ("arcs", None, None),
    ("arcs.n", "arc segments", "{:.0f}"),
    ("arcs.planet_share", "arc share of planet", "{:.4f}"),
    ("arcs.on_converging_share", "arcs on a converging boundary", "{:.3f}"),
    ("arcs.stranded_share", "arcs stranded > 3 spacings", "{:.3f}"),
    ("window.arc_births", "arc births (window)", "{:.1f}"),
    ("arcs.land_components", "land components (tect grid)", "{:.1f}"),
    ("arcs.land_lt1e3", "  < 1e3 km2", "{:.1f}"),
    ("arcs.land_1e3_1e4", "  1e3-1e4 km2", "{:.1f}"),
    ("arcs.land_1e4_1e5", "  1e4-1e5 km2", "{:.1f}"),
    ("arcs.land_1e5_1e6", "  1e5-1e6 km2", "{:.1f}"),
    ("arcs.land_gt1e6", "  > 1e6 km2", "{:.1f}"),
    ("arcs.arc_dominated", "arc-dominated land masses", "{:.1f}"),
    ("arcs.arc_only", "  touching no other continent", "{:.1f}"),
    ("dynamics", None, None),
    ("dyn.trench400_share", "coast <= 400 km of a trench (girdle)", "{:.3f}"),
    ("dyn.largest_noarc_share", "largest landmass w/o arc crust", "{:.3f}"),
    ("dyn.landmasses_noarc_ge_1pct", "  landmasses >= 1 % w/o arcs", "{:.1f}"),
    ("dyn.arc_crust_share", "arc crust share (thick ocean + born oceanic)", "{:.4f}"),
    ("window.cc_noarc_share", "C-C collision share, non-arc", "{:.3f}"),
    ("dyn.cc_noarc_conv_median_cmyr", "non-arc C-C closing median cm/yr", "{:.2f}"),
    ("dyn.cc_noarc_conv_p90_cmyr", "  ... p90 cm/yr", "{:.2f}"),
    ("dyn.cc_noarc_conv_km", "  ... converging length km", "{:.0f}"),
    ("dyn.oc_conv_median_cmyr", "O-C closing median cm/yr", "{:.2f}"),
    ("dyn.slab_ocean_speed_cmyr", "slab-attached ocean plates cm/yr", "{:.2f}"),
    ("dyn.n_slab_ocean", "  ... how many (>= 0.5 %)", "{:.1f}"),
    ("dyn.noslab_ocean_speed_cmyr", "slab-free ocean plates cm/yr", "{:.2f}"),
    ("dyn.size_exponent", "plate-size exponent N(>A) ~ A^x", "{:.2f}"),
    ("window.ev_collapse", "margin collapses (window)", "{:.1f}"),
    ("window.ev_micro", "microplate captures (window)", "{:.1f}"),
    ("window.ev_suture", "suture welds (window)", "{:.1f}"),
    ("window.ev_heal", "healed rifts (window)", "{:.1f}"),
    ("dyn.rift_pairs_open", "rifts holding their halves", "{:.1f}"),
    ("hypsometry (tect grid, m)", None, None),
    ("hyps.land_median_m", "land median m", "{:.0f}"),
    ("hyps.ocean_median_m", "ocean median m", "{:.0f}"),
    ("hyps.land_gt1km_pct", "% land > 1 km", "{:.1f}"),
    ("hyps.land_gt2km_pct", "% land > 2 km", "{:.1f}"),
    ("hyps.max_m", "max m", "{:.0f}"),
    ("hyps.min_m", "min m", "{:.0f}"),
)

FINAL_TABLE = (
    ("plates_born", "plates born", "{:.0f}"),
    ("plates_born_rift", "  by rift", "{:.0f}"),
    ("plates_born_split", "  by split_disconnected", "{:.0f}"),
    ("plates_born_other", "  other", "{:.0f}"),
    ("plates_died", "plates died", "{:.0f}"),
    ("lifetime_median_dead", "median life of the dead, steps", "{:.1f}"),
    ("lifetime_median_dead_myr", "  ... My", "{:.2f}"),
    ("short_lived_share_dead", "share living < 10 steps", "{:.3f}"),
    ("lifetime_median_all", "median life incl. alive, steps", "{:.1f}"),
    ("rifts", "rifts", "{:.1f}"),
    ("rift_far_share", "all rifts: collisions > 90 deg out", "{:.3f}"),
    ("rift_cc_share", "  ... of which C-C", "{:.3f}"),
    ("rift_halves_min_area_median", "smaller half's area (median)", "{:.3f}"),
    ("rift_cont_n", "continental rifts (halves >= 2 % continent)", "{:.1f}"),
    ("rift_first_myr", "  the first, My", "{:.0f}"),
    ("rift_cascade_150my", "  within 150 My of the first", "{:.1f}"),
    ("rift_free_median_cmyr", "force rifts: free opening cm/yr (median)", "{:.2f}"),
    ("rift_G0_median", "  ... G0 (median)", "{:.2f}"),
    ("rift_slow_phase_median_my", "  ... slow phase My (median)", "{:.1f}"),
    ("rift_peak_open_median_cmyr", "  ... peak opening cm/yr (median)", "{:.2f}"),
    ("pooled_slab_ocean_cmyr_0_1000", "pooled slab-attached ocean cm/yr, 0-1000", "{:.2f}"),
    ("pooled_slab_ocean_cmyr_1000_2000", "  ... 1000-2000", "{:.2f}"),
    ("pooled_slab_ocean_cmyr_2000_4000", "  ... 2000-4000", "{:.2f}"),
    ("pooled_slab_ocean_cmyr_4000_8000", "  ... 4000-8000", "{:.2f}"),
    ("pooled_cont_cmyr_2000_4000", "pooled continental plates cm/yr, 2000-4000", "{:.2f}"),
    ("rift_hemi_n", "rifts of a plate > half the planet", "{:.1f}"),
    ("rift_hemi_far_share", "  the first: collisions > 90 deg out", "{:.3f}"),
    ("rift_hemi_cc_share", "  ... of which C-C", "{:.3f}"),
    ("rift_hemi_area_before", "  ... the plate it cut (share)", "{:.3f}"),
    ("rift_hemi_step", "  ... at step", "{:.0f}"),
)


def default_launcher() -> list[str]:
    env = os.environ.get("TECT_SCORECARD_LAUNCHER")
    if env:
        return shlex.split(env)
    for p in HERE.parents:
        cand = p / "scratch" / "artifact" / "pyc"
        if cand.is_file() and os.access(cand, os.X_OK):
            return [str(cand)]
    return [sys.executable]


def cell(st: dict | None, fmt: str) -> str:
    if not st or st.get("mean") is None:
        return "-"
    m, lo, hi = st["mean"], st["min"], st["max"]
    if st.get("n", 1) > 1 and lo != hi:
        return f"{fmt.format(m)} [{fmt.format(lo)}..{fmt.format(hi)}]"
    return fmt.format(m)


def print_summary(summ: dict, title: str = "") -> None:
    cps = list(summ["checkpoints"].keys())
    if title:
        print(f"\n== {title}  (seeds {summ.get('seeds')}) ==")
    w0, wc, wt = 36, 30, 40
    print(f"{'metric':<{w0}}" + "".join(f"{'step ' + c:>{wc}}" for c in cps) + f"   {'Earth':<{wt}}")
    for key, label, fmt in TABLE:
        if label is None:
            print(f"-- {key}")
            continue
        row = f"{label:<{w0}}" + "".join(f"{cell(summ['checkpoints'][c].get(key), fmt):>{wc}}" for c in cps)
        print(row + f"   {D.EARTH_TARGETS.get(key, ''):<{wt}}")
    print("-- run totals (at the end)")
    for key, label, fmt in FINAL_TABLE:
        tgt = D.EARTH_TARGETS.get("final." + key, "")
        print(f"{label:<{w0}}{cell(summ['final'].get(key), fmt):>{wc}}   {tgt}")


def print_compare(summs: list[tuple[str, dict]]) -> None:
    """Side by side: one block per checkpoint step all the files share, one column per file."""
    common = [c for c in summs[0][1]["checkpoints"] if all(c in s["checkpoints"] for _, s in summs[1:])]
    w0, wc = 36, 28
    names = [(Path(n).name if Path(n).is_dir() else Path(n).parent.name) or n for n, _ in summs]
    for c in common:
        print(f"\n== step {c} ==")
        print(f"{'metric':<{w0}}" + "".join(f"{n[:wc - 2]:>{wc}}" for n in names) + "   Earth")
        for key, label, fmt in TABLE:
            if label is None:
                print(f"-- {key}")
                continue
            print(f"{label:<{w0}}" + "".join(f"{cell(s['checkpoints'][c].get(key), fmt):>{wc}}" for _, s in summs)
                  + f"   {D.EARTH_TARGETS.get(key, '')}")
    print("\n== run totals ==")
    for key, label, fmt in FINAL_TABLE:
        print(f"{label:<{w0}}" + "".join(f"{cell(s['final'].get(key), fmt):>{wc}}" for _, s in summs))


def load_summary(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    d = json.loads(p.read_text())
    if "checkpoints" in d:
        return d
    if "samples" in d:                      # a single seed's file
        return D.summarise([d])
    raise SystemExit(f"{path}: neither a summary nor a seed file")


def run_one(args, seed: int) -> dict:
    """One seed, in this process."""
    ns = argparse.Namespace(preset=args.preset, params=args.params, seed=seed, set=args.set)
    params = load_params(ns)
    steps = int(args.steps if args.steps is not None else params.tectonics.steps)
    params.tectonics.steps = steps
    myr = float(args.myr_per_step if args.myr_per_step is not None else getattr(params.tectonics, "myr_per_step", 0.15))
    t0 = time.time()
    sim = initialise(params, log=None)
    rep = D.observe(sim, steps, args.every, myr, log=lambda s: print(f"[seed {seed}] {s}", flush=True))
    rep.update({"seed": seed, "preset": args.preset, "params_file": args.params, "set": list(args.set),
                "steps": steps, "every": int(args.every), "seconds": time.time() - t0,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0})
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None, help="default 'earth' unless --params is given")
    ap.add_argument("--params", default=None, help="a world YAML instead of a preset")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=None, help="default: the preset's tectonics.steps")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE")
    ap.add_argument("--out", default=None, help="directory for seed_<s>.json and summary.json")
    ap.add_argument("--jobs", type=int, default=1, help="seeds run at once as subprocesses (0: in this process)")
    ap.add_argument("--myr-per-step", type=float, default=None, dest="myr_per_step")
    ap.add_argument("--launcher", default=None, help="command a seed subprocess is run with (default: "
                    "$TECT_SCORECARD_LAUNCHER, else scratch/artifact/pyc above this script, else this python)")
    ap.add_argument("--compare", nargs="+", default=None, metavar="SUMMARY", help="print these side by side and exit")
    ap.add_argument("--summarise-only", action="store_true", dest="summarise_only",
                    help="run nothing: rebuild DIR/summary.json from the seed files already in --out")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.preset is None and args.params is None:
        args.preset = "earth"

    if args.compare:
        print_compare([(p, load_summary(p)) for p in args.compare])
        return 0
    if args.out is None:
        raise SystemExit("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.worker is not None:
        rep = run_one(args, args.worker)
        (out / f"seed_{args.worker}.json").write_text(json.dumps(rep))
        return 0

    t0 = time.time()
    if args.summarise_only:
        pass
    elif args.jobs <= 0:
        for s in args.seeds:
            rep = run_one(args, s)
            (out / f"seed_{s}.json").write_text(json.dumps(rep))
    else:
        launcher = shlex.split(args.launcher) if args.launcher else default_launcher()
        base = [str(HERE), "--out", str(out.resolve()), "--every", str(args.every)]
        if args.preset:
            base += ["--preset", args.preset]
        if args.params:
            base += ["--params", str(Path(args.params).resolve())]
        if args.steps is not None:
            base += ["--steps", str(args.steps)]
        if args.myr_per_step is not None:
            base += ["--myr-per-step", str(args.myr_per_step)]
        for kv in args.set:
            base += ["--set", kv]
        queue, running, failed = list(args.seeds), [], []
        print(f"[scorecard] {len(queue)} seeds, {args.jobs} at a time, via {' '.join(launcher)}", flush=True)
        while queue or running:
            while queue and len(running) < args.jobs:
                s = queue.pop(0)
                log = open(out / f"seed_{s}.log", "w")
                cmd = launcher + base + ["--worker", str(s)]
                running.append((s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parents[1])), log))
            time.sleep(2.0)
            for item in list(running):
                s, proc, log = item
                if proc.poll() is not None:
                    log.close()
                    running.remove(item)
                    msg = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode}, see seed_{s}.log)"
                    if proc.returncode != 0:
                        failed.append(s)
                    print(f"[scorecard] seed {s} {msg} ({time.time() - t0:.0f}s)", flush=True)
        if failed:
            print(f"[scorecard] failed seeds: {failed}", flush=True)
    runs = []
    for s in args.seeds:
        f = out / f"seed_{s}.json"
        if f.exists():
            runs.append(json.loads(f.read_text()))
    if not runs:
        return 1
    summ = D.summarise(runs)
    summ.update({"preset": runs[0].get("preset"), "params_file": runs[0].get("params_file"), "set": runs[0].get("set"),
                 "steps": runs[0]["steps"], "every": runs[0].get("every"), "myr_per_step": runs[0]["myr_per_step"],
                 "spacing_km": runs[0]["spacing_km"], "earth_targets": D.EARTH_TARGETS,
                 "seconds": time.time() - t0})
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print_summary(summ, f"{summ['preset'] or summ['params_file']} {' '.join(summ['set'] or [])}  ->  {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
