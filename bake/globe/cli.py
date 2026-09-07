"""Command line entry points: ``globe-bake`` and ``globe-inspect``."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import PRESETS, STAGES, WorldParams


def _logger(verbose: bool):
    t0 = time.time()

    def logf(msg: str):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    return logf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="globe-bake", description="Bake a globe world (PLAN.md).")
    ap.add_argument("--world", required=True, help="world name (directory under --worlds-dir) or a path")
    ap.add_argument("--worlds-dir", default="worlds")
    ap.add_argument("--params", help="YAML parameter file")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None, help="parameter preset (default: 'default')")
    ap.add_argument("--seed", type=int, default=None, help="override world.seed")
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE", help="override a parameter, e.g. erosion.iterations=100")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default=None)
    ap.add_argument("--to", dest="to_stage", choices=STAGES, default=None)
    ap.add_argument("--only", choices=STAGES, default=None, help="run exactly one stage")
    ap.add_argument("--force", action="store_true", help="rerun stages even if done / params changed")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    params = load_params(args)
    world_dir = Path(args.world) if ("/" in args.world or args.world.startswith(".")) else Path(args.worlds_dir) / args.world
    from .pipeline import bake

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    from_stage, to_stage = args.from_stage, args.to_stage
    if args.only:
        from_stage = to_stage = args.only
    bake(world_dir, params, from_stage, to_stage, force=args.force, resume=not args.no_resume, logger=_logger(args.verbose))
    return 0


def load_params(args) -> WorldParams:
    if args.params:
        params = WorldParams.from_yaml(args.params)
        if args.preset:
            raise SystemExit("use either --params or --preset")
    else:
        params = PRESETS[args.preset or "default"]()
    if args.seed is not None:
        params.world.seed = int(args.seed)
    for s in args.set:
        key, _, val = s.partition("=")
        group, _, name = key.partition(".")
        grp = getattr(params, group)
        cur = getattr(grp, name)
        typ = type(cur)
        setattr(grp, name, typ(val) if typ is not bool else val.lower() in ("1", "true", "yes"))
    return params


def inspect_main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="globe-inspect", description="Dump stats / images for a baked world.")
    ap.add_argument("world", help="world directory")
    ap.add_argument("--field", action="append", default=[], help="field name to render (repeatable); default: all")
    ap.add_argument("--out", default=None, help="output directory for images (default: <world>/quicklook/inspect)")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args(argv)
    from .io.world_store import WorldStore
    from .viz import quicklook as ql

    store = WorldStore(args.world)
    params = store.params()
    grid = params.coarse_grid()
    print(f"world {store.root}: {grid.describe()}")
    for s, info in store.manifest.get("stages", {}).items():
        print(f"  stage {s:12s} done={info.get('done')} hash={info.get('hash')} seconds={info.get('seconds')}")
    names = args.field or store.field_names()
    out = Path(args.out or (store.quicklook_dir / "inspect"))
    for n in names:
        f = store.load_field(n, grid)
        st = f.stats()
        print(f"  {n:16s} {st['dtype']:8s} min={st['min']:.4g} max={st['max']:.4g} mean={st['mean']:.4g} std={st['std']:.4g}")
        if args.stats_only:
            continue
        if f.ncomp == 2:
            img = ql.render_vector(f, base=ql.render_scalar(f.vec_norm()))
        elif "id" in n or f.dtype.kind in "iu":
            img = ql.render_labels(f)
        elif n in ("height", "bedrock"):
            img = ql.render_height(f, 0.0, grid.cell_size_m)
        else:
            img = ql.render_scalar(f)
        ql.save_image(out / f"{n}.png", img)
    print(f"images in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
