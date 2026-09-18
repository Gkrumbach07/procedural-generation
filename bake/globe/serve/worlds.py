"""What a directory of worlds holds, and how a new one is started.

A world is a directory with a ``manifest.json`` (:class:`globe.io.world_store.
WorldStore`); :func:`scan` reads the directory the bakes live in and returns
one record per world -- its size, its seed, which stages are done and how
long each took, what there is to look at (the stage quicklooks, the viewer,
the zoom windows).  That is everything the planet page shows without opening
a world.

A world made here keeps its parameters in ``<world>/params.yaml``, and every
bake of it runs with ``--params`` pointing at that file, so the knobs a page
set are the knobs the next stage uses.  :data:`KNOBS` is the set the page
offers, with the defaults read from :class:`globe.config.WorldParams` itself
so they cannot drift from the code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import PRESETS, STAGES, WorldParams

#: a world's parameters as the page and the bakes read them
PARAMS_NAME = "params.yaml"
#: a name we are willing to make a directory for
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _knob(group: str, key: str, label: str, hint: str = "", lo=None, hi=None, step=None, choices=None) -> dict:
    d = WorldParams()
    value = getattr(getattr(d, group), key)
    k = {"group": group, "key": key, "name": f"{group}.{key}", "label": label, "hint": hint,
         "type": "int" if isinstance(value, bool) is False and isinstance(value, int) else
                 ("float" if isinstance(value, float) else "str"),
         "default": value}
    for n, v in (("lo", lo), ("hi", hi), ("step", step), ("choices", choices)):
        if v is not None:
            k[n] = v
    return k


#: the parameters the page offers.  Everything else keeps the preset's value:
#: these are the ones a planet is actually shaped by (docs/erosion-tuning.md,
#: docs/tectonics.md), and each is a knob whose effect is known.
KNOBS: list[dict] = [
    _knob("world", "seed", "Seed", "the whole world follows from it; a new seed is a new planet", lo=0, hi=2**31 - 1),
    _knob("world", "N_c", "Grid", "coarse cells a face; the cost of every stage goes with its square", choices=[128, 256, 512, 1024, 2048]),
    _knob("world", "cell_size_m", "Cell", "metres a coarse cell, so the planet's size", lo=10.0, hi=40000.0, step=1.0),
    _knob("world", "land_fraction", "Land", "share of the surface above sea level", lo=0.05, hi=0.8, step=0.01),
    _knob("tectonics", "steps", "Plate steps", "how long the plates run; the cycle needs a few thousand", lo=100, hi=8000, step=100),
    _knob("tectonics", "initial_plates", "Plates", "plates it starts with; more means smaller continents", lo=3, hi=24),
    _knob("tectonics", "continental_fraction", "Continental crust", "share of the crust that starts continental", lo=0.2, hi=0.9, step=0.05),
    _knob("tectonics", "cratons", "Cratons", "old hard cores the continents keep", lo=0, hi=64),
    _knob("tectonics", "rift_every", "Rift every", "steps between rifts; small means a busier planet", lo=50, hi=2000, step=50),
    _knob("climate", "T_eq", "Equator", "°C at the equator at sea level", lo=-10.0, hi=45.0, step=0.5),
    _knob("climate", "precip_mean", "Rain", "mean rain over the land, 1 = Earth's", lo=0.2, hi=3.0, step=0.1),
    _knob("climate", "lapse", "Lapse", "°C lost a kilometre of height", lo=3.0, hi=10.0, step=0.1),
    _knob("erosion", "iterations", "Erosion", "iterations of the coarse erosion", lo=10, hi=4000, step=10),
    _knob("refine", "refine_iterations", "Refine", "iterations of the fine grid's erosion", lo=10, hi=800, step=10),
]
_BY_NAME = {k["name"]: k for k in KNOBS}


def params_path(root: Path) -> Path:
    return Path(root) / PARAMS_NAME


def params_of(root: Path) -> WorldParams:
    """The world's parameters: its ``params.yaml`` where it has one, else the
    manifest's (a world baked before the page existed)."""
    root = Path(root)
    p = params_path(root)
    if p.exists():
        return WorldParams.from_yaml(p)
    return WorldParams.from_dict(json.loads((root / "manifest.json").read_text())["params"])


def write_params(root: Path, params: WorldParams) -> Path:
    """Keep ``params.yaml`` beside the world, for the next stage to run with."""
    params.validate()
    path = params_path(Path(root))
    params.to_yaml(path)
    return path


def ensure_params(root: Path) -> Path:
    """A world baked from the command line has no ``params.yaml``; give it the
    manifest's so the page can run its stages."""
    root = Path(root)
    if not params_path(root).exists():
        write_params(root, params_of(root))
    return params_path(root)


def apply(params: WorldParams, sets: dict) -> WorldParams:
    """Set the knobs of ``sets`` (``{"tectonics.steps": 2000, ...}``) on a
    copy of ``params``.  Unknown names and values out of a knob's range are
    refused -- this takes what a page sent."""
    out = WorldParams.from_dict(params.to_dict())
    for name, value in (sets or {}).items():
        knob = _BY_NAME.get(str(name))
        if knob is None:
            raise ValueError(f"{name}: not a parameter this page sets")
        if knob["type"] == "int":
            v = int(round(float(value)))
        elif knob["type"] == "float":
            v = float(value)
        else:
            v = str(value)
        if knob.get("choices") and v not in knob["choices"]:
            raise ValueError(f"{name}: {v} is not one of {knob['choices']}")
        if knob.get("lo") is not None and v < knob["lo"]:
            raise ValueError(f"{name}: {v} is below {knob['lo']}")
        if knob.get("hi") is not None and v > knob["hi"]:
            raise ValueError(f"{name}: {v} is above {knob['hi']}")
        setattr(getattr(out, knob["group"]), knob["key"], v)
    out.validate()
    return out


def create(worlds_dir: Path, name: str, preset: str = "earth", sets: dict | None = None) -> Path:
    """Make a world directory with the parameters a page asked for.  Nothing
    is baked: the stages are jobs of their own."""
    if not NAME_RE.match(name or ""):
        raise ValueError(f"{name!r}: a name is letters, digits, dot, dash or underscore")
    if preset not in PRESETS:
        raise ValueError(f"{preset!r}: not a preset ({', '.join(sorted(PRESETS))})")
    root = Path(worlds_dir) / name
    if root.exists():
        raise FileExistsError(f"{root} is already there")
    params = apply(PRESETS[preset](), sets or {})
    root.mkdir(parents=True)
    write_params(root, params)
    (root / "created.json").write_text(json.dumps({"preset": preset, "sets": sets or {}}, indent=1))
    return root


def _dir_bytes(path: Path, cap: int = 200_000) -> int:
    """Bytes under ``path``, giving up after ``cap`` files (a world holds
    thousands of tiles; the number is for a page, not for accounting)."""
    total = n = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
            n += 1
            if n >= cap:
                break
    return total


def quicklooks(root: Path) -> list[str]:
    """The stage pictures a world has, newest kind first."""
    d = Path(root) / "quicklook"
    if not d.is_dir():
        return []
    named = [f"quicklook/{s}.png" for s in STAGES if (d / f"{s}.png").exists()]
    iters = sorted((p for p in d.glob("erosion_iter*.png")), key=lambda p: p.name)
    return named + [f"quicklook/{p.name}" for p in iters[-1:]]


def describe(root: Path, sizes: bool = False) -> dict:
    """One world, as the page lists it."""
    root = Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError):
        manifest = {}
    stages = manifest.get("stages", {}) or {}
    params = manifest.get("params") or {}
    if not params and params_path(root).exists():
        params = WorldParams.from_yaml(params_path(root)).to_dict()
    world = params.get("world", {}) or {}
    done = [s for s in STAGES if (stages.get(s) or {}).get("done")]
    zooms = sorted(p.name for p in (root / "zoom").glob("*") if (p / "zoom.json").exists()) if (root / "zoom").is_dir() else []
    planet = sorted(p.name for p in (root / "zoom").glob("planet_R*") if (p / "planet.json").exists()) if (root / "zoom").is_dir() else []
    rec = {
        "name": root.name,
        "created": manifest.get("created"),
        "seed": world.get("seed"),
        "N_c": world.get("N_c"),
        "cell_size_m": world.get("cell_size_m"),
        "land_fraction": world.get("land_fraction"),
        "stages": {s: {"done": bool((stages.get(s) or {}).get("done")),
                       "seconds": (stages.get(s) or {}).get("seconds"),
                       "finished": (stages.get(s) or {}).get("finished")} for s in STAGES},
        "done": done,
        "next": next((s for s in STAGES if s not in done), None),
        "baked": bool(done),
        "viewer": (root / "viewer" / "index.html").exists(),
        "quicklooks": quicklooks(root),
        "zooms": zooms,
        "planet_levels": planet,
        "params_file": params_path(root).exists(),
    }
    if sizes:
        rec["bytes"] = _dir_bytes(root)
    return rec


def scan(worlds_dir: Path, sizes: bool = False) -> list[dict]:
    """Every world under ``worlds_dir``, newest first, then the directories
    that have parameters but nothing baked yet."""
    d = Path(worlds_dir)
    out = []
    for p in sorted(d.iterdir()) if d.is_dir() else []:
        if not p.is_dir():
            continue
        if (p / "manifest.json").exists() or params_path(p).exists():
            out.append(describe(p, sizes=sizes))
    out.sort(key=lambda r: (r.get("created") or "", r["name"]), reverse=True)
    return out


__all__ = ["PARAMS_NAME", "NAME_RE", "KNOBS", "params_path", "params_of", "write_params", "ensure_params",
           "apply", "create", "quicklooks", "describe", "scan"]
