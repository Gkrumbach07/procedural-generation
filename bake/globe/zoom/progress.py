"""Progress of a zoom bake, for whoever watches it (scripts/serve_world.py and
the globe viewer's bake panel).

A bake writes under ``<out>/progress/``:

* ``plan.json`` -- when it starts: every level with its expected work (cells
  x iterations, finer levels weighted up for their slower iterations), its
  tile count and state (``pending`` / ``running`` / ``done``), updated as
  each level starts and ends;
* ``L{R}_{a}_{b}.json`` -- each tile while it erodes, every couple of seconds:
  the iteration it is on.  Tiles run in worker processes, which find the
  directory through the environment (:data:`ENV`), so the planet bake, which
  never sets it, writes nothing.

:func:`read` turns them into a fraction, the level running and an estimate
of the time left.  Every write is a rename, so a reader never sees half a
file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ENV = "GLOBE_ZOOM_PROGRESS"
#: seconds between a tile's progress writes
TICK_S = 2.0
_last: dict[str, float] = {}


def _write(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def level_work(level, tiles: int = 1) -> float:
    """Expected work of a level: product cells x iterations, levels of 512
    and finer (whose iterations measured ~1.6x slower per cell) weighted up;
    a level cut from the planet (``iterations`` then unused) is ~0."""
    side = level.size if getattr(level, "size", 0) else level.cells * level.R
    return float(side) ** 2 * float(level.iterations) * (1.6 if level.R >= 512 else 1.0)


def write_plan(out: Path, entries: list[dict]) -> Path:
    """``entries``: ``{R, work, tiles, state}`` per level, coarse to fine."""
    d = Path(out) / "progress"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("L*_*.json"):
        f.unlink(missing_ok=True)
    _write(d / "plan.json", {"started": time.time(), "levels": entries})
    return d


def set_level(out: Path, R: int, state: str) -> None:
    p = Path(out) / "progress" / "plan.json"
    try:
        plan = json.loads(p.read_text())
    except (OSError, ValueError):
        return
    for e in plan["levels"]:
        if e["R"] == R:
            e["state"] = state
            e["t_" + state] = time.time()
    _write(p, plan)


def tile_tick(R: int, a: int, b: int, it: int, iterations: int) -> None:
    """Called by a tile each iteration; writes at most every :data:`TICK_S`
    (and on its last iteration) when :data:`ENV` names a directory."""
    d = os.environ.get(ENV)
    if not d:
        return
    key = f"{R}_{a}_{b}"
    now = time.time()
    if it + 1 < iterations and now - _last.get(key, 0.0) < TICK_S:
        return
    _last[key] = now
    try:
        _write(Path(d) / f"L{key}.json", {"R": R, "it": it + 1, "iterations": iterations, "t": now})
    except OSError:
        pass


def read(out: Path) -> dict | None:
    """``{fraction, level, level_index, levels, detail, elapsed_s, eta_s}``
    of the bake at ``out``, or None when it has written no plan."""
    d = Path(out) / "progress"
    try:
        plan = json.loads((d / "plan.json").read_text())
    except (OSError, ValueError):
        return None
    levels = plan["levels"]
    total = sum(max(e["work"], 1.0) for e in levels)
    done_work = 0.0
    cur, cur_idx, detail = None, None, ""
    for k, e in enumerate(levels):
        w = max(e["work"], 1.0)
        if e["state"] == "done":
            done_work += w
        elif e["state"] == "running":
            cur, cur_idx = e, k
            ticks = []
            for f in d.glob(f"L{e['R']}_*.json"):
                try:
                    ticks.append(json.loads(f.read_text()))
                except (OSError, ValueError):
                    pass
            tiles = max(int(e.get("tiles", 1)), len(ticks), 1)
            frac = sum(t["it"] / max(t["iterations"], 1) for t in ticks) / tiles
            done_work += w * min(frac, 1.0)
            its = sorted(ticks, key=lambda t: -t["t"])
            detail = f"{len(ticks)}/{tiles} tiles started" if tiles > 1 else ""
            if its:
                detail = (detail + ", " if detail else "") + f"iteration {its[0]['it']}/{its[0]['iterations']}"
    fraction = done_work / total if total > 0 else 0.0
    elapsed = time.time() - float(plan["started"])
    eta = elapsed * (1.0 - fraction) / fraction if fraction > 0.02 else None
    return {"fraction": round(fraction, 4), "level": None if cur is None else cur["R"], "level_index": cur_idx, "levels": [e["R"] for e in levels],
            "detail": detail, "elapsed_s": round(elapsed), "eta_s": None if eta is None else round(eta)}


__all__ = ["ENV", "TICK_S", "level_work", "write_plan", "set_level", "tile_tick", "read"]
