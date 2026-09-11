"""Stage runner: ordering, manifest, resume, quicklook hooks.

Each stage lives in ``globe.<pkg>.run`` and exposes::

    OUTPUTS: list[str]      # field names / relative paths it writes (hashed)
    def run(store: WorldStore, params: WorldParams, log=print) -> dict

``run`` reads its inputs from the store, writes its outputs and returns a
small JSON-able info dict recorded in the manifest.
"""
from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

from .config import STAGES, WorldParams
from .cubesphere import Grid
from .io.world_store import WorldStore

STAGE_MODULES = {
    "tectonics": "globe.tectonics.run",
    "climate": "globe.climate.run",
    "erosion": "globe.erosion.run",
    "hydro": "globe.hydro.run",
    "watersheds": "globe.hydro.watersheds",
    "refine": "globe.refine.run",
    "derive": "globe.derive.run",
    "tiles": "globe.refine.tiles",
}

log = logging.getLogger("globe")


def stage_module(stage: str):
    return importlib.import_module(STAGE_MODULES[stage])


def stage_range(from_stage: str | None, to_stage: str | None) -> list[str]:
    a = 0 if from_stage is None else STAGES.index(from_stage)
    b = len(STAGES) - 1 if to_stage is None else STAGES.index(to_stage)
    if b < a:
        raise ValueError("--to stage precedes --from stage")
    return list(STAGES[a : b + 1])


def bake(
    world_dir: str | Path,
    params: WorldParams,
    from_stage: str | None = None,
    to_stage: str | None = None,
    force: bool = False,
    resume: bool = True,
    logger=None,
) -> WorldStore:
    """Run ``[from_stage, to_stage]`` for the world at ``world_dir``.

    * A fresh world directory is created if needed.
    * If the directory exists with a different ``params_hash``, refuse
      unless ``force`` (then stages are rerun from ``from_stage``).
    * Stages before ``from_stage`` must be marked done in the manifest
      *with the same output-relevant parameters* (``STAGE_PARAM_GROUPS``).
    * With ``resume`` (default) stages already done inside the range and
      *not* explicitly requested by ``from_stage`` are skipped; pass
      ``force`` to rerun them.
    * Before a stage runs, it and every downstream stage are un-marked and
      the stage's ``OUTPUTS`` are deleted, so a crash mid-stage can never
      leave a 'done' marker over partial outputs, and stale files never
      leak into the new content hash.
    """
    logger = logger or (lambda msg: log.info(msg))
    params.validate()  # presets and --set mutate params after construction
    store = WorldStore(world_dir, create=True)
    grid = params.coarse_grid()
    stages = stage_range(from_stage, to_stage)

    existing = store.manifest.get("params_hash")
    if existing and existing != params.content_hash():
        if not force:
            raise RuntimeError(
                f"world {store.root} was baked with different params (hash {existing} != {params.content_hash()}); "
                "pass --force to rebake with the new params"
            )
        logger(f"params changed; invalidating stages from {stages[0]}")
        store.invalidate_from(stages[0], STAGES)
    store.init_manifest(params, grid)
    logger(f"world {store.root}: {grid.describe()}; fine N={params.N_fine}, T={params.world.T}, LODs 0..{params.max_lod}")

    first = STAGES.index(stages[0])
    for s in STAGES[:first]:
        if not store.stage_done(s, params):
            raise RuntimeError(
                f"cannot start from {stages[0]!r}: upstream stage {s!r} not done or baked with different params "
                f"in {store.manifest_path}"
            )

    for i, s in enumerate(stages):
        if resume and not force and store.stage_done(s, params) and not (i == 0 and from_stage is not None):
            logger(f"[{s}] already done (hash {store.stage_info(s)['hash']}), skipping")
            continue
        mod = stage_module(s)
        outputs = list(getattr(mod, "OUTPUTS", []))
        # un-mark this stage and everything downstream, and clear the stage's
        # outputs, *before* running so a crash cannot leave a stale 'done'
        store.invalidate_from(s, STAGES)
        store.clear_outputs(outputs)
        logger(f"[{s}] start")
        t0 = time.time()
        info = mod.run(store, params, logger) or {}
        dt = time.time() - t0
        try:
            write_stage_quicklook(mod, s, store, params, logger)
        except Exception as e:  # quicklooks must never break a bake
            logger(f"[{s}] quicklook failed: {e!r}")
        store.mark_stage(s, outputs, info, dt, params=params)
        logger(f"[{s}] done in {dt:.1f}s (hash {store.stage_info(s)['hash']})")
    if params.render.viewer:
        try:  # like quicklooks, the viewer must never break a bake
            from .viz.viewer import export_viewer

            export_viewer(store.root, formats=params.render.viewer_formats, log=logger)
        except Exception as e:
            logger(f"[viewer] export failed: {e!r}")
    return store


def write_stage_quicklook(mod, stage: str, store: WorldStore, params: WorldParams, logger) -> None:
    """Every stage writes ``quicklook/<stage>.png``.  A stage module may
    define ``quicklook(store, params, path)``; otherwise a default terrain
    view (height/bedrock + rivers + lakes, whatever exists) is rendered."""
    path = store.quicklook_path(stage)
    fn = getattr(mod, "quicklook", None)
    if fn is not None:
        fn(store, params, path)
    else:
        default_quicklook(store, params, path)
    logger(f"[{stage}] quicklook -> {path}")


def default_quicklook(store: WorldStore, params: WorldParams, path: Path) -> Path | None:
    from .viz import quicklook as ql

    grid = params.coarse_grid()
    name = "height" if store.has_field("height") else ("bedrock" if store.has_field("bedrock") else None)
    if name is None:
        return None
    h = store.load_field(name, grid)
    if store.has_field("sediment") and name == "height":
        h.data += store.load_field("sediment", grid).data
    q = store.load_field("discharge", grid) if store.has_field("discharge") else None
    w = store.load_field("water_surface", grid) if store.has_field("water_surface") else None
    return ql.quicklook_height(path, h, 0.0, grid.cell_size_m, discharge=q, water=w)


__all__ = ["bake", "stage_range", "stage_module", "STAGE_MODULES", "STAGES", "WorldStore", "Grid"]
