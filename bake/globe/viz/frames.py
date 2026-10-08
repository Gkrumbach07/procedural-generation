"""Timeline frames for the viewer, captured *during* a bake.

A frame is a small snapshot of the surface at one point of the run --
tectonics step or erosion iteration -- downsampled to ``render.frame_res``
cells per face and written to ``<world>/frames/<stage>/<key>.npz`` with a
JSON sidecar.  Capture only reads the simulation state, so a bake with and
without frames produces identical outputs, and every ``render.*`` knob is
excluded from the content hash (``config.RUNTIME_KNOBS``).

Arrays are interior-only ``(6, r, r)``, indexed ``[face, i, j]`` like every
coarse field.  ``height`` is float16: metres for erosion frames, bedrock
units for tectonics frames (the metres-per-unit scale is only fixed in
``tectonics.finalise``, so the exporter applies it from the manifest).
An erosion frame also carries the water and the ice the stage was working
with when it was taken -- ``lake`` (metres of standing water, float16) and
``ice`` (the share of a frame cell under ice, uint8) -- which the height
alone does not show: a lake stands above sea level, and the glacial pass is
a quarter of the run that looked like the rest of it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

STAGES = ("tectonics", "erosion")


def frames_dir(root, stage: str) -> Path:
    return Path(root) / "frames" / stage


def downsample(a: np.ndarray, res: int, how: str = "mean") -> np.ndarray:
    """``(6, N, N)`` -> ``(6, res, res)``.  Block mean / max when ``res``
    divides ``N``; strided nearest otherwise and always for ``"nearest"``
    (categorical fields)."""
    F, N = a.shape[0], a.shape[1]
    if res >= N:
        return a
    if how != "nearest" and N % res == 0:
        k = N // res
        b = a.reshape(F, res, k, res, k)
        return b.max(axis=(2, 4)) if how == "max" else b.mean(axis=(2, 4), dtype=np.float64)
    idx = np.minimum(((np.arange(res) + 0.5) * N / res).astype(np.int64), N - 1)
    return a[:, idx][:, :, idx]


class FrameRecorder:
    """Writes one stage's frames.  ``clear(after=k)`` removes frames with key
    ``> k`` -- a resumed run owns everything past its resume point."""

    def __init__(self, root, stage: str, res: int):
        assert stage in STAGES, stage
        self.dir = frames_dir(root, stage)
        self.stage = stage
        self.res = int(res)

    def clear(self, after: int | None = None) -> None:
        if after is None:
            shutil.rmtree(self.dir, ignore_errors=True)
            return
        for p in self.dir.glob("*.json"):
            if int(p.stem) > after:
                p.unlink(missing_ok=True)
                p.with_suffix(".npz").unlink(missing_ok=True)

    def write(self, key: int, arrays: dict[str, np.ndarray], **meta) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{int(key):05d}.npz"
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(p)
        p.with_suffix(".json").write_text(json.dumps({"key": int(key), "stage": self.stage, **meta}))
        return p


def list_frames(root, stage: str) -> list[tuple[int, Path, dict]]:
    """``[(key, npz_path, meta), ...]`` sorted by key; frames whose npz is
    missing (an interrupted write) are skipped."""
    out = []
    for j in sorted(frames_dir(root, stage).glob("*.json")):
        p = j.with_suffix(".npz")
        if p.exists():
            out.append((int(j.stem), p, json.loads(j.read_text())))
    return sorted(out, key=lambda t: t[0])


def clear_all(root) -> None:
    shutil.rmtree(Path(root) / "frames", ignore_errors=True)


# --------------------------------------------------------------------------
# capture helpers called from the stages
# --------------------------------------------------------------------------
def tectonics_frame(sim, rec: FrameRecorder, index: int, total: int) -> None:
    """Sea-levelled bed (bedrock units), raw plate id + 1, and the crust's age
    and kind on the tect grid -- so the timeline can run the sea floor being
    made at the ridges and eaten at the trenches, not only what is left of it
    at the end.  Age takes the nearest segment's, as the plate id does: a
    frame is a segment's worth of detail.  Kind is the continent the bed was
    cut with -- ``c > 0.5`` of the extent-weighted splat, the crust the
    finished map's ``crust_kind`` and the shelf mask are (and what
    scripts/tect_scorecard.py measures as the rendered continent) -- so the
    timeline's last frame and the map agree.  The nearest segment's label
    drew a Voronoi continent up to 0.005 of the planet off it (te/mass,
    Earth seed 1, steps 0-2000).  Without the extent balance
    (``tectonics.extent_split`` off: the classic dynamics, small and tiny)
    the frames keep the nearest segment's label, as they always drew it."""
    from ..tectonics.collision import build_tree, label_map_fast, splat
    from ..tectonics.run import frame_bed, inherited_age
    from ..tectonics.segments import OCEANIC

    tree = build_tree(sim.seg)
    shown = bool(getattr(sim.tp, "extent_split", False))
    if shown:
        bed, c = frame_bed(sim, tree, with_c=True)
    else:
        bed = frame_bed(sim, tree)
    idx, _ = label_map_fast(sim.seg, sim.grid, sim.r_cap, tree)
    pid = splat(sim.seg.plate_id, idx).astype(np.int32) + 1
    age = splat(sim.seg.age + inherited_age(sim), idx)
    kind = (c > 0.5).astype(np.int32) if shown else splat((sim.seg.kind != OCEANIC).astype(np.int32), idx)
    alive = np.nonzero(sim.plates.alive)[0].tolist()
    rec.write(index, {"height": downsample(bed, rec.res).astype(np.float16),
                      "plate": downsample(pid, rec.res, "nearest").astype(np.int16),
                      "crust_age": downsample(age, rec.res).astype(np.float16),
                      "crust_kind": downsample(kind, rec.res, "nearest").astype(np.uint8)},
              step=int(index), of=int(total), units="bedrock", alive=alive)


def standing_water(state) -> np.ndarray:
    """Depth of the lakes the erosion state holds, interior ``(F, N, N)`` in
    cell units; 0 on dry ground and on the sea.

    ``ErosionState.refresh_lakes`` leaves two kinds.  An overflowing lake is
    flagged (``lake_flag``) and stands at its spill point: the level the
    refresh solved (``lake_level``), while the surface is still the one it
    solved it on.  Later in the iteration, or with no level kept (a resumed
    state), the routing surface the flagged cells are crossed on stands in
    for it, which the datum hold carries with the ground -- a reading, not
    the level.  It rises ``erosion.route_eps`` a cell from the outlet, and
    its flood drains a closed basin below sea level at the basin's floor: on
    the small preset the 524 cells of lake at iteration 40, most of them in
    such a basin, read as 5, and the 4 cells the run ends with as 55.

    A closed lake has its level as the base level of its cells, so it is
    wherever the ground is below a base level that is not the sea's 0.  The
    sea is taken by that rule too -- the kernel's own, ``surface < base``
    with a base of 0 -- and not from ``state.sea``: a coastal cell that has
    gone under since the mask was made is sea to the kernel, not a lake (the
    mask drifts 50-120 cells an iteration at N_c = 1024 and jumps by
    thousands on a glacial pass, ``erosion.sea_mask_every``).  Before the
    first refresh there is neither kind."""
    I = state.interior
    surf = state.height[I] + state.sediment[I]
    depth = np.zeros(surf.shape, np.float64)
    if state.lake_flag is not None and state.route is not None:
        over = state.lake_flag[I] > 0
        fresh = getattr(state, "lake_level", None) is not None and getattr(state, "lake_level_at", None) == state.iteration
        level = np.asarray(state.lake_level, np.float64).reshape(surf.shape) if fresh else state.route[I]
        depth[over] = np.maximum(level[over] - surf[over], 0.0)
    base = state.base[I]
    closed = (surf < base) & (base != 0.0)
    depth[closed] = np.maximum(depth[closed], (base - surf)[closed])
    return depth


def erosion_frame(state, rec: FrameRecorder, iteration: int, total: int, params) -> None:
    """Surface ``height + sediment`` (metres, block mean), discharge (block
    max, so a channel one coarse cell wide survives the downsample), and what
    the height does not show: ``lake``, the standing water over the ground
    (:func:`standing_water`; metres, block mean, so a lake smaller than a
    frame cell is a shallower one), and ``ice``, the share of the frame cell
    under ice (0..255).

    Water no deeper than the marsh line is ground, as hydro has it
    (``hydro.marsh_depth``) -- taken cell by cell before the block mean, or
    the sheets a metre deep that hydro leaves dry would be the timeline's
    largest lakes and vanish on its last frame.

    The ice is the glacial pass's own (``glacial.carve``): the cold ground
    above the sea, and with ``erosion.glacial_sticky`` what the last pass
    carved under.  It is zero until the pass starts, ``erosion.glacial_from``
    of the way through the run; the sidecar says ``glacial`` from then on.
    With ``erosion.ice_history`` the ice line has a temperature of its own
    at every iteration (``glacial.ice_cooling``) and the ice is on every
    frame, as much as that time's cold holds; the sidecar's ``cooling_c`` is
    how far under the climate the line was read (negative: over it)."""
    from ..erosion import glacial

    ep, hp = params.erosion, params.hydro
    I = state.interior
    unit = state.height_unit_m
    surf = (state.height[I] + state.sediment[I]) * unit
    lake = standing_water(state) * unit
    lake[lake <= max(float(hp.lake_min_depth), float(getattr(hp, "marsh_depth", 0.0)))] = 0.0
    iced = bool(state.spherical) and int(ep.glacial_every) > 0 and float(ep.glacial_rate) > 0.0 \
        and int(iteration) >= float(ep.glacial_from) * int(total)
    history = bool(state.spherical) and bool(getattr(ep, "ice_history", False)) and getattr(state, "temp0", None) is not None
    ice = np.zeros(surf.shape, np.float32)
    if iced or history:
        on = glacial.ice_mask(state, float(ep.ice_evap))
        prev = getattr(state, "ice_prev", None) if bool(getattr(ep, "glacial_sticky", False)) else None
        if prev is not None:
            on = on | prev
        ice = on[I].astype(np.float32)
    rec.write(iteration, {"height": downsample(surf, rec.res).astype(np.float16),
                          "discharge": downsample(state.discharge[I], rec.res, "max").astype(np.float16),
                          "lake": downsample(lake, rec.res).astype(np.float16),
                          "ice": np.round(255.0 * downsample(ice, rec.res)).astype(np.uint8)},
              iteration=int(iteration), of=int(total), units="m", **({"glacial": True} if iced else {}),
              **({"cooling_c": round(float(state.ice_age), 2)} if history else {}))
