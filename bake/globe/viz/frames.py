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
    """Sea-levelled bed (bedrock units) and raw plate id + 1 on the tect grid."""
    from ..tectonics.collision import build_tree, label_map_fast, splat
    from ..tectonics.run import frame_bed

    tree = build_tree(sim.seg)
    bed = frame_bed(sim, tree)
    idx, _ = label_map_fast(sim.seg, sim.grid, sim.r_cap, tree)
    pid = splat(sim.seg.plate_id, idx).astype(np.int32) + 1
    alive = np.nonzero(sim.plates.alive)[0].tolist()
    rec.write(index, {"height": downsample(bed, rec.res).astype(np.float16),
                      "plate": downsample(pid, rec.res, "nearest").astype(np.int16)},
              step=int(index), of=int(total), units="bedrock", alive=alive)


def erosion_frame(state, rec: FrameRecorder, iteration: int, total: int) -> None:
    """Surface ``height + sediment`` (metres, block mean) and discharge (block
    max, so a channel one coarse cell wide survives the downsample)."""
    I = state.interior
    surf = (state.height[I] + state.sediment[I]) * state.height_unit_m
    rec.write(iteration, {"height": downsample(surf, rec.res).astype(np.float16),
                          "discharge": downsample(state.discharge[I], rec.res, "max").astype(np.float16)},
              iteration=int(iteration), of=int(total), units="m")
