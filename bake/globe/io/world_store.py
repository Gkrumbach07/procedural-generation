"""On-disk world layout (PLAN.md section 3) and manifest bookkeeping."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import STAGE_PARAM_GROUPS, WorldParams
from ..cubesphere import Grid
from ..field import FaceField


class WorldStore:
    def __init__(self, root: str | Path, create: bool = False):
        self.root = Path(root)
        if create:
            for d in ("coarse", "graph", "tiles", "quicklook", "checkpoints"):
                (self.root / d).mkdir(parents=True, exist_ok=True)
        self._manifest: dict | None = None

    # -- paths ----------------------------------------------------------
    @property
    def coarse_dir(self) -> Path:
        return self.root / "coarse"

    @property
    def graph_dir(self) -> Path:
        return self.root / "graph"

    @property
    def tiles_dir(self) -> Path:
        return self.root / "tiles"

    @property
    def quicklook_dir(self) -> Path:
        return self.root / "quicklook"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def quicklook_path(self, stage: str, suffix: str = "") -> Path:
        self.quicklook_dir.mkdir(parents=True, exist_ok=True)
        return self.quicklook_dir / (f"{stage}{('_' + suffix) if suffix else ''}.png")

    # -- manifest -------------------------------------------------------
    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            if self.manifest_path.exists():
                with open(self.manifest_path) as fh:
                    self._manifest = json.load(fh)
            else:
                self._manifest = {}
        return self._manifest

    def save_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(self.manifest, fh, indent=1, sort_keys=True, default=_json_default)
        tmp.replace(self.manifest_path)

    def init_manifest(self, params: WorldParams, grid: Grid) -> None:
        m = self.manifest
        m.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
        m["params"] = params.to_dict()
        m["params_hash"] = params.content_hash()
        m["seed"] = params.seed
        m["N_c"] = params.N_c
        m["R"] = params.world.R
        m["T"] = params.world.T
        m["N_fine"] = params.N_fine
        m["max_lod"] = params.max_lod
        m["R_planet"] = grid.R_planet
        m["cell_size_m"] = grid.cell_size_m
        m["halo"] = grid.H
        m.setdefault("stages", {})
        self.save_manifest()

    def params(self) -> WorldParams:
        return WorldParams.from_dict(self.manifest["params"])

    def stage_info(self, stage: str) -> dict | None:
        return self.manifest.get("stages", {}).get(stage)

    def stage_done(self, stage: str, params: WorldParams | None = None) -> bool:
        """True if ``stage`` is marked done.  With ``params`` also require
        that the stage was baked with the same output-relevant parameters
        (``STAGE_PARAM_GROUPS``); a missing or different per-stage
        ``params_hash`` counts as not done (cheap, no file hashing)."""
        info = self.stage_info(stage)
        if not (info and info.get("done")):
            return False
        if params is not None:
            groups = STAGE_PARAM_GROUPS.get(stage)
            if groups and info.get("params_hash") != params.group_hash(*groups):
                return False
        return True

    def mark_stage(self, stage: str, outputs: list[str], info: dict | None = None, seconds: float | None = None, params: WorldParams | None = None) -> None:
        st = self.manifest.setdefault("stages", {})
        groups = STAGE_PARAM_GROUPS.get(stage)
        st[stage] = {
            "done": True,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": seconds,
            "hash": self.hash_outputs(outputs),
            "params_hash": params.group_hash(*groups) if (params is not None and groups) else None,
            "outputs": list(outputs),
            "info": info or {},
        }
        self.save_manifest()

    def invalidate_from(self, stage: str, order: tuple[str, ...]) -> None:
        st = self.manifest.setdefault("stages", {})
        start = order.index(stage)
        for s in order[start:]:
            st.pop(s, None)
        self.save_manifest()

    # -- fields ---------------------------------------------------------
    def save_field(self, field: FaceField, name: str | None = None) -> None:
        field.save(self.coarse_dir, name)

    def load_field(self, name: str, grid: Grid, is_vector: bool | None = None) -> FaceField:
        return FaceField.load(self.coarse_dir, name, grid, is_vector=is_vector)

    def has_field(self, name: str) -> bool:
        return FaceField.exists(self.coarse_dir, name)

    def clear_outputs(self, outputs: list[str]) -> None:
        """Delete a stage's outputs before it reruns (mirrors
        :meth:`hash_outputs` resolution): a field name unlinks
        ``coarse/<name>.f{0..5}.npy``, a relative directory is removed
        recursively, a relative file unlinked; missing entries are no-ops.
        Keeps stale files (e.g. tiles of a previous LOD layout) from being
        hashed as part of the new output."""
        for name in outputs:
            if self.has_field(name):
                for k in range(6):
                    (self.coarse_dir / f"{name}.f{k}.npy").unlink(missing_ok=True)
                continue
            p = self.root / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()

    def field_names(self) -> list[str]:
        names = set()
        for p in self.coarse_dir.glob("*.f0.npy"):
            names.add(p.name[: -len(".f0.npy")])
        return sorted(names)

    # -- json -----------------------------------------------------------
    def write_json(self, relpath: str, obj: Any) -> Path:
        p = self.root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=1, default=_json_default)
        return p

    def read_json(self, relpath: str) -> Any:
        with open(self.root / relpath) as fh:
            return json.load(fh)

    def has(self, relpath: str) -> bool:
        return (self.root / relpath).exists()

    # -- hashing --------------------------------------------------------
    def hash_outputs(self, outputs: list[str]) -> str:
        """Content hash of a list of outputs.  Entries are field names
        (``coarse/<name>.f*.npy``), relative paths, or directory paths
        (hashed recursively, sorted)."""
        h = hashlib.sha256()
        for name in sorted(outputs):
            if self.has_field(name):
                for k in range(6):
                    _hash_file(h, self.coarse_dir / f"{name}.f{k}.npy")
                continue
            p = self.root / name
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        h.update(str(f.relative_to(p)).encode())
                        _hash_file(h, f)
            elif p.is_file():
                _hash_file(h, p)
            else:
                h.update(f"missing:{name}".encode())
        return h.hexdigest()[:16]


def _hash_file(h, path: Path, chunk: int = 1 << 20) -> None:
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
