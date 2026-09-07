"""World parameters (PLAN.md section 14), YAML loading, seeds.

One ``WorldParams`` holds every knob, grouped by stage.  Every random
number generator in the tool must come from :meth:`WorldParams.rng` so
that the same ``(seed, params)`` produces byte-identical output.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

STAGES = ("tectonics", "climate", "erosion", "hydro", "watersheds", "refine", "derive", "tiles")

#: Per-stage RNG offsets added to the world seed.
STAGE_SEED_OFFSETS = {
    "tectonics": 1000,
    "climate": 2000,
    "erosion": 3000,
    "hydro": 4000,
    "watersheds": 5000,
    "refine": 6000,
    "derive": 7000,
    "tiles": 8000,
    "stub": 9000,
}


@dataclass
class WorldGroup:
    seed: int = 0
    N_c: int = 1024
    cell_size_m: float = 50.0
    R: int = 4  # refinement factor
    T: int = 256  # tile edge (fine cells)
    land_fraction: float = 0.30
    halo: int = 4


@dataclass
class TectonicsParams:
    N_tect: int = 512
    segments: int = 20000
    initial_plates: int = 16
    steps: int = 1500
    convection: float = 10.0  # ★
    growth: float = 0.05  # ★ k_G
    dissolution_factor: float = 0.05  # ★
    deposit_density: float = 0.5  # k_D
    collision_radius_factor: float = 1.5  # × mean segment spacing
    gap_radius_factor: float = 1.5  # × mean segment spacing
    cascade_rate: float = 0.3  # ★
    cascade_threshold: float = 0.05  # in bedrock units before scaling
    cascade_passes: int = 3
    uplift_scale: float = 1.0
    uplift_window: int = 100  # k steps for (bedrock_now - bedrock_k_ago)/k
    uplift_baseline: float = 0.02  # small positive baseline, fraction of max uplift
    heat_diffusion: float = 0.5
    heat_noise_octaves: int = 3
    dt: float = 0.02
    damping: float = 0.05
    density_base: float = 0.5  # d_b in growth term
    height_scale_m: float = 4000.0  # maps bedrock units to metres
    smooth_sigma: float = 1.0  # final Gaussian (coarse cells)


@dataclass
class ClimateParams:
    T_eq: float = 28.0
    k_lat: float = 45.0
    lapse: float = 6.5  # °C per km
    m_ocean: float = 1.0
    k_base: float = 0.02
    k_oro: float = 0.5
    n_advect: int = 200
    precip_mean: float = 1.0
    k_evap: float = 0.001
    wind_speed: float = 1.0  # cells per advection step
    wind_deflection: float = 0.3
    itcz_width_deg: float = 10.0
    dry_band_deg: float = 25.0
    dry_band_strength: float = 0.5
    dt: float = 1.0


@dataclass
class ErosionParams:
    iterations: int = 800
    particles_per_cell: float = 0.25
    dt: float = 1.2  # ★
    density: float = 1.0  # ★
    friction: float = 0.05  # ★
    deposition_rate: float = 0.1  # ★
    evap_rate: float = 0.001  # ★
    k_mom: float = 1.0  # ★ momentumTransfer
    k_disc: float = 1.0
    ema: float = 0.1  # ★ map lerp
    thermal_rate: float = 0.5
    talus_slope_soft: float = 0.6  # rise/run
    talus_slope_hard: float = 1.2
    min_volume: float = 0.01  # ★
    max_steps: int = 0  # 0 -> 2 * N
    checkpoint_every: int = 50
    quicklook_every: int = 50
    slope_gain: float = 1.0  # multiplies the (dimensionless) slope in the particle force
    height_unit_m: float = 1.0  # kernel heights are height_m / height_unit_m
    chunk: int = 4096  # particles per parallel chunk (change-list capacity = chunk*max_steps)
    backend: str = "cpu"


@dataclass
class HydroParams:
    lake_min_depth: float = 0.5  # m
    river_threshold: float = 200.0  # cells of accumulation (× mean precip volume)
    requantile_land_fraction: bool = True


@dataclass
class WatershedParams:
    basin_max_cells: int = 512 * 512
    basin_min_cells: int = 64 * 64


@dataclass
class RefineParams:
    refine_iterations: int = 150
    detail_amp: float = 0.3
    halo_cells: int = 8
    workers: int = 0  # 0 -> os.cpu_count()
    feather_cells: int = 2
    particles_per_cell: float = 0.25


@dataclass
class DeriveParams:
    river_width_a: float = 2.0
    river_width_b: float = 0.5
    riparian_cells: int = 3
    wetland_cells: int = 3
    min_river_order: int = 1


@dataclass
class RenderParams:
    view_distance_m: float = 8000.0
    reanchor_distance_tiles: float = 1.0
    physics_radius_m: float = 500.0


@dataclass
class WorldParams:
    world: WorldGroup = field(default_factory=WorldGroup)
    tectonics: TectonicsParams = field(default_factory=TectonicsParams)
    climate: ClimateParams = field(default_factory=ClimateParams)
    erosion: ErosionParams = field(default_factory=ErosionParams)
    hydro: HydroParams = field(default_factory=HydroParams)
    watersheds: WatershedParams = field(default_factory=WatershedParams)
    refine: RefineParams = field(default_factory=RefineParams)
    derive: DeriveParams = field(default_factory=DeriveParams)
    render: RenderParams = field(default_factory=RenderParams)

    # -- convenience accessors ------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.world.seed)

    @property
    def N_c(self) -> int:
        return int(self.world.N_c)

    @property
    def N_fine(self) -> int:
        return int(self.world.N_c * self.world.R)

    @property
    def cell_size_m(self) -> float:
        return float(self.world.cell_size_m)

    @property
    def fine_cell_size_m(self) -> float:
        return float(self.world.cell_size_m) / self.world.R

    @property
    def R_planet(self) -> float:
        from .cubesphere import planet_radius

        return planet_radius(self.world.N_c, self.world.cell_size_m)

    def coarse_grid(self):
        from .cubesphere import get_grid

        return get_grid(self.world.N_c, self.world.halo, self.world.cell_size_m)

    def tect_grid(self):
        from .cubesphere import get_grid

        return get_grid(self.tectonics.N_tect, self.world.halo, self.world.cell_size_m * self.world.N_c / self.tectonics.N_tect, self.R_planet)

    def fine_grid(self):
        from .cubesphere import get_grid

        return get_grid(self.N_fine, self.world.halo, self.fine_cell_size_m, self.R_planet)

    @property
    def max_lod(self) -> int:
        """LOD levels 0..max_lod; max_lod has one tile per face."""
        tiles = self.N_fine // self.world.T
        lod = 0
        while tiles > 1:
            tiles //= 2
            lod += 1
        return lod

    def rng(self, stage: str, *extra: int) -> np.random.Generator:
        """Deterministic generator for a stage (and optional sub-keys such
        as a basin id or iteration index)."""
        if stage not in STAGE_SEED_OFFSETS:
            raise KeyError(f"unknown stage {stage!r}")
        return np.random.default_rng([int(self.world.seed) + STAGE_SEED_OFFSETS[stage], *[int(e) for e in extra]])

    def workers(self) -> int:
        return self.refine.workers or (os.cpu_count() or 1)

    # -- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorldParams":
        d = dict(d or {})
        kwargs = {}
        for f in fields(cls):
            sub = d.pop(f.name, {}) or {}
            if not isinstance(sub, dict):
                raise TypeError(f"params group {f.name!r} must be a mapping")
            typ = f.default_factory  # the group dataclass
            valid = {g.name for g in fields(typ)}
            unknown = set(sub) - valid
            if unknown:
                raise KeyError(f"unknown parameter(s) in {f.name!r}: {sorted(unknown)}")
            kwargs[f.name] = typ(**sub)
        if d:
            raise KeyError(f"unknown parameter group(s): {sorted(d)}")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorldParams":
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        return cls.from_dict(d)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def content_hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True, default=_json_default)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def group_hash(self, *groups: str) -> str:
        d = {g: getattr(self, g) for g in groups}
        s = json.dumps({k: dataclasses.asdict(v) for k, v in d.items()}, sort_keys=True, default=_json_default)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def with_overrides(self, **groups: dict) -> "WorldParams":
        d = self.to_dict()
        for g, sub in groups.items():
            d[g].update(sub)
        return WorldParams.from_dict(d)

    # -- presets ----------------------------------------------------------
    @classmethod
    def small_world(cls, seed: int = 0) -> "WorldParams":
        """PLAN section 15 test profile: the whole pipeline in ~1 minute."""
        p = cls()
        p.world = WorldGroup(seed=seed, N_c=128, cell_size_m=50.0, R=2, T=64, land_fraction=0.3)
        p.tectonics = dataclasses.replace(p.tectonics, N_tect=64, segments=1500, initial_plates=8, steps=300)
        p.climate = dataclasses.replace(p.climate, n_advect=60)
        p.erosion = dataclasses.replace(p.erosion, iterations=60, checkpoint_every=30, quicklook_every=30)
        p.watersheds = WatershedParams(basin_max_cells=48 * 48, basin_min_cells=8 * 8)
        p.refine = dataclasses.replace(p.refine, refine_iterations=20, halo_cells=4)
        return p

    @classmethod
    def tiny_world(cls, seed: int = 0) -> "WorldParams":
        """Even smaller profile for unit tests (seconds)."""
        p = cls.small_world(seed)
        p.world = WorldGroup(seed=seed, N_c=32, cell_size_m=50.0, R=2, T=16, land_fraction=0.3)
        p.tectonics = dataclasses.replace(p.tectonics, N_tect=32, segments=300, initial_plates=5, steps=60)
        p.climate = dataclasses.replace(p.climate, n_advect=20)
        p.erosion = dataclasses.replace(p.erosion, iterations=10, checkpoint_every=5, quicklook_every=5)
        p.watersheds = WatershedParams(basin_max_cells=12 * 12, basin_min_cells=3 * 3)
        p.refine = dataclasses.replace(p.refine, refine_iterations=5, halo_cells=2)
        return p


def _json_default(o):
    if is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


PRESETS = {"default": WorldParams, "small": WorldParams.small_world, "tiny": WorldParams.tiny_world}
