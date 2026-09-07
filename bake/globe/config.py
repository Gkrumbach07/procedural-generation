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

#: Parameter groups whose values determine a stage's output (own group +
#: world).  ``WorldStore.mark_stage`` records ``group_hash(*groups)`` per
#: stage and ``stage_done(stage, params)`` compares against it, so a resume
#: notices upstream parameter drift without hashing any files.
STAGE_PARAM_GROUPS = {
    "tectonics": ("world", "tectonics"),
    "climate": ("world", "climate"),
    "erosion": ("world", "erosion"),
    "hydro": ("world", "hydro"),
    "watersheds": ("world", "watersheds"),
    "refine": ("world", "refine"),
    "derive": ("world", "derive"),
    "tiles": ("world", "refine"),
}

#: Knobs that change *how* a bake runs but not its output; excluded from
#: ``content_hash`` / ``group_hash`` so changing them never refuses a resume.
#: ``erosion.chunk`` / ``erosion.backend`` stay in the hash (per-batch
#: change-list application and CUDA float atomics can change the output).
ALL = object()
RUNTIME_KNOBS: dict[str, Any] = {
    "refine": {"workers"},
    "erosion": {"checkpoint_every", "quicklook_every"},
    "render": ALL,
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
    uplift_window: int = 100  # k steps: uplift = uplift_scale * (bedrock_now - bedrock_k_ago) / erosion.iterations
    uplift_baseline: float = 0.02  # small positive baseline, fraction of max uplift
    heat_diffusion: float = 2e-4  # rad² per unit time (× dt per step; sub-stepped for stability)
    heat_noise_octaves: int = 3
    dt: float = 0.02
    damping: float = 0.05
    density_base: float = 0.5  # d_b in growth term
    height_scale_m: float = 4000.0  # maps bedrock units to metres
    smooth_sigma: float = 1.0  # final Gaussian (coarse cells)
    # -- sphere-specific knobs (see globe/tectonics/plates.py for units) --
    force_scale: float = 5e-3  # angular acceleration per unit convection*∇heat (heat/rad); pixel -> radian unit change
    max_speed: float = 0.5  # cap on |omega| (rad per unit time; 0 = none)
    initial_speed: float = 0.05  # |omega| of the random initial plate rotations (rad per unit time)
    initial_thickness: float = 0.2  # crust thickness at t = 0
    new_thickness: float = 0.05  # thickness of crust spawned at divergent boundaries
    gap_cooling: float = 0.02  # heat removed per gap cell when new crust forms
    subduction_heating: float = 0.02  # peak heat added (Gaussian blob, 1 spacing wide) per subducted segment
    spawn_spacing_factor: float = 0.85  # min spacing of new segments, × mean segment spacing
    boundary_width_factor: float = 3.0  # hardness: boundary_proximity falls to 0 at this × spacing from a foreign plate
    collision_zone_factor: float = 2.0  # uplift clamped >= 0 within this × spacing of a subduction of the uplift window
    label_every: int = 1  # rebuild the label map every n steps (1 = every step)


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
    # evap = k_evap*max(T,0) is a dimensionless spatial multiplier on
    # erosion.evap_rate: ~1 at a warm sea-level cell (T_eq), 0 where T <= 0.
    k_evap: float = 1.0 / 28.0
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
    friction: float = 0.25  # ★ 0.05 in McDonald 2020 with sub-cell steps; 0.25 (SimpleHydrology) with unit steps, see erosion/particle.py
    deposition_rate: float = 0.1  # ★
    # ★ McDonald evapRate; per-step particle decay is
    # volume *= 1 - dt*evap_rate*evap[cell], evap = climate multiplier (~1)
    evap_rate: float = 0.001
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
    slope_gain: float = 2.0  # multiplies the gravity force (tangential surface normal) in the particle direction update
    erodibility: float = 0.2  # c_eq = erodibility * dh * (1 + k_disc * erf(q / disc_saturation)); 1.0 = McDonald 2022
    height_unit_m: float = 0.0  # kernel heights are height_m / height_unit_m; 0 or 1 -> use cell_size_m (cell units)
    disc_saturation: float = 32.0  # discharge (volume units ~ upstream cells) at which erf(q/disc_saturation) saturates the entrainment term
    max_erode: float = 0.25  # cap on terrain removed per particle-step (cell units); safety against blow-ups
    flood_every: int = 10  # recompute the particle routing surface (epsilon priority flood, erosion/route.py) every k iterations; 0 = steer on the raw terrain
    route_eps: float = 1e-3  # minimum drop per cell (cell units) of the routing surface across lakes
    pit_steps: int = 16  # kill a particle after this many consecutive uphill steps (stuck in a pit)
    chunk: int = 512  # particles per parallel chunk (change-list capacity = chunk*(max_steps+1) entries of 24 B); terrain is frozen within a chunk
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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` unless the world/tile geometry is consistent.

        PLAN section 3: the 2x LOD pyramid must reach exactly one tile per
        face, so ``N_fine = N_c*R`` must be a multiple of ``T`` and
        ``N_fine / T`` a power of two.  Halo / N_tect limits are checked by
        ``Grid`` (N >= 2H); ``N_c % N_tect`` is not required (PLAN 6.3
        resamples).  Called on construction and again by ``bake()`` because
        presets and ``--set`` mutate params after construction."""
        w = self.world
        if w.N_c < 1 or w.R < 1 or w.T < 1:
            raise ValueError(f"world.N_c, world.R and world.T must be >= 1 (got N_c={w.N_c}, R={w.R}, T={w.T})")
        n_fine = self.N_fine
        if n_fine % w.T != 0:
            raise ValueError(
                f"N_fine = N_c*R = {w.N_c}*{w.R} = {n_fine} is not a multiple of the tile edge T={w.T}; "
                "PLAN section 3 requires an exact tiling so the LOD pyramid reaches 1 tile per face"
            )
        n = n_fine // w.T
        if n & (n - 1) != 0:
            raise ValueError(
                f"tiles per face at LOD 0 = N_fine/T = {n_fine}/{w.T} = {n} is not a power of two; "
                "PLAN section 3 requires the 2x LOD pyramid to reach exactly 1 tile per face "
                f"(got N_c={w.N_c}, R={w.R}, T={w.T})"
            )

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
        """Deterministic generator for a stage and optional sub-keys (basin id,
        iteration index, ...).  Keys of different length never collide
        (``rng(s) != rng(s, 0)``) and negative ids (ocean = -1) are allowed.

        The first entropy word is PLAN section 4's ``seed + stage_offset``;
        the ``len(extra)`` word defeats SeedSequence's zero-padding and the
        two's-complement mask makes negative keys valid and deterministic."""
        if stage not in STAGE_SEED_OFFSETS:
            raise KeyError(f"unknown stage {stage!r}")
        mask = (1 << 64) - 1
        key = [(int(self.world.seed) + STAGE_SEED_OFFSETS[stage]) & mask, len(extra), *[int(e) & mask for e in extra]]
        return np.random.default_rng(key)

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

    def hashed_dict(self, groups: tuple[str, ...] | None = None) -> dict:
        """``to_dict()`` restricted to ``groups`` (all if None) with the
        ``RUNTIME_KNOBS`` removed — the part of the params that determines
        the baked output."""
        d = self.to_dict()
        out = {}
        for g, sub in d.items():
            if groups is not None and g not in groups:
                continue
            knobs = RUNTIME_KNOBS.get(g)
            if knobs is ALL:
                continue
            out[g] = {k: v for k, v in sub.items() if not (knobs and k in knobs)}
        return out

    def content_hash(self) -> str:
        """Hash of every output-relevant parameter (runtime knobs excluded)."""
        s = json.dumps(self.hashed_dict(), sort_keys=True, default=_json_default)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def group_hash(self, *groups: str) -> str:
        """Hash of the named parameter groups (runtime knobs excluded)."""
        s = json.dumps(self.hashed_dict(tuple(groups)), sort_keys=True, default=_json_default)
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
