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
    # ``erosion.resume`` only chooses whether ``checkpoints/`` is read, never
    # what is computed: in the hash it would make toggling it invalidate the
    # whole world (``bake`` refuses without ``--force``, and ``--force``
    # without ``--from`` reruns from tectonics).
    "erosion": {"checkpoint_every", "quicklook_every", "resume"},
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
    """PLAN section 6 knobs.  Lengths on the sphere are chord/arc lengths in
    radians; "spacing" is the mean segment spacing ``sqrt(4*pi/M)`` (radians)
    so every *_factor knob is resolution independent.  Time is measured in
    tectonic steps (age, diffusion, speeds); there is no separate dt."""

    N_tect: int = 512
    segments: int = 20000
    initial_plates: int = 16
    steps: int = 1500
    convection: float = 10.0  # ★
    growth: float = 0.05  # ★ k_G (thickness units per step)
    dissolution_factor: float = 0.05  # ★
    deposit_density: float = 0.5  # k_D
    plate_size_jitter: float = 0.35  # spread of initial plate sizes: per-plate distance weights are 1 ± this, so 0 tiles the sphere evenly and ~0.8 reproduces Earth's hierarchy (a few plates covering most of the surface, microplates between). Earth spans ~94x largest:smallest with its top 7 plates over 92% of the globe; 0.35 gives a near-uniform 3.2x, which leaves every landmass a single collision zone
    collision_radius_factor: float = 1.0  # × spacing: segments of different plates closer than this collide
    gap_radius_factor: float = 1.0  # × spacing: cells farther than this from every segment are divergent gaps
    overlap_fraction: float = 0.5  # segments of different plates closer than this × collision radius collide even when not approaching (no interleaving along transform boundaries)
    splat_sigma_factor: float = 0.5  # sigma (× spacing) of the Gaussian blend of the nearest segments when splatting segment values to the tect grid
    splat_knn: int = 12
    cascade_rate: float = 0.3  # ★
    cascade_threshold: float = 0.05  # bedrock units (thickness·(1−density)) per cell of neighbour distance on the tect grid
    cascade_passes: int = 3
    uplift_scale: float = 1.0
    uplift_window: int = 100  # k steps: uplift = uplift_scale * (h_now - h_k_ago) [per segment, metres] / erosion.iterations
    uplift_baseline: float = 0.02  # positive baseline outside collision zones, fraction of the 99th-percentile uplift
    heat_diffusion: float = 2e-5  # rad² per step (explicit, sub-stepped for stability)
    heat_grid_divisor: int = 4  # the heat field lives on an N_tect / divisor grid (low-frequency driver; cheap diffusion)
    heat_relax: float = 0.002  # per-step relaxation of heat towards the initial background field (keeps the sources from saturating)
    strata_period: float = 40.0  # bedrock fabric: age interval (tectonic steps) between successive hard bands.  Crust accretes at plate boundaries, so lines of equal age are the strata it was laid down in and bands run parallel to the boundary that made them — narrow where accretion was fast, wide where it was slow (median ~7 coarse cells at the defaults).  0 = no fabric
    strata_amp: float = 0.35  # how far the fabric swings hardness either side of the base value.  Without it hardness is a smooth two-tone blend of crust age and density (autocorrelation still 0.53 at 64 cells) with no structure at drainage-basin scale, so every continent develops the same radial network
    heat_noise_octaves: int = 3
    heat_noise_freq: float = 1.5  # base lattice frequency of the initial heat noise (features ~ 1/freq of the diameter)
    damping: float = 0.05  # omega *= (1 - damping) per step
    density_base: float = 0.5  # d_b in the growth term
    height_scale_m: float = 4000.0  # metres per bedrock unit when both relief_m and relief_spacings are 0
    relief_m: float = 0.0  # explicit override, metres: if > 0, scale bedrock so the 99.9th percentile of land sits at this height
    relief_spacings: float = 1.5  # when relief_m == 0: that percentile sits at this many mean segment spacings (metres), so the vertical scale follows the horizontal one at every preset (0 = use height_scale_m)
    smooth_sigma: float = 1.0  # final Gaussian on the tect grid (tect cells); resampling to the coarse grid is cubic
    # -- sphere-specific knobs (see globe/tectonics/plates.py for the force model) --
    force_scale: float = 3e-4  # plate angular acceleration in spacings/step² per unit (convection × |∇heat| [heat per radian] / mass per area)
    max_speed: float = 0.3  # cap on plate speed, spacings per step (0 = none); keep < collision_radius_factor / 2
    initial_speed: float = 0.1  # speed of the random initial plate rotations, spacings per step
    initial_thickness: float = 0.4  # crust thickness at t = 0
    max_thickness: float = 1.0  # crystallisation growth is faded by exp(-thickness / max_thickness)
    ridge_height: float = 0.15  # thermal buoyancy of new crust, bedrock units (ridges at rifts; decays with age)
    ridge_age: float = 150.0  # e-folding age (steps) of the buoyancy
    new_thickness: float = 0.05  # thickness of crust spawned at divergent boundaries
    gap_cooling: float = 0.05  # peak heat removed (Gaussian blob, 1 spacing wide) per segment of new crust spawned at a rift
    subduction_heating: float = 0.01  # peak heat added (Gaussian blob, 1 spacing wide) per subducted segment
    spawn_spacing_factor: float = 0.85  # min spacing of new segments, × spacing
    relax_rate: float = 0.1  # per-step segment height cascade rate (0 = off); moves rate*(Δh-thr)/2/knn to each lower neighbour
    relax_threshold: float = 0.15  # maximum stable slope, bedrock units per spacing
    relax_knn: int = 8
    belt_width_factor: float = 1.0  # sigma (× spacing) of the Gaussian that shares subducted mass among the survivor's same-plate neighbours (mountain belt width)
    boundary_width_factor: float = 3.0  # hardness: boundary_proximity falls to 0 at this × spacing from a foreign plate
    collision_zone_factor: float = 2.0  # uplift clamped >= 0 within this × spacing of a subduction of the uplift window
    area_blend: float = 0.05  # rolling blend of the measured Voronoi area per step (PLAN: 0.99 rolling == 0.01)
    label_every: int = 1  # rebuild the label map (gaps, areas) every n steps; collisions and forces run every step


@dataclass
class ClimateParams:
    T_eq: float = 28.0
    k_lat: float = 45.0
    lapse: float = 6.5  # °C per km
    m_ocean: float = 1.0
    # Moisture rain-out is parameterised in physical length (climate/precipitation.py), so the
    # picture is the same at any N_c / cell size:
    #   frac = min(1, 1 - exp(-step_m/L_base) + k_oro*(1 - exp(-rise_m/oro_height_m)))
    # step_m = |wind|*dt*cell_size_m per sweep; L_base = moisture_reach_m or moisture_reach_frac*R_planet.
    # (PLAN 7's per-step k_base is 1 - exp(-step_m/L_base): 0.02 at 50 m cells needs L_base = 2.5 km.)
    moisture_reach_frac: float = 0.5  # background rain-out e-folding fetch as a fraction of R_planet (continents scale with the planet)
    moisture_reach_m: float = 0.0  # ... or an explicit fetch in metres (> 0 overrides the fraction)
    k_oro: float = 1.0  # max orographic rain-out fraction per sweep; 1 = pure exp(-climb/oro_height_m) moisture loss
    oro_height_m: float = 2000.0  # orographic scale height: an air parcel keeps exp(-dh/oro_height_m) of its moisture after climbing dh
    calm_floor: float = 0.25  # the background rain-out uses a step length of at least calm_floor*wind_speed cells (still air still rains out; no zero-rain lines in the calm belts)
    n_advect: int = 0  # advection sweeps; 0 = sweep until the moisture field is stationary (see advect_tol / n_advect_max_factor)
    advect_tol: float = 1e-3  # auto mode stops when the max moisture change over land per sweep < advect_tol*m_ocean
    n_advect_max_factor: float = 4.0  # auto-mode cap: n_advect_max_factor*N/(wind_speed*dt) sweeps
    precip_mean: float = 1.0
    # evap = k_evap*max(T,0) is a dimensionless spatial multiplier on
    # erosion.evap_rate: ~1 at a warm sea-level cell (T_eq), 0 where T <= 0.
    k_evap: float = 1.0 / 28.0
    wind_speed: float = 1.0  # cells per advection step (keep < halo - 1 so the semi-Lagrangian departure point stays in the halo)
    wind_deflection: float = 0.3  # k: steering around terrain, t = k*slope/(1+k*slope) (see climate/wind.py)
    itcz_width_deg: float = 10.0  # sigma of the ITCZ wet band (Gaussian in latitude)
    dry_band_deg: float = 25.0  # centre latitude of the subtropical dry bands
    dry_band_strength: float = 0.5  # precip multiplier at the dry-band centre is 1 - strength
    dt: float = 1.0  # advection step (departure = pos - wind*dt)
    # -- additions (climate/*.py) ------------------------------------------
    wind_meridional: float = 0.4  # meridional / zonal speed ratio of the surface branches of the three cells
    band_blend_deg: float = 6.0  # width of the smooth zonal-sign transitions at 30 and 60 degrees (calm belts)
    pole_taper_deg: float = 3.0  # wind speed tapers smoothly to 0 within this angle of a pole
    deflection_smooth: int = 2  # 3x3 binomial passes on the surface before the deflection gradient
    rise_smooth: int = 1  # 3x3 binomial passes on the surface before the windward-rise gradient
    precip_floor: float = 0.1  # background rain added on land (fraction of the land mean of the advected rain) before the latitude prior
    precip_smooth: int = 1  # 3x3 binomial passes on the rain (spillover: orographic rain drifts a few cells) before the prior / normalisation
    itcz_strength: float = 1.0  # precip multiplier at the equator is 1 + strength
    dry_band_width_deg: float = 8.0  # sigma of the dry bands


@dataclass
class ErosionParams:
    iterations: int = 800
    particles_per_cell: float = 0.25
    dt: float = 1.2  # ★
    density: float = 1.0  # ★
    friction: float = 0.25  # ★ inertia: the previous (unit) direction enters the direction update with weight 1 - dt*friction before gravity / momentum are added (the step is always one cell, so friction cannot limit a terminal speed); 1/dt = no inertia, see erosion/particle.py
    deposition_rate: float = 0.1  # ★
    # ★ McDonald evapRate; per-step particle decay is
    # volume *= 1 - dt*evap_rate*evap[cell], evap = climate multiplier (~1)
    evap_rate: float = 0.001
    k_mom: float = 1.0  # ★ momentumTransfer
    k_disc: float = 1.0
    ema: float = 0.1  # ★ map lerp
    thermal_rate: float = 0.5
    creep_rate: float = 0.1  # hillslope creep: a second thermal pass with talus 0 at this rate (linear diffusion of the surface; submerged cells are inert); 0 = off.  Without it every particle path incises its own rill (drainage density saturates at one channel per ~3 cells, parallel micro-rills instead of a trunk network); 0.3 over-smooths the divides (docs/erosion-tuning.md)
    thermal_max: float = 1.0  # cap on the material a cell sheds per thermal pass (cell units): a tectonic cliff relaxes at a bounded rate instead of collapsing in one iteration
    talus_slope_soft: float = 0.6  # rise/run
    talus_slope_hard: float = 1.2
    min_volume: float = 0.01  # ★
    max_steps: int = 0  # 0 -> 2 * N
    checkpoint_every: int = 50
    quicklook_every: int = 50
    slope_gain: float = 2.0  # multiplies the gravity force in the particle direction update
    slope_saturation: float = 0.0  # > 0: gravity = slope_gain * s / sqrt(s^2 + slope_saturation^2) along the downhill direction (terminal-velocity flow; gentle slopes still steer); 0 = tangential surface normal (McDonald)
    erodibility: float = 0.2  # c_eq = erodibility * dh * (1 + k_disc * f(q)), f set by disc_exponent below; 1.0 = McDonald 2022
    cover_depth: float = 0.1  # alluvial cover scale (cell units): sediment this thick fully shields the bedrock below, and a thinner film shields it in proportion (Sklar & Dietrich cover effect), so erodibility blends from 1 (sediment) to (1 - hardness) (bare rock).  0 = the bare-rock-only gate, under which hardness reached 0.1 % of land cells (half the land carries < 40 cm of sediment, but any film counted as full cover) and every continent eroded at the same rate
    height_unit_m: float = 0.0  # kernel heights are in cell units (height_m / cell_size_m): every cap, talus slope and gravity term is defined in them; 0 or 1 (= cell_size_m) are the only accepted values, anything else raises
    disc_saturation: float = 32.0  # discharge scale (volume units ~ upstream cells) of the entrainment term: erf(q/disc_saturation) when disc_exponent is 0, (q/disc_saturation)^disc_exponent otherwise
    disc_exponent: float = 0.5  # > 0: unsaturated power-law entrainment c_eq = erodibility*dh*(1 + k_disc*(q/disc_saturation)^disc_exponent) (stream-power concavity theta = 0.5: a trunk river carries its load at a gentler slope than a rill, so long profiles are concave, valleys widen downstream and a channel survives on a floodplain); 0 = the saturating erf law, under which every plain became an alluvial fan and no channel could meander
    max_erode: float = 0.25  # cap on terrain removed per particle-step (cell units) at trace time (against the chunk-start terrain)
    iter_erode: float = 0.5  # net erosion a cell may receive per iteration (cell units), enforced against the live terrain in apply order; the shortfall cancels the particle's later deposits (erosion/particle.py apply_changes)
    iter_deposit: float = 1.0  # net deposition a cell may receive per iteration (cell units); the excess moves back up the particle's path, the remainder waits in the per-cell `pending` stockpile (released at this rate)
    ocean_deposition_rate: float = 0.3  # a particle that reaches the sea keeps walking downslope on the seafloor, deposit-only, dropping this fraction of its load per step (submarine fan); no erosion, no discharge track below sea level
    ocean_steps: int = 64  # at most this many seafloor steps (then the rest waits in the cell's pending stockpile, re-injected next iteration)
    fan_room: float = 1.0  # a seafloor step may settle at most this much (cell units) on a flat sea floor per particle-step (the drop to the previous cell when larger); the sea-level ceiling, the fan_slope descent and iter_deposit still bound the pile in apply_changes.  0.02 (= DEP_FLOOR) throttled offshore dispersal to ~1 cell unit per stockpile and iteration, so river mouths parked most of their load in `pending`
    fan_slope: float = 0.05  # a submarine fan descends at least this much per cell away from its source (cell units per cell): the deposit ceiling of a seafloor step is the previous path cell minus this, and never above -DEP_FLOOR
    glacial_every: int = 10  # run the glacial pass (erosion/glacial.py) every k iterations; 0 = off.  Ice is where the mean annual temperature is at or below freezing (evap <= 0), which at the defaults is ~9 % of land, close to Earth's glaciated fraction
    ice_evap: float = 0.0  # ice forms where the climate field `evap` is at or below this.  `evap` is k_evap*max(T,0), so 0 is exactly the freezing line and a positive value is a warmer equilibrium-line altitude (more of the world glaciated).  This is the FIRST-ORDER control on lakes: measured across three seeds, lake area swung 5x with the seed (0.14 %, 0.31 %, 0.74 % of land) but at most 43 % with glacial_from/glacial_every, and one seed had no land below +1.6 C at all, so no ice and no glacial lakes were possible however the other knobs were set
    glacial_from: float = 0.75  # start glaciating this far through the run (fraction of erosion.iterations); 0 = glaciate throughout.  Earth is lake-rich because glaciation was *recent* — the basins ice cut ~10 ka ago have not had time to fill, and lake lifetime is short next to landscape evolution time.  Carving throughout instead gives the fluvial system the whole rest of the run to drain and backfill every basin
    glacial_rate: float = 1.0  # bed lowered per glacial pass (cell units) at the reference ice flux, scaled by sqrt(discharge/disc_saturation) and by (1 - 0.5*hardness).  Unlike every other erosional term this one has NO base-level limit — ice flows uphill out of a basin — which is what leaves the closed depressions that become lakes
    glacial_ramp: int = 4  # cells over which the carve ramps up from the ice margin inward.  Erosion that only scales with ice flux deepens a valley monotonically downstream, which drains; tapering it to zero at the snout leaves a rock lip with the deepest point inside the ice, i.e. a closed basin
    glacial_max: float = 0.4  # cap on the bed a cell loses in one glacial pass (cell units)
    moraine_frac: float = 1.0  # fraction of the excavated rock deposited on the ice margin as moraine (the rest is lost); 1.0 keeps the pass mass-conserving, and the moraine dams valleys leaving an ice field, which is the second way ice makes lakes
    resume: bool = True  # resume from checkpoints/ whose parameter + upstream + kernel-version hash matches; False recomputes from bedrock
    flood_every: int = 10  # recompute the particle routing surface (epsilon priority flood, erosion/route.py) every k iterations; 0 = steer on the raw terrain
    route_eps: float = 1e-3  # minimum drop per cell (cell units) of the routing surface across lakes
    pit_steps: int = 16  # kill a particle after this many consecutive uphill steps (stuck in a pit)
    chunk: int = 2048  # particles per parallel chunk (change-list capacity = chunk*(max_steps+16) entries of 24 B, ~100 MB at N_c=1024); terrain is frozen within a chunk.  2048 traces ~20 % faster than 512 (fewer parallel launches, less tail imbalance) with the same morphology on the single-face tests (docs/erosion-tuning.md)
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
    feather_cells: int = 8  # fine cells over which a basin's refined detail ramps in from its (frozen) divide; keep >= 2R (a coarse cell): a 2-cell ramp reads as a crease in the LOD-0 hillshade
    particles_per_cell: float = 0.25


@dataclass
class DeriveParams:
    """PLAN section 11 knobs (globe/derive).  Rivers come from the *fine*
    discharge: cells above a discharge threshold chosen so that a given
    fraction of the land is river (density matched to the coarse channel
    network by default), thinned to centrelines."""

    river_width_a: float = 2.0  # w = a * (Q / Q_thr)^b fine cells (Q_thr = the river discharge threshold)
    river_width_b: float = 0.5
    riparian_cells: int = 3  # coarse cells from a channel / river mask cell (x R at fine resolution)
    wetland_cells: int = 3  # coarse cells from a lake cell
    min_river_order: int = 1  # rivers of lower Strahler order are dropped from rivers.json / the mask
    # -- river extraction (derive/rivers.py) --------------------------------
    river_mask_fraction: float = 0.0  # fraction of fine land cells above the discharge threshold; 0 = auto: river_fraction_scale x the coarse channel fraction of land
    river_fraction_scale: float = 1.5  # auto mode: a thresholded discharge blob is wider than a 1-cell D8 channel
    river_hysteresis: float = 2.5  # connectivity threshold = the discharge of river_fraction x this fraction of land; low-threshold blobs survive only if they contain a high-threshold cell (1 = off)
    river_fallback_fraction: float = 0.03  # auto mode when the coarse graph has no channels (stub hydro)
    discharge_smooth_cells: float = 1.0  # Gaussian sigma (fine cells) applied to the discharge before thresholding; 0 = off
    min_river_cells: float = 8.0  # x R^2: smaller discharge blobs are dropped, smaller holes are filled before thinning
    spur_cells: float = 3.0  # x R: skeleton spurs (endpoint -> junction) shorter than this are pruned
    max_width_cells: float = 24.0  # cap on the river width (fine cells)
    river_point_step: float = 2.0  # polyline vertex spacing (fine cells) after Catmull-Rom smoothing
    graph_match_cells: int = 2  # a polyline takes order / edge_id from a coarse drainage edge within this many coarse cells
    # -- biomes (derive/biomes.py) ------------------------------------------
    precip_scale_cm: float = 100.0  # annual precipitation (cm) at the land *mean* rain rate (wetness = 1; climate.precip_mean by contract)
    precip_gamma: float = 0.5  # P_cm = precip_scale_cm * wetness^gamma: compresses the skewed rain rate (coasts 10-30x the mean, interiors 0.05x) into the Whittaker range
    precip_max_cm: float = 400.0  # cap on P_cm (the Whittaker bins end at 220 cm); 0 = none
    alpine_min_m: float = 0.0  # alpine override: surface above this (metres) AND temperature below alpine_T; 0 = use alpine_min_relief_frac
    alpine_min_relief_frac: float = 0.6  # when alpine_min_m == 0: the alpine threshold is this fraction of the achieved land relief (99.9th percentile of the land surface), so it follows tectonics.relief_spacings
    alpine_T: float = 0.0  # degC
    cliff_slope: float = 1.6  # rise/run above which a cell is bare cliff (and vegetation is 0)
    cliff_max_fraction: float = 0.03  # cliffs cover at most this fraction of the land: the effective threshold is max(cliff_slope, that land quantile of the coarse slope)
    slope_stencil: int = 0  # half-width (fine cells) of the fine slope stencil; 0 = R (coarse-scale slope on the fine grid)
    soil_full_depth_m: float = 1.0  # sediment depth at which the soil factor of the vegetation reaches 1 (derive/soil.py)
    lake_min_cells: float = 1.0  # x R^2: smaller fine lake pieces are ignored


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
        p.climate = dataclasses.replace(p.climate, n_advect=0)  # auto: sweep until stationary (cap 4*N)
        p.erosion = dataclasses.replace(p.erosion, iterations=60, checkpoint_every=30, quicklook_every=30)
        p.watersheds = WatershedParams(basin_max_cells=48 * 48, basin_min_cells=8 * 8)
        p.refine = dataclasses.replace(p.refine, refine_iterations=20, halo_cells=4, feather_cells=4)  # 2R
        return p

    @classmethod
    def tiny_world(cls, seed: int = 0) -> "WorldParams":
        """Even smaller profile for unit tests (seconds)."""
        p = cls.small_world(seed)
        p.world = WorldGroup(seed=seed, N_c=32, cell_size_m=50.0, R=2, T=16, land_fraction=0.3)
        p.tectonics = dataclasses.replace(p.tectonics, N_tect=32, segments=300, initial_plates=5, steps=60)
        p.climate = dataclasses.replace(p.climate, n_advect=0)
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
