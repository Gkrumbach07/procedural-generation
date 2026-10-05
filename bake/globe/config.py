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
    # (dyn-minimal: tectonics.myr_per_step converts the physical parameters, so it is hashed)
    "render": ALL,
}


@dataclass
class WorldGroup:
    """The default world is Earth: radius is derived as
    ``N_c * cell_size_m * 4 / 2pi``, so 1024 cells of 9773 m give 6371 km.

    `land_fraction` is honoured exactly only while `tectonics.shelf_fraction`
    is 0; with shelf mode on (the default) sea level is measured against the
    continental crust instead and the land area falls out of it. See
    docs/crust-types.md."""

    seed: int = 0
    N_c: int = 1024
    cell_size_m: float = 9773.0  # -> R_planet 6371 km
    R: int = 2  # refinement factor
    T: int = 64  # tile edge (fine cells)
    land_fraction: float = 0.30
    halo: int = 4


@dataclass
class TectonicsParams:
    """PLAN section 6 knobs.  Lengths on the sphere are chord/arc lengths in
    radians; "spacing" is the mean segment spacing ``sqrt(4*pi/M)`` (radians)
    so every *_factor knob is resolution independent.  Time is measured in
    tectonic steps (age, diffusion, speeds); there is no separate dt."""

    N_tect: int = 256
    segments: int = 20000
    initial_plates: int = 9  # (dyn-minimal: 1 supercontinent + 8 ocean plates -- 10 smaller ones had their ridges reach the girdle in 30-40 My; shipped 4) plates at step 0: ONE for the assembled supercontinent, the rest tiling the ocean. A supercontinent is a single rigid block -- nothing inside it collides until it breaks up -- so rifting is what raises this count, which is the right causal order
    steps: int = 4000  # 1500 ended the bake mid-dispersal with no belt younger than the breakup; 3000 is past the first reassembly (docs/plate-forces.md section 7), and 4000 is where `variable_extent` leaves the best hypsometry Earth scale has measured (band error 12.7, section 4d).  Not higher: at 8000 the continents drain to 25 % of the crust and 12000 to 22 %, an Earth-scale divergence `small` does not show (section 4e)
    convection: float = 10.0  # ★
    myr_per_step: float = 0.15  # million years per tectonic step.  The dynamics read it (synth-dyn): every physical knob below -- My, cm/yr -- is converted to steps and radians per step through it (TectonicSim.steps_of / cmyr), so it is hashed, and the scorecard (scripts/tect_scorecard.py, globe/tectonics/diagnostics.py) reports in it.  The classic dynamics (CLASSIC_DYNAMICS) never read it.  Calibration: the model has no time unit of its own -- speeds are in segment spacings per step -- and three independent readings of the shipped Earth preset (20000 segments, spacing ~160 km) agree on it: plate speeds against Earth's (slab-attached ~8 cm/yr, continents ~3), the mean age of the ocean floor against Earth's 64 My, and ridge_age (400 steps, subsidence done) against the ~80 My half-space cooling takes -- each gives 0.09-0.25 My/step, centred near 0.15, so 4000 steps are ~600 My. Under the classic dynamics a run at another segment count has another spacing and so another calibration (speeds scale with the spacing); the synth-dyn forces are in physical units, but other resolutions are not measured
    growth: float = 0.0  # ★ k_G (thickness units per step).  0: continental crust changes by tectonics, not by crystallising out of the mantle everywhere -- 0.05 inflated the median thickness to 2x its birth value over a run
    dissolution_factor: float = 0.05  # ★
    deposit_density: float = 0.5  # k_D
    plate_size_jitter: float = 0.80  # spread of initial plate sizes: per-plate distance weights are 1 ± this, so 0 tiles the sphere evenly and ~0.8 reproduces Earth's hierarchy (a few plates covering most of the surface, microplates between). Earth spans ~94x largest:smallest with its top 7 plates over 92% of the globe; 0.35 gives a near-uniform 3.2x, which leaves every landmass a single collision zone
    # --- intraplate relief (globe/tectonics/intraplate.py) ---------------
    reorganise_every: int = 0  # steps between plate reorganisations (0 = never). Earth's interiors are former boundaries; with a fixed configuration an interior is never a boundary and so is never uplifted (measured: 6 m of local relief over 107 km across 88 % of land)
    reorganise_plates: int = 0  # plate count to re-cluster into (0 = keep initial_plates)
    rift_every: int = 0  # (dyn-minimal: rifts are triggered by insulation, see rift_mode; shipped 600) steps between rifting one plate in two (0 = never); opens new boundaries inside old interiors.  600 over 4000 steps with 4 initial plates is the combination earth-v13 through v16 were baked with
    rift_plates: int = 2  # at most this many plates rift per event; the actual number is 1..this, and the targets are drawn at random weighted by area rather than always being the largest. Deterministic argmax targeting sliced the same supercontinent every event, which reads as the whole map coming apart on a schedule
    rift_zigzag: float = 0.35  # spherical-noise perturbation of the rift plane. A spreading centre is a staircase of ridge segments offset by transforms, not a smooth arc; 0 gives the old straight cut
    rift_zigzag_freq: float = 6.0  # lattice frequency of that perturbation: higher = shorter ridge segments between offsets
    rift_speed_factor: float = 1.0  # × max_speed: the rate at which the two halves of a rifted plate move apart. It used to multiply `convection`, which is not a speed: 10 spacings per step against a cap of 0.3, so the one step that opened a rift rotated each half 14.4° (~1600 km) before the cap could apply
    plate_split_every: int = 1  # steps between checking whether a plate has been cut into disconnected pieces (0 = never). A plate is a rigid rotation, which preserves distances, so a piece severed by a trench can never drift away from its parent: measured at step 800 of the Earth preset, one plate was four pieces 46-83° apart moving as one body, which is what interleaves the ocean plates into ribbons
    variable_extent: bool = True  # give every segment its own extent (steradians of ground, Segments.ext) instead of the one global spacing, so that a continent-on-continent collision can take *area* away and thicken the crust on what is left -- crustal shortening -- and extension can give it back. Off (the default): extent is 4 pi / M everywhere, every ratio below is exactly 1, and the stage is bit-identical, so no baked world is invalidated. On (the default since earth-v16, and what `earth-v14`/`v15` were baked with): every field the stage writes moves, and every number tuned against equal-area segments (convection, force_scale, heat_insulation, orogen_decay, continental_fraction...) was re-measured for it -- the values around it here are those.  Off is still bit-identical to every world baked before v14, so setting it False in a params.yaml reproduces them. Why it exists: with fixed extent a collision can only delete the loser (the pair's area halves; continental crust falls to 0.17 of the segments by 1500 steps) or keep it whole (area is created from nothing; the share climbs to 0.83), and neither leaves a planet that can run several supercontinent cycles -- docs/plate-forces.md section 4b
    extent_min: float = 0.25  # a segment may shorten to this share of its birth extent before it is merged away into the survivor (below it the splat kernel would be narrower than the cloud's own packing length, which shows through as worms -- see splat_sigma_factor)
    extent_max: float = 3.0  # and may spread to this share before it splits in two, so extension cannot make one segment cover a continent
    arc_thickness: float = 0.55  # the column an island arc is born with, x continental_thickness, at the belt's density; 0 = the slab's own (0.2, oceanic density: the rule before v14, and bit-identical). An arc is crust made from the mantle -- ~20 km thick and continental in composition when it forms, not a relabelled 7 km slab -- and with fixed area the difference never showed because most arcs were deleted in the next collision; with variable_extent they survive, and at Earth scale 5.9 % of continental segments ended thinner than ocean floor, filled the submerged-shelf quantile on their own and put sea level 1.8 km too low. 0.55 is 20 of 35 km. The mass drawn from the mantle is ledgered (`arc_mantle`)
    extent_thin_floor: float = 0.5  # continental crust stops taking extent -- stops stretching -- at this thickness (x the initial continental column), and a gap beside it becomes sea floor instead. Continental crust on Earth rifts to about half its thickness and then breaks; with no floor, extension went on thinning it without limit: at Earth scale 5.9 % of continental segments ended thinner than ocean floor, that paper-thin crust set the shelf quantile, and sea level sat 1.8 km lower than the fixed-area run's (a plateau of a continent over a 2.4 km sea, docs/plate-forces.md section 4d)
    extent_sigma_floor: float = 1.0  # the smallest splat sigma a shortened segment may have, in *design* spacings. The kernel must stay at least as wide as the local point spacing (splat_sigma_factor) whatever the extent says, so shortening never uncovers the packing; where the floor binds, the footprint stops following the extent (logged as `sigma_floored`)
    weld_steps: int = 60  # steps a continental segment that has shortened onto another plate stays that plate's, whatever the shape of the segment cloud says. Without it split_disconnected hands the welded sliver straight back to the plate it came from (it is embedded in it), the same pair collides again the next step, and a run with continental_shortening on went from 9 collisions a step to 100 -- with the island arcs that ride on ocean-ocean collisions turning the planet 98 % continental by 4000 steps
    plate_split_min: int = 16  # segments a severed piece needs to become a plate in its own right (it inherits its parent's motion); smaller fragments are welded onto the plate around them. With `variable_extent` on it is the *floor* under `plate_min_area`, in design segments' worth of ground (summed extent / (4 pi / segments)): below a handful of points a plate is the sampling, not the planet, which is what keeps `small` (1500 segments, where 8e-4 of the sphere is 1.2 of them) at its old 16. Off, it is the whole rule, a count, as it always was
    plate_min_area: float = 0.005  # (2026-10-02: 0.005, ~2.5 Mkm2 -- smaller severed pieces weld to the plate around them instead of racing off as plates; plate births 1-1.4 -> 0.3-0.6 per 10 My, plates living 15-40 My; Cocos-sized plates (0.6 %) still exist) with `variable_extent` on, the ground a severed piece needs to become a plate, as a fraction of the sphere (summed extent / 4 pi; at least `plate_split_min` design segments' worth). A size, not a count: 16 segments is this at the shipped 20000 and a quarter of it at 80000, where seed 2 ran 35/48/83 plates at steps 1000/1250/1500 with the count against 6/15/14 with it scaled to 64 (and 10/15/11 at 20000). 8e-4 is exactly 16 segments at 20000, so the threshold did not move at the shipped resolution, only what it is measured in: a piece of 16 shortened slivers is no longer a plate. 0.010 sr, 4.1e5 km2 at Earth's radius: a microplate
    rift_min_area: float = 4.0e-4  # the ground a plate needs before a rift may cut it, as a fraction of the sphere (at least 8 design segments' worth, the hard-coded count this replaces and still the whole rule with `variable_extent` off). 4e-4 is those 8 at 20000
    hotspots: int = 12  # fixed points in the mantle frame (plumes).  They thicken crust drifting over them only with hotspot_rate > 0; with `volcanoes` on each one builds a chain of edifices on the plate passing over it (Hawaii-Emperor).  Earth has ~10-15 strong plumes (Courtillot et al. 2003).  The classic dynamics keep 0
    hotspot_rate: float = 0.0  # thickness added per step at a hotspot centre, tapering to 0 at its rim
    hotspot_radius_factor: float = 3.0  # × mean segment spacing: hotspot radius
    # --- crust types (globe/tectonics/segments.py) -----------------------
    # Earth's hypsometry is bimodal -- continental shelf and abyssal plain
    # ~4.5 km apart -- because it carries two kinds of crust with different
    # *fates*, not merely different numbers.  Continental crust cannot
    # subduct, so it survives and thickens; oceanic crust always can, so it
    # is born thin at a ridge and destroyed at a trench without ever
    # thickening.  Measured with one crust type, our height distribution was
    # a single broad hump (thickness a continuum 0.18-8.3, density 0.20-0.96)
    # and the ocean spanned 1846 m against Earth's ~3000.
    continental_fraction: float = 0.49  # (2026-10-03: 0.49 -- the supercontinent starts at mean 1.01 columns (35 km) of crust, and collisions thicken it to the dead-belt floor's steady state (~1.28, 45 km; Earth's continental crust averages 38-41 km) at constant volume, so its area settles: 0.40 -> 0.32 by 600 My, 0.49 -> 0.39-0.40 (land 26 %; Earth 0.40 / 29 %) on seeds 0 and 1423. A start sized for the settled state, not a drain: volume holds at 1.01-1.03 x. dyn-minimal: 0.40 Pangaea-sized; shipped 0.75) fraction of the initial crust seeded continental, as one assembled supercontinent. Earth's continental crust including shelves is ~40 % of the surface; this starts a little above it because collision thickening consumes area (measured over 1500 steps: 0.70 -> 0.60, 0.55 -> 0.36 -- the loss scales with the perimeter-to-area ratio, so a smaller continent loses proportionally more).  0.75 with `variable_extent` settles at 40.5 % of cells continental at 4000 steps, which is Earth's number
    craton_roughness: float = 0.80  # relative noise on each craton's own distance field, so nuclei are ragged rather than discs. Measured as the coefficient of variation of the centroid-to-edge radius (a disc is 0, real cratons 0.25-0.45): 0.152 at 0, with 13 of 22 nuclei under 0.15; 0.321 here, with 1. Higher fragments them -- at 1.2 the largest connected component falls to 73 % of a craton
    margin_taper: float = 0.35  # outermost fraction of the continent whose crust is stretched thin -- the continental shelf and slope. Earth's drowned margin is ~29 % of the continental crust
    margin_thinning: float = 0.45  # crustal thickness at the very edge of that taper, x normal. Earth's rifted margins run 35 km -> 10 km
    craton_fraction: float = 0.45  # fraction of the continental crust that is Archean craton -- old, thick, strong nuclei welded together by weaker mobile belts. Rifting is steered around them (see intraplate.rift), which is why Gondwana split between Amazonia, West Africa, Congo and Kalahari rather than through them
    supercontinent_roughness: float = 0.45  # spherical-noise perturbation of the initial continent's margin, radians. 0 gives a circular cap; this gives embayments and promontories
    cratons: int = 24  # number of proto-craton seeds the initial continental crust is grown from; fewer/larger gives a supercontinent, more/smaller a scatter of microcontinents
    craton_thickness: float = 1.25  # x continental_thickness for Archean cores. Earth's cratonic crust is 40-45 km against 30-35 for younger continental crust. Thickness here buys *strength*, not height: `craton_density` is raised to match, so a craton floats at the same level as a belt
    belt_thickness: float = 0.92  # x continental_thickness for the mobile belts welded between the cratons: younger and thinner, but lighter, so it sits at the same height
    continental_spread: float = 0.10  # relative fbm variation on continental thickness, so the distribution has width rather than two spikes
    continental_thickness: float = 1.0  # initial thickness of continental crust (Earth ~35 km)
    continental_density: float = 0.804  # mobile-belt crust, normalised to mantle = 1: Airy height = 0.92 * (1 - 0.804) = 0.180
    craton_density: float = 0.856  # Archean crust is thicker but *denser* -- a mafic granulite lower crust that younger crust does not have -- so 1.25 * (1 - 0.856) = 0.180, the same height. This is why Earth's shields are low: the Canadian Shield averages ~300 m, the Baltic ~200 m, the West African ~300 m, all with 40+ km of crust under them. Elevation on Earth comes from crust being thickened *now* (Tibet, the Andes) or recently (the Urals, the Appalachians), not from being old. Giving cratons the extra buoyancy of their extra thickness put them 1.6 km above the belts, which drowned the belts and left the continents reading as a scatter of circular islands (measured: cratons 0.1 % drowned and 70 % of all land, belts 55.7 % drowned)
    oceanic_thickness: float = 0.20  # initial thickness of oceanic crust (Earth ~7 km, i.e. 1/5 of continental)
    oceanic_density: float = 0.88  # Airy height 0.024 -- the ~7.5x gap that makes the histogram bimodal
    arc_accretion: float = 0.15  # fraction of a subducting *oceanic* slab welded onto the overriding plate as an arc; the rest returns to the mantle. 1.0 (the old behaviour) makes the crust a monotone accumulator
    shelf_fraction: float = 0.33  # if > 0, sea level drowns this fraction of the CONTINENTAL crust and the land area falls out, instead of `world.land_fraction` of the surface being forced dry. Earth is ~0.275 (continental crust incl. shelves ~40 % of the globe, land 29 %); 0.33 is what `variable_extent`'s hypsometry wants, measured -- it drowns the shelves the extent model actually builds and took the Earth-scale band error from 21.7 to 12.7 (docs/plate-forces.md section 4d). Required once the hypsometry is bimodal: an area quantile has to cut a hump whose size varies +-0.1 between seeds, and when it misses it lands in the trough (measured: land under 1 km of 78.0 / 75.6 / 17.4 % across three seeds of one configuration)
    orogen_decay: float = 0.006  # per step, the fraction of a belt's height above `orogen_floor_m` that goes back to the mantle. An orogen is only high while convergence feeds it; without this every belt a world ever built stays at full height and 31.2 % of the land ends up above 2 km against Earth's ~11 %.  0.006 with `variable_extent` on: swept against 0.0045/0.005/0.0055 at Earth scale, which buy 1.3-1.6 points of bedrock above 2 km -- inside the spread those runs show between each other -- for 2-3 km more peak, past Earth's 8849 m in all three (section 4e).  Our high ground is short because the belts are narrow, which is not this knob's to fix
    orogen_floor_m: float = 1200.0  # the height a dead belt settles at -- the `ural` profile's crest. The Urals and the Appalachians are low welts, not nothing
    flat_slab_age: float = 60.0  # a subducting slab younger than this (steps) is buoyant enough to shallow out, which jumps deformation far inland and builds a wide, low `laramide` belt instead of a narrow `andean` one. The Farallon flat slab under North America is the type case
    orogen_along_strike: float = 1.5  # how far, in segment spacings, one collision's cross-section fades out of its own plane. A ball query is a disc, so without this an isolated event paints a 3300 km circle of plateau instead of a point on a belt
    orogen_width_scale: float = 1.0  # every zone of the orogen cross-sections (orogeny.TYPES, in km: andean ~1,050, himalayan ~1,350, laramide ~1,750 wide) scaled by this. The heights are only the weights the accreted crust is shared by, so a narrower profile stacks the same mass onto fewer segments. At 1 the belts above 1500 m come out ~1,700-1,900 km wide on the earth preset against the Andes' 200-700 and the Rockies' 110-480 km; the splat's ~137 km blur and the relax limiter floor a belt at ~2-3 spacings (320-480 km) whatever this is. 1 = the profiles as written
    max_crust_thickness: float = 2.3  # continental thickness (× the 1.0 initial) above which the root delaminates; 0 = no limit. 2.3 is ~80 km, the thickest crust on Earth (southern Tibet). Unlimited, a few segments stacked to 8.3x and squashed the vertical scale everyone else shares. At 2.0 the 4-5 km land was 74-100 % crust at the cap on every seed (3-4 km: 12-74 %): orogens are mostly craton-derived (density ~0.84, not the belts' 0.804), and 2x of that stands only ~3.7 km above sea level, so the 4-5 km band held 0.2-0.9 % of the land against Earth's 1.7 (docs/ocean-depth.md)
    delamination: float = 0.05  # fraction of the excess over max_crust_thickness shed to the mantle per step
    arc_birth: float = 0.0  # (te/arcs: 0; the classic dynamics keep 0.05) probability that an ocean-on-ocean subduction converts the survivor to continental crust by a coin flip. It made one-segment continental specks -- 24-36 % isolated, the rest one-segment chains, 3.4-5.1 % of the planet by 600 My against Earth's ~1 % of intra-oceanic arc crust, none of them ever subducted -- and drew their extra crust from the mantle (`arc_mantle`). Arcs are now oceanic crust thickened at the volcanic front (`arc_front`), and they become continental only by docking (`arc_dock_km`). Was 0.20 when the old scattered-blob start needed propping up; with a supercontinent start it no longer sustains the continental area at all (52.1 % at 0 against 55.6 % at 0.20) and only scatters specks -- 312 continental components at 0.20 against 62 at 0, for 3.6 % of the area.  With te/mass's sinks at 0.05 (margin erosion calibrated without it) it was a second continental source nothing balanced: extent 0.43 / 0.45 / 0.49 and volume 1.12 / 1.20 / 1.35 at steps 1000 / 2000 / 4000 (Earth seed 4), 0.58 and 1.55 by step 8000 (seeds 0, 1)
    differentiation: float = 0.0  # fraction of the gap to `density_continental` a survivor closes per collision (0 = off (the default until the end-to-end measurement below is in: the coarse-grid prototype measured beta 3.71 -> 1.84, but that was a different amplitude basis and did not check whether the variance survives erosion)). Collision alone only averages density, so the elevation histogram stays one narrow spike; Earth is bimodal because thickened crust partially melts, the light granitic fraction stays and the dense residue is lost to the mantle
    density_continental: float = 0.30  # density floor differentiation drives collided crust toward: granitic continental crust, which floats high
    animate_frames: int = 0  # capture this many animation frames DURING the run and write quicklook/tectonics.webp (0 = off (the default until the end-to-end measurement below is in: the coarse-grid prototype measured beta 3.71 -> 1.84, but that was a different amplitude basis and did not check whether the variance survives erosion)). Re-simulating for an animation afterwards costs a second full run -- 39 minutes at Earth scale
    animate_width: int = 900  # animation frame width in pixels
    animate_fps: float = 12.0  # animation playback rate
    collision_radius_factor: float = 1.0  # × spacing: segments of different plates closer than this collide
    gap_radius_factor: float = 1.0  # × spacing: cells farther than this from every segment are divergent gaps
    overlap_fraction: float = 0.5  # segments of different plates closer than this × collision radius collide even when not approaching (no interleaving along transform boundaries)
    splat_sigma_factor: float = 1.0  # sigma (x mean segment spacing) of the Gaussian blend used to reconstruct the tect grid from the segment cloud. Must be >= the spacing: a reconstruction kernel narrower than its samples resolves the samples, and Poisson-disc packing has a characteristic length, so the bedrock came out covered in worms. Measured at Earth defaults, 0.5 gave 84 m of relief at the segment scale and 1.0 gives 42 m, saturating past 1.5 -- and once erosion drops the land median to ~40 m those worms *are* the coastline. (That last sentence blamed the wrong worms for the *lace*: with every within-kind height replaced by its kind mean the coast comes out rougher, so the lace is the flank of the crust-type step as the truncated kNN blend resolves it -- see splat_knn_base and docs/coast-fringe.md section 5)
    splat_knn: int = 12
    splat_knn_base: int = 0  # 0 = off (bit-identical bedrock). N > splat_knn: the *base* height (min(h, belt height) on continental crust, all of oceanic) is blended from the N nearest segments at the same sigma and only the orogenic excess (h - base) from the splat_knn nearest, so belts keep their width. Why: at sigma = 1 spacing the 12th neighbour still carries 2.6 % mean / 5.7 % max of the normalised weight, so the blend jumps whenever a segment enters or leaves the twelve, and those jumps on the flank of the continental/oceanic step are the lacy coastline -- not per-segment height history (replacing every within-kind height by its kind mean makes the coast rougher, docs/coast-fringe.md section 5). The crust-type boundary (c, crust_kind, the shelf mask) stays on the splat_knn blend. Measured with 48 on `small` seeds 0 / 1, tectonics only: coast L/sqrt(A) at equal area -5.6 % / -2.9 %, fingers 1.25 -> 0.94 / 1.11 -> 0.99 %, belt top within 1 %, land median +3 / +1 m, sea median +20 / -4 m; on tiny -1.2 / -10.8 %. It is a mild version of the refuted symmetric ramp (the land median rises 1..10 %, the sea shallows 1..4 %), fingers are not consistently better, and a 300-step Earth-parameter toy lost 10-13 % of its low belts, so it stays off until an Earth bake measures coastal belt heights and the 2-4 km bands
    splat_kernel: str = "gaussian"  # reconstruction kernel of the splat above (globe/tectonics/collision.py splat_weights), for every field finalise rasterises from the cloud (height, crust-type fraction, dh, age, density). 'gaussian' (default, bit-identical): the truncated splat_knn Gaussian, which jumps where a segment enters or leaves the list. 'wendland': Wendland C2 (1 - d/h)^4 (4 d/h + 1), compact support h = 3.454 sigma (the truncated Gaussian's central weight, collision.wendland_support), the whole support always gathered, so no jump. 'tapered': the Gaussian times (1 - (d/d_k)^2)^2 at the splat_knn-th neighbour. Measured on `small` seeds 0 / 1 and `tiny` seeds 0 / 1, tectonics only (docs/coast-fringe.md section 6): wendland coast L/sqrt(A) at equal area -5.0 / -2.7 / -1.8 / -7.5 %, vertical scale within +0.9 %, belt p99 within +2.2 %, fingers better on three worlds and worse on tiny seed 1 (6.00 -> 6.37 %), the sea median 0.4-2.0 % shallower on all four, and in shelf mode (small, shelf_fraction 0.275) the coast gain is -0.4 % with fingers worse -- it fails the shipping conditions on tiny seed 1 and on the sea, so the default stays gaussian. tapered narrows the kernel (vertical scale -38..-50 %) and roughens the coast +7..+19 %: refuted
    margin_sigma_factor: float = 0.0  # x mean segment spacing: sigma of the wide kernel that re-positions the continental/oceanic step at the margin (globe/tectonics/run.py margin_ramp). The narrow splat above resolves the individual boundary segments, so the shelf edge came out scalloped one segment at a time -- on earth-v5 the -500..-2000 m isolines sit a median 0.4-0.7 spacings from the crust-type boundary, and the viewer's light band along every coast is that step. Raise-only: the oceanic side climbs to the smooth ramp (a continental rise), land, belts and sea level are untouched. Measured on `small` seed 0: shelf-edge isoline L/sqrt(A) 7.79 -> 6.38, coastline and crust-type boundary unchanged. 0 = off (bit-identical bedrock)
    cascade_rate: float = 0.3  # ★
    cascade_threshold: float = 0.05  # bedrock units (thickness·(1−density)) per cell of neighbour distance on the tect grid
    cascade_passes: int = 3
    uplift_scale: float = 1.0
    uplift_window: int = 100  # k steps: uplift = uplift_scale * (h_now - h_k_ago) [per segment, metres] / erosion.iterations
    uplift_baseline: float = 0.02  # positive baseline outside collision zones, fraction of the 99th-percentile uplift
    heat_diffusion: float = 2e-5  # rad² per step (explicit, sub-stepped for stability)
    heat_grid_divisor: int = 4  # the heat field lives on an N_tect / divisor grid (low-frequency driver; cheap diffusion)
    heat_insulation: float = 0.003  # per-step drift of the heat *background* (the field heat_relax pulls towards): down under continental crust, up under ocean floor. Plates are pushed towards hot, so this is a supercontinent insulating the mantle beneath it and being pushed apart by the upwelling, then the ocean it opens cooling the field back. With 0 the background is the step-0 noise for the whole run and every fragment is drawn back to the same spot (docs/plate-forces.md)
    heat_relax: float = 0.002  # per-step relaxation of heat towards the initial background field (keeps the sources from saturating)
    strata_period: float = 40.0  # bedrock fabric: age interval (tectonic steps) between successive hard bands.  Crust accretes at plate boundaries, so lines of equal age are the strata it was laid down in and bands run parallel to the boundary that made them — narrow where accretion was fast, wide where it was slow (median ~7 coarse cells at the defaults).  0 = no fabric
    strata_amp: float = 0.35  # how far the fabric swings hardness either side of the base value.  Without it hardness is a smooth two-tone blend of crust age and density (autocorrelation still 0.53 at 64 cells) with no structure at drainage-basin scale, so every continent develops the same radial network
    heat_noise_octaves: int = 3
    heat_noise_freq: float = 1.5  # base lattice frequency of the initial heat noise (features ~ 1/freq of the diameter)
    damping: float = 0.05  # omega *= (1 - damping) per step
    density_base: float = 0.5  # d_b in the growth term
    height_scale_m: float = 26400.0  # metres per bedrock unit when both relief_m and relief_spacings are 0
    relief_m: float = 0.0  # explicit override, metres: if > 0, scale bedrock so the 99.9th percentile of land sits at this height
    relief_spacings: float = 0.0  # when relief_m == 0: that percentile sits at this many mean segment spacings (metres), so the vertical scale follows the horizontal one at every preset (0 = use height_scale_m)
    # --- fractal detail (globe/tectonics/run.py inject_detail) -----------
    detail_amp: float = 0.0  # detail added to the bedrock, as a fraction of the LOCAL relief over detail_relief_cells. Erosion reworks the spectrum it is handed but cannot add variance that was never there: measured beta 12.99 leaving tectonics, 6.47 after erosion and 6.0-6.3 after refine at any R, where real topography is ~2. 0 = off (the default until the end-to-end measurement below is in: the coarse-grid prototype measured beta 3.71 -> 1.84, but that was a different amplitude basis and did not check whether the variance survives erosion)
    detail_cells: float = 16.0  # coarse cells in the longest injected wavelength; shorter octaves follow. Wavelengths above this belong to tectonics and injecting them would fight the plate-scale relief
    detail_octaves: int = 5  # octaves below detail_cells (5 reaches half a cell)
    detail_relief_cells: int = 9  # window (coarse cells) the local relief is measured over, so plains stay flat and mountains get rough
    ranges_amp: float = 0.8  # ranges and basins along the orogenic belts' strike (globe/tectonics/ranges.py): oriented Gabor noise, crests along the belt, amplitude this x the belt's height above ranges_base_m (so +-amp of it at 2 sigma). The segment cloud is 160 km apart and its splat blurs ~137 km, so a belt leaves tectonics as one smooth swell ("one big mountain") that erosion can only carve, not divide into ranges. 0 = off; 0.8 is what earth-v11 was baked with and what its ranges come from
    ranges_wavelength_km: float = 80.0  # range-to-range spacing across the belt (the carrier wavelength; the across-belt envelope is the same)
    ranges_elongation: float = 3.0  # along-belt envelope / across-belt envelope where the strike is clear (structure-tensor coherence 1); round where it is not
    ranges_base_m: float = 400.0  # belt height is measured above this; plains and cratons below it get no ranges
    ranges_orient_sigma: float = 6.0  # tect cells (39 km on the earth preset) the structure tensor is averaged over (earth-v11: 6 -- a belt's strike is steadier over 230 km than over 78, and the crests follow it instead of wandering)
    ranges_strength_km: float = 500.0  # wavelength of the smooth field that sets how strong the ranges are along a belt, so they come and go; 0 = uniform
    ranges_basin: float = 0.5  # a basin's depth below the belt's swell as a fraction of a crest's height above it (basins fill)
    ranges_cap_m: float = 3000.0  # the belt height the range amplitude follows is capped here, so the tallest swells do not carry the tallest crests (earth preset at ranges_amp 0.4 without a cap: peaks 10-11 km)
    smooth_sigma: float = 1.0  # final Gaussian on the tect grid (tect cells); resampling to the coarse grid is cubic
    # -- sphere-specific knobs (see globe/tectonics/plates.py for the force model) --
    force_scale: float = 3e-4  # plate angular acceleration in spacings/step² per unit (convection × |∇heat| [heat per radian] / mass per area)
    slab_pull: float = 0.0  # force on the plate whose ocean floor is going down at a trench: per subducted oceanic segment, directed from the plate's centre of mass towards the trench, in units of the heat force on one segment under a unit gradient (heat per radian), scaled by slab age (age / ridge_age, clipped to 1: an old, cold slab is the heavy one). This is what keeps a plate going where it has started going -- on Earth the dominant driving force -- and without it a rifted half falls straight back onto the fixed heat attractor (docs/earth-v3-review.md). 0 = off
    trench_suction: float = 0.0  # × slab_pull: the same pull, on the *overriding* plate, towards the trench where its neighbour goes down. Slab pull alone only moves the ocean plates (a continent has no slab of its own), so a rifted half sits still while the sea floor beyond it subducts under its far margin; suction is what carries it out over the closing ocean (extroversion), and on Earth it is the second force on the overriding plate after ridge push. 0 = off
    ridge_push: float = 0.0  # force per segment of new crust spawned at a divergent boundary, on the plate it joins, directed away from the ridge (towards the plate's centre of mass), same units as slab_pull. Keeps a rift opening once it has opened: the crust it makes pushes its own two plates apart. On Earth about a tenth of slab pull. 0 = off
    continental_shortening: float = 0.0  # what a continent-on-continent collision does to the losing segment. 0: it is deleted and all of its mass goes to the winner (the crust doubles, the area halves -- the continental share of the crust fell 0.68 -> 0.36 over 3000 steps, docs/plate-forces.md). > 0: this fraction of its mass and thickness goes into the belt, it keeps the rest, stays alive, and joins the winner's plate -- crustal shortening that conserves area, the front of the losing plate accreting segment by segment onto the other, which is also how the two become one plate again without a rule saying so
    suture_collisions: float = 0.0  # continent-on-continent collisions between one pair of plates, summed with a memory of `suture_window` steps, at which the two plates weld into one (the smaller joins the larger; the pole is the inertia-weighted mean). A collision deletes the losing segment, so two continents that keep converging eat each other indefinitely -- the continental share of the crust halved over 3000 steps -- where on Earth the suture locks and the crust is conserved (India still converges, but the Tethyan crust between it and Asia is gone, not India). Welding is also what closes the cycle: the fragments become one plate again. 0 = off
    suture_continental: float = 0.5  # a plate welds only if at least this share of its area is continental. Island arcs are continental crust (`arc_birth`), so without the gate every ocean plate that lands an arc on a continent welds to it and the run ends as one plate
    suture_cooldown: float = 400.0  # steps after a rift during which the two halves cannot weld back together. The zig-zag cut grinds as the halves open, and those contacts are continent-on-continent, so without this every rift re-welded within 50 steps and nothing ever broke up
    suture_window: float = 300.0  # memory (steps) of the suture count: it decays by 1 / suture_window per step, so contacts have to be sustained, not accumulated over a whole run
    max_speed: float = 0.3  # cap on plate speed, spacings per step (0 = none); keep < collision_radius_factor / 2
    initial_speed: float = 0.1  # speed of the random initial plate rotations, spacings per step
    initial_thickness: float = 0.4  # crust thickness at t = 0
    max_thickness: float = 1.0  # crystallisation growth is faded by exp(-thickness / max_thickness)
    ridge_height: float = 0.1155  # thermal buoyancy of young *oceanic* crust, bedrock units: half-space cooling, buoy = ridge_height * max(0, 1 - sqrt(age / ridge_age)), so a ridge crest stands this high above crust of age >= ridge_age. 0.1155 x 26400 m = 3050 m, the ridge-crest-to-old-floor step of the GDH1 age-depth curve (Stein & Stein 1992: 2600 m at the crest, 5650 m asymptote; docs/ocean-depth.md)
    abyss_depth: float = 0.062  # thermal subsidence of fully cooled oceanic lithosphere, bedrock units, subtracted from all oceanic crust at finalise. The crust-only Airy column floats old sea floor at 0.024 against a sea level near 0.176, i.e. -4018 m, where Earth's abyssal plain starts; this puts it at GDH1's -5650 m and, with ridge_height, the crest at -2600 m. Read at finalise and by the viewer's tectonics frames (`frame_bed`), never by the simulation step: plate motion is untouched. 0 = the crust-only floor (docs/ocean-depth.md)
    ridge_age: float = 400.0  # age (steps) at which oceanic crust has finished subsiding. Must be comparable to the seafloor's actual lifetime or the term is dead: measured at 150 against a median crust age of 1500 it carried 0.1 % of the height variance
    new_thickness: float = 0.05  # thickness of crust spawned at divergent boundaries
    gap_cooling: float = 0.05  # peak heat removed (Gaussian blob, 1 spacing wide) per segment of new crust spawned at a rift
    subduction_heating: float = 0.01  # peak heat added (Gaussian blob, 1 spacing wide) per subducted segment
    spawn_spacing_factor: float = 0.85  # min spacing of new segments, × spacing
    relax_rate: float = 0.1  # per-step segment height cascade rate (0 = off (the default until the end-to-end measurement below is in: the coarse-grid prototype measured beta 3.71 -> 1.84, but that was a different amplitude basis and did not check whether the variance survives erosion)); moves rate*(Δh-thr)/2/knn to each lower neighbour
    relax_threshold: float = 0.15  # maximum stable slope, bedrock units per spacing
    relax_knn: int = 8
    orogen_shaping: float = 0.25  # how strongly a collision belt is pulled into its type's cross-section (0 = the old symmetric Gaussian bump). A real orogen is asymmetric and digs a foreland moat in front of itself; a Gaussian can represent neither. See globe/tectonics/orogeny.py
    belt_width_factor: float = 1.0  # sigma (× spacing) of the Gaussian that shares subducted mass among the survivor's same-plate neighbours (mountain belt width)
    boundary_width_factor: float = 3.0  # hardness: boundary_proximity falls to 0 at this × spacing from a foreign plate
    collision_zone_factor: float = 2.0  # uplift clamped >= 0 within this × spacing of a subduction of the uplift window
    area_blend: float = 0.05  # rolling blend of the measured Voronoi area per step (PLAN: 0.99 rolling == 0.01)
    label_every: int = 1  # rebuild the label map (gaps, areas) every n steps; collisions and forces run every step
    # --- dyn-minimal: an Earth-like start, one-sided insulation, boundary forces, force-driven rifts ---
    # (globe/tectonics/forces.py).  Every new parameter is in physical units, converted with
    # myr_per_step, so the time step can change later.
    # (myr_per_step is defined above; with dyn-minimal the simulation reads it, so it is no longer reporting-only)
    start_mode: str = "pangaea"  # 'pangaea': supercontinent on one plate in a superocean of `initial_plates - 1` ocean plates, an ocean-floor age structure, a circum-supercontinent subduction girdle and force-balanced omegas; 'classic': the shipped random-pole start
    ocean_plate_max: float = 0.22  # largest initial ocean plate, share of the sphere (Pacific today ~0.20); the tiling is redrawn until it fits
    start_age_cmyr: float = 3.0  # half spreading rate that ages the initial ocean floor: age = distance to the nearest ridge / this
    start_age_max_my: float = 180.0  # oldest initial ocean floor (Jurassic Pacific ~180 My)
    start_insulation: float = 0.35  # share of the full insulation deficit already under the supercontinent at step 0 (it has been assembled for a while)
    girdle_heat: float = 0.25  # initial heat ring (a downwelling) along the supercontinent's margin: the slabs of the subduction girdle
    heat_noise_amp: float = 0.15  # amplitude of the fbm in the neutral mantle (the shipped start was all fbm, amplitude 0.5)
    insulation_time_my: float = 100.0  # > 0: one-sided insulation -- under continents the background relaxes to `insulation_floor` (an upwelling, a repeller) with this time constant, under ocean back to the neutral mantle.  0: the shipped rule (ocean -> 1, an attractor everywhere)
    insulation_floor: float = 0.0
    slab_force: float = 1.5  # (2026-10-02: 1.5 with boundary_shear -- the fault friction holds every plate to its neighbours, and at 0.6 slab-attached plates fell to 3.5-5.5 cm/yr; 1.5 gives 6-8, Earth ~8) slab pull per unit trench length at a saturated, old slab, in heat units (Stokes: the same torque as a heat step this size across the trench); 0 = off
    slab_sat_km: float = 400.0  # slab length at which the pull saturates (the upper-mantle slab)
    slab_detach_my: float = 20.0  # e-folding time of a slab that is no longer fed (detachment)
    slab_age_my: float = 80.0  # slab pull grows as sqrt(age / this), floored at slab_age_floor
    slab_age_floor: float = 0.3
    slab_speed_cmyr: float = 18.0  # (2026-10-02: 18 with slab_force 1.5 and boundary_shear; the fault friction, not this, now sets the speeds) (synth-dyn 9; dyn-minimal 7) slab resistance (bending, interface) per unit trench length, set so an old saturated slab alone moves its trench at this speed: slab-attached plates saturate here whatever their size (Earth: 7.9-8.1 cm/yr median)
    boundary_drag_km: float = 600.0  # plate boundaries drag like a strip this wide of oceanic lithosphere on its base (0 = off)
    plate_response_my: float = 15.0  # (2026-10-02: 15 -- the momentum the smooth community simulations give their plates; with it plates >= 1 % turn a median 0.3 deg per 0.75 My (was 0.9) and at most 6-10 deg per 10 My (was 16)) > 0: e-folding time (My) over which a plate's motion follows the force balance (plate momentum); 0 = the damping rate
    rift_response_my: float = 3.0  # with plate_response_my: the response time of a rift's two halves, from the cut until rift_free_my after it broke through (0 = no exception)
    rift_free_my: float = 20.0
    boundary_shear: float = 1.0  # (2026-10-02: 1.0 with k 15 -- small plates had moved 7-11 cm/yr against their neighbours, 120-134 deg off their heading; now 4-5 cm/yr and 21-71 deg, and plate-pair boundaries flip converging/spreading 1-3 % of 0.75 My frames instead of 3-8 %) share of the boundary drag that is fault friction between the two plates (a coupling on their relative along-strike motion) rather than drag on the mantle; 0 = the old absolute drag
    boundary_shear_k: float = 15.0  # ...and its strength, x the boundary drag per unit length
    collision_drag: float = 120.0  # converging continent-continent contacts resist the relative motion with this x the boundary drag per unit length (0 = off)
    basal_drag_continental: float = 2.0  # > 0: plates drag on the mantle like their lithosphere -- oceanic as its column, continental this x that (x1.5 under cratons) -- instead of in proportion to the crust's column mass (0 = the shipped I)
    cc_heating: bool = False  # continental losers also heat the field (shipped True): a collision has no slab and makes no downwelling
    rift_mode: str = "force"  # 'force' (synth-dyn): an insulated continental plate is cut where the force balance, released along the cut, opens it everywhere (forces.release; see rift_deficit ...); 'insulation' (dyn-minimal): it rifts when the insulation deficit under it, times sqrt(its continental share), exceeds rift_stress, through its upwelling; 'clock': the shipped rift_every
    rift_stress: float = 0.42
    rift_min_cont: float = 0.06  # continental area (share of the sphere) below which a plate never rifts by insulation
    rift_refractory_my: float = 45.0  # a plate born from a rift may not rift again for this long
    rift_check_every: int = 10
    rift_strength: float = 4.0  # a new rift couples its halves' normal motion with this x the smaller half's own drag (spread over the rift's contacts), holding the opening back to ~1/(1+G)...
    rift_weaken_km: float = 120.0  # (synth-dyn: with rift_neck_power 4 and rift_break_factor 2.5; dyn-minimal: power 2, factor 3) ...weakening as exp(-(opening / this)^2): necking -- slow phase, then fast (Brune 2016)
    rift_cut_zigzag: float = 0.06
    margin_collapse_my: float = 100.0  # (synth-dyn: 100 with the force test, margin_collapse_max_my 160 untested; dyn-minimal: 150, untested) ocean floor riding a continental plate detaches as its own plate, with a seed slab under the margin, once the floor along the margin is this old (median, +-15 % per plate): the Wilson cycle's closing half (0 = off)
    margin_collapse_every: int = 50
    margin_collapse_min: float = 0.005  # ... and only if the plate carries at least this much ocean floor (share of the sphere)
    margin_collapse_slab_km: float = 150.0  # the seed slab: ~the underthrusting that makes subduction self-sustaining (Gurnis 2004)
    spawn_gate: str = "pair"  # what a gap between plates needs to spawn sea floor. 'nearest' (shipped): its nearest segment moves away from it; 'outflow' (dyn-minimal): the net outflow of its 6 nearest; 'pair' (the arcs track's spawn_relative + spawn_normal, synth-dyn's default): the two plates around it separate there, their velocities at the cell taken across the boundary normal -- no new floor in the hole a slab leaves at a trench
    closure_ocean_only: bool = True  # the extent closure rescales only the sea floor (the books track's fix, te/books), so a girdle consuming floor faster than ridges make it does not inflate the continents
    # --- synth-dyn (on top of dyn-minimal) ---
    tectonic_radius_km: float = 6371.0  # the sphere every physical tectonics knob (km, cm/yr) is converted on: Earth's, whatever the rendered planet's radius, so a toy body (small, tiny: 4 km) runs Earth's angular rates instead of freezing (dyn-minimal converted through R_planet).  0 = the planet's own radius
    slab_onset_km: float = 0.0  # > 0: slab pull switches on smoothly between 0.5x and 1.5x this much slab (Gurnis et al. 2004: ~100-150 km of underthrusting before a slab sustains itself); 0 = linear from zero (dyn-minimal)
    rift_neck_power: float = 4.0  # a rift's strength necks as exp(-(opening / rift_weaken_km)^power): 2 is dyn-minimal's Gaussian, higher is a sharper end to the slow phase
    rift_break_factor: float = 2.5  # a rift's coupling is dropped (it is a ridge) once it has opened this x rift_weaken_km
    # the force rift test (rift_mode 'force'): a cut is released in the force balance and fails if its tension beats its strength
    rift_deficit: float = 0.66  # the time gate: a continental plate is tested once the mean insulation deficit under its continent (0-1), x (its continental share / rift_size_ref)^rift_size_exponent, passes this (x +-15 % per plate).  0.66 at 0.4 of the sphere is dyn-minimal's 0.42 with sqrt
    rift_size_ref: float = 0.4
    rift_size_exponent: float = 0.0  # dyn-minimal's sqrt (0.5) could never re-rift a 0.2 half (ceiling 0.44 against 0.42 +-15 %): dispersal stopped at 2-4 landmasses
    rift_min_open_cmyr: float = 1.0  # a cut qualifies only if, released, its halves open at least this fast on average
    rift_candidates: int = 24  # candidate cuts per tested plate (half through its upwelling, half through points drawn by deficit / strength)
    rift_tension: float = 0.0  # optional floor on the chosen cut's tension -- free opening (cm/yr) x the halves' reduced drag (sr of oceanic lithosphere) / cut length (rad) / the cut's mean strength (2-9 at the start of every seed, flat in time: the girdle's pull, not the insulation, so it cannot be the clock); the best qualifying cut by this score is the one that rifts
    rift_max_angle: float = 80.0  # every crossing of the cut within this many degrees of the cut's centre (no far side)
    rift_max_conv: float = 0.02  # at most this share of the cut (by length) may converge when released
    rift_end_open: float = 0.4  # the cut's 10th-percentile opening must be at least this x its mean (no hinge at one end)
    rift_half_min: float = 0.2  # each half keeps this share of the plate's continent...
    rift_min_half: float = 0.012  # ...and at least this much continent (share of the sphere)
    rift_slow_cmyr: float = 0.8  # a new rift's strength G0 is calibrated so it opens at this rate at first (Brune 2016: < 1 cm/yr for 20-25 My)
    rift_strength_max: float = 60.0  # cap on G0
    rift_abort_my: float = 40.0  # a rift that has opened less than half of rift_weaken_km after this long has failed and heals (its halves are one plate again); 0 = never
    rift_stall_km: float = 0.0  # (te/dyn, experimental) > 0: a rift past half of rift_weaken_km that opened less than this over the last rift_abort_my has stalled, and its halves' coupling is released (they are two plates with an ordinary boundary); 0 = it holds until it opens rift_break_factor x rift_weaken_km
    rift_craton_strength: float = 3.0  # lithospheric strength against a mobile belt's 1
    rift_suture_strength: float = 0.5  # crust assembled within rift_suture_my
    rift_suture_my: float = 150.0
    rift_ocean_strength: float = 1.5
    margin_collapse_test: bool = True  # an old margin fails only if, released, its floor (with a seed slab's pull) closes on the continent: the forced part of initiation (Gurnis et al. 2004)
    margin_collapse_min_cmyr: float = 0.0  # ... at a mean approach above this
    margin_collapse_max_my: float = 160.0  # ... or, untested, once the margin's floor is this old (spontaneous foundering of the oldest floor; Earth keeps almost none older than ~180-200 My)
    # small plates live and die (dyn-events' merge_microplates)
    micro_area: float = 0.005  # a plate below this share of the sphere with no live slab of its own is passive... (0 = off)
    micro_life_my: float = 12.0  # ...and once passive this long is captured by the neighbour it shares most boundary with and is not converging on (Morra 2013: small plates live 10-20 My); remnants below plate_min_area are captured at once
    micro_conv_cmyr: float = 0.2
    micro_every: int = 10
    suture_time_my: float = 20.0  # > 0: two plates meeting along >= suture_min_km of continent-continent contact with a median relative normal speed below suture_rate_cmyr for this long are welded into one (a finished collision; the next rift reopens the young suture); 0 = off
    suture_min_km: float = 800.0
    suture_rate_cmyr: float = 1.0
    # The C-C contacts of an assembled continent keep closing and count as collisions every step
    # (synth-dyn risk 1: 0.66-0.84 of all collisions after 150 My).  Not the damping lag:
    # measured on seed 1 at 375-480 My the contacts close at a median 0.79 cm/yr and the force
    # balance's own terminal velocities at 0.75-0.79 -- the collisional drag slows them, as
    # designed.  And the share is a count: every converging C-C pair counts every step, a trench
    # segment once when it goes down.  By the plates' convergence (the census's boundary length
    # x closing rate, window cc_kin_share) C-C is 0.10-0.15 of it over 150-600 My on seeds 0-3,
    # where the count says 0.57-0.82.  (window cc_kernel_share, the C-C share of the ground the
    # collision *kernel* consumed, reads 0.02-0.03 there: the kernel's bookkeeping, not the
    # convergence -- it subducts 3.5-4.2x the O-C/O-O convergence; see diagnostics.py.)  The
    # persistence weld here, rift_stall_km and collide_frozen_ids are experimental and off: a
    # 1500 km / 50 My weld never fired over 375-480 My on seed 1, where every long contact was a
    # rift's half
    suture_persist_my: float = 0.0  # > 0: two plates whose continent-continent contact stays >= suture_persist_km for this long are welded, at any closing rate below suture_persist_cmyr -- a contact-persistence weld; the young suture's strength (rift_suture_strength) lets the force rift reopen it.  0 = off
    suture_persist_km: float = 1500.0
    suture_persist_cmyr: float = 0.0  # ... with the median closing over the contact below this at every check (0 = any rate)
    collide_frozen_ids: bool = False  # (te/dyn, experimental) True: the collision kernel reads every pair's plates as they were when the step's collisions began.  False (the shipped rule, and CLASSIC_DYNAMICS): a continental loser relabelled onto the survivor's plate collides again in the same call with its former plate-mates behind it, as the survivor's -- a within-step chain (32-37 % of C-C collisions, seeds 1/2/4/1423 at 225-525 My).  Off because removing the chain alone moves the Earth numbers away from Earth: at 600 My (the preset's 4000 steps) on seeds 0/1/2/3/4/1423 the ocean floor's mean age went from 71/65/52/62/65/75 My live to 76/89/76/88/89/126 frozen (Earth 64), and land above 2 km from 5.3/14.7/1.4/11.3/12.3/7.8 % to 2.2/3.4/2.7/1.9/1.4/1.0 % (Earth 13.4).  On one state the chain makes up for an under-take in the primary contacts: the kernel's C-C shortening against the census's C-C length x closing rate (window cc_take) is 0.84-0.92 live and 0.64-0.68 frozen on seeds 2/4/1423 at 225-525 My, and 1.08 / 0.90 on two halves closing at 1-8 cm/yr; over whole runs both settle near 0.7 (pooled per window, seeds 2/3: 0.66-0.73 live, 0.63-0.71 frozen) -- the overlap frozen ids leave builds up until the lens-capped takes catch up -- so the deficit is the kernel's either way.  At 1200 My (seeds 0/1) the comparison is mixed: ocean age 84/88 live, 73/90 frozen; land above 2 km 23.3/33.6 % live, 15.6/8.3 % frozen; continental extent 0.371/0.326 live, 0.385/0.383 frozen.  The C-C share of collisions by count fell with frozen ids (0.77/0.80/0.65/0.76 -> 0.72/0.73/0.71/0.70 over 150-600 My, seeds 0-3), the cycle intact, but the count is the misleading measure (above).  Before turning it on: a primary-take correction that brings cc_take within ~10 % of 1, then a gate on cc_take, ocean age and land above 2 km on seeds 0-3
    orogen_push: float = 0.0  # > 0: thickened crust at a continent-continent contact pushes the plates apart, this much per unit contact length at full thickening (heat units, as slab_force): the orogen's gravitational potential energy balancing the collision (Tibet against India; Copley et al. 2010)
    orogen_push_th0: float = 1.2  # ...from this continental column (x continental_thickness)...
    orogen_push_dth: float = 0.8  # ...rising to full over this much more
    ocean_tiling: str = "zipf"  # 'zipf': the initial ocean plates get rank^-ocean_plate_alpha target sizes in [ocean_plate_min, ocean_plate_max] (dyn-forcebalance's power-law tiling: one Pacific-like plate, then a tail); 'cluster': dyn-minimal's redrawn cluster_plates
    ocean_plate_alpha: float = 1.0
    ocean_plate_min: float = 0.012
    # --- island arcs (te/arcs; globe/tectonics/volcanoes.py) ---
    # An intra-oceanic arc is two things at two scales: a ridge of arc crust (20-35 km thick,
    # 100-200 km wide, crest 1-3 km deep) that the segment cloud carries as oceanic crust
    # thickened at the volcanic front, and the volcanoes standing on it (cones 10-30 km across,
    # 50-100 km apart, mostly seamounts) that no reconstruction of a 160 km cloud can show, so
    # they are point edifices riding the plates that finalise stamps at their own size.  Every
    # length here is km on the tectonic sphere (tectonic_radius_km), every time My
    # (myr_per_step), every height m at Earth's vertical scale (volc_ref_m_per_unit).
    crust_km: float = 35.0  # km of crust per thickness unit (continental_thickness 1.0 ~ 35 km), for the knobs below given in km
    arc_front: float = 1.0  # (classic 0) share of what an ocean-under-ocean subduction hands the overriding plate (arc_accretion of the slab) that goes to its VOLCANIC FRONT -- the overriding plate's oceanic segment nearest the point arc_front_km behind the trench -- as arc crust, volume for volume, instead of being spread over the island-arc belt profile (50-370 km behind the trench, 2 spacings wide), over which it never thickened past ~10 km.  Earth adds 30-90 km3 of arc crust per km of arc per My in a band 100-150 km wide over the slab's 100 km contour; arc_accretion 0.15 of a 7 km slab at 5 cm/yr is ~52
    arc_front_km: float = 180.0  # trench to volcanic front, km (Earth 100-300, mode ~180)
    arc_keep: float = 0.25  # (classic -1 = off) when an oceanic column thinner than arc_dock_km subducts, the overriding plate scrapes off this share of its arc crust (the column above oceanic_thickness) and arc_accretion of the sea floor under it; the rest goes down (subduction erosion: little of a subducting ridge or arc is offscraped).  Off, a subducting arc's crust went down like sea floor.  0.5 (the prototype's) kept the arc crust in the system: with whole-arc docking it piled up as plateaus of arc crust in the oceans and docked continents from 0.40 to 0.48 of the planet in 600 My.  With the terranes riding their own plates (intraplate.split_disconnected), 0.5 against 0.25 on Earth seeds 0-4 and 1423 at 600 My: intra-oceanic arc crust 1.4-2.0 % against 1.0-1.8 % of the planet, arc thickness p50 20-25 km against 20-27, crest p50 -2.2 to -3.0 km against -2.2 to -3.1 (means -2.70 / -2.82 km, inside the seeds' spread), stranded arc crust 0.32-0.58 against 0.15-0.52, docked terranes 2.1-3.8 % against 1.8-3.3 % and continental volume 0.99-1.24x against 0.94-1.23x step 0: more crust in the system and no better crests
    arc_dock_km: float = 17.5  # (classic 0 = off) an oceanic column at least this thick (km) on the down-going side of an approaching pair does not subduct: it docks.  Onto ocean floor it joins the overriding plate as a terrane; onto a continent it turns continental (booked `docked`) and the same contact shortens it into the margin (arc-continent collision: Taiwan).  A docked terrane is never handed back: while it is welded the other side goes down under it, its weld is renewed while the contact lasts, and a thick unwelded arc meeting it docks onto it -- unless its own plate's crust around it has gone, when it is welded onto the plate it sits in (intraplate.split_disconnected: exempt, terranes rode their own plates' poles through foreign plates, 7 % of all subduction going down under them); two terranes of two plates in contact stack (an arc-arc collision: the thinner shortened into the thicker).  Earth: crust thicker than ~17 km jams a trench (Cloos 1993)
    arc_dock_keep: float = 0.5  # of a docking arc only this share accretes (its ground with variable_extent, else its column); the rest -- forearc, dense lower crust, the slab under it -- goes down the trench, booked `subducted` (collisions subduct much of an arc: the Luzon arc under Taiwan, the Halmahera arc under the Sangihe).  1.0 accreted whole arcs: onto continents 12 % of the planet's ground and +27 % of the continental crust in 600 My on the first Earth seed, and onto ocean plates plateaus of arc crust no trench could take.  Lost at a terrane's first dock onto ocean floor and at its dock onto a continent, not when the same terrane docks onto an ocean plate again (collision.collide).  With 0.5 and arc_keep 0.25, Earth seeds 0 / 1 / 2 / 3 / 4 / 1423 at 600 My: continental extent 0.378 / 0.384 / 0.344 / 0.405 / 0.414 / 0.366 (synth-dyn seeds 0-3 0.364 / 0.376 / 0.373 / 0.379), volume 1.10 / 1.11 / 0.94 / 1.21 / 1.23 / 1.04 x step 0 (docking +0.18-0.29 of the continental books' 4.1-4.2, slab accretion +0.89-1.38, orogen decay -0.58 to -1.45), docked terranes 2.5 / 2.2 / 2.2 / 2.2 / 3.3 / 1.8 % of the planet, intra-oceanic arc crust (>= 14 km) 1.21 / 1.77 / 1.65 / 0.95 / 1.63 / 1.37 % at p50 20-27 km
    arc_max_km: float = 35.0  # (classic 0 = off) root foundering: an oceanic column thicker than this (km) sheds its excess to the mantle with the e-folding time arc_founder_my (booked in `delaminated`, counted in `arc_foundered`).  Earth's intra-oceanic arcs are 20-35 km thick however long they have been active; denser cumulates founder below that
    arc_founder_my: float = 3.0  # e-folding time of arc root foundering above arc_max_km, My (was `delamination`'s 0.05 a step, 3 My at 0.15 My/step, which a run at another step length would not have kept)
    terrane_relax_my: float = 30.0  # (classic 0 = off) a docked terrane (Segments.terrane: arc crust at slab density 0.88, which stood 1-2 km below the shelves and pulled the shelf-mode sea level down) loses its dense mafic root to the mantle with this e-folding time -- density towards continental_density at constant thickness, booked in `residue`, counted in `terrane_relaxed` -- as accreted arcs on Earth turn into andesitic continental crust
    arc_ridge_km: float = 60.0  # (classic 0 = off) finalise redraws the arc crust on ocean floor (the column above oceanic_thickness) as a ridge this sigma across strike (km) and one along-strike spacing along it, volume for volume, in place of what the one-spacing splat made of it (a swell 400 km across at half the column's relief)
    arc_water_loading: float = 1.45  # the arc ridge's relief above the ocean floor is Airy relief under water: a column stands rho_m / (rho_m - rho_w) = 3.3 / 2.27 = 1.45x as high in the sea as its (1 - density) buoyancy gives in air, which the shared vertical scale (set on land) does not know.  Without it a 28 km arc stood 1.9 km over the abyssal floor against Earth's ~3.7 km, and the crests sat at -2.7 to -3.0 km (p50) on the first Earth runs, the deep edge of Earth's 1-3 km.  With it, Earth seeds 0 / 1 / 2 / 3 / 4 / 1423 at 600 My: crest p50 -3.11 / -2.76 / -2.17 / -3.02 / -3.06 / -2.79 km, 39-74 % of the arc crust's ground 1-3 km deep (arc crust p50 20-27 km thick: the crests follow the thickness, not this factor)
    volcanoes: bool = True  # (classic False) volcanic edifices as points riding the plates: arc volcanoes at the volcanic front of persistent ocean-ocean trenches where the overriding crust is an arc, fed by the slab going down beneath them, and hotspot volcanoes over the `hotspots` plumes.  Finalise stamps them as cones into the bedrock; extinct ones go through erosion (uplift, replay), active ones are added back on top after it (the `volcano_active` field)
    volc_slab_g: float = 0.5  # an arc vent is founded or fed only where the slab under the trench (TectonicSim.slab, min(S / slab_sat_km, 1)) is at least this: >= 200 km of slab, i.e. the same trench converging >= 1 cm/yr for 4-15 My.  Sliver contacts that consume a few segments make no volcanoes
    volc_sep_km: float = 60.0  # along-strike spacing of arc vents: a slab event feeds the nearest standing vent of its plate within this distance, else founds one, so the vents fill the front at 1-2x this (random sequential packing: mean ~1.34x).  Earth's arc volcanoes stand 50-100 km apart; measured nearest-vent spacing p10 / p50 / p90 61-62 / 72-76 / 109-113 km on Earth seeds 0-4 and 1423 at 600 My -- set by this, not emergent
    volc_active_my: float = 6.0  # an arc vent not fed for this long is extinct
    volc_height_m: float = 2600.0  # median height of an active arc edifice above the arc ridge, m (Mariana / Izu volcanoes rise 2-3.5 km off the arc crest), scaled by sqrt(slab flux / a 5 cm/yr trench's) in [0.35, 1.6] and a lognormal(0.3) per vent.  Earth seeds 0 / 1 / 2 / 3 / 4 / 1423 at 600 My: 2.23 / 1.20 / 2.96 / 0.48 / 0.70 / 1.95 km2 of arc island per km of converging ocean-ocean boundary (Earth 0.4-9: Marianas 0.36, Tonga 0.9, Antilles / Aleutians / Vanuatu 7-9), 64-94 % of them standing on the arc ridge (ground above -3 km).  3200 m gives 1.2-1.5x the island area, with 54-86 % on the ridge -- more vents standing out of the abyssal floor rather than more of the arcs above the sea
    volc_height_max_m: float = 4500.0  # cap on an arc edifice above its ridge, m
    volc_flank_deg: float = 11.0  # arc edifice flank slope: base radius = height / tan (2.6 km -> 13 km)
    volc_decay_my: float = 8.0  # an extinct arc edifice's height e-folding time, My (subsidence and erosion: remnant-arc volcanoes are guyots)
    volc_hot_height_m: float = 4500.0  # median hotspot edifice above the sea floor, m (Hawaii ~9 km, Canaries / Reunion ~7, most < 4)
    volc_hot_height_max_m: float = 9000.0
    volc_hot_flank_deg: float = 6.0  # shield volcano flanks
    volc_hot_decay_my: float = 20.0  # a hotspot edifice decays from the moment the plate carries it off the plume (Kauai 1.6 km high at 5 My; guyots after ~30 My)
    volc_hot_active_my: float = 2.0  # ...and is active (not eroded: added after the erosion stage) for this long after its last feed
    volc_hot_sep_km: float = 80.0  # a plume founds its next edifice once the last has drifted this far off it...
    volc_hot_jitter: float = 0.4  # ...x U(1 - this, 1 + this), drawn per edifice, so a chain is not a string of equal beads at one cadence
    volc_hot_size_jitter: float = 0.3  # lognormal sigma of a hotspot edifice's size, per edifice (on top of a per-plume lognormal 0.35)
    volc_continental: bool = False  # also stamp the volcanoes of continental arcs and continental hotspots (Andes, Cascades, Yellowstone); off: only intra-oceanic arcs and ocean-floor hotspots
    volc_ref_m_per_unit: float = 26400.0  # the vertical scale (m per bedrock unit) the edifice heights are quoted at -- the Earth preset's height_scale_m; a planet whose finalise scale differs (a toy body) gets its cones scaled by scale / this, and their footprint follows the tectonic sphere
    # --- continental mass and visible area (te/mass: proto/crit-mass on these dynamics) ---------
    # Every knob is in physical units (km of crust, My) converted through crust_km (above) and
    # myr_per_step.  CLASSIC_DYNAMICS turns all of them off (the shipped sinks: everything
    # thickened crust loses goes to the mantle, and extent is invisible)
    orogen_return: float = 0.85  # share of the crust orogenic collapse takes off a belt that stays continental -- gravitational collapse and erosion spread it as ground (ext) at constant volume; the rest goes to the mantle (eclogitised roots, subducted sediment).  Earth's collisional loss is ~0.4 km3/yr of 3-5 gross (Scholl & von Huene 2009).  0 = the shipped sink, all of it to the mantle (relax_orogens)
    orogen_floor_cols: float = 0.15  # (2026-10-01: 0.15 -- with collapse at 200 My, seeds 0-3 at 600 My finalise to land bands 0-1/1-2/2-3/3-4 km of 59-77/18-23/4-8/0.7-5.7 % against Earth's 71/15/7.5/3.8; at 0.02 the integrated planet was 94-100 % below 1 km) > 0: the height a dead belt settles at, as a thickness of belt crust above `belt_thickness` (x continental_thickness), instead of `orogen_floor_m` through height_scale_m -- so small and Earth run the same crust physics (the metre floor was 2.45 columns on small and 1.15 on Earth, so only Earth ever decayed). Earth's shipped 1200 m is 0.232.  0.02 (~100 m above unthickened crust): everything thickened collapses back to it, so it is the continents' resting column and a higher floor is a thicker continent for the same crust -- at 0.04 the median continental height sat on the floor by 600 My and the mean column had risen 18 % (te/mass, Earth seed 1, arc_birth 0)
    orogen_collapse_my: float = 200.0  # (2026-10-01: 200 -- Earth's orogens lose their topography over 100-200 My (Clift et al. 2009); 40 flattened the integrated planet to 94-100 % of land below 1 km, 100 to 86-93 % at floor 0.02) > 0: e-folding time (My) of a belt's height above its floor -- the one rated thinning process (collapse spreads crust into ground at constant volume, `orogen_return` of it; 2-3 km3/yr kept as ground, 1.5-2.5 sr per 8000 steps).  25, 60 and 75 My moved neither the extent (within ~0.02) nor the land above 2 km (0-4 % at arc_birth 0, where too little C-C shortening reaches it -- 1.3-2.7 sr per 8000 steps, ~half an Earth rate -- and shape_belt spreads it over 500-1750 km belts).  0 = `orogen_decay` per step (0.006 = 25 My at 0.15 My/step)
    orogen_fold_cap: float = 1.0  # > 0: a belt's fold-and-thrust moves at most this x the volume the collision handed the survivor from its foreland into the range -- as fast as the plates converge, not once per contact per step (creeping C-C contacts thinned their forelands to 0.2-0.5 columns, 9-12 % of the continents, and dragged the shelf sea level down).  0 = the shipped rule (up to 0.4 of each foreland column per event)
    margin_stretch: float = 0.6  # > 0: a gap opening between two plates with continental crust on both sides is a continental rift, and the crust it opens in is the margins stretched until they break: the new segment there is continental, on the ground the gap opened (the design cell every new segment gets), its crust drawn at constant volume from the non-craton continental crust within margin_width_km beyond the rift's edge, which thins with it -- none of it below this column (x continental_thickness); once the zone cannot give the new cell this column the rift has broken up and the gap spawns sea floor.  Rifted margins thin from 35 km to ~10 km over 100-300 km before breakup (Brune 2016), ~20-25 km averaged over a segment's 160 km, and are the shelves.  The rate is the divergence's: a gap opens beside the last new segment only once the plates have drawn ~a spacing apart (Earth seed 5, 3000 steps: 6.4 margin segments per cell of newly opened continental-rift gap, against 10.8 sea-floor segments per cell at the ridges; zone median 8 segments reaching 215 km, max 331; zone column 0.89 -> 0.75 at the median per new segment, none below 0.6, no craton thinned).  Measured (Earth seeds 0-3 and 1423, 8000 steps): 1.18-1.32 sr of rifted margin per run, extent 0.362-0.397 at every 250-step sample.  Off (seed 1): extent 0.349 at step 8000 -- a continental rift spawned sea floor against unthinned margins, and nothing made thin continental crust once the start's tapered margins were thickened.  (Its first form handed the nearest continental segment a whole cell per step while the absorbed gap stayed open, thinning one segment from 1.27 to 0.64 columns in a step, cratons included; review of te/mass, seed 5)
    margin_width_km: float = 150.0  # with margin_stretch: how far beyond the rift's edge, either side, the margin stretches -- the non-craton continental crust within this of the rift gives each new margin segment its crust, so the zone thins together.  Rifted margins are 100-300 km wide (Brune 2016).  It sets how much ground a rift makes before it breaks (the zone's ground x (1 / margin_stretch - 1)); Earth seed 1, 8000 steps: 0 (the edge segment alone) 0.49 sr, extent 0.349-0.395, ending 0.358; 150 1.21 sr, 0.368-0.395; 300 1.97 sr, 0.378-0.400, but its wider deep margins pulled the shelf sea level down (land median 809 m, ocean median -3.38 km at step 8000, against 311 m and -3.78 km)
    delamination_return: float = 0.8  # share of the crust the thickness cap (max_crust_thickness, delamination) takes that stays continental as ground -- lateral flow and plateau collapse (Tibet extruding east) -- instead of foundering.  0 = the shipped sink: all of it to the mantle; the shipped cap took 5-8 km3/yr in a collision phase
    margin_erosion_km: float = 0.9  # subduction erosion + sediment subduction where sea floor goes down under continental crust: km of the overriding margin's crust removed per km of sea floor subducted under it (a volume per unit area of slab), taken as ground at the margin's own column -- the trench migrates into the continent; booked `margin_eroded`.  Earth's best estimate for subduction erosion including the subducted sediment is 1.2-1.5 km per km of slab (~0.7 forearc erosion + ~0.5-0.8 sediment; Scholl & von Huene 2009, ~3 km3/yr at 55,000 km of trench); the model has no separate sediment flux and applies this only under continents, so 0.9 is that whole budget at 0.6-0.75 of Earth's estimate.  Balanced against the merged additions (slab accreted onto continents plus the island arcs that dock onto them, te/arcs) so the continents grow by Earth's few tenths of a km3/yr: on Earth seeds 0 / 1 / 2 / 3 over 0-4000 steps, gross recycling 3.27 / 3.39 / 3.29 / 3.31 km3/yr (margin 2.92 / 2.88 / 2.86 / 2.95, the rest collapse and the cap not kept as ground) against additions 3.73 / 3.91 / 3.82 / 3.80 -- growth 0.46 / 0.52 / 0.53 / 0.49 km3/yr, continental volume 1.04x and extent 0.370 / 0.375 / 0.358 / 0.372 at 600 My.  0.6 (te/mass's calibration with no docking) grew them 1.52 / 1.35 / 1.44 km3/yr (volume 1.12-1.13x by 600 My); 1.0 gave 0.36 / 0.14 / 0.27 (and 0.25 on seed 3) with extent 0.357-0.369.  0 = off
    extent_split: bool = True  # make extent visible (globe/tectonics/extent.py): per plate, while its continents own more ground than the map gives them (the rolling Voronoi area, Segments.area), the segment with the most unseen ground sheds a child -- its own column, its own crust, no ground made -- onto its plate's margin sea floor; while they own less, a coastal segment hands its crust to a continental neighbour and becomes sea floor.  Needs closure_ocean_only
    extent_band: float = 1.0  # the balance's hysteresis, in cells: a plate splits only while its unseen ground holds this many of the child's cell and retreats its coast only while it is short by this many of the coastal cell
    extent_split_at: float = 0.75  # ...and a segment is a parent only once its own unseen ground (ext - its cell) holds this share of the child's cell, so a parent is not left far under its own cell
    extent_residence_my: float = 10.0  # a split child is not retreated, and a retreated coast is not split onto, within this long (My): the coast does not flicker
    extent_split_active: bool = True  # a plate with no sea floor of its own at its margin grows into a neighbour's (the trench pushed back) instead of keeping ground the map cannot show
    extent_merge: bool = True  # with extent_split: a plate whose continents own less ground than their cells show (margin erosion, shortened C-C losers) retreats its coast
    extent_every_my: float = 1.0  # the balance runs this often (My; every 7 steps at 0.15 My/step): it flips ~0.5 points a step and the rolling area it reads moves on a 3 My e-fold, so every step only cost (0.023 s of a 0.12 s step) and churned more (retreats undoing a child 348 vs 45 per 4000 steps, seed 1)
    extent_split_rate_my: float = 2000.0  # cost bound: at most this many splits (and as many coast retreats) per My -- 2100 a pass at extent_every_my 1 (ceil(2000 x 0.15 x 7)); it never binds (~3 splits per My are made)
    ocean_ext_relax_my: float = 10.0  # > 0 (with closure_ocean_only): the sea floor's extent relaxes toward its own cell (the rolling Voronoi area) with this e-folding time before the closure rescales it, so the closure's factor does not compound on old sea floor (a segment living 4000 steps gained e^1.6 x; one old slab carried 6 design extents into a single event).  And no sea floor carries more than extent_max.  0 = the closure's uniform factor alone



#: The values that restore the shipped (pre-synth-dyn) tectonic dynamics: the random-pole
#: supercontinent start at 0.75, the heat-gradient update with no boundary forces, the
#: two-sided insulation, the rift clock, the nearest-segment spawn test, the proportional
#: extent closure, no margin collapse or microplate capture, the coin-flip arcs (no arc
#: front, docking, foundering, volcanoes or plumes) -- and the shipped crust sinks (orogen
#: decay and the thickness cap to the mantle, no margin erosion, invisible extent).  ``small``
#: and ``tiny`` use them (their tests are calibrated to those dynamics); a world YAML can set
#: them too.
CLASSIC_DYNAMICS = dict(plate_min_area=8.0e-4, plate_response_my=0.0, start_mode="classic", rift_mode="clock", continental_fraction=0.75, initial_plates=4,
                        rift_every=600, slab_force=0.0, boundary_drag_km=0.0, collision_drag=0.0,
                        basal_drag_continental=0.0, insulation_time_my=0.0, cc_heating=True, margin_collapse_my=0.0,
                        spawn_gate="nearest", closure_ocean_only=False, micro_area=0.0, orogen_push=0.0,
                        suture_time_my=0.0, suture_persist_my=0.0, rift_stall_km=0.0, collide_frozen_ids=False,
                        arc_birth=0.05, arc_front=0.0, arc_keep=-1.0, arc_dock_km=0.0, arc_max_km=0.0,
                        terrane_relax_my=0.0, arc_ridge_km=0.0, volcanoes=False, hotspots=0,
                        # te/mass: the shipped sinks and invisible extent
                        orogen_return=0.0, orogen_floor_cols=0.0, orogen_collapse_my=0.0, delamination_return=0.0,
                        margin_erosion_km=0.0, extent_split=False, ocean_ext_relax_my=0.0, orogen_fold_cap=0.0,
                        margin_stretch=0.0)


def classic_dynamics(tp: "TectonicsParams", **over) -> "TectonicsParams":
    """``tp`` with the shipped dynamics (:data:`CLASSIC_DYNAMICS`), then ``over``."""
    return dataclasses.replace(tp, **{**CLASSIC_DYNAMICS, **over})


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
    creep_rate: float = 0.02  # hillslope creep: a second thermal pass with talus 0 at this rate (linear diffusion of the surface; submerged cells are inert); 0 = off (the default until the end-to-end measurement below is in: the coarse-grid prototype measured beta 3.71 -> 1.84, but that was a different amplitude basis and did not check whether the variance survives erosion).  Without it every particle path incises its own rill (drainage density saturates at one channel per ~3 cells, parallel micro-rills instead of a trunk network); 0.3 over-smooths the divides (docs/erosion-tuning.md).  earth-v11 was baked at 0.02 and the difference is visible: at 0.1 the divides come out rounded and the planet reads as smoother
    thermal_max: float = 50.0  # cap on the material a cell sheds per thermal pass (cell units): a tectonic cliff relaxes at a bounded rate instead of collapsing in one iteration
    talus_slope_soft: float = 0.6  # rise/run
    talus_slope_hard: float = 1.2
    min_volume: float = 0.5  # retired, ignored: kept only so manifests written before `min_volume_frac` still load.  It was a length (metres, divided by the cell size) standing in for a *volume*, which shrinks with the cell area: at 76 m cells every particle was born below it and died in 3 steps (docs/zoom-windows.md)
    min_volume_frac: float = 5e-5  # a particle dies of evaporation once its volume falls below this fraction of the iteration's spawn volume.  Relative, so it means the same at every cell size: the spawn volume is the rain per cell over particles per cell, which scales with the cell area.  5e-5 is the margin the old 0.5 m had on the earth preset's 9.8 km grid (earth-v9: spawn volume 0.91, 0.5 / 9773 = 5.1e-5 cell units = 5.6e-5 of it) -- ln(1 / 5e-5) / (dt * evap_rate) ~ 8,300 steps at evap 1, past max_steps, so the planet pass is unchanged in practice
    max_steps: int = 0  # 0 -> 2 * N
    checkpoint_every: int = 50
    quicklook_every: int = 50
    slope_gain: float = 2.0  # multiplies the gravity force in the particle direction update
    slope_saturation: float = 0.0  # > 0: gravity = slope_gain * s / sqrt(s^2 + slope_saturation^2) along the downhill direction (terminal-velocity flow; gentle slopes still steer); 0 = tangential surface normal (McDonald)
    erodibility: float = 0.2  # c_eq = erodibility * dh * (1 + k_disc * f(q)), f set by disc_exponent below; 1.0 = McDonald 2022
    cover_depth: float = 5.0  # alluvial cover scale (cell units): sediment this thick fully shields the bedrock below, and a thinner film shields it in proportion (Sklar & Dietrich cover effect), so erodibility blends from 1 (sediment) to (1 - hardness) (bare rock).  0 = the bare-rock-only gate, under which hardness reached 0.1 % of land cells (half the land carries < 40 cm of sediment, but any film counted as full cover) and every continent eroded at the same rate
    height_unit_m: float = 0.0  # kernel heights are in cell units (height_m / cell_size_m): every cap, talus slope and gravity term is defined in them; 0 or 1 (= cell_size_m) are the only accepted values, anything else raises
    disc_saturation: float = 32.0  # discharge scale (volume units ~ upstream cells) of the entrainment term: erf(q/disc_saturation) when disc_exponent is 0, (q/disc_saturation)^disc_exponent otherwise
    disc_exponent: float = 0.5  # > 0: unsaturated power-law entrainment c_eq = erodibility*dh*(1 + k_disc*(q/disc_saturation)^disc_exponent) (stream-power concavity theta = 0.5: a trunk river carries its load at a gentler slope than a rill, so long profiles are concave, valleys widen downstream and a channel survives on a floodplain); 0 = the saturating erf law, under which every plain became an alluvial fan and no channel could meander
    disc_saturation_cells: float = 0.0  # > 0: disc_saturation is this many cells of upstream area (x the iteration's rain per active cell) instead of a fixed discharge.  McDonald's erf(0.4 q) in his per-cycle volume units (512 particles over a 512^2 map) saturates at ~1280 cells.  A fixed discharge means a different area at every refinement: 32 is 3,300 km^2 of rain on the earth preset, so in a 76 m zoom window no stream ever reached the entrainment boost and water ran off as a sheet (docs/zoom-windows.md).  0 = use disc_saturation
    momentum_saturation_cells: float = 0.0  # > 0: the stream momentum push reaches half strength at this many cells of upstream area (particle.trace_particles mom_scale); McDonald's is ~512.  0 = at one spawn volume (the kernel before 11): full strength on every rill, which in a zoom window lines particles up into parallel sheets
    slope_limit_erode: float = 0.0  # > 0: soillib's pit-free erosion limit -- a cell loses at most this x cell diagonal x its downhill slope per iteration (particle.slope_erode_cap; soillib 0.25), so the bottom of a pit cannot be deepened.  0 = off
    slope_limit_deposit: float = 0.0  # > 0: soillib's deposition limit -- a cell gains at most this x cell diagonal x 0.3 (soillib's critical sediment slope) per iteration (soillib 0.25); min with iter_deposit.  0 = off
    slope_limit_exit: float = 0.02  # downhill slope assumed towards a neighbour off the kernel array (soillib exitSlope)
    max_erode: float = 12.5  # cap on terrain removed per particle-step (cell units) at trace time (against the chunk-start terrain)
    iter_erode: float = 25.0  # net erosion a cell may receive per iteration (cell units), enforced against the live terrain in apply order; the shortfall cancels the particle's later deposits (erosion/particle.py apply_changes)
    iter_deposit: float = 50.0  # net deposition a cell may receive per iteration (cell units); the excess moves back up the particle's path, the remainder waits in the per-cell `pending` stockpile (released at this rate)
    ocean_deposition_rate: float = 0.3  # a particle that reaches the sea keeps walking downslope on the seafloor, deposit-only, dropping this fraction of its load per step (submarine fan); no erosion, no discharge track below sea level
    ocean_steps: int = 64  # at most this many seafloor steps; what is still carried is offered to the last sea cell, and the surplus splits between that cell's pending stockpile and `lost_offshore` (`offshore_writeoff`)
    fan_room: float = 50.0  # a seafloor step may settle at most this much (cell units) on a flat sea floor per particle-step (the drop to the previous cell when larger); the sea-level ceiling, the fan_slope descent and iter_deposit still bound the pile in apply_changes.  0.02 (= DEP_FLOOR) throttled offshore dispersal to ~1 cell unit per stockpile and iteration, so river mouths parked most of their load in `pending`
    fan_slope: float = 0.05  # a submarine fan descends at least this much per cell away from its source (cell units per cell): the deposit ceiling of a seafloor step is the previous path cell minus this, and never above the waterline floor (`dep_floor_m`).  Setting it to 0 recovers 11 % of the offshore loss on the small preset: it clamps the walk, but what strands the load is a shelf already filled to the floor (a sea death has no fan ceiling; docs/sea-death-stockpile.md)
    offshore_writeoff: float = 0.25  # a particle that dies in the sea with a load its cell will not take (a shelf filled to `dep_floor_m` below the waterline, or the `iter_deposit` rate cap) parks (1 - this) of the surplus in the death cell's `pending`, to be re-injected next iteration as a seafloor particle, and writes this fraction off as `lost_offshore`.  1.0 is the old rule (delete the whole surplus: 3141 Mm of crust over the earth-v5 bake, docs/lakes-in-erosion.md); 0 would let the stockpile on a full shelf grow without bound (the deadlock commit b4f24d4 removed).  In between, a stockpile that keeps failing decays geometrically, so pending on the seafloor is bounded by inflow / this.  Measured on the small preset (docs/sea-death-stockpile.md): 0.25 keeps 25 % of what the old rule deleted at 60 iterations and 33 % at 120 (a parked stockpile gets more chances the longer the run), seafloor pending plateaus at ~5 k m on ~900 of 74 k sea cells, ocean deaths +3-4 %, seconds per iteration unchanged
    glacial_every: int = 10  # run the glacial pass (erosion/glacial.py) every k iterations; 0 = off (the default until the end-to-end measurement below is in: the coarse-grid prototype measured beta 3.71 -> 1.84, but that was a different amplitude basis and did not check whether the variance survives erosion).  Ice is where the mean annual temperature is at or below freezing (evap <= 0), which at the defaults is ~9 % of land, close to Earth's glaciated fraction
    ice_evap: float = 0.0  # ice forms where the climate field `evap` is at or below this.  `evap` is k_evap*max(T,0), so 0 is exactly the freezing line and a positive value is a warmer equilibrium-line altitude (more of the world glaciated).  This is the FIRST-ORDER control on lakes: measured across three seeds, lake area swung 5x with the seed (0.14 %, 0.31 %, 0.74 % of land) but at most 43 % with glacial_from/glacial_every, and one seed had no land below +1.6 C at all, so no ice and no glacial lakes were possible however the other knobs were set
    glacial_from: float = 0.75  # start glaciating this far through the run (fraction of erosion.iterations); 0 = glaciate throughout.  Earth is lake-rich because glaciation was *recent* — the basins ice cut ~10 ka ago have not had time to fill, and lake lifetime is short next to landscape evolution time.  Carving throughout instead gives the fluvial system the whole rest of the run to drain and backfill every basin
    # `glacial_rate` and `glacial_max` are lengths (metres; LENGTH_PARAMS_M).  They used to be
    # consumed in cell units, tuned on the small preset's 50 m cells, so at the `earth` preset's
    # 9773 m cells they meant 9773 m and 3909 m a pass, a factor of 195: one pass took 2263 m off
    # the highest point in fifty iterations (docs/missing-relief.md), planed the waterline flat in
    # a single step, and against the isostatic rebound ran away to 22.8 km peaks and -17.5 km pits
    # by iteration 800 (docs/streaks-and-flats.md).
    glacial_rate: float = 50.0  # bed lowered per glacial pass (metres) at the reference ice flux, scaled by sqrt(discharge/disc_saturation) and by (1 - 0.5*hardness).  Unlike every other erosional term this one has NO base-level limit — ice flows uphill out of a basin — which is what leaves the closed depressions that become lakes
    glacial_ramp: int = 4  # cells over which the carve ramps up from the ice margin inward.  Erosion that only scales with ice flux deepens a valley monotonically downstream, which drains; tapering it to zero at the snout leaves a rock lip with the deepest point inside the ice, i.e. a closed basin
    glacial_max: float = 20.0  # cap on the bed a cell loses in one glacial pass (metres; LENGTH_PARAMS_M)
    glacial_sticky: bool = True  # a cell glaciated in one pass stays glaciated while it is still cold, even once its bed is carved below sea level.  Off, the ice margin is the coastline in cold lowlands and it migrates inland pass by pass as the carve drowns the outer steps, printing land/sea bands and concentric arcs (docs/streaks-and-flats.md).  Persisted in checkpoints
    moraine_frac: float = 1.0  # fraction of the excavated rock deposited on the ice margin as moraine (the rest is lost); 1.0 keeps the pass mass-conserving, and the moraine dams valleys leaving an ice field, which is the second way ice makes lakes
    isostasy: float = 0.8  # (needs the glacial lengths in metres: against the old cell-unit glacial scale it ran away to 22.8 km peaks.)  Airy compensation of the mass surface processes move: an eroded column rebounds by this fraction of the rock removed, a loaded one subsides by it.  0.8 = continental crust over mantle (tectonics.continental_density 0.804), i.e. eroding 1 km of rock lowers the surface ~200 m.  Without it erosion lowers the surface one-for-one and planes a continent to base level (docs/missing-relief.md).  0 = off
    flexure_km: float = 60.0  # the rebound is regional, not per cell: Gaussian sigma of the flexural response (a ~30 km elastic plate).  A locally-rebounding valley could never be incised; a flexural one lifts the peaks around it.  In metres, so it means the same thing at every cell size
    isostasy_every: int = 20  # apply the accumulated rebound every k iterations: the flexural smoothing is ~250 monotone diffusion steps at the earth preset (~16 s), so this keeps it near +0.8 s per iteration
    uplift_max_m: float = 0.0  # ceiling on the tectonic uplift a cell receives per erosion iteration (metres; LENGTH_PARAMS_M); 0 = off, the default since `uplift_mode` 'replay': the start is lowered by the same capped field, so the cap no longer bounds the end state (with no erosion it is `bedrock` either way) and only moves where the crests spend the run -- nearer their final height, under erosion for longer.  On the small preset, replay at 2 m/it against 0 keeps 0.74 / 0.52 % of land above half the bedrock maximum against 1.10 / 1.11 % (seeds 0 / 1; bedrock 2.66 / 2.78 %), and neither puts a cell above the bedrock's 99.9th percentile (docs/uplift-replay.md).  What follows is why it existed under 'stack' (2 m/it was its default there).  The `uplift` field is the height a segment gained over the last `tectonics.uplift_window` steps spread over `iterations`, with no bound: a belt still rising at the last tectonic step keeps rising for the whole stage, and at a ridge crest nothing opposes it (no discharge, talus never reached at 9.8 km cells, the glacial carve tapers to zero there).  earth-v5: 4.51 m/it x 800 = 3.6 km stacked on a 6.7 km bedrock, top cell 9862 m = 5798 bedrock + 3162 uplift + 866 datum - 36 m of erosion, 2.2 % of land above 5 km (Earth 0.3 %); sweep-od006 8.52 m/it -> 14,780 m.  Global pass only (a window applies its uplift as given); the cap is taken before the mean is removed, so the applied field stays mass-free.  The value is a trade-off, not a guard on a few crests: on earth-v5 a 1 m/it cap touches 9.3 % of land cells but those are 52 % of the land above 2 km and 83 % of the 4-5 km band (the median capped cell loses ~700 m over the run), while 2 m/it touches 4.1 % of land and, by the static estimate, takes the share above 5 km from 2.2 to ~1.6 % and the maximum from 9862 to ~9070 m without reaching into the 2-4 km bands (docs/uplift-ceiling.md).  Note it is per iteration: the total a cell may gain over the stage is this times `iterations`, so the same value is a different ceiling on a preset with another iteration count
    uplift_mode: str = "replay"  # how the tectonic `uplift` field enters the stage. 'stack' (the rule until docs/uplift-replay.md): erosion starts from `bedrock` -- the crust at the last tectonic step, which already contains the last `tectonics.uplift_window` steps of orogeny -- and applies the field on top for all `iterations`, so that window is counted twice (earth-v5's top cell: 5798 m bedrock + 3162 m applied uplift).  'replay': erosion starts from the crust as it stood at the reference step, `bedrock` less the total the stage will apply (the same mean-free, capped field `apply_uplift` adds, times `iterations`), with the datum held on it, and replays the window during the run, so with no erosion the surface ends exactly at `bedrock` (up to the rigid datum hold).  Small preset, seeds 0 / 1: the uncapped stack ends at 744 / 890 m on a 585 / 584 m bedrock maximum with 0.57 % of land above the bedrock's 99.9th percentile; replay at 455 / 479 m and 0 %, land median 22.0 / 40.2 m against the capped stack's 22.9 / 39.3, ocean median within 6-15 m of the bedrock's instead of 40 m below it, a third less `lost_offshore` (small-preset numbers; at Earth scale the estimate on earth-v7 is an ocean median ~450 m shallower, -3195 -> ~-2750 m against Earth's ~-4070 (docs/ocean-depth.md), because the window's sea-floor subsidence was double-counted as well).  Global pass only; a window (refine) is untouched.  Replay makes `iterations` part of the checkpoint identity (the start state depends on it)
    resume: bool = True  # resume from checkpoints/ whose parameter + upstream + kernel-version hash matches; False recomputes from bedrock
    sea_mask_every: int = 10  # recompute the erosion-side sea mask and per-basin base level (erosion/maps.py refresh_base) every k iterations; 0 = off, i.e. the sea is `surface < 0` and every closed basin below the waterline is a marine sediment sink for the whole stage.  Measured at N_c=1024: the mask costs 0.40 s against a 5.4 s iteration and drifts 50-120 cells an iteration -- but it jumps 1k-19k cells on exactly the iterations the glacial pass (`glacial_every` 10) and the isostatic rebound (`isostasy_every` 20) run, because those are the passes that move the surface in a lump and open or sever a strait.  So the stride has to divide both, and 10 also keeps it in step with `flood_every`: the routing flood seeds on this mask and must never be handed a stale one.  0.74 % of the stage (docs/erosion-and-the-sea.md)
    basin_fill_grade: float = 0.1  # > 0: before the first iteration, the closed basins the tectonic bedrock arrives with are laid with sediment up to a plain rising this many metres per km from each basin's outlet (erosion/maps.py fill_basins); 0 = off.  0.1 m/km is a large lowland river's plain (the Amazon and Mississippi run 0.02-0.1)
    basin_fill_min_m: float = 1.0  # ...only basins deeper than this below their spill (metres)
    lake_balance: bool = True  # every flood_every iterations, solve each depression's water level with hydro's evaporation balance (hydro/balance.py) on the eroding surface and make the particles treat it as water: the load drops at the shore on entering a lake (a delta), the crossing exchanges nothing with the bed, the bed is never eroded below the level, and a lake drawn down below its rim is a sink like the sea. Off, a lake is a pit the routing surface carries particles across and hydro floods afterwards -- three unrelated water surfaces (docs/earth-v3-review.md section 2)
    lateral_rate: float = 0.0  # a particle cutting its bed also cuts the bank on the outside of its bend, at this rate x |sin(turn)| x the bed's erosion: 0 = off (the channel can only cut down, never sideways, so it never meanders).  Capped by how far the bank stands above the bed, like every other erosion
    window_lake_evap: float = 0.0  # windows with `window_lakes`: open-water evaporation per *window* cell, in the units of the window's own rain (`zoom_params` sets it to hydro.lake_evap / R², the coarse-cell value spread over the fine cells).  A depression settles where its inflow equals this summed over the water it covers, instead of filling to its rim (McDonald's water layer; ErosionState.refresh_lakes_window).  0 = fill to the spill point
    window_lakes: bool = False  # windows (refine / zoom): every depression the routing flood fills deeper than hydro.lake_min_depth is a flagged, overflowing lake, as `lake_balance` does on the planet (ErosionState.refresh_lakes_window).  Off: the refine stage is unchanged until measured (docs/zoom-windows.md)
    lake_fill: bool = False  # a lake keeps the sediment its rivers bring it: the load its shore has no room for is parked on the lake (`ErosionState.lake_load`, never more than the lake's `lake_room`) and every lake refresh lays it towards a plain rising `basin_fill_grade` from the outlet, far side first (`ErosionState.settle_lake_loads`); a frozen lake takes nothing.  Global pass only, and only with `lake_balance`.  OFF: on the quarter-resolution world it took lakes from 0.88 to 0.83 % of the land (0.50 % with the marsh rule), but at full resolution earth-v20 came out with 2.98 % against earth-v19's 3.13 %, six lakes over 100,000 km2 where there were four, and more water deeper than 10 m (1.89 % against 1.36 %): the sediment laid is a load, the crust under it subsides, and terminal basins downstream of lakes that no longer evaporate their inflow stand higher (a plateau lake of 33,000 km2 at 1,544 m became 231,000 km2 at 1,644 m).  Not ready until that is understood
    lake_trap: float = 0.9  # the fraction of a particle's load deposited at the shore cell on entering a lake (McDonald 2020 drops 90 % at the drain)
    dep_floor_m: float = 1.0  # depth of water the kernel keeps over any deposit under the waterline, metres: a submerged cell fills to this far below its water surface and no further, and mass wasting stops there too. It was a length in cell units (0.02 cells: 1 m on the 50 m preset it was tuned on, 195 m at the earth preset), which forbade deposition on the whole shelf -- no delta, coastal plain or shelf wedge could build (docs/crust-audit.md, docs/earth-bake.md)
    flood_every: int = 10  # recompute the particle routing surface (epsilon priority flood, erosion/route.py) every k iterations; 0 = steer on the raw terrain
    route_eps: float = 0.05  # minimum drop per cell of the epsilon-filled routing surface particles steer on (metres; LENGTH_PARAMS_M).  Its job is to let a particle cross a *lake* towards the outlet.  It was 1e-3 in cell units -- 5 cm per cell at the 50 m cells it was tuned on, but 9.77 m per cell at the earth preset, which raised the routing surface above the terrain over 93 % of the land and up to 2.6 km above it, and drew the radial streaks (docs/streaks-and-flats.md)
    pit_steps: int = 16  # kill a particle after this many consecutive uphill steps (stuck in a pit)
    chunk: int = 2048  # particles per parallel chunk (change-list capacity = chunk*(max_steps+16) entries of 24 B, ~100 MB at N_c=1024); terrain is frozen within a chunk.  2048 traces ~20 % faster than 512 (fewer parallel launches, less tail imbalance) with the same morphology on the single-face tests (docs/erosion-tuning.md)
    backend: str = "cpu"


# --------------------------------------------------------------------------
# cell-size dependence
# --------------------------------------------------------------------------
#: Erosion parameters that are physical *lengths*.  The kernel works in cell
#: units (``height_unit_m == cell_size_m``), so a length written straight into
#: it means "this many cell widths" and silently rescales with the grid: the
#: shipped 0.1 of ``cover_depth`` is 5 m of alluvium at the 50 m cells these
#: were tuned on and **1.0 km** at the 9.8 km cells an Earth-radius world
#: needs.  Same for the per-iteration caps, which become kilometres and stop
#: capping anything.
#:
#: So these are declared in **metres** on `ErosionParams` and divided by the
#: cell size here.  Their defaults are the values they were tuned at (the old
#: cell-unit number x 50 m), so a 50 m world is unchanged and every other cell
#: size now means the same physical thing.  Slopes (``talus_slope_*``,
#: ``fan_slope``), rates and ratios are genuinely dimensionless and carry over
#: untouched.
LENGTH_PARAMS_M = ("cover_depth", "max_erode", "iter_erode", "iter_deposit",
                   "thermal_max", "fan_room", "route_eps",
                   "glacial_rate", "glacial_max", "uplift_max_m")


def cell_units(ep: "ErosionParams", name: str, cell_size_m: float) -> float:
    """One length parameter (metres) in the kernel's cell units at this grid."""
    return float(getattr(ep, name)) / float(cell_size_m)


@dataclass
class HydroParams:
    lake_min_depth: float = 0.5  # m
    marsh_depth: float = 3.0  # m.  Standing water no deeper than this on the coarse grid is marsh, not lake: hydro leaves the cell dry in `water_surface`, marks it in `marsh`, and derive classes it wetland.  A cell here is kilometres across, and half a metre of water over one is a plain tilted by a metre, not a lake: 30 % of earth-v19's lake area was under 3 m deep (0.93 % of the land), sheets of up to 220,000 km2 that came and went with the last isostasy pass.  Earth's own are the Pantanal, the Sudd, the West Siberian and Hudson Bay lowlands.  0 = every depth of standing water is a lake
    river_threshold: float = 200.0  # cells of accumulation (× mean precip volume)
    requantile_land_fraction: bool = True
    lake_evap: float = 6.0  # open-water evaporation off a lake, in units of the land-mean *precipitation* depth at evap == 1 (a 28 C cell).  A closed basin settles where its inflow equals `lake_evap * evap` summed over the water it covers, instead of filling to its rim -- which is why the Caspian stands 28 m *below* sea level.  0 = the spill-point fill the priority flood alone gives.  Derived, not tuned: Earth land precipitation ~800 mm/yr, open-water evaporation where closed basins sit 1000-2000 mm/yr (Caspian ~1000, Chad ~2200), and `flow_acc` routes *precipitation* as if all of it were runoff where Earth yields only ~0.31-0.36 of it, so the number is (E/P)/runoff_ratio = 3.5-8.1, centre 5.8.  Measured on earth-full: the share of land draining internally is 0.4 % at 2, 9.3 % at 5, 35 % at 6 and flat above -- Earth is 18 %, and no value lands there because one basin takes 29 % of the land's precipitation and flips across 5-6 (hydro/balance.py, docs/erosion-and-the-sea.md)
    ocean_min_fraction: float = 0.02  # a body of water below sea level is *ocean* only if it covers at least this fraction of the globe (the largest one always counts). Everything else below sea level is a closed basin on land -- a Caspian, not a gulf -- and the flood fills it to its spill point as a lake. Measured on the first Earth bake before this existed: 504 landlocked basins, 6.9 M km² (4.5 % of the land), all classified as ocean


@dataclass
class WatershedParams:
    basin_max_cells: int = 512 * 512
    basin_min_cells: int = 64 * 64


@dataclass
class RefineParams:
    refine_iterations: int = 150
    detail_amp: float = 0.3
    coast_taper_m: float = 40.0  # the detail noise fades to nothing at sea level and is full this many metres above or below it. The noise is ridged (sharp creases at the two-cell wavelength) with an amplitude set by the coarse slope, which is largest at the shelf step, so added to a surface hovering within metres of zero it turned every coast into a two-cell fringe of inlets that read as a delta (docs/earth-v3-review.md section 3). 0 = no taper
    halo_cells: int = 8
    workers: int = 0  # 0 -> os.cpu_count()
    feather_cells: int = 8  # fine cells over which a basin's refined detail ramps in from its (frozen) divide; keep >= 2R (a coarse cell): a 2-cell ramp reads as a crease in the LOD-0 hillshade
    particles_per_cell: float = 0.25
    hardness_smooth_cells: float = 0.0  # Gaussian sigma (coarse cells, seamless over cube edges) of the hardness every level below the coarse grid sees (refine, zoom windows, planet levels): tectonics' strata bands are closed loops of equal crust age, 1-2 coarse cells wide (aliased) on 5-15 % of land, and they print into the fine surface as rings and zebra patches of stipple -- through the detail noise's (0.5 + 0.5 hardness) and, twice as much, through the kernel's bedrock erodibility (1 - hardness). 0 = off
    hardness_max: float = 1.0  # cap on that hardness: rock at 1 does not erode, so a hard band keeps all its noise while a soft one is smoothed. 1 = off


@dataclass
class DeriveParams:
    """PLAN section 11 knobs (globe/derive).  Rivers come from one of two
    sources (``river_source``): the coarse drainage graph, each reach traced
    through the fine grid and clipped at the lakes (the default), or the
    *fine* discharge thresholded so that a given fraction of the land is
    river and thinned to centrelines (the earlier path, kept for
    comparison)."""

    river_source: str = "graph"  # 'graph': rivers are graph/drainage.json's reaches traced through the fine grid, split at the lakes, width in metres from the reach's mean discharge (falls back to 'discharge' when the graph has no edges, e.g. stub hydro or a tiny world with no catchment over hydro.river_threshold); 'discharge': threshold the fine discharge and skeletonise (docs/earth-v3-review.md section 2: hair-thin rivers unrelated to the hydro flood, no respect for lakes)
    river_width_m_a: float = 30.0  # graph source: width_m = max(river_width_min_m, a * (Q / Q_ref)^river_width_b) with Q_ref = hydro's river_threshold_volume, so a reach at the channel threshold is a metres, not one fine cell. Leopold & Maddock's w ~ Q^0.5
    river_width_min_m: float = 30.0  # graph source: a first-order stream is tens of metres, not a cell; drawn into fine/river_mask as at least the centreline cell whatever the cell size (4.9 km at Earth)
    river_width_a: float = 2.0  # discharge source: w = a * (Q / Q_thr)^b fine cells (Q_thr = the river discharge threshold)
    river_width_b: float = 0.5  # exponent of both width laws
    riparian_cells: int = 3  # coarse cells from a channel / river mask cell (x R at fine resolution)
    wetland_cells: int = 3  # coarse cells from a lake cell
    min_river_order: int = 1  # rivers of lower Strahler order are dropped from rivers.json / the mask
    # -- river extraction (derive/rivers.py) --------------------------------
    river_mask_fraction: float = 0.0  # fraction of fine land cells above the discharge threshold; 0 = auto: river_fraction_scale x the coarse channel fraction of land
    river_fraction_scale: float = 1.5  # auto mode: a thresholded discharge blob is wider than a 1-cell D8 channel
    river_hysteresis: float = 2.5  # connectivity threshold = the discharge of river_fraction x this fraction of land; low-threshold blobs survive only if they contain a high-threshold cell (1 = off)
    river_fallback_fraction: float = 0.03  # auto mode when the coarse graph has no channels (stub hydro)
    discharge_smooth_cells: float = 1.0  # Gaussian sigma (fine cells) applied to the discharge before thresholding; 0 = off (the default until the end-to-end measurement below is in: the coarse-grid prototype measured beta 3.71 -> 1.84, but that was a different amplitude basis and did not check whether the variance survives erosion)
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
    # -- the HTML viewer (globe/viz/viewer.py); all hash-exempt ------------
    #: export ``<world>/viewer/`` at the end of every bake
    viewer: bool = True
    #: extra exports beside it: any of "equirect", "anim" (comma separated)
    viewer_formats: str = ""
    #: timeline frames captured during tectonics (0 = none)
    tectonics_frames: int = 60
    #: capture an erosion frame every this many iterations (0 = none)
    erosion_frame_every: int = 10
    #: cells per face of a timeline frame
    frame_res: int = 256
    #: cells per face of the final frame: 0 = the full resolution of its
    #: source (N_c, or N_c x R with viewer_refined)
    viewer_final_res: int = 0
    #: draw the final frame from the refined grid (fine/) when refine has
    #: run, instead of the coarse one.  It was off while the fine frame showed
    #: refine's own defects -- land specks offshore (the upsample overshooting
    #: a shallow shelf) and rivers blurred into the coarse discharge by the
    #: basin feather, ending short of the sea -- fixed by the coast pass, the
    #: unfeathered discharge and lake outflow (docs/viewer-rivers.md)
    viewer_refined: bool = True


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
        um = self.erosion.uplift_mode
        if um not in ("stack", "replay"):
            raise ValueError(f"erosion.uplift_mode must be 'stack' or 'replay' (got {um!r}); see docs/uplift-replay.md")
        ow = float(self.erosion.offshore_writeoff)
        if not (0.0 < ow <= 1.0):
            raise ValueError(f"erosion.offshore_writeoff must be in (0, 1] (got {ow}): 0 re-creates the unbounded "
                             "seafloor stockpile the write-off exists to prevent, and a value above 1 drains pending below zero")
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
        # A 4 km body, so none of the Earth-scale settings apply. Relief goes
        # back to following the tectonic pattern's own horizontal scale;
        # rifting is *required*, not optional: a supercontinent that never
        # rifts is one rigid plate where nothing happens, so crust age,
        # density and boundary proximity all stay uniform and the hardness
        # field goes flat (measured land-spread 0.053 at rift_every 0 against
        # 0.107 at 100). Three rifts over 300 steps is enough to give the
        # world some internal structure to test against;
        # and `shelf_fraction` is off because sea level measured against the
        # continental crust is the right model for a planet that *has*
        # continents, which a 4 km body does not -- while the tests need
        # `land_fraction` as a hard guarantee, and shelf mode makes the land
        # area an output.
        # the toy bodies keep the shipped dynamics (CLASSIC_DYNAMICS): their tests are calibrated
        # to the random-pole start and the rift clock
        p.tectonics = dataclasses.replace(
            classic_dynamics(p.tectonics), N_tect=64, segments=1500, initial_plates=8, steps=300,
            relief_spacings=1.5, height_scale_m=4000.0, shelf_fraction=0.0, rift_every=100)
        p.climate = dataclasses.replace(p.climate, n_advect=0)  # auto: sweep until stationary (cap 4*N)
        p.erosion = dataclasses.replace(p.erosion, iterations=60, checkpoint_every=30, quicklook_every=30, basin_fill_grade=0.0)  # toy bodies have no continental basins to fill
        p.watersheds = WatershedParams(basin_max_cells=48 * 48, basin_min_cells=8 * 8)
        p.refine = dataclasses.replace(p.refine, refine_iterations=20, halo_cells=4, feather_cells=4)  # 2R
        return p

    @classmethod
    def tiny_world(cls, seed: int = 0) -> "WorldParams":
        """Even smaller profile for unit tests (seconds)."""
        p = cls.small_world(seed)
        p.world = WorldGroup(seed=seed, N_c=32, cell_size_m=50.0, R=2, T=16, land_fraction=0.3)
        # 60 steps, so it needs its own rift interval to see any at all
        p.tectonics = dataclasses.replace(p.tectonics, N_tect=32, segments=300, initial_plates=5, steps=60,
                                          rift_every=20)  # inherits small_world's toy-body scale
        p.climate = dataclasses.replace(p.climate, n_advect=0)
        p.erosion = dataclasses.replace(p.erosion, iterations=10, checkpoint_every=5, quicklook_every=5, basin_fill_grade=0.0)  # toy bodies have no continental basins to fill
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


#: `default` is Earth (see `WorldGroup`); `small` and `tiny` are toy bodies for
#: tests and quick plumbing checks, not for judging terrain -- their landmasses
#: are a few km across, which is far too small for a drainage network to
#: develop any scale range (docs/terrain-realism.md).
PRESETS = {"default": WorldParams, "earth": WorldParams, "small": WorldParams.small_world, "tiny": WorldParams.tiny_world}
