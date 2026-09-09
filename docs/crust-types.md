# Crust types, and why the hypsometry was unimodal

Earth's surface elevation histogram has **two humps** — a continental
platform around sea level and an abyssal plain near −4000 m, about 4.5 km
apart, with comparatively little area between them. Ours had one. This is
the measurement that found out why, and the design that followed from it.

## The measurement

An Earth-radius world (`N_tect=256`, 20,000 segments, 1500 steps,
`relief_m=9000`), instrumented at the end of the run:

| | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| age | 35 | 211 | 1026 | 1500 | 1500 | 1500 | 1500 |
| thickness | 0.001 | 0.177 | 0.662 | 1.592 | 2.460 | 4.718 | 8.324 |
| density | 0.160 | 0.200 | 0.345 | 0.463 | 0.626 | 0.878 | 0.955 |
| height | 0.001 | 0.128 | 0.314 | 0.648 | 0.991 | 1.933 | 3.096 |

and the height histogram was a single broad hump peaking at 0.28 with a
long tail — no gap anywhere.

**Four findings, in order of how much they mattered.**

**1. There was no crust type.** Every segment obeyed the same law.
`crystallise` grows thickness by `G = k_G (1−T)(1−T−d_b)` at a density read
from the local heat, so a segment's state was simply a running integral of
the heat it had drifted through. Heat is continuous, therefore thickness
was continuous (0.18 → 8.3), therefore density was continuous
(0.20 → 0.96), therefore height was continuous. A continuum in, a continuum
out. Nothing in the model ever said *this is ocean floor*.

**2. The age-depth buoyancy term was dead.** It existed —
`ridge_height · exp(−age / ridge_age)` — but at `ridge_age = 150` against a
median crust age of 1500, `exp(−10) ≈ 4.5e−5`. Only **10 %** of the crust
was younger than `3·ridge_age`. It carried **0.1 %** of the height variance
(`var(buoy) = 0.0004` against `var(h_iso) = 0.374`). Not absent, not
starved of old crust — mis-scaled to the point of irrelevance, and not
fixable by turning the knob up, because a ±0.15 perturbation on a field
spanning 0 → 5.5 cannot make an ocean either way.

**3. Age did not predict thickness** (r = **0.055**). On Earth age predicts
seafloor depth almost exactly — depth ∝ √age is one of the most robust
relations in geophysics. Ours had no such relation because there was no
conveyor for it to describe.

**4. The crust was not recycled.** Over 1500 steps: 21,660 segments
spawned, 23,479 subducted — about one crust's worth of turnover, but
distributed as froth along boundaries rather than as a conveyor. **54 % of
the final crust was primordial**, and spawned crust died at nearly the same
rate (62 %) as primordial crust (51 %). Earth resurfaces ~100 % of its
ocean floor every ~200 My and ~0 % of its continents. That asymmetry is
the whole mechanism, and we had none of it.

## The design

The two populations exist because of an **irreversibility**, not a
difference of degree:

* Continental crust is too buoyant to subduct. It survives indefinitely
  and thickens with every orogeny.
* Oceanic crust is dense enough to sink. It is created at a ridge and
  destroyed at a trench without ever getting the chance to thicken — 7 km
  at birth and 7 km at death.

So `kind` (`OCEANIC` / `CONTINENTAL`) is a per-segment flag that **gates
the laws**, not a label applied to the result:

| where | rule | why |
|---|---|---|
| `initialise` | `seed_cratons` grows `continental_fraction` of the cloud from `cratons` weighted seeds | a per-segment random draw gives salt-and-pepper crust whose continents are one segment wide; real cratons are coherent |
| `spawn_segments` | new crust is **always oceanic**, at `oceanic_density` | it used to take `deposit_density(T)`, so crust born at a ridge — where heat is by construction highest — came out *light*. Backwards, and it meant the model could not make ocean floor at all |
| `crystallise` | grows **continental crust only** | ocean floor is a chilled melt layer, not an accreting pile. This is what holds the seafloor at its birth thickness and stops it drifting up into the continental range |
| `_apply_collisions` | crust type decides who subducts, *before* density | a continent cannot go down under ocean floor at any density. This is the irreversibility |
| `_apply_collisions` | an oceanic slab hands over only `arc_accretion` of itself; the rest returns to the mantle | at 100 % the crust was a monotone accumulator — thickness 8.3 from an initial 0.4 |
| `_apply_collisions` | continent-on-continent keeps **everything** — nothing subducts, the crust doubles | that is what a Tibet is |
| `_apply_collisions` | ocean-on-ocean converts the survivor to continental with probability `arc_birth` | island arcs are how continental crust is actually born. Without a birth channel the continental area can only shrink from the initial seeding |
| `delaminate` | continental thickness above `max_crust_thickness` decays to the mantle | see below |
| `ridge_buoyancy` | `max(0, 1 − √(age/ridge_age))`, oceanic only | the actual half-space cooling result, and it reaches zero at a finite age, so `ridge_age` means "subsidence is done" and can be read off the seafloor's measured lifetime |
| `spread_collisions`, `relax_segments`, `segment_cascade` | refuse to move thickness across a type boundary | the continent-ocean contact is a ~4 km step in bedrock height, far above any plausible cascade threshold, so an unrestricted kernel drains every coastal continental segment into the seafloor beside it — exactly the leak that closes the gap crust types exist to open |

### Three defects the first implementation had

Recorded because two of them were mass-conservation bugs that the
`nan`-free, finite output would not have revealed.

1. **`spread_collisions` spread mass that was never accreted.** It handed
   the survivor's neighbours the *whole* subducted slab, while
   `_apply_collisions` had transferred only `arc_accretion` of it. A mass
   source at every trench. Fixed by passing the same fraction through.
2. **`_segment_cascade` drove thickness negative.** It clamped nothing,
   which was survivable when every segment was ≥ 0.4 thick and is not when
   ocean floor is 0.2. Measured `thickness p1 = −0.030`, `height p1 =
   −0.009`. Fixed with the same `0.5 × thickness` clamp `_relax_kernel`
   already had.
3. **Continental stacking was unbounded.** Every orogeny doubles the
   survivor and nothing stopped it, so a handful of segments reached
   thickness 8.3 against a continental median of 2.2. That tail is not
   harmless: `finalise` pins the vertical scale to a high percentile of
   land, so a few runaway spikes squash the entire map beneath them — the
   land/ocean gap came out at **400 m, worse than before crust types
   existed**. Earth saturates near 70 km even under Tibet, because a thick
   root enters the eclogite field, becomes denser than the mantle around
   it, and founders. `delaminate` sheds the excess to the mantle.

## The vertical scale is derivable, not a knob

Our height law `h = t(1 − ρ)` **is** Airy isostasy with ρ normalised to
mantle = 1, so it already has a physical metres-per-unit. Work it out from
Earth both ways:

```
continental: 35 km crust, ρ = 2.7/3.3  ->  H = 6.36 km   vs our h = 0.180
oceanic:      7 km crust, ρ = 2.9/3.3  ->  H = 0.848 km  vs our h = 0.024
             ratio 7.5                             ratio 7.5
=> 1 bedrock unit = 35.4 km, from either population
```

The ratio matches to two digits, which is the check that
`continental_density` / `oceanic_density` were chosen consistently. And
because both sides agree, the scale is not free:

| quantity | Earth | bedrock units |
|---|---|---|
| ridge thermal buoyancy | 3.0 km | **0.085** (`ridge_height`) |
| Tibet, 70 km crust | 12.7 km | 0.360 (so `max_crust_thickness` = 2.0) |
| abyssal → Everest | 11.9 km | 0.335 |

So an Earth-radius world should set `relief_m = 0`, `relief_spacings = 0`
and `height_scale_m = 35354` rather than rescaling to a target relief.
`relief_spacings` remains right for the small presets, where the point is to
keep relief proportional to a 4–32 km body — see docs/world-scale.md.

Measured, switching an Earth-scale run from `relief_m = 9000` to the
isostatic scale moved the land/ocean gap from **2398 m to 4581 m** against
Earth's 4540. Pinning a percentile of land was letting a handful of runaway
orogen spikes set the scale for everything beneath them.

## `land_fraction` has to cut *inside* the continental population

This is the largest error left, and it is a configuration mistake rather
than a missing mechanism.

On Earth continental crust including the drowned shelves is ~40 % of the
surface while land is 29 %: sea level sits **within** the continental hump,
so normal continent is only ~100 m above water and the shelves are under it.

Ours ends a 1500-step run at 19–23 % continental against
`world.land_fraction = 0.30`. To expose 30 % of the surface, sea level must
therefore drop *below* the continental base and into the ocean floor — and
every continent then stands its full 0.18 units, **6.4 km**, clear of it:

| | land p50 | land < 1 km |
|---|---|---|
| ours (cont. fraction 0.19, land fraction 0.30) | 3283 m | 37.9 % |
| Earth | ~300 m | 71 % |

So `continental_fraction` must *end* above `land_fraction`, not start there.
It decays over a run — collisions consume continental segments, `arc_birth`
makes new ones — toward an equilibrium set by the ratio of continent-continent
to ocean-ocean collisions:

| `arc_birth` | continental fraction at step 1500 (from 0.35) |
|---|---|
| 0.02 | 0.186 |
| 0.08 | 0.199 |
| 0.20 | 0.226, and nearly flat over the last 500 steps |

Raising `arc_birth` alone is a weak lever because the process is
self-limiting: more continent means more continent-continent collisions to
consume it. The seed has to be higher too.

## Sea level cannot be placed by surface area

The fix above -- seed more continental crust so `land_fraction` cuts inside
the hump -- works on the seed it was tuned on and fails on the next one.
Three seeds of one otherwise identical Earth-scale configuration, at a fixed
`land_fraction = 0.25`:

| | seed 0 | seed 1 | seed 2 | Earth |
|---|---|---|---|---|
| continental fraction at step 1500 | 0.351 | 0.462 | **0.260** | ~0.40 |
| land under 1 km | 78.0 % | 75.6 % | **17.4 %** | 71 % |
| ocean median | −3672 m | −3573 m | **−1555 m** | −3700 m |

Seed 2's margin over `land_fraction` was 0.01, so its quantile landed in the
trough between the humps and every continent read as a plateau again. **No
value of the knob fixes this**: an area quantile has to cut a hump whose
size varies by ±0.1 between seeds, and the knob is a constant.

`tectonics.shelf_fraction` measures against the thing that varies instead --
drown that fraction of the *continental* crust and let the land area fall
out. That is what sea level physically is: Earth's oceans hold just enough
water to cover the shelves, ~27 % of the continental crust, leaving 29 % of
the globe dry. Same three seeds, `shelf_fraction = 0.275`:

| metric | area quantile | **shelf mode** | Earth |
|---|---|---|---|
| land under 1 km | 57.0 ± **28.0** | **75.6 ± 3.7** | 71 |
| ocean median | −2933 ± **976** | **−3703 ± 21** | −3700 |
| land median | 989 ± **1031** | **276 ± 36** | ~300 |
| land/ocean gap | 3923 ± 109 | **3979 ± 56** | ~4000 |

Ocean-depth scatter falls from ±976 m to ±21 m. Land *area* is now an output
(23.6 ± 5.5 %) rather than a forced constant, which is the trade: the shape
of the hypsometry is stable and the amount of land varies with how much
continental crust the run happened to keep.

## One number is calibrated, not derived

`height_scale_m` ships at **26400**, not the 35354 the isostasy above gives.

Earth's crustal densities predict a 5510 m continent-to-abyssal step; Earth's
observed step is ~4000 m. The 34 % discrepancy is real -- water loading of
the ocean basins accounts for roughly 1.2 km of it, and the rest is that
"continental crust" is 35 km at ρ = 2.7 only as a global average. So the
derivation fixes the *ratio* between the two crust types, which is what
makes the histogram bimodal, and the absolute scale is calibrated against
Earth's measured hypsometry. Do not present 26400 as falling out of the
physics; it does not.

## Configuration

`PRESETS["earth"]` carries all of it. Tectonics only -- erosion is
calibrated in cell units at ~50 m cells and does not transfer to the 9.8 km
cells an Earth-radius world needs (docs/world-scale.md section 4), so run it
with `--to tectonics` until that is settled.

## Continent shape: what moves it, measured

The hypsometry knobs above are settled. These decide whether the land comes
out as one blob or as several plausible continents, and they **interact** --
sweeping them one at a time is misleading.

Single seed, otherwise the `earth` configuration, at 1500 steps
(`biggest` is the share of all land in the largest connected mass; Earth's
largest is Afro-Eurasia at ~20 %):

| config | biggest | cont. frac | land % | ocean p50 | land <1 km |
|---|---|---|---|---|---|
| base | 32.9 % | 0.351 | 23.3 | −3713 | 77.2 % |
| `cratons` 28 | **37.8 %** | 0.466 | 30.6 | −3397 | 79.6 % |
| `plate_size_jitter` 0.8 | 25.4 % | 0.541 | 35.9 | −3815 | 85.4 % |
| `rift_every` 600 | 22.9 % | 0.350 | 23.2 | −3445 | 65.9 % |
| `rift_every` 400 | 22.7 % | **0.235** | 16.9 | **−1834** | **14.6 %** |
| `rift_every` 400 + jitter 0.8 | 19.7 % | 0.417 | 27.0 | −3646 | 62.5 % |
| **Earth** | **20.3 %** | **0.40** | **29.0** | **−3700** | **71 %** |

Three things worth keeping:

* **`cratons` runs backwards.** 28 seeds gave a *more* concentrated
  supercontinent than 12, not a more fragmented one. Seeding more separate
  continents does not produce more continents, because without a way to
  split, crust only ever merges -- so more seeds simply merge sooner.
* **Rifting alone at 400 wrecks the planet** (continental fraction 0.235,
  land under 1 km 14.6 %): it consumes continental crust faster than
  `arc_birth` replaces it. Paired with `plate_size_jitter = 0.8` it does
  not, because larger plates carry less boundary per unit area and so fewer
  destructive collisions. Neither knob is usable alone at that rate.
* **The single-seed result did not hold.** `rift 400 + jitter 0.8` reading
  19.7 % against Earth's 20.3 % looked like a near-exact match. Across three
  seeds it is **26.0 ± 7.0 %**, and on one of them rifting achieved
  essentially nothing (35.8 % against a base of 37.5 %).

What *is* stable under rifting is everything the crust types fixed --
land/ocean gap 4000 ± 127 m, continental fraction 0.40 ± 0.05, ocean median
−3491 ± 112 m -- so rifting buys fragmentation on average without
disturbing the hypsometry. It costs a little low ground: 61.8 % of land
under 1 km against the base's ~78 %, where Earth is 71 % (comparable error,
opposite sign).

## The land texture is the segment cloud

Local relief on land, measured on the coarse grid:

| | 49 km window | 166 km window |
|---|---|---|
| 20,000 segments | 96 m | 363 m |
| 200,000 segments | **181 m** | 503 m |

The amplitude is plausible for continental interiors, but it **rises with
segment count**, which is the signature of the point cloud rather than of
resolved geology: the splat sigma is a fraction of the segment spacing, so
more segments means finer *and stronger* packing noise, never less. It is
Poisson-disc texture with no drainage structure and no orientation, sitting
exactly where erosion would otherwise organise relief. Not worth chasing
inside the tectonics stage.

Note also that straight coastlines are **not** an artifact: each cube face
is a gnomonic projection, on which great circles are exactly straight lines,
so a plate boundary near a great circle has to render straight. Real rifted
margins look like this too.
