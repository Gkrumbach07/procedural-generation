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
