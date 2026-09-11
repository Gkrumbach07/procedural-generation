# Auditing the crust: belt types, the supercontinent, and the ocean floor

Four things `docs/earth-bake.md` left open were open because nobody had
counted anything: which belts a world builds, whether the supercontinent
cycle happens, why the ocean is shallow, and where erosion particles die.
This is what the counting found. Every number is one Earth-scale run of the
shipped `earth` preset (1024² per face, 9773 m cells, 1500 steps, seed 0),
and the instruments are in `bake/scripts/` so they can be re-run.

## The belt census

`orogeny.shape_belt` now takes an optional `census` dict and records what
`classify` decided for every collision. `scripts/supercontinent.py` prints
it. The whole 1500-step run, 101,105 classified collisions:

| | count | share |
|---|---|---|
| `laramide` | 53,328 | **52.7 %** |
| `andean` | 32,308 | 32.0 |
| `himalayan` | 7,456 | 7.4 |
| `ural` | 0 | 0 |
| ocean under ocean (`pair_oo`) | 40,866 | **40.4 %** |
| ocean under continent (`pair_oc`) | 44,770 | 44.3 |
| continent into continent (`pair_cc`) | 7,456 | 7.4 |
| continent under ocean (`pair_co`) | **0** | 0 |

`pair_co = 0` is the one number that comes out exactly right by
construction: a continent never subducts under ocean floor at any density,
which is the irreversibility `docs/crust-types.md` built the whole crust-type
model around. It holds over 101,105 events.

Every other row in the table is a defect.

### `ural` is not a missing branch

It was on the open list as "`classify` never returns `ural`", and the
reading behind that — four types named, three returned, so one is unwired —
is wrong. **A Ural is not a kind of collision; it is the end of one.** The
Urals were a full continent-continent collision, and what makes them Urals
rather than a Himalaya is a quarter of a billion years of standing still
afterwards. A `classify` branch returning `ural` would build a belt that was
born dead.

The type *is* used, in the place where that age belongs:
`TYPES["ural"].crest_m` is `orogen_floor_m = 1200 m`, the height
`relax_orogens` decays every unfed belt down to. A former orogen is produced
by the decay, not classified into existence. `classify`'s docstring now says
so, so the next reader does not re-open this.

### `island_arc` is the branch that really was missing

**40 % of all collisions are ocean on ocean**, and every one of them was
building an `andean` or `laramide` cross-section: a 6000 m or 3500 m
plateau, 1050 to 1750 km wide, laid out over *ocean floor*. Both of those
profiles describe an arc standing on thick continental crust — the Andes are
ocean under continent — and there is no continent under an intra-oceanic
arc. The Marianas are a trench, a narrow ridge that mostly stays submerged,
and a back-arc basin behind it, with about 2.5 km of relief on a 4 km-deep
plain.

`TYPES["island_arc"]` is that cross-section (500 km wide, 2500 m crest,
−2000 m trench, a −500 m back-arc), and `classify` returns it whenever the
overriding plate is oceanic — checked *before* `flat_slab`, which was
otherwise routing buoyant slabs to a continental profile they had no
business building here either.

The census after the change, same preset and seed:

| | before | after |
|---|---|---|
| `laramide` | 53,328 | 26,178 |
| `andean` | 32,308 | 25,748 |
| `himalayan` | 7,456 | 7,762 |
| `island_arc` | — | **50,202** |

`andean + laramide` is now exactly `pair_oc` (51,926), which is the
invariant to check: every continental-margin belt has a continent under it
and nothing else does.

### `flat_slab_age` makes the exception the rule

`laramide` is 53,328 belts against `andean`'s 32,308: **62 % of every
ocean-consuming collision is classified flat-slab.** On Earth flat-slab
subduction is the exception — the Peruvian and Chilean segments, the
Farallon slab that built the Rockies — perhaps a tenth of convergent margin
length.

The knob is not obviously wrong; the crust it reads is. `flat_slab_age = 60`
steps against `ridge_age = 400` (the age at which sea floor has finished
subsiding, ~80 My on Earth) makes the threshold ~12 My, which is a
defensible buoyancy limit. But the age histogram of subducting slabs is:

| slab age at subduction (steps) | count | share |
|---|---|---|
| 0–15 | 35,030 | **40.9 %** |
| 15–30 | 8,940 | 10.4 |
| 30–60 | 9,358 | 10.9 |
| 60–120 | 11,340 | 13.2 |
| 120–240 | 11,450 | 13.4 |
| 240–400 | 5,298 | 6.2 |
| ≥ 400 | 4,220 | 4.9 |

(`island_arc` takes the ocean-on-ocean events out of this comparison but
does not change the ratio it is about: after the fix `laramide` is 26,178
against `andean`'s 25,748, still **50 % of every continental-margin arc**.)

**Two fifths of all subducted ocean floor is under 15 steps old** — crust
spawned into a gap and eaten again almost immediately. That is the "froth
along boundaries rather than a conveyor" that `docs/crust-types.md`
diagnosed from the other end (54 % of the final crust primordial), still
present after crust types fixed the population structure. The conveyor is
short, so the crust arriving at a trench is young, so the flat-slab branch
fires, so `laramide` dominates. Raising `flat_slab_age`'s threshold would
hide that; it would not fix it.

## The supercontinent cycle does not happen

`scripts/supercontinent.py` measures the order parameter the cycle is about:
the largest connected continental mass as a share of all continental *area*,
sampled 75 times through the run. It works on the segment cloud, not the
rendered grid, so sea level and the splat kernel cannot colour the answer,
and it reports three link radii so the answer is not an artifact of one
threshold. Earth today is ~0.37 (Afro-Eurasia of all continental crust).

Over 1500 steps the largest mass never falls below **0.79**, and:

* **0 of 1500 steps** have the crust apart (largest mass < 0.7).
* The single deepest sample, 0.791 at step 1260, is back to 0.935 twenty
  steps later. That is a collision healing, not an ocean opening.
* The link radius does not rescue it: at 1.1 spacings the largest mass runs
  0.89–0.93 over the second half of the run against 0.95–0.96 at 1.6.
* Landmasses holding ≥ 1 % of continental area: **1** for almost the
  whole run, never more than 2.

So the honest status is not "breakup verified, reassembly unverified".
**Breakup does not happen either.** What `docs/crust-types.md` measured as
"biggest 27.0 ± 6.1 %" is the largest *emergent* landmass as a share of
*land*, which is a different quantity: one continental mass, cut into 15
visible pieces by where sea level happens to fall. The map looks like
several continents; the crust is one.

`rift` does what it says — it splits a plate and gives the halves new poles,
and the earlier fix that stopped it re-drawing every plate's pole is real —
but splitting a *plate* is not splitting a *continent*, and nothing in the
run drives the halves far enough apart to open an ocean between them.

## The ocean is shallow because there is no abyss

Measured on the coarse bedrock of the tectonics stage (the shipped
classifier, world `w-dec008`), the whole sea floor:

| | ours | Earth |
|---|---|---|
| ocean median | −2859 m | −3700 |
| ocean p5 | −3914 | — |
| deepest cell anywhere | **−4020** | −11,000 |
| ocean shallower than −2500 m | **66 %** | ~13 |

**There is a hard floor at −4020 m**, and it is not a coincidence: oceanic
crust floats at `h_ocean = 0.024` bedrock units and sea level sits at
0.1762, so `(0.024 − 0.1762) × 26400 = −4018 m` is exactly where fully
subsided sea floor has to sit. The entire depth range of the ocean is the
2244 m of `ridge_height` thermal buoyancy above that floor. Earth's abyssal
plain runs to −6000 m and its trenches to −11,000.

So the model cannot make an abyssal plain deeper than −4 km whatever else is
fixed, and only 22.8 % of the sea floor by area has reached `ridge_age` and
finished subsiding at all (median age 169 steps of 400).

`scripts/ocean_depth.py` splits the sea floor by the crust under it, using
the new `diagnostics/crust_kind` field — `finalise` already computed that
mask to place sea level in shelf mode and simply never wrote it down.
Without it, "the ocean is too shallow" cannot be separated from "much of the
ocean is drowned continent", and the two want opposite fixes.

Split — on the `island_arc` world, the only one baked since the field
existed; its ocean median is −2716 m, so its gap to Earth is 984 m:

| | % of globe | median | p10 | deepest |
|---|---|---|---|---|
| drowned continent | 10.5 | −488 m | −1799 | −3511 |
| **oceanic crust** | **61.7** | **−2840 m** | −3651 | **−4126** |
| Earth's abyssal plain | ~60 | ~−4300 | | −6000 |

**The drowned shelf is not the explanation.** It is 15 % of the ocean by
area and lifts the ocean median by **+124 m** of that 984 m. 28 % of the
continental crust is under water, against `shelf_fraction = 0.275` — the
knob does what it says.

It is, though, the wrong *shape*. Drowned continent has a median depth of
**−488 m** and a p10 of −1799, and only **2.5 % of the globe** is drowned
continent shallower than −200 m against Earth's ~8 % of shelf. So the model
does not have a shelf in the sense that matters — a wide, nearly flat
platform a hundred metres under water — it has a ramp that goes from the
coast to 1800 m in the outer tenth. `margin_taper`/`margin_thinning` set
that ramp, and nothing has ever checked its *gradient* against a real
margin, only its existence.

(`docs/earth-bake.md` measured "the shelf is 8.33 % of the surface, squarely
inside Earth's 7–8 %". That counted every cell between −200 m and 0
regardless of the crust beneath it, and it was post-erosion, where sediment
has filled the inner ramp. The 2.5 % here is pre-erosion and continental
only. Both are right; they are different quantities, and the second is the
one a shelf is.)

That is a candidate mechanism for the offshore-sediment bias
`docs/earth-bake.md` left open — a margin with no flat platform on it gives
a particle nowhere to stop between the coast and deep water — and it is a
candidate, not a finding, until the death census says where particles
actually stop.

The sea floor itself is what sits high, and two things hold it there:

* **It cannot go deeper than about −4.0 km.** Fully subsided ocean floor is
  pinned at `(h_ocean − sea_level) × height_scale_m = (0.024 − 0.1762) ×
  26400 = −4018 m`, and the measured minimum over the whole planet is
  −4020 m. Earth's abyssal plain *starts* around there and runs to −6000.
  Whatever else is fixed, the model has no abyss.
* **Most of it never finishes subsiding.** Median sea-floor age is 169 steps
  against `ridge_age = 400`, and only **22.8 % of it by area** is fully
  subsided. The median sea floor is therefore still carrying ~35 % of its
  ridge buoyancy — several hundred metres of it — where Earth's median sea
  floor (~60 My of ~80) carries ~13 %.

Both come back to the same short conveyor that makes 41 % of subducted
slabs younger than 15 steps.

### Two explanations tested and eliminated

Neither of the mechanisms that looked most likely survives measurement, and
both were cheap to check.

**Arc plateaus on the sea floor.** Before `island_arc` existed, 40 % of
collisions were laying a 6000 m or 3500 m plateau over ocean crust, which
looks like an obvious way to fill an ocean basin. A/B of the two
classifiers, everything else identical:

| | ocean median | max | >5 km band | land % |
|---|---|---|---|---|
| `andean`/`laramide` on ocean crust | −2859 m | 6619 | 0.7 | 28 |
| **`island_arc`** | **−2710 m** | 7349 | 1.1 | 28 |

The ocean came out **shallower**, not deeper. The reason is in
`shape_belt`'s own design and is worth stating because it applies to every
future profile edit: *a profile's heights set the belt's shape, not its
height.* The accreted mass is fixed by the collision and is shared over the
positive part of the profile, so narrowing the footprint — 500 km for an
island arc against 1050 for an Andes — puts the same mass on fewer segments
and makes the belt **taller**. Editing `crest_m` to lower a belt does
nothing of the kind.

`island_arc` is kept anyway: an intra-oceanic arc that is 500 km wide with a
trench and a back-arc is right, and one that is a 1750 km Colorado Plateau
is not. But it is not the ocean-depth fix, and the search should move to the
sea-floor floor and the age distribution above.

**The drowned shelf**, above: +124 m of 984.

## Where particles die

`stats["deaths"]` counts *why* a particle stopped and attaches no position
to it, so "86 % of the sediment is in deep water" had nothing to check
against. `scripts/fork_erosion.py --deaths` adds the position: it reads the
raw change list through a new `diag` hook on `maps.run_iteration`, so the
census is of the particles the run actually used rather than a re-traced
copy of them. A particle's seafloor steps carry `cl_vol == 0` and its final
deposits `cl_vol < 0`, so the death cell is the first entry with
`cl_vol < 0` and the seafloor walk is the count of zeros before it.

From the baseline run's own log, before any of that: at iteration 67 of
800, **94.7 % of all 1.57 M particles die in the ocean** and 5.3 % in a pit
— and those two account for every particle to within one, so `age`, `exit`,
`evap` and `stop` are all zero. The share rises as the landscape smooths
(84.7 % ocean at iteration 8). So essentially the entire
sediment budget is delivered by particles that reach the sea and then walk
the sea floor for up to `ocean_steps = 64` steps — 625 km at 9773 m cells —
before their last deposit. Whether that walk ends on the shelf or in the
abyss is the whole of the offshore-sediment question, and it is what the
census measures.

## Hotspots are written and switched off; LIPs do not exist

Filed as "LIPs and hotspot tracks never started", which is half right.

`intraplate.apply_hotspots` exists, is called from `TectonicSim.step`, and
does the thing that makes a track rather than a blob: the spots are fixed in
the mantle frame while the plates move over them, so a plate crossing one
comes out with a line of thickened crust behind it. It ships **disabled** —
`tectonics.hotspots = 0` and `hotspot_rate = 0.0` — and no run has ever
measured what it produces. Turning it on is a parameter change, not a
feature.

Two things to know before turning it on. It adds thickness to whatever crust
drifts over a spot with no crust-type gate, which is right for Hawaii on
ocean floor and right for Yellowstone on continent, but `relax_orogens` only
decays *continental* height, so an oceanic swell it raises will never come
down. And the addition accumulates every step, so a slow plate builds a
plateau where a fast one builds a chain — which is the real behaviour, but
it means `hotspot_rate` cannot be read off a target height.

Large igneous provinces are genuinely absent: nothing in the model produces
the short, enormous burst at plume initiation that makes a Deccan or a
Siberian Traps, and a LIP is not a slow track with the rate turned up.
