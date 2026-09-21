# Slab pull, and what welds a continent back together

docs/earth-v3-review.md ended with a build order for the supercontinent
cycle: slab pull first. This is that work, measured the way the review
measured the problem: `scripts/supercontinent.py --preset earth --steps
3000`, the same simulation the bake runs, sampled every 50 steps. The order
parameter is the largest connected continental mass as a share of all
continental area (1.0 = one supercontinent), reported at link radii of
240, 256, 319 and 479 km so that a split cannot be an artifact of one
threshold; alongside it the continental share of the segment cloud, which
is the crust that has not been deleted.

The baseline, no new forces (from the review):

| step | largest mass @ 240 km | @ 479 km | continental share | plates |
|---|---|---|---|---|
| 1500 | 0.91 | 0.93 | 0.505 | 10 |
| 2450 | **0.54** | 0.94 | 0.371 | 8 |
| 3000 | 0.56 | 0.60 | 0.364 | 10 |

One breakup, at the sixth rift, and a continent that has lost half its
area to continent-on-continent collisions by then.

## 1. Slab pull

`tectonics.slab_pull` (`plates.slab_pull_torques`). Every *oceanic*
segment that goes down at a trench pulls its own plate towards that
trench: a force `slab_pull · w(age) · area` along the tangent from the
plate's centre of mass to the segment, `w = clip(age / ridge_age, 0, 1)`.
The torque this puts on the plate induces, at the trench, a velocity along
the force, so the plate's edge moves into the trench. The age weight is
there because 27 % of subducting slabs at the defaults are younger than 15
steps, crust spawned at a ridge and eaten a moment later, and those should
not steer a plate. A continental loser is crustal shortening, not a slab,
and pulls nothing.

The gain is in the units of the heat force on one segment under a unit
gradient. The heat torque is summed over every segment of a plate; the
slab torque over the few dozen that subduct per step; so the gain has to
be in the hundreds before the two are comparable. Measured over the first
150 steps, `slab_pull = 1000` gives a slab-to-heat torque ratio of ~4,
`300` of ~1.

| `slab_pull` | first split (largest mass < 0.7) | minimum | re-assembles? | continental share @ 3000 | plates @ 3000 |
|---|---|---|---|---|---|
| 0 | step 2450 | 0.54 | no (0.56 at 3000) | 0.364 | 10 |
| 100 | 2600 | 0.46 | no | 0.371 | 9 |
| 300 | 1750 (partial, 0.71) | 0.69 | yes, 0.84 by 2650 | 0.372 | 8 |
| **1000** | **1450** | 0.47 | **yes, 0.94 by 1750; splits again 2050 and 2650 (0.34); 0.66 at 2950** | **0.246** | 12 |

At 1000 the cycle is there: split, reassembly within 300 steps, split
again, twice. The link radius says the oceans it opens are real ones: at
step 1450 the largest mass is 0.47 at every radius up to 479 km, where the
baseline's 200 km rifts were invisible above 256 km. But the continent
comes out of it at a quarter of the segments. A faster cycle is more
continent-on-continent collisions, and each of those deletes a segment.

## 2. Trench suction and ridge push: both rejected

Slab pull moves ocean plates. A continent has no slab, so a rifted half
sits still while the sea floor beyond it subducts under its far margin;
on Earth what carries it out over the closing ocean is ridge push behind
it and trench suction ahead of it. Both were added
(`tectonics.trench_suction`, the same pull on the overriding plate;
`tectonics.ridge_push`, a push off the ridge on the plate every spawned
segment joins) and both are off, because in this model they shatter the
plates:

| run | plates @ 550 | plates @ 3000 | continental share @ 3000 | largest mass @ 3000 |
|---|---|---|---|---|
| slab 300, suction 0.5, ridge 30 | 41 | 43 | 0.322 | 0.45 |
| slab 1000, suction 0.5, ridge 100 | 51 | **144** | 0.146 | 0.12 |
| slab 300, suction 1.0, ridge 30 | 41 | 61 | 0.391 | 0.61 |

Suction alone, at 0.5 with slab 1000, holds 30-34 plates through the
first 1500 steps against 8-13 without it. What it does is pull the
overriding plate's edge into the trench where its neighbour is already
going down, so both plates converge on the same line, the collision rate
there multiplies, and `split_disconnected` then finds the pieces the
trench has cut off. The forces are physical; the plate-splitting rule
turns them into fragmentation. They stay in the code at 0 for when that
rule is revisited.

## 3. Suturing: a collision between continents ends by welding them

`tectonics.suture_collisions` (`intraplate.suture`). Continent-on-continent
collisions are counted per plate pair with a memory of `suture_window =
300` steps; when a pair's count passes the threshold the smaller plate
joins the larger and the merged plate turns about the inertia-weighted
mean of the two poles. The crust is untouched. This is the missing half
of the cycle in two senses: it is what stops two continents from eating
each other indefinitely (India still converges on Asia, but the crust
that is gone is the Tethyan floor between them, not India), and it is what
makes the fragments *one plate again*, which is what "reassembly" means.

Only continents weld. Island arcs are continental crust (`arc_birth`), so
the first version let every ocean plate that landed an arc on a continent
weld to it, and three runs ended with one, one and two plates.
`suture_continental = 0.5` gates the weld on both plates being at least
half continental by area.

Without the gate, at `slab_pull = 1000`, threshold 200:

| step | largest mass @ 240 km | continental share | plates | sutures so far |
|---|---|---|---|---|
| 1750 | 0.955 | 0.653 | 5 | |
| 2050 | 0.767 | 0.640 | 6 | |
| 2350 | **0.474** | 0.637 | 3 | |
| 2650 | **0.882** | 0.621 | 3 | |
| 2950 | 0.588 | 0.579 | 2 | 7 |

The continental share holds at 0.58-0.62 against 0.25 without suturing,
and the crust splits, re-forms and splits again. The plate count is the
leak the gate closes.

With the gate, the same threshold fires **once** in 3000 steps and the
continent is chewed to 0.15 of its area again. The count was never the
continents: the per-pair contact rate, logged every 50 steps of a
`slab_pull = 1000` run, is 1-4 continent-on-continent contacts per step
with peaks of 10-12 at the collision moments, and plate ids change every
time `split_disconnected` fires, so a count keyed on a pair rarely
accumulates. The thresholds have to be sized to that (see the end).

Sized to it, with the gate and a `suture_cooldown` of 400 steps after a
rift (the zig-zag teeth grind as a rift opens, and those contacts are
continent-on-continent, so without the cooldown every rift re-welded
within 50 steps), the weld does what it says and the cycle dies of it:

| `slab_pull`, threshold / window | sutures | largest mass, min over the run | continental share @ 3000 | `pair_cc` |
|---|---|---|---|---|
| 1000, 20 / 50 | 7 | 0.88 | 0.644 | 3,716 |
| 1000, 40 / 100 | 6 | 0.79 | 0.594 | 4,053 |
| 600, 20 / 50 | 4 | 0.69 | 0.566 | 5,694 |

Continent conserved, collisions down to a third of the baseline, and
**no breakup**: the largest mass never leaves 0.88-0.95 until the last
few hundred steps. In the slab-pull-only run the split that reached 0.47
at step 1450 was the third rift's halves opening while the first two
rifts' halves were still apart; with welding on, the halves of the 400
and 800 rifts are one plate again by 1150, and the 1200 rift has a
single rigid continent to work on. Welding is right where a collision is
a collision, and wrong where it is a rifted pair that the fixed heat
field has dragged back together, and this model cannot tell the two
apart because they are the same event. That is the argument for fixing
the field before fixing the weld (section 5).

## 4. Segment-level shortening: rejected, and why

The other way to conserve area is to not delete the losing segment:
`tectonics.continental_shortening` keeps a continental loser alive,
moves that fraction of its mass into the belt, and hands it to the
winner's plate (crustal shortening as accretion, the front of one plate
welding onto the other segment by segment). It conserves the continent
(0.59-0.65 of the segments at 3000, every run) and at `slab_pull = 1000`
it produced a textbook cycle: 0.69 at 850, 0.91 at 1150, 0.53 at 1300,
0.97 at 1700, 0.72 at 2650. It also produced **14 million**
continent-on-continent collisions against 9 thousand in the baseline. A
surviving loser is still inside the collision radius of its winner, so
the same pair is a collision again every step until nothing is left of
it. Retreating the loser to the edge of the radius cut that to 1.3-6
million and blew the plate count up to 96-257: the retreated segment is
now labelled with the winner's plate and sits inside the loser's, so
`split_disconnected` sees an orphan every step and either welds it back
(and it collides again) or, when enough have gathered, makes a plate of
them. The knob stays in the code at 0 with this note. The physics it
wants is a plate boundary that *moves* with the suture, which this
model's rigid-plate-plus-splitting machinery cannot express one segment
at a time.

## 4b. The weld, and why several cycles still do not run (2026-09-18)

Asked for several supercontinent cycles, measured on `small` (N_c 128, 1500
segments, seed 0, 4000 steps), continental share of the segments:

| variant | cont @4000 | cratons | plates | collisions/step | M |
|---|---|---|---|---|---|
| shipped (`continental_shortening = 0`) | **0.178** | 56 | 15 | 10 | 1127 |
| shortening 0.25, before the weld | 0.984 | 1421 | **1** | 224 | 3701 |
| shortening 0.25, weld 60 | 0.895 | 534 | 51 | 748 | 2224 |
| shortening 0.25, weld permanent | 0.984 | 1421 | 1 | 183 | 3701 |
| shortening 0.25, weld 60, `arc_birth = 0` | 0.833 | 556 | 30 | 202 | 1566 |

The shipped default is the failure the user saw: a continent-on-continent
collision deletes the loser, so continental crust halves about every 1500
steps and settles near 0.17 -- `worlds/long-plates` (6000 steps) ended with
18 % continental crust and 13 % land, against earth-v11's 39 % at 3000.
Cratons go the same way (405 -> 56): two cratons in contact make one of them
the loser.

`weld_steps` (new) fixes what section 4 rejected shortening for.  A welded
loser lies inside the plate it came from, so `split_disconnected` welded it
back every step and the pair collided again; keeping it on the winner's plate
for 60 steps holds the plate count at 51 instead of 1 and keeps the craton
count from running away.  It is inert with shortening off: nothing is welded,
and no world already baked changes.

It is not enough.  Even with the plates healthy and island arcs off, the
continental share still climbs 0.60 -> 0.83, because **a segment is a fixed
unit of area**.  A continent-on-continent collision can delete the loser (the
area of the pair halves) or keep it (the area of the pair is unchanged while
its crust thickens, so area is created from nothing) -- and nothing in
between, since the area a segment covers is the segment itself, not its
`area` field, which the splat does not read.  Real shortening reduces area as
it thickens crust.  Making that possible means the blend weighting a segment
by an area it can lose, and only then is a cycle that assembles, rifts and
re-assembles worth looking for (the supercontinent index |mean position of the
continental segments| stayed 0.1-0.5 in every run above, with no periodicity).

Until then: 3000-4000 steps is the honest range, where continental crust is
still 0.39-0.56 of the coarse cells.

## 4c. Extent: the cycle, measured (2026-09-19)

`tectonics.variable_extent` gives every segment the ground it covers
(`Segments.ext`, steradians) and makes the splat weight it by that, so a
collision can spend *area* -- the overlap of the two discs, which is the
boundary length times the convergence -- instead of choosing between deleting
the loser and keeping it whole.  Extension is the same process backwards: a
void inside a continent is taken by the crust around it, which thins by as
much, rather than being filled with new crust.  `small`, 4000 steps:

| | continental share | plates | largest landmass |
|---|---|---|---|
| shipped (a contact deletes the loser) | 0.178, still falling | 15 | 1.00 -> 0.17, no recovery |
| `continental_shortening = 0.25` | 0.98 | 1 | never breaks up (min 0.94) |
| `variable_extent` | 0.41-0.48, steady | 17 | 0.48 / 0.28 / 0.43 / 0.20 / 0.36 |

It also unlocks `slab_pull`, which section 3 rejected at gain 1000 for
"shredding the continent to 0.25" -- that was the deletion, not the force.
With the ground conserved, `variable_extent` + `slab_pull = 1000` over **9000
steps** holds the continental extent at 0.36-0.48 the whole way and cycles:

```
biggest 1.00 0.96 0.54 0.53 0.35 0.24 0.15 0.42 0.38 0.60 0.37 0.36 0.39 0.42
        0.45 0.36 0.24 0.37 0.60 0.28 0.29 0.45 0.40 0.29 0.18 0.19 0.43 0.23
        0.25 0.69 0.36                                   (every 300 steps)
```

Breakups (largest mass < 0.30) at 1800, 4800, 5700, 7200, 8100; assemblies
(> 0.45) at 2700 (0.60), 4200, 5400 (0.60), 8700 (**0.69**).  Four assemblies
and five breakups in one run, on a period of 1500-3000 steps.  The script's
own verdict still reads "breaks up and stays apart" because its threshold for
*together* is 0.7 and these assemblies reach 0.60-0.69; on `small` (1500
segments) a gathered crust is a looser thing than on Earth, and the number to
watch is the trajectory, not the verdict.

### The crust-age map the run makes for itself

`scripts/crust_age_map.py` runs the plates (no bake) and maps the crust's age
with a ramp per kind, `--prehistory 0` turning off the invented continental
past so the map shows only what the run produced.  `small`, 9000 steps:

| | continental crust | continental age IQR | ocean age p50 | assemblies |
|---|---|---|---|---|
| shipped | 5.0 % | 3951 (0.44 runs) | 110 | none |
| `variable_extent` | 21.9 % | 3470 (0.39) | 117 | 2 |
| + `slab_pull = 300` | **26.7 %** | 1544 (0.17) | 59 | (peak 0.57 at 4000) |
| + `slab_pull = 1000` | 21.2 % | 324 (0.04) | 16 | 4 (peak 0.69) |

With extent on, the map is the one the reference maps look like and nobody
drew it: the oceans in bands from the ridge out, the continents with dark old
cores and pale belts welded round them.  The shipped model has the age
structure too -- and 5 % of a planet's crust left to show it on.  The
trade-off along the slab_pull axis is cycles against age: the harder the
planet is driven, the more it assembles and the younger everything gets.

So `inherited_age` now fades out by how much structure the run made for
itself (`PREHISTORY_FADE`, the interquartile spread of the continental ages as
a share of the run): a short run that starts assembled gets the whole invented
past, a cycling one gets none of it, measured 1064 steps on 65 % of segments
against zero.

Known cost, not yet addressed: `slab_pull = 1000` recycles the sea floor hard
-- median floor age 21 steps against 81 without it, and 0.1 % of it fully
subsided against 3.7 % -- so the ocean comes out uniformly young and the
crust-age map loses its stripes.  Gain 300 is milder (p50 58).  Next: an Earth
bake with the knob on, which needs every number tuned against equal-area
segments re-measured (convection, force_scale, heat_insulation, orogen_decay,
continental_fraction, shelf_fraction).

## 4d. The Earth bake with extent on: what it costs, and what is left (2026-09-20)

`worlds/earth-v14` is `earth-v13`'s parameters plus `variable_extent`, one
variable changed, 64 minutes.  Two bugs the `small` preset structurally could
not show turned up first (section 4b's table was measured before them):

* the ground a contact consumes was the overlap of the two discs, a
  penetration depth set by the collision radius and so by the segment size --
  at 20000 segments a quarter of the area per unit of boundary that it is at
  1500.  It is the margin's width times the convergence now, which is the
  physical rate and scale-free: `small` 0.673 against `earth` 0.699 at 400
  steps, where it had been 0.71 against 0.76 and diverging;
* continental stretch drew on what *every* collision destroyed, so ground a
  trench swallowed went to the continents instead of coming back at a ridge:
  +6.5 steradians of a 12.6 planet over 400 Earth-scale steps.  Only what
  crustal shortening itself consumed is available to stretch now (+0.08);
* and the closure was thinning the crust by its own renormalisation factor,
  which is one-signed and compounded: the first bake came out with its ocean
  at -1461 m and refine finishing in a sixth of its usual time because there
  was no relief left to refine.

With all three fixed, `earth-v14` against `earth-v13`:

| | v13 | v14 | Earth |
|---|---|---|---|
| land | 40.5 % | **30.6 %** | 29 % |
| continental crust | 56.0 % | 41.5 % | ~40 % |
| plates | 11 | 22 | ~15 major |
| land median | 311 m | **2133 m** | ~800 m |
| land above 2 km | 6.7 % | **54.7 %** | 13.3 % |
| ocean median | -4065 m | **-2367 m** | -3700 m |
| bedrock range | -5705..8704 | -3876..11498 | -10900..8848 |
| continental crust age | invented (14817-32859) | **1772-32707, partly its own** | - |

So the area behaviour is right and the hypsometry is not: v14 is a planet of
plateaus over a shallow sea.  `continental_fraction` is not the lever -- swept
at Earth scale with extent on, raising it *lowers* the final continental crust
and shallows the ocean further (0.75 -> 41.5 % and -2367 m, 0.85 -> 34.0 % and
-1637, 0.92 -> 26.0 % and -1539), which is the opposite of its meaning with
fixed area.  The crust columns themselves are not the problem: at 1200 Earth
steps the two agree (continental thickness p50 0.936 both, ocean 0.200 both,
continental height p50 +0.178 both), so what moved is where sea level lands on
them and how sharp the crust-type boundary is -- the power-cell blend resolves
a margin more crisply than the distance-only one, and the shelf ramp that gave
v13 its low land goes with it.

It was not a sweep; it was three more bugs, each found by ledgering the crust
by phase (`thin_*`, the count of continental segments each phase leaves
thinner than half a column) and looking at the thin segments themselves:

1. **Island arcs were relabelled ocean floor.** An ocean-ocean collision made
   the survivor continental and left it its 0.2 column at oceanic density.
   With fixed area most were deleted in the next collision; with extent they
   survive, and at Earth scale 5.9 % of continental segments were thinner than
   ocean floor -- crust sitting at abyssal height that filled the 27.5 %
   submerged-shelf quantile on its own and put sea level 1.8 km too low.
   `arc_thickness` (0.55 of a column, at the belt's density, the difference
   from the mantle and ledgered as `arc_mantle`) births an arc as arc crust.
   On `small` that took the thin count from 28 to zero.
2. **The stretch budget never reached the kernel** (`spent` allocated and read
   but not passed), so continental extension had been off since it was
   budgeted.  Wired, with the thinning floor (`extent_thin_floor`: crust at
   half a column stops stretching and the gap becomes a rift).
3. **The belt code spread what it assumed, not what moved.**
   `spread_collisions` and `orogeny.shape_belt` lay out the crust a collision
   handed the survivor, and both computed it by the fixed-area rule (the whole
   of a dead loser's column), then clamped the removal to leave the survivor
   1e-3.  With extent the survivor was handed a sliver, so the belt code
   stripped every winner it touched: 1189 of 8062 continental segments thinner
   than half a column at Earth scale, old, at full extent, 44 % cratons.  The
   kernel now reports per pair what it handed over and both routines use it.
   Same run: 183 thin, and collisions remove thin segments (-255) rather than
   make them.

Earth scale, 4000 steps of plates, after all three:

| | fixed area | extent as baked in v14 | extent now | Earth |
|---|---|---|---|---|
| land | 40.5 % | 30.6 % | 27.1 % | 29 % |
| land median | 311 m | 2133 m | 1221 m | ~800 m |
| land above 2 km | 6.7 % | 54.7 % | 22.6 % | 13.3 % |
| ocean median | -4065 m | -2367 m | **-3728 m** | -3700 m |
| bedrock range | -5705..8704 | -3876..11498 | -5332..9589 | -10900..8848 |

And the cycle on `small` (9000 steps, `slab_pull = 300`, all fixes): extent
share 0.37-0.45 throughout, three assemblies (0.60, 0.49, 0.49) and five
breakups, ocean age median 72 steps at the end.  `earth-v14` is re-baking with
all of it; what it shows decides whether `variable_extent` becomes the default.

## 4e. What the decay knob cannot do, and the age a crust-age map plots (2026-09-20)

Three questions after `earth-v15` (band error 12.7 against Earth, but light up
high: 6.8 % of the land above 2 km against 13.3 %, peaks to 5601 m against
8849).

### `orogen_decay` between 0.004 and 0.006: refuted

Four Earth-scale runs, 4000 steps, plates only, v15's parameters otherwise
(`/tmp/hyps.py`-style probe, bedrock, area-weighted):

| `orogen_decay` | land | bedrock median | above 2 km | ocean median | max |
|---|---|---|---|---|---|
| 0.0045 | 26.0 % | 899 m | 14.2 % | -4411 m | 9447 m |
| 0.005 | 27.8 % | 899 m | 13.9 % | -4243 m | 10229 m |
| 0.0055 | 26.5 % | 849 m | 14.0 % | -4005 m | 10093 m |
| **0.006** (v15) | 27.1 % | 932 m | 12.6 % | -3965 m | **7217 m** |

Lowering it buys 1.3-1.6 points of bedrock above 2 km -- inside the spread
these runs show between each other, since the trajectory is chaotic and the
ocean median wanders 450 m across the four -- and costs 2 to 3 km of peak,
past Earth's 8849 m in all three.  So the missing high ground is not the
decay's to give: our belts are too *narrow*, not too short, and what Earth has
above 2 km is mostly plateau.  `orogen_decay` stays at 0.006; the next attempt
at the 2-4 km band should widen belts, not raise them.

### A longer run does get rid of the invented past -- and costs the planet

The invented past (`run.inherited_age`) fades out on its own once a run makes
a quarter of a run's worth of continental age spread.  It does -- and the
planet goes with it (Earth scale, plates only, v15's parameters):

| steps | land | continental crust | above 2 km | ocean median | max | age IQR |
|---|---|---|---|---|---|---|
| 4000 | 27.1 % | 40.5 % | 12.6 % | -3965 m | 7217 m | 0.00 runs |
| 8000 | 17.3 % | 25.0 % | 31.1 % | -3133 m | 6976 m | 0.62 runs |
| 12000 | 15.7 % | 22.1 % | 31.9 % | -2400 m | 11744 m | 0.90 runs |

The continents drain, the crust they had piles onto what is left, and the sea
floor shallows by 1.5 km as the area it has to cover grows.

`small` does not do this: over 8000 steps its continental extent settles at
0.38-0.45 and its crust volume holds (9.3-11.4), oscillating, with no trend.
So this is another Earth-scale divergence of the kind section 4d found three
of, not the model running down.  Worth noting from the same ledger: what gives
continental area back is mostly not extension.  Over 8000 steps on `small`,
collisions spend 11.5 steradians of continental extent and rift spawning
returns 1.27 -- the balance comes from the closure that holds the total at
4 pi, which returns area in proportion to extent, and so favours whichever
kind covers more ground.  Earth scale is where that asymmetry should bite
hardest, and it is the first place to look.

**So: 4000 steps stands at Earth scale.**

### `Segments.rework`: the age of the last assembly

Which leaves the real problem, and it is not the run's length.  The land was
one flat colour because `age` is when the *rock* formed, and a continent's
rock all forms at step 0: at 4000 steps on Earth, three quarters of the
continental area read exactly 4000 and the interquartile spread was **zero**.

A crust-age map of the land does not plot that.  It plots when the crust was
last *assembled* -- a craton's own number, an orogen's last orogeny.  So
`Segments.rework` carries that: the same number as `age` until a collision
stacks foreign crust into the column, and then the two mixed by mass
(`rework *= 1 - received/total`), zero for an island arc, which is crust being
made now.  Nothing but `crust_age` reads it -- no force, height or plate moves
because it exists, and the ocean floor is still mapped at `age`, which for the
sea floor is the same thing.

`small`, 4000 steps, continental area by assembly age:

| | protolith `age` | assembly `rework` |
|---|---|---|
| younger than a quarter of the run | 9.8 % | 16.1 % |
| a quarter to three quarters | 14.7 % | 52.6 % |
| three quarters to all of it | 0.0 % | 24.9 % |
| never assembled since step 0 | **75.5 %** | **6.4 %** |

and the spread the fade measures goes from nothing to 0.32 of a run, so the
invented past switches itself off -- at 4000 steps, where the planet is right,
instead of at 8000, where it is not.  The map shows dark cores with paler
belts round them and whole young blocks where a margin has been active for the
length of the run.

### What the ledger was hiding

Turning `variable_extent` on by default made the mass ledger stop closing,
and neither reason was the default flip's fault -- both were bugs the old
defaults could not show.

The **invariant was wrong**.  Shortening moves crust between columns of
different extent, so the plain sum of the column masses moves when nothing
has been created or destroyed.  What that mode conserves is the sum of
`ext * mass` -- `Segments.crust_mass`, as `crust_volume` is for thickness --
and the ledger is now kept in whichever of the two its mode conserves.

And **`arc_mantle` was counted twice**: the collision phase's own before-and-
after difference already contains what the arcs drew from the mantle.  It
could only show once `arc_thickness` was non-zero, and it was zero.

The closure now has a ledger term of its own, `extent_closed`: it rescales
`ext` globally and carries crust with it.  That figure is the one to watch
for the drain above -- it says how much of the planet's crust the change of
units is moving, where `extent_close` only says how far the factor is from
one.  The ledger closes to 1e-16 in both modes.

### What ships

`earth-v15`'s configuration is the preset now -- `variable_extent` on,
`arc_thickness` 0.55, `shelf_fraction` 0.33, `orogen_decay` 0.006,
`initial_plates` 4, `steps` 4000, `rift_every` 600, `continental_fraction`
0.75 -- so `--preset earth` reproduces it exactly and every new planet starts
there.  `variable_extent: false` in a params.yaml is still bit-identical to
the model every world before `earth-v14` was baked with, so those reproduce
too.  `earth-v16` is the bake: the same planet as v15, with the crust-age map
this section is about, the 1.2 km planet level and the detail tiles.


## 5. The field that moves: continental insulation

`tectonics.heat_insulation`. The heat background that `heat_relax` pulls
the live field towards is no longer the step-0 noise for the whole run:
every step it drifts *down* under continental crust and *up* under ocean
floor, by `heat_insulation` per step (the crust is rasterised from the
label map onto the heat grid). Plates are pushed towards hot in this
model, so a supercontinent cools the attractor under itself and its
pieces are pushed off it towards the ocean; the ocean floor they leave
behind warms and draws the next plates in. That is a supercontinent
insulating the mantle beneath it, with the model's sign convention.

| run | first split (< 0.7) | trajectory | continental share @ 3000 | `pair_cc` |
|---|---|---|---|---|
| insulation 0.002, no slab pull | **1150** | 0.77, 0.61 @1450, **0.96 @1750**, 0.58 @2050, 0.76 @2350, 0.38 @2650, 0.31 @2950 | 0.434 | 7,615 |
| insulation 0.0005, slab 1000 | 1750 | 0.47, 0.63, 0.62, 0.41, 0.50 | 0.205 | 16,220 |
| insulation 0.002, slab 1000 | 850 | 0.72, 0.95 @1150, 0.66, 0.71, 0.40, 0.56, 0.28, 0.19 | 0.155 | 16,779 |

Insulation alone does what slab pull was reached for: the continent
splits at the third rift, reassembles to 0.96 by 1750, and splits again,
with *fewer* continent-on-continent collisions than the baseline (7,615
against 8,930) because the halves are pushed apart rather than dragged
back. Add slab pull at 1000 and the collisions double and the continent
is shredded to a fifth. The dispersal is the field's job; slab pull at
the gain that moves continents is too violent for a crust that cannot
lock.

### Insulation with the weld

The weld was argued for above on the grounds that reassembly under a
moving field would be a real collision. Measured, it still takes the
breakup with it:

| insulation, `slab_pull`, threshold / window | sutures | largest mass, min | continental share @ 3000 | `pair_cc` |
|---|---|---|---|---|
| 0.002, 0, 20 / 50 | 6 | 0.85 | 0.588 | 4,245 |
| 0.002, 300, 20 / 50 | 6 | 0.77 | 0.619 | 5,035 |
| 0.001, 0, 20 / 50 | 7 | 0.40 (2350, 2950) | 0.572 | 4,627 |
| 0.002, 0, 40 / 100 | 5 | 0.48 (2950) | 0.608 | 3,778 |

The weld keeps 0.57-0.62 of the continent and halves the collisions, and
the crust spends the run at 0.85-0.96 with a dip or two. Insulation
without it spends the run splitting and re-forming at 0.43. Those are the
two ends of one trade: a continent that locks when it collides is a
continent that stays locked while the rifts that would free it fire into
a single rigid block. `suture_collisions` ships at 0 with this table; the
weld is right, the rift needs to win against it, and that is a rift
problem (a rift that opens *into* a suture, not across a random plane)
for another session.

### Rate, seed, and what ships

| insulation (no slab pull, no weld) | first split (< 0.7) | largest mass @ 1500 | minimum | continental share @ 3000 | `pair_cc` |
|---|---|---|---|---|---|
| 0.002 | 1150 | 0.96 (re-formed 1750) | 0.31 | 0.434 | 7,615 |
| **0.003** | **1450** | **0.62** | 0.35 | 0.490 | 6,849 |
| 0.004 | 1950 | 0.96 | 0.22 | 0.490 | 7,128 |
| 0.002, seed 2 | 1950 | 0.97 | 0.30 | 0.491 | 6,640 |

Every rate and seed gives the cycle; when in the run it turns is set by
which rift the field has undermined by the time it fires, which is why
the first split lands anywhere from 1150 to 1950. `heat_insulation`
ships at **0.003**: at seed 0 that is the one whose first split
(0.62 at 1450, 0.47 at 1700) is inside the 1500-step bake, so the bake
ends with the interior ocean open rather than with the fragments back
together, which is what "we still have a supercontinent at the end" was
about. Tectonics-only bakes at the `earth` preset show it directly in
the viewer: a rift ocean opening diagonally across the continent from
about step 500, a basin hundreds to a thousand kilometres wide by 1000,
and two landmasses with a second arm of ocean between them at 1500 --
the middle panel of the coalescence diagram, where `earth-v3` had a
single continent with a scar across it.

Shipped: `heat_insulation = 0.003`. Off, with their measurements above:
`slab_pull`, `trench_suction`, `ridge_push`, `continental_shortening`,
`suture_collisions`. Continental share at 3000 steps is 0.49 against the
baseline's 0.36; the deletion in continent-on-continent collisions is
still there and is the thing to fix next, by a rift that opens along a
suture so that a locked pair can be unlocked.

## 6. `worlds/earth-v4`: the bake, and what the breakup costs

The full `earth` bake with `heat_insulation = 0.003` and nothing else
changed (60 minutes: erosion 3185 s, refine 210 s). The timeline shows
the rift ocean opening across the continent by step 500 and the final
frame is several continents with real oceans between them and island
arcs in the old superocean, against `earth-v3`'s one continent with a
scar. Tectonics census against `earth-v3`:

| | earth-v3 | earth-v4 |
|---|---|---|
| plates alive at 1500 | 10 | 17 |
| collisions | 55,203 | 62,754 |
| ocean under ocean / under continent / continent on continent | 53 / 38 / 9 % | 48 / 48 / **4 %** |
| tectonic maximum | 8783 m | **4884 m** |
| lakes / rivers / max Strahler order | 2219 / 7976 / 6 | 1534 / 6133 / 4 |

And the hypsometry after erosion (`scripts/hypsometry.py`):

| band | Earth | earth-v3 | earth-v4 |
|---|---|---|---|
| 0-1 km | 71.6 | 76.7 | **91.0** |
| 1-2 | 15.4 | 8.7 | 6.0 |
| 2-3 | 7.5 | 7.4 | **1.8** |
| 3-4 | 3.8 | 4.2 | **0.8** |
| 4-5 | 1.7 | 1.9 | 0.3 |
| >5 | 0.3 | 1.1 | 0.0 |
| max | 8849 m | 8783 | **4884** |
| ocean median | -3700 m | -3100 | -2379 |

**The mountains went with the grinding.** `earth-v3`'s high ground
was built by continent-on-continent collisions, and most of those were
the rifted halves being dragged back onto each other by the fixed heat
field, the very thing this work removed. With the halves pushed apart
instead, continent-on-continent falls from 9 % to 4 % of collisions, no
Himalayan belt is built in 1500 steps, and the highest point on the
planet is an Andean crest at 4884 m. Land above 2 km is 2.9 % against
Earth's 13.3 %. The ocean is shallower for the right reason (the interior
oceans are young floor that has not subsided) and the rivers are shorter
because the continents are.

So the cycle and the mountains are, at these settings, in tension: the
first needs the continents to disperse, the second needs them to
collide. Earth has both because its collisions are a *later* phase of
the same cycle (India left Gondwana and hit Asia 100 My later). Two
ways to get both, to be measured next: run past the reassembly
(`steps` 2500-3000, where the 3000-step trajectories re-form the
continent at 0.9 and split it again), or raise the Andean belt so
ocean-under-continent margins carry the high ground, as the Andes do at
6960 m. Either way `orogen_decay` and the belt profiles were tuned on a
world whose collisions were 9 % continental and will want re-measuring.

## 7. Steps, decay, and `earth-v5`

"Just do more steps until it gets interesting." Tectonics-only Earth
bakes at 2500, 3000 and 4000 steps with `heat_insulation = 0.003`
(bedrock hypsometry, before erosion):

| | Earth | 1500 (`earth-v4`) | 2500 | 3000 | 4000 | 2500, decay 0.004 | **3000, decay 0.004** |
|---|---|---|---|---|---|---|---|
| 1-2 km | 15.4 | 6.0 | 19.1 | 25.7 | 29.0 | 21.8 | 28.1 |
| 2-3 | 7.5 | 1.8 | 3.3 | 1.5 | 2.4 | 5.2 | **10.2** |
| 3-4 | 3.8 | 0.8 | 0.9 | 0.8 | 0.9 | 2.0 | **3.8** |
| 4-5 | 1.7 | 0.3 | 0.1 | 0.0 | 0.5 | 0.5 | 0.9 |
| max m | 8849 | 4884 | 5475 | 3839 | 5292 | 5661 | 6747 |
| land % | 29.2 | 42 | 35 | 31 | 25 | 30 | 24 |

Length alone does not bring the mountains back: at `orogen_decay =
0.008` a belt is gone 125 steps after the convergence that built it
stops, so whatever the run's length the high ground is what collided in
the last two hundred steps, and at 3000 that happened to be little.
4000 steps scatters the crust into an archipelago (25 % land in dozens
of pieces). Halving the decay keeps the belts of the first reassembly
standing, and 3000 steps at 0.004 puts the 2-4 km bands on Earth's
before erosion has touched them. Both are the defaults now; tectonics
costs 277 s at Earth scale instead of 120 s.

`worlds/earth-v5` is the full bake (with docs/lakes-in-erosion.md as
well). The eroded surface at iteration 800:

| band | Earth | earth-v4 | **earth-v5** |
|---|---|---|---|
| 0-1 km | 71.6 | 94.2 | **67.7** |
| 1-2 | 15.4 | 3.5 | 18.7 |
| 2-3 | 7.5 | 1.3 | **6.4** |
| 3-4 | 3.8 | 0.6 | **2.8** |
| 4-5 | 1.7 | 0.2 | 2.3 |
| >5 | 0.3 | 0.1 | 2.1 |
| land median | ~350 m | 62 | 655 |
| max | 8849 m | 6021 | 9862 |
| ocean median | -3700 m | -2457 | -3247 |
| within ±50 m of sea level | 1-2 % | 20 | 7 |

Four to six continents with mountain belts on their collision margins,
an ocean nearly at Earth's depth, and a hypsometric curve that is on
Earth's from 0 to 4 km. Two things stand out on the other side: 2.2 % of
the land above 5 km against Earth's 0.3 (the reassembly belts at 0.004
are now too well kept, or erosion is not reaching them), and a land
median of 655 m against ~350 -- too much of the crust is standing at
1-2 km, which is where the 3000-step bedrock already had 28 %. The
decay is a knob with two numbers now and wants sweeping between 0.004
and 0.008 on the full bake.

### The decay sweep, on the full bake

`worlds/sweep-od006` and `sweep-od008`: the `earth` preset baked end to
end with only `orogen_decay` changed (the same source as `earth-v5`,
which is the 0.004 arm). Eroded surface at iteration 800:

| band | Earth | 0.004 (`earth-v5`) | 0.006 | 0.008 |
|---|---|---|---|---|
| 0-1 km | 71.6 | 67.7 | 84.6 | 91.2 |
| 1-2 | 15.4 | 18.7 | 9.3 | 6.8 |
| 2-3 | 7.5 | **6.4** | 2.3 | 0.8 |
| 3-4 | 3.8 | **2.8** | 1.2 | 0.6 |
| 4-5 | 1.7 | 2.3 | 0.8 | 0.3 |
| >5 | 0.3 | 2.1 | 1.8 | 0.2 |
| land median | ~350 m | 655 | **359** | 157 |
| max | 8849 m | 9862 | 14,780 | 6674 |
| ocean median | -3700 m | -3247 | -3373 | -3286 |

0.008 is the world before this work: no mountains. 0.006 lands the
median on Earth's and empties the 2-4 km bands to a third of Earth's;
0.004 fills those bands and lifts the median to 655 m. The two cannot be
had from this knob alone, and **the excess above 5 km is not the decay's
either**: 0.006 keeps 1.8 % of the land above 5 km with a highest point
of 14.8 km, where 0.004 tops out at 9.9 km. A few peaks that high are a
defect of their own, and it has since been measured on the peaks
themselves (docs/uplift-ceiling.md): it is the `uplift` field -- the last
100 tectonic steps' rate, applied for all 800 iterations with no ceiling
onto ridge crests nothing erodes -- and not a stack of overlapping
profiles, except for the 1.0 % of od006's *bedrock* that already stands
above 5 km (max 8612 m) before erosion starts. The top cell of 0.004 is
5798 m of bedrock + 3162 m of applied uplift + 866 m of datum hold; od006's
is 8544 + 6855 - 619 m of net erosion. `erosion.uplift_max_m` (2 m per
iteration) was the ceiling, and halved the excess (docs/uplift-ceiling.md).
The window turned out to be counted twice, once in `bedrock` and once over
the run. `erosion.uplift_mode = 'replay'` starts erosion from the
reference-step crust instead, and the cap is off by default
(docs/uplift-replay.md).
0.004 stays the default: the 2-4 km bands are what a viewer sees as
mountains, and the median is the 1-2 km band's problem to fix by
erosion, not by taking the belts down faster.

