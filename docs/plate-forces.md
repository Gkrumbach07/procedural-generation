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

