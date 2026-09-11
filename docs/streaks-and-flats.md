# Radial streaks, the concentric ripple, and the continent-sized tidal flat

Three artifacts in the corrected-glacial Earth-scale world at iteration 800:
straight rays fanning across the lowlands, a set of concentric rings, and a
plain the size of Russia sitting within a few metres of sea level. All
three are chased here with measurements on a fresh bake of the shipped
`earth` preset (`worlds/w-base`, seed 0, current code including the
`island_arc` classifier, so the numbers differ from `docs/earth-bake.md`'s
world while the phenomena are the same).

The instruments are in `bake/scripts/`: `fork_erosion.py --fresh` runs any
erosion variant from iteration 0 on a world's own tectonics and climate —
verified bit-identical to `bake.py` on the same parameters — so a fix that
acts from the first iteration can be tested on exactly the same world
rather than argued about.

## The radial streaks are the epsilon routing surface

### Particles do not steer on the terrain

`erosion/route.py` builds an epsilon-filled priority-flood surface that
particles steer on wherever it stands above the terrain: `max(route,
surface)`. Its purpose, in its own docstring, is to let a particle cross a
*lake* "on its flat-with-epsilon water surface towards the outlet". The
flood enforces a minimum drop of `route_eps` from every cell to the one it
was reached from:

```python
if v < h + eps:
    v = h + eps
```

`route_eps = 1e-3` is a height in cell units. At the 50 m cells the erosion
model was tuned on that is **5 cm per cell**. At the `earth` preset's
9773 m cells it is **9.77 m per cell**, and wherever the real terrain falls
less than that the flood raises the routing surface above it. Measured on
the baseline:

| | iteration 50 | iteration 150 |
|---|---|---|
| land where the terrain falls less than `route_eps` per cell | 41.3 % | 41.5 % |
| land where the routing surface stands above the terrain | **92.3 %** | **93.4 %** |
| routing surface above terrain, median | 93 m | 67 m |
| … p90 | 1218 m | 967 m |
| … maximum | **2658 m** | 2256 m |

The routing surface covers nearly the whole continent, and in places it
stands more than two and a half kilometres above the ground: a flow path
250 cells from its outlet accumulates 250 × 9.77 m of forced rise. That
surface is a cone of constant slope rising away from every outlet, so the
flow lines it produces are straight rays converging on the outlet — the
classic priority-flood epsilon artifact, applied here to the entire land
surface rather than to lakes.

### The measurement that says so

One particle pass on the iteration-50 state through `maps.run_iteration`'s
`diag` hook, measuring for every land step the cosine between the step and
three directions:

| terrain slope | cos(step, terrain downhill) | cos(step, momentum) | steps on route | cos(step, route downhill) |
|---|---|---|---|---|
| 0 – 1e-5 | −0.07 | 0.95 | 99.8 % | **0.83** |
| 1e-5 – 1e-4 | 0.01 | 0.93 | 99.4 % | **0.78** |
| 1e-4 – 3e-4 | 0.67 | 0.95 | 99.6 % | 0.85 |
| 3e-4 – 1e-3 | 0.53 | 0.92 | 98.3 % | 0.79 |
| 1e-3 – 3e-3 | 0.37 | 0.90 | 94.4 % | 0.71 |
| 3e-3 – 1e-2 | 0.47 | 0.90 | 92.8 % | 0.66 |

On gentle ground a particle's step is uncorrelated with the direction the
terrain falls (cos ≈ 0) and strongly aligned with the routing surface's
fall (0.78–0.85). Particles are steering downhill — on a phantom surface.

Two hypotheses were tested and set aside on the way, and both are worth
recording because both looked right:

* **Ballistic particles.** The kernel's direction update is
  `normalize(0.7 · previous + momentum push + 2.4 · slope · downhill)`, and at
  Earth-scale slopes (land median 1.6e-3) the gravity term is 10⁻⁴–10⁻² of
  the inertia term, so particles *should* barely turn. A path-straightness
  test could not confirm it: chord/arc came out 0.91–0.96 in every slope
  bin, which is what a perfectly straight *digital* line scores, so the
  metric was saturated and said nothing either way.
* **`slope_saturation`**, the kernel's own terminal-velocity option for
  exactly this, set to an Earth-scale 3e-4. One pass on the iteration-50
  state made alignment with the terrain *worse* on steep ground (0.47 →
  −0.01) and did nothing on the flat (0.01 → 0.04). It makes gravity
  stronger on the terrain, but the particles were not steering on the
  terrain. The run was stopped.

### The fix under test

`route_eps` at its 50 m-cell meaning — 5 cm per cell, `5.116e-6` cell units
at this grid — run from iteration 0 as `forks/vr-eps5cm`.

**Predicted before it ran:** the share of land steering on the routing
surface falls from ~93 % to the genuinely closed depressions (a few per
cent); the routing surface's p90 height above the terrain falls from
~1 km to metres; the straight rays disappear from the discharge field. The
risk is on genuinely flat ground, where a particle with no epsilon slope to
follow has almost nothing to steer on at all: expect more `stop` and `pit`
deaths and possibly sheet deposition where there were rays.

**Result at iteration 100**, in the 256 × 256-cell window where the
baseline's routing surface stands highest above the land (a mean of
**1739 m** over a plateau that is itself flat to within ~85 m):

| iteration 100 | shipped (9.77 m/cell) | 5 cm/cell |
|---|---|---|
| routing surface above terrain, median over land | 73 m | **7 m** |
| … p90 | 1057 m | **50 m** |
| land median | 510 m | 570 m |
| land above 2 km | 14.0 % | 14.2 % |
| highest point | 7544 m | 7544 m |
| globe within ±50 m of sea level | 3.29 % | 3.26 % |

In the shipped run the routing surface is flagged over the whole window as
a smooth geometric pyramid, and the discharge field under it is a sheet of
thousands of dead-straight parallel rays cut by a straight line where two
branches of the flood tree meet; the momentum field is one uniform
direction. With the epsilon at its 50 m meaning the routing fill follows
real valleys in branching, organic shapes, and the discharge field is a
dendritic network — a sinuous trunk river, converging tributaries, no
parallel rays and no flood boundary. The hypsometry is unchanged to within
a few per cent, and the tidal-flat numbers do not move at all: the routing
surface is not what makes the flat.

**The predicted risk is real, and bounded.** One particle pass on each
run's iteration-300 state:

| one pass, iteration 300 | 9.77 m/cell | 5 cm/cell |
|---|---|---|
| mean steps per particle | 70.6 | **146.6** |
| die in a pit | 1.38 % | **6.81 %** |
| stop (no motion at all) | 0 | 0.00 % |
| clamped deposit entries | 18.6 M | 55.5 M |

With no phantom slope to ride, a particle on genuinely flat ground wanders
about twice as far and is five times likelier to end in a closed pit —
where it drops its load, which is how closed basins fill. None stalls
outright. The costs are a longer particle phase and, possibly, pit-fill
artifacts of their own; the render at iteration 100 shows none, and the
800-iteration run is the check.

**At iteration 800 the fix holds.** Both runs carry the shipped glacial
pass, so they differ only in `route_eps`:

| iteration 800 | shipped (9.77 m/cell) | 5 cm/cell |
|---|---|---|
| routing surface above terrain, median over land | 113 m | **11 m** |
| … p90 | 1317 m | **25 m** |
| land median | 484 m | 565 m |
| land above 2 km | 4.80 % | 5.45 % |
| highest point | 5192 m | 5220 m |
| globe within ±50 m of sea level | 6.40 % | 7.16 % |
| sea-level change per 1 % of land fraction | 37 m | 43 m |

The phantom surface stays gone for the whole run, and the hypsometry and
the waterline move by less than a point: the routing fix is neutral on
everything but the streaks.

The render at 800, in the window where the shipped routing surface stands
highest (face 3, cells 692–947 × 484–739, a mean 2125 m above a plateau), is
the iteration-100 picture again. Shipped: the routing fill flagged over
100 % of the window as one smooth pyramid, a discharge sheet of parallel
rays cut by a straight diagonal where two branches of the flood tree meet,
and one uniform momentum direction. With the epsilon at 5 cm: a dendritic
network with two meandering trunk rivers and converging tributaries, a
routing fill that follows the valleys, and sediment laid along the channels
and in fans rather than in sheets. The pit-fill artifacts the one-pass
death count warned about do not appear.

## The continent-sized tidal flat

### It is made in one step, by the first glacial pass

The previous account (fluvial planing to base level, held there by
`DEP_FLOOR` and `hold_datum`) is true of the slow part and misses the
part that makes the picture. Checkpoints either side of the first glacial
pass, which fires on the step that *produces* iteration 600:

| globe within … of sea level | iteration 550 | iteration 600 |
|---|---|---|
| ±1 m | 0.47 % | **7.78 %** |
| ±20 m | 4.86 % | 8.81 % |
| ±50 m | 8.26 % | 10.13 % |

**4.94 % of the globe — 25.2 M km², every cell of it cold enough for ice —
went from more than 50 m above sea level to within ±1 m of it in that one
step**, losing a mean of 454 m of relief (p90 1087 m, max 3806 m). And
289,124 of those 289,174 cells sit at 0.000 m: not *near* sea level, on it.

The mechanism is three pieces, each measured elsewhere:

1. At the shipped `glacial_rate`/`glacial_max` one pass can carve 3909 m
   (docs/missing-relief.md), so it carves everything cold down to its
   floor.
2. That floor is `room = surface + DEP_FLOOR`, and `DEP_FLOOR` is 195.5 m
   at this grid — so every carved cell lands at exactly `−DEP_FLOOR`, one
   plane flat to float precision.
3. `hold_datum` puts exactly 30 % of the cells above zero by subtracting
   the k-th surface value. With a quarter of a million cells sharing one
   value, that order statistic falls *inside* the plane, and the whole
   plane is shifted onto zero.

The baseline's whole trajectory (surface, `compare.py`) shows the step for
what it is:

| baseline | iter 400 | 550 | **600** | 800 |
|---|---|---|---|---|
| globe within ±1 m of sea level | 0.39 % | 0.47 % | **7.78 %** | 0.13 % |
| globe within ±50 m | 6.90 % | 8.26 % | 10.13 % | 6.40 % |
| cells straddling the waterline | 0.87 % | 0.90 % | 4.66 % | 3.33 % |
| sea-level change per 1 % of land fraction | 64 m | 54 m | **0.08 m** | 37 m |
| highest point | 9114 m | 9647 m | 9790 m | **5192 m** |

At iteration 600 moving sea level by eight *centimetres* changes the land
fraction by a whole point: the waterline is sitting in a plane. By 800 the
datum has drifted the plane slightly off zero, but it is still inside the
±50 m band and the conditioning never recovers. (The same step takes the
highest point from 9790 m to 5192 m: docs/missing-relief.md.)

The stipple in the rendered globe is float rounding deciding, cell by cell,
whether a surface that is level to under a millimetre is land or sea. The
renderer is right; there is no land/sea boundary to draw there because the
model has placed a continent-sized plane exactly on the waterline.

### The slow part, and what isostasy does to it

Before any glacial pass the flat grows steadily in the baseline — the
globe within ±50 m of sea level runs 2.36 % at iteration 50, 4.68 % at 200,
5.83 % at 300 — while the datum loses its conditioning (114 → 74 m of
sea-level change per point of land fraction). That is the fluvial pass
planing to base level with nothing to stop it: the kernel lowers the
surface one metre for every metre of rock it removes.

`erosion.isostasy = 0.8` makes a column rebound by 0.8 of the rock it loses,
spread by a 60 km flexural Gaussian (Airy with the tectonics stage's own
densities; eroding a kilometre lowers a continent ~200 m). Run from
iteration 0 on the identical world, one variable changed, at iteration 300:

| iteration 300 | baseline | isostasy 0.8 | route fix only | Earth |
|---|---|---|---|---|
| land above 2 km | 8.9 % | **14.7 %** | 9.2 % | 13.3 % |
| land median | 383 m | 504 m | 427 m | ~350 m |
| globe within ±50 m of sea | 5.83 % | **2.24 %** | 5.61 % | 1–2 % |
| … within ±20 m | 3.22 % | 0.97 % | 3.27 % | |
| sea-level change per 1 % of land | 74 m | **156 m** | 79 m | |
| routing surface above terrain, p90 | 942 m | 1045 m | **23 m** | |

The two fixes act on separate defects and it shows: the routing epsilon
moves only the routing numbers, and isostasy moves only the planing ones.
Isostasy also does more than hold the lowlands off the waterline — it
holds the *high ground* up, at 14.7 % of land above 2 km against Earth's
13.3 %. That is the fluvial/uplift imbalance docs/missing-relief.md ranked
as the first problem (it removes 9.0 of the 10.3 points of high ground that
erosion takes), and a missing isostatic response is its mechanism: without
one, uplift on the order of a kilometre over the run can never keep pace
with erosion that lowers the surface one-for-one.

### Isostasy defeats the plane even with the shipped glacial pass

The same first shipped pass, 3909 m a pass, with and without the rebound
(iteration 600):

| after the first shipped glacial pass | no isostasy | isostasy 0.8 | Earth |
|---|---|---|---|
| globe within ±1 m of sea level | 7.78 % | **0.03 %** | |
| globe within ±50 m | 10.13 % | **2.43 %** | 1–2 % |
| largest set of cells sharing one surface value | 376,532 | **25** | |
| sea-level change per 1 % of land fraction | 0.08 m | **123 m** | |
| cells straddling the waterline | 4.66 % | 0.88 % | |
| land above 2 km | 5.35 % | 12.41 % | 13.3 % |
| land median | 317 m | 412 m | ~350 m |
| highest point | 9790 m | **12,014 m** | 8849 m |

The pass still carves everything cold down to its floor, but the flexural
rebound applied on the same step lifts the carved ground back up by a
spatially varying amount, so there is no single plane for `hold_datum`'s
quantile to land in. The waterline stays well-conditioned and the ±50 m
band stays near Earth's.

The last row is the cost: with the rebound the highest point overshoots
Earth by more than three kilometres. That is the known effect — incising
valleys unloads the plate and the flexural response lifts the peaks between
them — working harder than Earth's does.

**The overshoot is confined to the top band.** The highest point is one
cell, so here is the whole curve, bedrock, % of land:

| | 0–1 km | 1–2 | 2–3 | 3–4 | 4–5 | >5 | 99.9th pct | max |
|---|---|---|---|---|---|---|---|---|
| Earth | 71.6 | 15.4 | 7.5 | 3.8 | 1.7 | **0.34** | | 8849 |
| shipped 400 | 80.2 | 12.0 | 4.5 | 2.0 | 0.8 | 0.61 | 7210 | 9114 |
| isostasy 400 | 67.6 | 18.3 | 7.2 | 4.0 | 1.7 | 1.32 | 8624 | 9935 |
| route + isostasy 400 | 65.9 | 19.6 | 7.4 | 4.1 | 1.7 | 1.36 | 8669 | 9942 |
| shipped 600 | 82.4 | 11.7 | 3.8 | 1.4 | 0.4 | 0.35 | 6711 | 9790 |
| isostasy 600 | **71.1** | **16.4** | **5.8** | **3.5** | **1.7** | 1.45 | 9465 | 12,014 |

With isostasy every band from sea level to 5 km sits within about one point
of Earth — the shipped world is short in every band above 1 km — and the
excess is roughly 1.1 % of land above 5 km. The shipped world already
carries 1.8× Earth's share there at iteration 400, before any glacial pass.
Isostasy doubles an existing tail; it does not create one.

**The tail is the active orogens, lifted whole.** 88 % of the >5 km land at
iteration 600 lies in the top tenth of land by tectonic uplift (43 % in the
top hundredth); over all land that share is 9.7 %. At those cells the world
with isostasy stands a median 2706 m above the world without it. Averaged
over a 127 km box around each cell the gain is 2647 m, so 98 % of it is
regional — the rebound lifts the whole range, not selected spikes.

That is the expected steady state, not a defect in the smoother. With a
rebound fraction ρ a range stops rising when erosion runs at
`uplift / (1 − ρ)` — five times the uplift at ρ = 0.8 — and that needs far
steeper, higher ground than the world without rebound, where erosion only
has to match the uplift. It is what Earth does too. What caps Earth's ranges
near 5 km of mean elevation is missing here: the lower crust flowing out
from under a thick plateau, and a glacial buzzsaw that works at the
snowline. The corrected glacial pass is the only candidate for the second;
its effect at 800 is in the comparison below.

**Isostasy must not ship without the glacial units fixed.** The same run,
carried on past 600 with the shipped glacial pass:

| bedrock, min / max | 600 | 700 | 800 |
|---|---|---|---|
| shipped | −3709 / 9790 | −6309 / 5086 | −8889 / 5192 |
| glacial units fixed, no isostasy | −3873 / 9812 | −3845 / 10,021 | −3862 / 10,096 |
| isostasy + shipped glacial | −7603 / 12,014 | −12,405 / 19,567 | **−17,525 / 22,833** |

Both ends run away, and neither is a numerical spike. The highest cell sits
in a 225-cell block standing at 19–22 km that rose about 3 km in a hundred
iterations while the glacial pass cut its edge down from 13–19 km to 5–9 km.
Carving up to 3909 m a pass off the flanks unloads the plate, and the
rebound lifts the whole block, uncarved centre included. The lowest cell is
a glacial pit with its bedrock at −17.5 km under 18.3 km of sediment: the
shipped pass has no base level, so it digs the same hole every pass and
deposition keeps refilling it (the shipped run digs one to −8.9 km without
isostasy). With the glacial units corrected and no isostasy, both ends stay
put. The two fixes are coupled: with the shipped glacial scale, isostasy
cannot be switched on.

The first of Earth's two missing caps has a hook already. Tectonics caps continental crust at
`max_crust_thickness` (2× normal, the root delaminating above it), but the
erosion stage receives `uplift` as a fixed metres-per-iteration field and
applies it for all 800 iterations whatever the range has become. With the
rebound on, a range that erosion has lifted to plateau height keeps
receiving its full tectonic rate. Tying the erosion-stage uplift to that
same ceiling is the natural cap for this tail. It is a new mechanism and is
not attempted here.

**Predicted before it ran**, for erosional isostasy at 0.8 (continental
Airy) with a 60 km flexural response: the land median stays well above the
baseline's (erosion now lowers the surface by ~0.2 of what it removes, not
all of it); the share of the globe within ±50 m of sea level stays near its
tectonic value instead of climbing; the datum stays well-conditioned (tens
of metres of sea-level change per point of land fraction, not ~2); and the
high ground above 2 km survives better, because a flexural response to
valley incision lifts the interfluves and peaks around it. The risk is the
other side of the same coin: a continent that no longer planes down may
now overshoot Earth's low bands instead of undershooting them.

## The concentric ripple is the glacial pass printing into the plane

Found by reproducing the state in the rendered globe: fork the baseline's
iteration-600 checkpoint (which already contains the first shipped glacial
pass, and so the plane) to 650 with the corrected glacial values and the
old routing epsilon pinned. The rings appear there, in a window that is
100 % cold and 95 % *submerged*, and a one-pass probe places them in the
**bedrock**, not the sediment.

The attribution is an A/B from the same checkpoint, 50 iterations each,
identical except for the glacial pass:

| bedrock, the ring window | iteration 600 | 650, glacial ON | 650, glacial OFF |
|---|---|---|---|
| concentric rings, arcs, diagonal stripes | none | **present** | none |

With glaciation off, fifty iterations of particles, mass wasting, uplift
and datum holds leave the plane visibly untouched; with it on, the oval ring
set, arcs and a field of diagonal stripes appear. The rings are drawn by the
glacial pass.

The mechanism follows from the plane. `ice_mask` is *cold and above sea
level*, and on a surface level to float precision at exactly 0 m, which
cells are above zero is decided by rounding: the ice mask is a noisy set of
patches. `ice_depth` then shrinks each patch one 8-neighbour ring at a
time and quantises the carve into `glacial_ramp + 1` = 5 levels, so every
pass carves a set of concentric steps around every patch edge — and the
diagonal stripes are the 45° faces of those Chebyshev rings. The next pass
sees a different mask (the carved cells are now below zero, and the datum
has moved), so ring sets accumulate pass after pass.

The ring is therefore a symptom, not a separate defect: no plane, no
flickering ice mask, no rings.

**And no plane forms at all if the glacial units are corrected before the
first pass.** Forking from iteration 550 — before any glacial pass — with
the corrected values (≤ 59 m a pass at this grid), everything else
identical:

| iteration 600 | corrected from 550 | shipped |
|---|---|---|
| globe within ±1 m of sea level | **0.09 %** | 7.78 % |
| globe within ±50 m of sea level | 6.86 % | 10.13 % |
| largest set of cells sharing one surface value | **9** | 376,532 |
| the ring window, elevation p5..p95 | **41 .. 810 m** | −0.0 .. 0.0 m |

The first corrected pass reaches 483,088 ice cells and carves at most 59 m;
it cannot flatten 450 m of relief, so the plane is never made and the ring
window is still land with 800 m of relief.

**That is not the end of the rings, though.** Run on to iteration 650, the
corrected arm grows its own banded structure in the same window: alternating
land and sea stripes following the contours, and a dense fingerprint of
concentric arcs in bedrock, none of which is there at 550. So "no plane, no
rings" is too strong — the plane makes the rings dramatic (concentric
around every noise-defined ice patch) but is not their only source. The
candidate for the rest is the same taper acting on an ice margin that *is*
the coastline: `ice_mask` requires the surface to be above zero, so in cold
lowlands the edge of the ice is the waterline; the quantised taper carves a
staircase inward from it; and the carve's floor (`surface + DEP_FLOOR`,
195.5 m below sea level at this grid) lets the outer steps be cut below the
sea, where they become water and move the margin — and the next staircase —
inland. Each pass would leave a band parallel to the coast. A glaciation-off
control from 550 is needed before that is more than a candidate.

The control, both arms forked from the same iteration-550 state, 100
iterations, identical but for the glacial pass:

| the ring window | 550 | 650, corrected glacial ON | 650, glacial OFF |
|---|---|---|---|
| bedrock high-pass std | 19.5 m | **31.6 m** | 19.8 m |
| cells on a land/sea boundary | 0.34 % | **12.8 %** | 0.34 % |
| window below sea level | 1.5 % | **22.4 %** | 1.5 % |

With glaciation off the window is essentially unchanged from 550; with the
corrected pass on, five passes drown 21 % of it and draw the zebra land/sea
banding and the fingerprint arcs. The bands are 5–8 cells apart, where the
quantised taper only makes one-cell steps, so the quantisation is not the
main cause: the *migrating margin* is. A cell carved below sea level stops
counting as ice, the margin jumps inland, and the next pass's staircase
starts from there. (A real tidewater glacier does not stop being a glacier
because it has cut its bed below the sea.)

**Fix under test: a sticky ice mask** (`erosion.glacial_sticky`) — a cell
glaciated in one pass stays glaciated while it is still cold, whatever its
bed has been carved to, so the margin is fixed by climate and the original
terrain rather than by the carving itself, and repeated passes deepen one
stable basin, which is the overdeepening the pass was built to make.

Forked from the same iteration-550 state, 100 iterations, the corrected
glacial values in both arms, only `glacial_sticky` differing:

| iteration 650 | margin migrates | sticky ice | glacial off |
|---|---|---|---|
| ring window: cells on a land/sea boundary | 12.8 % | **2.9 %** | 0.34 % |
| ring window: below sea level | 22.4 % | 19.0 % | 1.5 % |
| lakes, % of land (globe) | 1.73 % | **2.59 %** | 0.32 % |
| number of lakes | 2316 | 1724 | 502 |
| deepest lake | 169 m | **314 m** | 25 m |

The zebra banding is gone — the drowned ground comes out as a few coherent
basins, fjord- and lake-shaped, instead of stripes along the coast — and
the concentric arcs along the coast go with it. The pass still overdeepens
(19 % of the window is below sea level) and it now makes *more* lake: fewer,
larger, deeper basins, which is what repeated passes on a stable margin
should do.

(Lakes here are true closed depressions — a priority flood of the surface
with an epsilon of 10 µm, fill deeper than 1 m. The obvious alternative,
the routing surface's fill, is useless in these arms: they pin the shipped
9.77 m routing epsilon, whose phantom surface stands above ~93 % of all
land in every arm, the glaciation-off one included.)

**One glacial artifact survives sticky ice**: a dense fingerprint of
concentric arcs deep inside the ice, present in both glaciation-on arms and
absent with it off. It is not the quantised taper — a one-pass probe there
finds the window 100 % ice and every cell at taper 1.0, with the carve near
its 59 m cap. The carve is `rate · √discharge · …`, so where it is not capped
it prints the *discharge* field into the bedrock, and these arms carry the
shipped routing epsilon, whose discharge field is the ray-and-fan artifact
of the first section. The working reading is that the fingerprint is the
routing streaks made permanent by the ice, which makes it the routing fix's
job — tested by running both fixes together.

**The combined run settles it.** V5 carries all four fixes — `route_eps` at
5 cm, isostasy 0.8, the glacial units corrected, sticky ice — from iteration
200. The same window at 550 (before glaciation) and at 650 (two passes in):

| ring window | V5 550 | V5 650 | sticky ice alone, 650 |
|---|---|---|---|
| bedrock high-pass std | 37.6 m | 68.8 m | 31.7 m |
| land/sea boundary cells | 0.36 % | **0.35 %** | 2.88 % |
| below sea level | 1.6 % | **1.5 %** | 19.0 % |
| globe: lake share of land | 2.46 % | 2.96 % | 2.59 % |
| globe: lakes | 1854 | 2270 | 1724 |
| globe: deepest lake | 275 m | 618 m | 314 m |

The glaciation no longer moves the coast at all, and there are no rings.
The high-pass field doubles, but the render shows why: it is a dendritic
valley network running downslope, sharper and denser than before the ice,
not bands parallel to the margin — the ice is deepening real valleys. The
straight diagonal fingerprint that sticky ice alone left behind (the old
routing's discharge, printed by the √q term) is gone with the routing fix,
which confirms that reading. The glaciation also does what it is for: more
lakes, deeper ones, and more of the land in them.

(A secondary point stands on its own: a
taper quantised into five one-cell steps will terrace an ice margin on any
terrain, and a continuous distance would not.)

## What ships

**All four fixes, at iteration 800, against every partial combination.**
Surface metrics except where marked bedrock:

| iteration 800 | Earth | shipped | route | isostasy | route + iso | glacial units | **all four** |
|---|---|---|---|---|---|---|---|
| 0–1 km, % of land (bedrock) | 71.6 | 74.5 | 67.2 | 73.6 | 71.3 | 85.0 | **68.4** |
| 1–2 | 15.4 | 20.3 | 26.8 | 16.5 | 18.3 | 8.9 | **17.3** |
| 2–3 | 7.5 | 3.8 | 4.3 | 5.5 | 5.7 | 3.6 | **6.3** |
| 3–4 | 3.8 | 1.2 | 1.4 | 2.7 | 2.8 | 1.5 | **4.1** |
| 4–5 | 1.7 | 0.2 | 0.3 | 1.2 | 1.2 | 0.5 | **2.0** |
| >5 | 0.34 | 0.01 | 0.01 | 0.56 | 0.59 | 0.45 | **1.98** |
| land above 2 km | 13.3 | 4.80 | 5.45 | 9.41 | 9.51 | 5.54 | **13.32** |
| land median, m | ~350 | 484 | 565 | 565 | 601 | 188 | 500 |
| globe within ±1 m of sea | | 0.13 | 0.11 | 0.18 | 0.15 | 1.83 | 0.14 |
| globe within ±50 m | 1–2 | 6.40 | 7.16 | 4.38 | 4.33 | 11.45 | **3.97** |
| cells straddling the waterline | | 3.33 | 3.54 | 1.62 | 1.69 | 3.98 | **1.46** |
| sea level per 1 % of land, m | | 37 | 43 | 37 | 39 | 2.6 | 48 |
| routing surface above terrain, p90 m | | 1317 | 25 | 1239 | 32 | 1070 | **12** |
| ocean median, m | −3700 | −1927 | −1918 | −2497 | −2483 | −2525 | **−2755** |
| lowest / highest bedrock, m | / 8849 | −8889 / 5192 | −7427 / 5220 | −17,525 / 22,833 | −15,871 / 22,938 | −3862 / 10,096 | −8588 / 12,973 |

Every partial combination fails somewhere the full one does not: without
the routing fix the phantom surface is back (p90 1.2–1.3 km); isostasy with
the shipped glacial scale runs away; the glacial units without isostasy
plane the lowlands (median 188 m, 11 % of the globe within 50 m of the
sea). Together, every band from sea level to 5 km sits within about a point
of Earth's, the land above 2 km is 13.32 % against 13.3, the waterline is
the best-conditioned of the six, and the ocean is 800 m deeper than
shipped, closer to Earth's. The one band that is worse is the top: 1.98 %
of land above 5 km against Earth's 0.34, still thickening between 600
(1.53 %) and 800, with a highest point of 12,973 m. That is the orogen tail
above, and its cap is the tectonic ceiling the erosion-stage uplift does
not yet respect.

**The glacial lengths go into metres at the values they were tuned at.**
`LENGTH_PARAMS_M` declares every length at its old cell-unit value × 50 m,
so a 50 m-cell world stays bit-identical; for the glacial pass that is 50 m
and 20 m a pass. V5 ran the stronger 147 m / 59 m, so V6 repeats it from
V5's own iteration-550 checkpoint (all four fixes, before any ice) with
50 m / 20 m. After the first glacial pass the two are the same world:

| iteration 600 | V5, 147 / 59 m a pass | V6, 50 / 20 m a pass |
|---|---|---|
| lowest / highest bedrock | −7471 / 11,436 m | −7476 / 11,427 m |
| globe within ±1 m of sea level | 0.04 % | 0.04 % |
| globe within ±50 m | 3.04 % | 3.14 % |
| land above 2 km | 13.1 % | 13.0 % |
| land above 5 km | 1.53 % | 1.53 % |
| land median | 492 m | 487 m |
| largest set of cells sharing one surface value | 20 | 23 |

The earlier forks without isostasy said the same (`metres` against
`earthlike` in docs/missing-relief.md: 7124 against 7055 m at the top,
4.4 % above 2 km in both). The per-pass value is not what the world is
sensitive to; the units were.

V6 run on to 800 holds that, and is slightly better at the waterline:

| iteration 800 | V5, 147 / 59 m a pass | V6, 50 / 20 m (shipped) |
|---|---|---|
| bedrock bands 0–1 / 1–2 / 2–3 / 3–4 / 4–5 / >5 km, % of land | 68.4 / 17.3 / 6.3 / 4.1 / 2.0 / 1.98 | 69.3 / 16.8 / 6.1 / 4.0 / 2.0 / 1.94 |
| land above 2 km | 13.32 % | 13.09 % |
| globe within ±50 m of sea level | 3.97 % | 3.84 % |
| cells straddling the waterline | 1.46 % | 1.05 % |
| sea-level change per 1 % of land fraction | 48 m | 94 m |
| routing surface above terrain, p90 | 12 m | 8 m |
| lowest / highest bedrock | −8588 / 12,973 m | −8472 / 12,910 m |

**The small preset takes isostasy and sticky ice without trouble.** Its
worlds are spherical too, so the two defaults reach them. Baked through
`scripts/bake.py --preset small --to erosion`, the eroded surface
(`coarse/height` + `sediment` — `hypsometry.py`'s world mode reads the
tectonic `bedrock`, which the two bakes share):

| small preset, seed 0, eroded surface | off | isostasy 0.8 + sticky |
|---|---|---|
| globe within ±1 m of sea level | 5.73 % | **2.64 %** |
| globe within ±50 m | 41.3 % | 39.7 % |
| land median | 14.9 m | 18.9 m |
| highest bedrock | 608 m | 678 m |
| erosion stage | 3.8 s | 11.6 s |

Half the surface moves by more than a metre and the same way it does at
Earth scale — the waterline flat halves and the land stands a little
higher — with nothing running away. The cost is the flexural smoother:
at 50 m cells `flexure_km` asks for more than the planet, the sigma is
capped at N/8 = 16 cells, and 60 iterations apply it three times. The
tiny preset's 10 iterations never reach `isostasy_every = 20`, so it is
unchanged.

**The defaults, then.** `erosion.route_eps = 0.05` m (already shipped);
`erosion.glacial_rate = 50` m and `erosion.glacial_max = 20` m, in
`LENGTH_PARAMS_M`; `erosion.isostasy = 0.8`; `erosion.glacial_sticky =
True`. The four ship together or not at all — every partial combination in
the table above fails somewhere. A new test pins the glacial lengths to
metres at two cell sizes, the isostasy default test now asserts 0.8 and
that 0 still switches it off, and the whole suite passes. The open cost is
the orogen tail above 5 km.
