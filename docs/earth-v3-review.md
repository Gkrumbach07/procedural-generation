# Earth 3 reviewed: the tectonic start, the hydrology, and the coast

`worlds/earth-v3` is the first bake with the connected-ocean mask inside
erosion and the lake balance in hydro (docs/erosion-and-the-sea.md). Looked
at in the timeline viewer it raised five things, and this document runs each
of them to ground: what the code does, what a measurement says, and what to
build. The five, in the order they were raised:

1. the starting plates and the supercontinent cycle;
2. rivers that cut through lakes on land bridges, hair-thin rivers, and no
   sense of a lake having an inflow and an outflow;
3. the fringed, fingered coast that reads as a delta and is not one;
4. a render mode with elevation and biome together;
5. whether tectonics should simply run longer.

Numbers are the `earth` preset (1024² per face, 9773 m cells, 1500 tectonic
steps, 800 erosion iterations), seed 0, read off `earth-v3`'s manifest and
graphs, plus one tectonics-only re-run at 3000 steps.

## 1. The starting state, and why the ocean plates fight each other

### What `initialise` builds

* One continent: a noise-perturbed spherical cap holding
  `continental_fraction = 0.60` of the sphere, centred on a random unit
  vector (`plates.seed_supercontinent`). It is deterministic given the seed;
  the shape is random in the sense that the seed picks it, but there is
  exactly one shape family, a cap with a rough margin.
* **One plate holds the whole continent.** `supercontinent_plates` gives the
  continent plate 0 and clusters the remaining 40 % of the sphere into
  `initial_plates - 1 = 7` oceanic plates by spherical k-means. No boundary
  crosses the continent until the first rift at step 400.
* Every plate gets a uniformly random Euler pole at `initial_speed = 0.1`
  spacings per step (16 km/step).
* The only driving force afterwards is the gradient of a heat field:
  `f_i = area_i · ∇heat`, plates pushed towards hot. The heat field is an
  fbm drawn once at step 0 and kept as `heat_bg`; every step relaxes the
  live field back to it (`heat_relax = 0.002`, a 500-step memory). There is
  no slab pull, no ridge push, no mantle drag. Subduction warms the field a
  little and new crust cools it, but `heat_relax` erases both.
* A rift fires every `rift_every = 400` steps on one or two plates drawn
  **at random, weighted by segment count**, so the continent is the likely
  but not certain target. The halves open about `com × n` at `max_speed`
  (fixed in docs/plates-rifts-and-water.md) and then fall back under the
  heat force within `1 / damping = 20` steps.

That is what the two plate pictures show. At step 0 the seven ocean plates
are packed into a 40 % cap with random poles, so every one of their mutual
boundaries converges somewhere; the continent sits rigid and untouched for
400 steps. By the first rift the ocean plates have interleaved and eaten
into one another while the continent has not moved as a whole. The belt
census of the bake says the same:

| collision pair (`earth-v3`) | count | share |
|---|---|---|
| ocean under ocean (`pair_oo`, island arcs) | 29,115 | 52.7 % |
| ocean under continent (`pair_oc`, Andean + Laramide) | 21,101 | 38.2 % |
| continent on continent (`pair_cc`, Himalayan) | 4,894 | 8.9 % |

More than half of everything the model builds is an island arc.

### Against the supercontinent cycle

The Pastor-Galán / Britannica picture has three ingredients the model
lacks:

* **A superocean with a subduction girdle around the continent.** Ocean
  floor moves *towards* the continent and goes down under its rim; the
  ocean plates are not converging on each other in the middle of the ocean.
  Here the ocean poles are random, so the girdle is a coincidence when it
  occurs.
* **An interior ocean that keeps opening.** Once the continent rifts, the
  fragments keep going, for thousands of kilometres, until they collide on
  the far side (extroversion), or the interior ocean grows subduction zones
  and closes again (introversion), or they are caught in a north-south
  belt (orthoversion). All three need a force that *carries* a fragment
  away from the rift. The model has a one-step opening impulse and then the
  same fixed attractor that held the continent together in the first place;
  the fragments are pulled back onto it. docs/crust-audit.md measured the
  result: the oceans the cycle opens are 176-256 km wide.
* **The exterior ocean closes.** With no slab pull there is nothing that
  makes a plate go on being consumed at the trench it is attached to, so
  oceans neither open to Atlantic width nor close to nothing.

### Does running longer help? Measured

`scripts/supercontinent.py --preset earth --steps 3000` (the same
simulation the bake runs, sampled every 50 steps; largest continental mass
as a share of continental area, at four link radii):

| step | largest mass (link 240 km) | at 479 km | continental share of segments | plates |
|---|---|---|---|---|
| 100 | 0.99 | 1.00 | 0.632 | 8 |
| 550 | 0.96 | 0.97 | 0.685 | 13 |
| 1000 | 0.94 | 0.94 | 0.641 | 9 |
| 1500 | 0.91 | 0.93 | 0.505 | 10 |
| 2000 | 0.87 | 0.92 | 0.426 | 13 |
| 2400 | 0.69 | 0.94 | 0.378 | 7 |
| **2450** | **0.54** | 0.94 | 0.371 | 8 |
| 2600 | 0.55 | **0.58** | 0.363 | 9 |
| 3000 | 0.56 | 0.60 | 0.364 | 10 |

Two readings:

* **Yes, eventually.** The rift at step 2400, the sixth, is the first that
  sticks: the largest mass halves at 2450 and is still apart at 3000, at
  every radius up to 479 km. The script's verdict is "breaks up and stays
  apart at every radius measured". Nothing of the kind happens in the first
  1500 steps, which is the bake, and the 5 rifts before it all healed.
* **At the cost of half the continent.** The continental share of segments
  rises to 0.685 by step 550 as island arcs are born, then falls
  monotonically to 0.364. A continent-continent collision deletes the
  losing segment and hands its mass to the winner as thickness, so every
  rift that heals shortens the continent by the width of the belt it
  builds. 8,930 such collisions over the run. Earth's continental area is,
  to first order, conserved; here it halves.

So "run tectonics longer" on its own buys one breakup at twice the cost and
with a smaller continent; the split is real but it is not a cycle, and it is
not the thing the graphic shows.

### What to build, in order

1. **Slab pull.** At every collision where a segment of plate A subducts, add
   a force on A directed from A's centre of mass towards the trench, scaled
   by the slab's age (older, denser floor pulls harder). This is the
   dominant plate-driving force on Earth and the one that makes a plate
   *keep* going where it has started going: a rifted half with a trench on
   its far side drifts into the superocean and closes it (extroversion,
   the common case on Earth). It needs no new state; the collision list
   already has the pairs and `age`. Measure with `supercontinent.py` at
   3000 steps: the largest mass should fall below 0.7 within a few hundred
   steps of a rift and the link radius at which it recovers should be
   thousands of kilometres, not hundreds.
2. **A heat background that moves.** Keep `heat_relax` but relax towards a
   background that itself drifts: re-draw the fbm phase slowly, or advect
   it. A fixed attractor is what reassembles the fragments on the same
   spot; a wandering one gives the cycle somewhere else to go.
3. **The start.** Seed the heat background hot under the continent (a
   supercontinent insulates the mantle; that is the standard story for why
   it breaks up) and cold under the superocean, and set the ocean plates'
   initial poles so they move towards the continent rather than at random.
   That puts the subduction girdle in place at step 0 and stops the
   ocean-on-ocean mashing. Randomise the continent's shape family too: a
   cap is the only shape it can be today; two or three caps merged, or a
   cap with a deep embayment, would give the rifts something to follow.
4. **Conserve continental area.** When continent meets continent, thicken
   the belt but do not delete the segment; let the orogen's extra thickness
   decay (it already does, `orogen_decay`) and spawn continental rather
   than oceanic crust in the gaps that open behind a belt. Measure the
   continental share of segments over 3000 steps; it should hold near 0.6.

Slab pull is the lever; the other three shape where it takes the plates.

## 2. Hydrology: the rivers and the lakes come from three different places

The complaint was that rivers cut through lakes on little land bridges,
that lakes do not read as having an inflow and an outflow, that some rivers
are hair-thin, and that the whole thing feels inconsistent. It is, and by
construction: nothing that draws a river and nothing that draws a lake
shares a water surface.

| what | where it comes from |
|---|---|
| rivers in the viewer (before this change) | erosion's particle `discharge` EMA, thresholded at the 97th percentile of land |
| lakes in the viewer | hydro's priority flood after erosion, drawn down by `lake_evap`, `depth > 0.5 m` |
| the drainage graph (`graph/drainage.json`) | D8 on the flood-filled DEM, with `lake_in` / `lake_out` nodes on every channel that enters or leaves a lake, and Strahler order |
| fine rivers (`graph/rivers.json`, `fine/river_mask`) | the *fine* discharge, Gaussian-smoothed, thresholded so a fixed fraction of the land is river, skeletonised; the lake mask is never subtracted, segments are never split at a lake, and the coarse graph is consulted only to copy an order number |
| fine lakes (`graph/lakes.json`) | refine re-floods every basin on the fine grid **to the spill point**, discarding the coarse balance; 510,341 fine lake cells against 124,276 coarse |
| lakes inside erosion | none. `flood_every = 10` computes a *routing* surface (a priority flood with epsilon) that particles steer on so they cross a pit to its outlet; nothing is stored, nothing fills, nothing evaporates |

Each artifact follows directly:

* **Rivers through lakes on land bridges.** A depression the flood filled
  has high discharge across its whole flat, the smoothed threshold keeps
  it, the skeleton runs a centreline straight through, and the lake mask is
  not subtracted from the river mask. Where the fill is shallower than
  `lake_min_depth = 0.5 m` the cell is drawn as land, so one lake becomes
  several pieces with a river visibly bridging them.
* **Lakes without inflow or outflow.** The coarse graph has them
  (`earth-v3`: 1,805 `lake_in` and 1,625 `lake_out` nodes) but derive never
  reads them, and `graph/lakes.json` records only an `outlet` cell.
* **Hair-thin rivers.** `width = clip(2 · (Q / Q_thr)^0.5, 1, 24)` in fine
  cells, so anything at the threshold is one 4.9 km cell wide and a river
  at four times the threshold is four. In `earth-v3` the median river
  polyline is 12 cells long and order 1 is 62 % of all rivers. In the
  viewer the width was a byte smoothstep on the discharge texture and had
  no physical meaning at all.
* **Ruler-straight rivers on the plains.** D8 on a filled DEM routes a flat
  towards its flood parent, so across the planed lowlands (the ±50 m band
  that docs/missing-relief.md is about) the channel is a diagonal line for
  hundreds of kilometres. The river-order layer added below makes this
  obvious on `earth-v3`'s southern plain.
* **Basins stop at cube-face edges.** None of the 937 watersheds in
  `earth-v3` spans two faces (checked on `basin_id`); the basin layer shows
  the seams as straight meridians through the continent. That is the
  partition, not the flow: `flow_dir` is cross-face.

### What McDonald does, and what he says he would do

* The 2020 post (docs/ref, "Procedural Hydrology: Dynamic Lake and River
  Simulation") floods **during** the particle loop: a drop that stops
  raises a test plane over an 8-way flood-fill of its pit until the volume
  under the plane matches the drop's volume; if the fill reaches a cell
  below the plane and below the start (a drain) the drop and its overflow
  are moved to the drain and carry on descending, dropping 90 % of their
  sediment on the way. Pools are a map that later drops interact with
  (they stop on entering one; friction and evaporation are reduced on
  established streams). He gives the reason for doing it inside the loop:
  it makes the water time-dependent and lets particles interact.
* The 2023 post (the model this bake uses) **removed lakes entirely**:
  "I decided to remove lakes from the simulation entirely until I revisit
  them in the future", because the flood-fill was expensive, its surface
  was decoupled from the simulation's timescale, and the drainage edge
  cases were intractable. His stated plan is a droplet that carries both
  soil and water with an equilibrium mass-transfer law, so water moves
  between the particle and a map representation and "when a particle flows
  over and out of a lake, it can carry that water with it naturally". He
  also lists sea level and shores as the thing he has to do before calling
  the work done.

Our erosion loop is already halfway to the 2020 version: the routing flood
is a priority flood over the whole grid every ten iterations, and particles
do cross lakes on it. What is missing is that the surface is not water.

### The design: one water surface, owned by erosion

1. **Lake level is erosion state.** Alongside `route`, keep a `lake_level`
   per depression, solved every `flood_every` by the same balance hydro
   uses (`hydro/balance.py`): inflow from the discharge EMA against
   evaporation over the cells under the level. Between solves the level is
   fixed. A depression whose balance is above its spill point is
   overflowing and its level *is* the spill point, as now.
2. **Particles treat it as water.** Entering a lake a particle deposits its
   load over the first cells it crosses (the delta), then rides the flat
   to the outlet as it does today on `route`; erosion is disabled below
   the lake level (a local base level), and the outlet cell erodes under
   the overflow discharge, so a lake drains itself by cutting its sill,
   which is how most real lakes end. The glacial pass may still deepen a
   basin below its level.
3. **Hydro reads, it does not recompute.** The flood in `hydro/run.py`
   becomes a check; `water_surface`, the lake polygons and the graph's
   `lake_in` / `lake_out` nodes come from the state erosion left.
4. **Refine inherits the level.** `basin_job` currently re-floods each basin
   to the spill point and throws the coarse balance away. It should take
   the coarse `lake_level` as a boundary condition, fill to *that*, and
   erode the fine grid against it. That closes the 4x gap in fine lake
   cells.
5. **Derive traces rivers from the graph, not from a threshold.** Reaches
   from `flow_dir` and `flow_acc`, split at every `lake_in` / `lake_out`,
   with the lake polygon from the same water surface. Width from discharge
   with a floor in metres (a first-order stream is tens of metres, not a
   cell), drawn as a line with a width attribute rather than burned into
   a raster; at 9.8 km cells every river on the planet is sub-cell anyway.

Cost: the balance solve is one sort per depression and already runs in
hydro in 0.13 s at Earth scale; the routing flood already runs. The new
kernel work is the deposit-on-entry and the no-erosion-below-level tests,
both inside the existing per-particle sample.

### What the viewer does now

The viewer's rivers now come from hydro's `flow_acc` on the final frame,
drawn at exactly hydro's `river_threshold_volume`, coloured by Strahler
order; before hydro has run (the timeline frames) they still come from the
erosion discharge. Three layers were added: **Satellite** (below),
**River order** (the graph's reaches painted by order, 1 pale to 6 deep)
and **Basins** (the watershed partition). Lakes and rivers on the final
frame now agree by construction, because they are the same flood.

## 3. The fringed coast is noise crossing sea level, not a delta

The light, fingered fringe along every shore is fine cells within a few
metres of zero. Three things put them there:

* `refine/upsample.py` samples the coarse surface with a bicubic spline and
  then adds **ridged** fbm detail noise to land only, with amplitude
  proportional to the coarse slope. At the shelf step the slope is the
  largest on the map, so the noise is tens of metres, ridged noise is a
  field of sharp creases at the two-cell wavelength, and it is added to a
  surface that hovers around zero. The ocean side gets none of it, so the
  fringe is one-sided.
* Nothing can be deposited in the top 195 m of the water column
  (`DEP_FLOOR` is a length in cell units, docs/crust-audit.md), so the
  shelf never builds up to a coastal plain that would put the shoreline
  on a slope instead of on a flat.
* The fine sea is "below zero and within one coarse cell of open ocean", so
  each crease is its own inlet.

Fixes, cheapest first: taper the refine noise amplitude to zero within a
few metres of sea level and use plain rather than ridged noise there; put
`DEP_FLOOR` in metres so the shelf can fill; classify the fine sea by
connectivity. The satellite layer shades shallow water by depth so the
fringe reads as a shelf rather than as a delta, but the fringe is data and
the render only changes what it looks like.

## 4. Elevation plus biome: the satellite layer

Added to the viewer as the layer after Elevation. Ground colour is taken
continuously from the climate channels the frame already carries
(temperature and precipitation, the biome texture's sister channels), from
sand through dry grass and shrub to closed forest by rainfall, fading to
tundra and permanent snow with cold, with the ice, alpine and cliff classes
overriding; height then fades the ground to bare rock and snow. Water is
the water mask: lakes one blue, ocean from a shelf turquoise to abyssal
navy by depth. Rivers are drawn on top by order. It is a class-free
colouring on purpose: the biome raster's 30 km riparian corridors and
wetland halos are legend colours, not what the ground looks like, and
they dominated a first version that used the palette.

## 5. Run tectonics longer?

Answered in section 1: a 3000-step run does break the continent at step
2450 and keeps it apart, but the continent is 36 % of the segments by
then, and the five rifts before that one all healed. Length is not the
missing ingredient; the driving force is. Once slab pull is in, re-measure
at 1500 and 3000 and choose the step count from the cycle, not the other
way round.
