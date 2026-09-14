# The coast fringe: smoothing the continental margin

> **Verdict: the ramp ships off** (`margin_sigma_factor = 0`, bit-identical
> to the previous output). Two adversarial reviews of the change below
> found that its headline number does not hold up: the -18 % in the
> shelf-edge isoline length on `small` is read at a fixed depth while the
> ramp shallows 62 % of the sea (sea median -516 -> -407 m), so the mask
> grows and the isoline shrinks whatever its shape; at equal area the
> change is -5 % with the finger fraction unchanged. The correction also
> has no compact support (a Gaussian tail of 2.5 spacings reaches 31 % of
> Earth's ocean floor), costs +30 % of the tectonics stage through the
> timeline frames, and the step it re-positions is not sign-definite
> (young, buoyant ocean floor stands above a thinned margin base; that is
> gated now). What survives is section 1's diagnosis and
> `scripts/coastline.py` as the instrument. Section 4's reading of the
> *coastline's* lace as per-segment height history was wrong and is
> retracted there: the lace is the flank of the crust-type step as the
> truncated 12-neighbour blend resolves it. Section 5 has the one lever
> that reduces it with the crust boundary bit-identical
> (`tectonics.splat_knn_base`, -5.6 % / -2.9 % coast L/sqrt(A) at equal
> area on `small` seeds 0 / 1) and why it also ships off.

The viewer showed a spiky, fingered light-blue band along every coastline,
in tectonics-only bakes as much as in finished worlds (docs/earth-v3-review.md
section 3 blamed refine's noise and then erosion's floor before
docs/lakes-in-erosion.md section 4 traced it to the tectonics stage).  This
note is the diagnosis on data, the change (`tectonics.margin_sigma_factor`,
`margin_ramp` in `globe/tectonics/run.py`) and what it measured.

## 1. Where the band comes from

The height field is *not* rasterised through the nearest-segment label
map -- `label_map_fast` only splats plate ids and collision zones.  It is
the kNN Gaussian blend `SmoothSplat` (12 nearest segments, sigma one
segment spacing) of the per-segment height, and that blend is exactly

    bed = c * h_cont + (1 - c) * h_ocean

with `c` the continental fraction of the nearest segments.  `c` steps from
1 to 0 over about one spacing, at the positions of the individual boundary
segments, and the continental/oceanic step it multiplies is the largest
step on the map (~0.06 bedrock units; 1.5 km on Earth).  So the shelf edge
is resolved one segment at a time and comes out scalloped at the 160 km
scale.  The viewer shades the sea by `(depth / ocean_bottom)^0.6`, so the
light band is the drowned margin and its outer edge is the -1000..-2500 m
isolines.

Measured on `worlds/earth-v5` (read-only), distance from the isolines to
the crust-type boundary in segment spacings:

| isoline | median distance | within one spacing |
|---|---|---|
| coast (0 m) | 1.05 | 49 % |
| -1000 m | 0.49 | 77 % |
| -1500 m | 0.39 | 78 % |

The band is ~1.2 spacings (~200 km) wide at the coast; oceanic crust within
1 / 2 / 3 spacings of a margin is 11 / 22 / 31 % of the oceanic crust.

The margin taper itself (`plates.seed_supercontinent`: the outer 35 % of
the continent at 0.45x thickness) is per-segment *mass* -- it feeds
collisions, the cascade, the orogeny floors and the ledger -- so moving it
onto the label map ("approach B") would change the simulation and every
number tuned in docs/plate-forces.md.  It is left alone.

## 2. The change: re-position the step, raise-only

`margin_ramp` keeps the narrow blend everywhere and adds a correction near
the type boundary.  `cw` is `c` blurred on the tect grid by
`margin_sigma_factor` spacings (a Gaussian on the FaceField, no second kNN
query); the correction

    corr = (cw - c) * (h_base_cont - h_ocean)

is what the blend would have added had it seen the boundary at the wide
resolution.  Two constraints shape it:

* **Belts pass through untouched.**  Only the continental *base*
  (`min(h, 0.18)`, the nominal belt/craton height both float at) enters
  the step; anything above it is orogenic thickening and keeps the narrow
  resolution.  Without the split the ramp lifted the belts' surroundings and
  the vertical scale (re-derived from the 99.9th land percentile) moved
  +50 m/unit on tiny.
* **Raise-only.**  `max(corr, 0)`: where `cw > c` (the oceanic side) the
  bed climbs to the ramp -- a continental rise; where `cw < c` (the
  continental side) nothing happens.  So land, belts and sea level are as
  the blend made them, and the correction is exactly zero where `cw == c`
  (the interior of either crust).

The narrow `c` is what `crust_kind` and the shelf mask use, so the
crust-type boundary and the placement of sea level in shelf mode are
unchanged.  `frame_bed` calls the same helper, so the viewer's timeline
frames stay identical to the finished map.  `margin_sigma_factor = 0` is
bit-identical to the old output (verified: the `small` stage hash
`12861d1dbb782755` with and without the field).

### The symmetric variant, and why it is not the default

Dropping the `max()` smooths more (shelf edge L/sqrt(A) 7.79 -> 4.35 on
`small`) but lowers the continental side: sea level -67..-92 m on `small`
and -273..-340 m on an Earth-parameter toy, land median +193 m, belt bases
-160 m median / -430..-530 m p10 inside the ramp (cross-sections keep their
shape, the absolute height drops), and it lightens the whole ocean in the
viewer.  It fails the "belts and away-from-coast untouched" constraints and
would move the hypsometry docs/lakes-in-erosion.md measured.

## 3. Measured

`scripts/coastline.py` reports, for the mask `bedrock > level`, `L/sqrt(A)`
(land/sea 4-neighbour edges within faces over the square root of the cells
above the level; a disc is 3.54) and the *finger fraction* (share of the
mask removed by a binary opening with a disc of radius half a segment
spacing -- area in spits and scallops narrower than one segment), at the
coast, at fixed depths and at the shelf edge (midpoint of the median
drowned-continental and median oceanic-crust bed, taken from the first
world so a before/after pair is read at one level), plus the crust-type
boundary as the must-not-change control.

`small`, seed 0, tectonics only (spacing 3.73 tect = 7.46 coarse cells,
opening disc r = 3.7):

    python scripts/bake.py --world /tmp/cf-before --preset small --seed 0 --only tectonics --set tectonics.margin_sigma_factor=0
    python scripts/bake.py --world /tmp/cf-after  --preset small --seed 0 --only tectonics
    python scripts/coastline.py /tmp/cf-before /tmp/cf-after

| | f = 0 (before) | f = 2 | **f = 2.5 (default)** | f = 3.5 |
|---|---|---|---|---|
| shelf edge (-435 m) L/sqrt(A) | 7.787 | 6.949 | **6.384** (-18 %) | 5.331 |
| shelf edge fingers / inlets | 0.75 / 0.71 % | 0.65 / 0.66 % | **0.63 / 0.66 %** | 0.50 / 0.62 % |
| -500 m L/sqrt(A) | 7.531 | 6.077 | **5.395** | 4.332 |
| coast L/sqrt(A), fingers | 12.917, 1.25 % | 12.912, 1.23 % | **12.913, 1.22 %** | 12.931, 1.20 % |
| crust boundary L/sqrt(A) | 7.859 | 7.859 | **7.859** | 7.859 |
| tect cells raised / median lift | 0 | 36.8 % / 71 m | **37.9 % / 95 m** | 39.7 % / 139 m |
| tectonics stage, 61 timeline frames + finalise | 4.7 s | 11.4 s | **16.9 s** | 29.9 s |

At f = 2.5 on `small`: sea level +0.00003 units (+0.2 m), scale 7131 ->
7132 m/unit, land fraction 29.67 -> 29.66 %, land median 35.0 -> 34.9 m,
collision-zone land median 68.5 -> 70.3 m and p99 528.6 -> 529.1 m.  The
raw ramp is exactly zero on land; the residual land change (|d| median
0.2 m, p99 10 m, max 62 m, all within a couple of coarse cells of a shore)
is the existing 1-cell Gaussian and cascade downstream of it.  The sea
side is the point: 60 % of sea cells rise by > 1 m, the sea median goes
-516 -> -407 m and the ocean-crust median -718 -> -589 m -- on `small`
nearly all the ocean is within 3 spacings of a margin.  On Earth 31 % of
the oceanic crust is, already at a median -3001 m against -3654 m beyond
(earth-v5), so expect roughly +100 m on the ocean median and up to several
hundred metres right beside margins.  docs/crust-audit.md already flags
"the model has no abyss"; check `scripts/ocean_depth.py` on the next Earth
bake.

Earth-parameter toy (N_c 128, N_tect 64, 1500 segments, 300 steps, shelf
mode, 26400 m/unit), prototype: shelf edge (-1534 m) 5.496 -> 4.779
(-13 %), fingers 0.38 -> 0.29 %, sea level +13 m (raised cells that are
drowned continental: bays), land median 685 -> 672 m, sea median
-2082 -> -1783 m, belt cells +0 m.

`worlds/earth-v5` as baked (before the change; the scale the viewer
showed): coast 25.97 / 0.27 %, -1000 m 23.88 / 0.28 %, shelf edge
(-1962 m) 21.75 / 0.19 %, -2500 m 25.42 / 0.26 %, crust boundary 25.36 /
0.50 %.  Seeing the change at that scale needs a full rebake.

### Cost

Three Gaussian blurs at sigma = f x spacing in tect cells; the per-pass
radius is capped at H - 1 = 3 cells so the pass count grows as sigma², and
the timeline frames pay it too (`frame_bed`, 60 frames at Earth):

| | f = 2 | f = 2.5 | f = 3.5 |
|---|---|---|---|
| N_tect 256 (Earth), per call | 0.52 s (67 passes) | 0.81 s (105) | 1.65 s (205) |
| x (1 finalise + 60 frames) | 32 s | 49 s | 101 s |
| N_tect 64 (`small`), per call | 0.11 s | 0.20 s | 0.41 s |

About +50 s on the 179 s Earth tectonics stage.  If that ever matters,
blur on the N_tect/4 heat grid and resample; do not drop it from
`frame_bed`, the frames must match the final map.

### Hashes

`margin_sigma_factor` is a `TectonicsParams` field, so `content_hash` and
`group_hash('world', 'tectonics')` change for every existing world: `bake`
refuses them without `--force`, and `--force` reruns from tectonics.
Erosion checkpoints invalidate themselves through the tectonics output
hash; the erosion `KERNEL_VERSION` is not touched.

## 4. The coastline's own lace (retracted reading)

The ramp works below sea level and leaves the coast where it was (coast
L/sqrt(A) 12.917 -> 12.913).  The first version of this section read the
lacy shoreline on earth-v5 as the narrow splat resolving *per-segment
history* -- arc accretion, the taper, `continental_spread` -- from a
segment-scale residual (bed minus a 1.5-spacing Gaussian) of 245 m std on
the drowned shelf and 186 m on low land against an 11 m/cell regional
slope.  Both halves of that were wrong, and this section is the corrected
measurement (scratch scripts `earth_residual*.py`, `decompose.py`; the
reference worlds read-only).

**The residual was mostly the step.**  On `worlds/earth-v5`, bed minus a
1.5-spacing Gaussian: drowned shelf 380 m and low land (0-500 m) 307 m
*including* the step; more than 2 spacings from the crust-type boundary
169 m and 141 m; open oceanic crust > 2 spacings 161 m; within 1 spacing of
the boundary 558 m.  The within-kind worm amplitude is ~150 m std on both
crusts, not 245 / 186, and the residual-over-slope arithmetic away from
the step is 141 m / 4.8 m per cell = 29 cells = 1.8 spacings.

**The coast sits on the step.**  Coast cells are a median 0.58 spacings
from the crust-type boundary (EDT of a 3x3-eroded `crust_kind` boundary),
70 % within one spacing, 94.7 % of them on continental crust; on every
Earth world the coastline is about as convoluted as the crust-type boundary
itself (coast / boundary L/sqrt(A): earth-v5 25.97 / 25.36, sweep-od006
28.16 / 26.12, sweep-od008 26.76 / 22.81, earth-v4 18.18 / 17.15).

**Within-kind history is not the cause.**  With every continental *base*
height (`min(h, belt height)`) replaced by the continental mean, belts kept
and every oceanic height replaced by the oceanic mean -- zero within-kind
history -- the coast comes out *rougher*: `small` seed 0 12.917 -> 15.621,
seed 1 13.497 -> 14.082, tiny 7.861 -> 8.111 and 7.961 -> 7.437.  What is
left when the heights are constants per kind is `c`, the Gaussian-kNN blend
of the 0/1 kind labels, whose flank isolines resolve the individual
boundary segments: the lace is the step's flank, and the coast sits on it.

**The ceiling, and the candidates that smooth within-kind heights.**  A
global low-pass of the earth-v5 bedrock at equal land area gives coast
L/sqrt(A) 25.97 -> 24.13 / 21.46 / 19.50 / 17.56 at sigma 0.5 / 1 / 1.5 /
2 spacings (`small` seed 0: 12.92 -> 11.74 / 10.26 / 8.63 / 7.54), but that
low-passes the step itself, which section 2 refuted.  Everything that
smooths the heights but keeps the step (5 test worlds: `small` seeds 0 and
1, `small` with `shelf_fraction` 0.275, tiny seeds 0 and 1; equal-area
coast L/sqrt(A), crust_kind must be identical):

| candidate | coast L/sqrt(A) | what else moved | verdict |
|---|---|---|---|
| (b) same-kind segment-graph average of the base near a type boundary, 1-3 passes | -1..-5 % (small), -4..-8 % (tiny 1) | vertical scale +3..+13 % (boundary belt segments had their base averaged); fingers 1.20..1.32 % vs 1.25 | null |
| (a) per-kind wide (2.5 sp, knn 48) blend of the base, narrow residual attenuated within 1-2 sp of the boundary | -1 % | scale +8..+9 %; fingers 1.25..1.38 % | null |
| (c) sigma 1.5 (knn 24), `c` kept narrow | -14 % | scale +79..+103 %, land median 35 -> 81 m, land in the 0.4-0.6 km band 4.3 -> 20.8 % | fails |
| (d) `c` = Phi(signed Voronoi distance / s), s fitted to the existing `c` | 12.92 -> 12.94..13.61, shelf edge 7.79 -> 9.6 | worse everywhere | refuted |
| Voronoi-area weighting of the kNN weights | 12.85 / 13.58 / 9.88 / 7.98 / 7.87 vs 12.92 / 13.50 / 9.96 / 7.86 / 7.96 | scale -6..-20 % | null |
| knn 48 for the whole height, `c` from knn 12 | -7.9 / -6.0 / -2.8 / -4.5 / -13.7 % | scale +8..+15 % (the peaks drop ~10 % in units) | see section 5 |

The symmetric wide `c` (section 2's refuted variant) as a diagnostic:
-14 % at 1.5 spacings and -8..-29 % at 2.5, with the land median 35 -> 55
-> 66 m and the sea median -516 -> -381 -> -277 m.  The kNN blend's own
irregular-sampling error on a unit ramp is 0.09 spacings of ramp on
`small` and 0.12-0.13 on tiny.

## 5. The truncation: `tectonics.splat_knn_base`

At sigma = 1 spacing the 12th nearest segment still carries 2.6 % mean /
5.7 % max of the normalised weight (`SmoothSplat` with `splat_knn` 12, on
every test world).  So `c` and the bed jump whenever a segment enters or
leaves the twelve, and on the flank of the step those jumps are the lace.
Blending the whole height from 48 neighbours removes them (previous row of
the table) but also lowers the peaks ~10 % in units, because a belt one
spacing wide averaged over a 48-segment disc is flattened.  The lever that
keeps the belts is the split `margin_ramp` already had for the step
height:

    base   = min(h, h_belt) on continental crust, h on oceanic
    bed    = blend_N(base) + blend_12(h - base)
    c      = blend_12(kind)                 (crust_kind and the shelf mask: bit-identical)

`splat_knn_base = N` (default 0 = off, the old `blend_12(h)` byte for
byte: the `small` stage hash `12861d1dbb782755` seed 0 and
`30b47dc82ff71e76` seed 1 with the field at 0 equal the bakes before the
field existed).  `finalise` and `frame_bed` build the pair through one
helper (`splat_blends`), so the timeline's last frame is the finished map;
`dh`, age and density stay on the narrow blend, so uplift and hardness are
untouched.  The manifest records `splat_knn_base` (12 when off) and
`splat_base_fraction` (|base| / (|base| + |excess|) of the rasterised bed;
0.966 / 0.971 on `small`).

### Measured

`small`, seeds 0 and 1, tectonics only (about 3-4 s each, 61 timeline
frames + finalise; +1.1 s with the knob on):

    P=.venv/bin/python; S=scratch/lace
    $P scripts/bake.py --world $S/small0-off --preset small --seed 0 --only tectonics --set tectonics.splat_knn_base=0
    $P scripts/bake.py --world $S/small0-on  --preset small --seed 0 --only tectonics --set tectonics.splat_knn_base=48
    $P scripts/bake.py --world $S/small1-off --preset small --seed 1 --only tectonics --set tectonics.splat_knn_base=0
    $P scripts/bake.py --world $S/small1-on  --preset small --seed 1 --only tectonics --set tectonics.splat_knn_base=48
    $P scripts/coastline.py --equal-area $S/small0-off $S/small0-on
    $P scripts/coastline.py --equal-area $S/small1-off $S/small1-on
    $P scripts/hypsometry.py --preset small $S/small0-off $S/small0-on $S/small1-off $S/small1-on

`--equal-area` reads the after-world at the level that gives it the
before-world's land cell count (the reviewers' first condition on section
3; here that level is 0.0 m on seed 0 and 0.1 m on seed 1, so the physical
coast and the equal-area coast are the same to three figures).

| | seed 0 off | seed 0 **48** | seed 1 off | seed 1 **48** |
|---|---|---|---|---|
| coast L/sqrt(A) at equal area | 12.917 | **12.197** (-5.6 %) | 13.497 | **13.113** (-2.8 %) |
| coast at 0 m | 12.917 | 12.195 | 13.497 | 13.108 |
| coast fingers / inlets | 1.25 / 0.95 % | **0.94 / 0.92 %** | 1.11 / 0.75 % | **0.99 / 0.67 %** |
| coast cells | 1537 | 1446 | 1678 | 1643 |
| shelf edge (first world's level) L/sqrt(A) | 7.787 (-435 m) | 7.592 | 6.045 (-521 m) | 5.938 |
| own shelf edge | -435.0 m | -423.5 m | -521.2 m | -538.8 m |
| -500 m L/sqrt(A) | 7.531 | 7.414 | 6.082 | 5.986 |
| crust boundary L/sqrt(A), fingers | 7.859, 0.65 % | **identical** | 6.065, 0.33 % | **identical** |
| `crust_kind`, `collision_zone` | | array_equal | | array_equal |
| land % | 29.67 | 29.68 | 30.86 | 30.88 |
| scale m/unit (top of land in units) | 7131.3 | 7095.6 (-0.5 %) | 10173.6 | **10549.8 (+3.7 %)** |
| land median / mean m | 35.0 / 53 | **38.0** / 55 | 62.1 / 79 | 63.1 / 81 |
| sea median m | -516.4 | **-495.6** | -731.4 | -735.6 |
| +-50 m of sea, % of globe | 30 | 28 | 23 | 23 |
| belt (collision-zone land) median / p90 / p99 / p99.9 m | 68.5 / 327 / 528.6 / 578 | 71.4 / 329 / 527.3 / 580 | 84.8 / 232 / 489.3 / 572 | 87.6 / 243 / 496.2 / 576 |
| collision zone, all cells, median / p99 m | -521 / 370 | -502 / 368 | -785 / 330 | -793 / **347 (+5 %)** |
| land |d| median / p99 / max m | | 6.5 / 26 / 42 | | 6.4 / 39 / 64 |
| sea d median (p10 / p90) m | | +12.3 (-5 / +32) | | -13.3 (-33 / +7) |
| hypsometric bands | all 0-1 km | unchanged | all 0-1 km | unchanged |

The same split measured in-process on the other test worlds (prototype
through the real `finalise`): `small` with `shelf_fraction` 0.275 (Earth's
sea-level mode) 9.955 -> 9.831 (-1.2 %), fingers 1.07 -> 1.09 %, scale
-0.6 %; tiny seed 0 7.861 -> 7.767 (-1.2 %), fingers 9.9 -> 9.6 %; tiny
seed 1 7.961 -> 7.103 (-10.8 %), fingers 6.0 -> 6.3 %; an Earth-parameter
toy (Earth defaults, N_c 128, N_tect 64, 1500 segments, 300 steps, shelf
mode, fixed 26400 m/unit) seed 0 3.469 -> 3.430 (-1.1 %), fingers
0.36 -> 0.42 %, land median 265 -> 292 m, sea median -2464 -> -2430 m, no
collision zone formed; seed 1 4.541 -> 4.425 (-2.6 %), fingers 0.32 ->
0.31 %, land median 214 -> 217 m, **belt median 285 -> 257 m (-10 %), belt
p99 462 -> 401 m (-13 %)**, top p99.9 428 -> 433 m.  `knn 24` for the base
gets 60-90 % of the knn-48 gain (the 24th neighbour's weight is 0.003);
knn 96 = knn 48.

### Where it hurts, and why it ships off

* It is a mild version of section 2's refuted symmetric ramp: the wider
  kernel puts more weight at 2-4 spacings, so the continental side near
  the margin comes down and the oceanic side up.  Land median +1..+10 %
  and sea median 1..4 % shallower across the test worlds (small seed 0:
  land 35 -> 38 m, sea -516 -> -496 m); on seed 1 the sea median went the
  other way (-731 -> -736 m) while the -1000 m isoline lengthened
  (3.79 -> 4.51).
* Fingers are not consistently better: down on small 0 / 1 and tiny 0, up
  on tiny 1, the shelf-mode small and the Earth toy seed 0, flat on toy
  seed 1.  The reviewers asked for this explicitly; it is not delivered.
* The belts: on `small` the belt-land p99 is within 0.3 / 1.4 % and the top
  of land within 0.5 % on seed 0, but seed 1's top of land (in units) drops
  3.7 % and its all-cell collision-zone p99 rises 5 %; the 300-step toy,
  whose belts barely stand above the base (`h_belt`), lost 10-13 % of them
  -- a young accretionary belt is mostly *base* under this split and the
  wide kernel averages it with its surroundings.  On Earth's fixed 26400
  m/unit scale the coastal belts sit on a base the wide kernel lowers by
  ~0.005-0.009 units (130-240 m), and the 2-4 km bands docs/plate-forces.md
  tuned could move.
* Cost at Earth (synthetic 17286-segment cloud, 6x256x256): one k=48 query
  0.15 s against 0.07 s for k=12, one blend 0.067 s against 0.021 s, `nb`
  and `w` 302 MB transient against 75 MB; times 1 finalise + 60 timeline
  frames, about +10-15 s on a 164 s stage.  On `small` the stage went
  3.1 -> 4.3 s.

So `splat_knn_base` stays 0.  Default-on is contingent on an Earth-scale
tectonics-only bake with 48 against `worlds/earth-v5/coarse/bedrock`
showing, with `coastline.py --equal-area`, `hypsometry.py` (2-4 km bands,
belt heights via `collision_zone` on land) and `ocean_depth.py`: coastal
belt heights and the 2-4 km bands unchanged, fingers not worse, and the
coast L/sqrt(A) gain surviving Earth's shelf mode (where `small` gained
the least, -1.2 %).  Earth-v5 before numbers to compare against: coast
25.965 / 0.27 %, shelf edge (-1962 m) 21.749 / 0.19 %, crust boundary
25.358 / 0.50 %, land 24.6 %, continental 33.5 %.

### Hashes

Like `margin_sigma_factor`, the new field changes `content_hash` and
`group_hash('world', 'tectonics')` for every existing world (they already
mismatch, their manifests predate `margin_sigma_factor`); with the knob at
0 the bedrock is bit-identical so erosion checkpoints, which invalidate
through the tectonics output hash, do not move.  The erosion
`KERNEL_VERSION` is not touched.
