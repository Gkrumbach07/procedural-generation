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
> area on `small` seeds 0 / 1) and why it also ships off. Section 6
> removes the truncation without widening the kernel
> (`tectonics.splat_kernel`): a Wendland C2 kernel of matched width takes
> -5.0 / -2.7 / -1.8 / -7.5 % off the coast on `small` 0 / 1 and `tiny`
> 0 / 1 with the vertical scale within 1 %, but it makes tiny seed 1's
> fingers worse and shallows the sea 0.4-2.0 % on all four worlds, so the
> default stays `gaussian`; the tapered Gaussian is refuted.

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

## 6. Removing the truncation at matched width: `tectonics.splat_kernel`

> **Verdict: null result, the default stays `gaussian`.** A compactly
> supported Wendland C2 kernel whose width is matched to the truncated
> Gaussian takes the jump out and smooths the coast at equal area on all
> four test worlds (-5.0 / -2.7 / -1.8 / -7.5 %) with the vertical scale
> within 1 % and the belt p99 within 2.2 %, but it fails two of the
> shipping conditions: on `tiny` seed 1 the fingers get worse (6.00 ->
> 6.37 %) and the land and belt medians rise 8 %, and the sea median is
> shallower on every world (0.4-2.0 % in metres, 0.6-2.9 % in bedrock
> units) -- small, but systematic.  In Earth's shelf mode the coast gain
> nearly vanishes (-0.4 %, fingers worse).  The tapered Gaussian removes
> the jump too and is refuted: it narrows the kernel.

Section 5 blamed the lace on the truncation and removed it by widening
the kernel (48 neighbours), which also shallowed the sea and lowered
young belts.  What was left untried was a kernel whose weights reach zero
before the neighbour list ends, so a segment entering or leaving the list
contributes nothing at that moment, at the *same* width.  Both
candidates, behind `splat_kernel` (`globe/tectonics/collision.py`
`splat_weights`, used by `SmoothSplat`, built for every blend in
`splat_blends` -- the height, the crust-type fraction `c`, `dh` for
uplift, age and density for hardness -- so `crust_kind`, uplift and
hardness stay consistent with the bed, in `finalise` and in every
timeline frame):

* **`wendland`**: `w = (1 - d/h)^4 (4 d/h + 1)` for `d < h`, gathered from
  a kNN list sized `ceil(1.5 * expected + 16)` for the mean density and
  re-queried at twice the count, per chunk of 65536 cells, while any
  row's farthest listed neighbour is not beyond `h`.  So the whole support
  is always inside the list, by construction rather than by a margin.
* **`tapered`**: the Gaussian times `(1 - (d / d_12)^2)^2`, `d_12` the
  12th neighbour's distance, so the last listed neighbour weighs zero.

The manifest records `splat_kernel` and `splat_support_covered` (the share
of tect cells whose first list already held the whole support).
`'gaussian'` is the old code path byte for byte: stage hashes
`12861d1dbb782755` (`small` seed 0) and `30b47dc82ff71e76` (seed 1), the
same as sections 3 and 5.

### The width

The obvious matches for `h` are to the *untruncated* Gaussian: equal 2-D
second moment (`h = 3.795 sigma`) or equal half-weight radius (`3.752
sigma`).  Both are wider than the kernel in use.  At sigma = 1 spacing the
12 neighbours sit inside a disc of radius 1.95 sigma, which leaves the
Gaussian a second moment of 1.34 sigma^2 instead of 2 sigma^2.  Measured
at `h = 3.795 sigma`: the peaks come down, and the vertical scale (pinned
to the 99.9th land percentile) rises to compensate:

| `h = 3.795 sigma` | small 0 | small 1 | tiny 0 | tiny 1 |
|---|---|---|---|---|
| coast L/sqrt(A) at equal area | -8.1 % | -6.5 % | -4.8 % | -13.7 % |
| fingers | 1.25 -> 0.97 % | 1.11 -> 0.86 % | 9.91 -> 8.76 % | 6.00 -> 6.11 % |
| scale m/unit | +14.8 % | +17.8 % | +22.2 % | +20.7 % |
| land median / belt median m | +25 / +14 % | +14 / +14 % | +32 / +31 % | +39 / +41 % |
| sea median in units | 4.2 % shallower | 3.5 % | 7.9 % | 8.4 % |

That is section 5's wide kernel again, and it fails the same way.  The
width that matters for the peaks is the kernel's *central weight*: a
feature one spacing wide keeps about `w(0) / sum(w)` of its amplitude.
The 12-truncated Gaussian renormalised over its disc (`pi R^2 = 12
spacing^2`) has a peak of `1 / (2 pi sigma^2 (1 - exp(-R^2 / 2 sigma^2)))`
and the 2-D Wendland C2 has `7 / (pi h^2)`, so `h^2 = 14 sigma^2 (1 -
exp(-R^2 / 2 sigma^2))`, which is `h = 3.454 sigma`
(`collision.wendland_support`, from the design spacing so it is the same
in every frame).  Checked against the discrete cloud (scratch
`criteria.py`): the `h` at which the mean nearest-neighbour weight over
the tect cells equals the Gaussian's is 3.456 / 3.464 / 3.470 / 3.473
sigma.  A sweep around it in-process through the real `finalise` (scratch
`proto.py`) shows the change in vertical scale crossing zero there on all
four worlds, and the trade the verdict rests on:

| `h` / sigma | coast at equal area (s0 / s1 / t0 / t1) | scale | tiny 1 fingers (6.00 %) | sea median m, shallower by (negative: deeper) |
|---|---|---|---|---|
| 3.40 | -4.6 / -2.3 / -0.9 / -6.1 % | -2.0..-3.1 % | 5.79 % | 2.5 / 3.3 / 3.8 / 4.1 % |
| **3.45** | -4.9 / -2.5 / -1.8 / -7.5 % | -0.1..+0.7 % | 6.37 % | 0.8 / 0.6 / 1.8 / 2.1 % |
| 3.50 | -5.0 / -3.0 / -1.8 / -7.5 % | +2.0..+3.4 % | 6.32 % | -0.6 / -1.6 / 0.0 / 0.5 % |

No width fixes both: narrower brings tiny 1's fingers back but costs sea
depth and moves land and belt medians by up to 6 %, while wider deepens
the sea in metres only because the scale went up (in units it is still
shallower) and does not bring the fingers back.
`3.454` ships as the `wendland` width.

The list: the support holds a mean 32-34 segments (min 22, max 51) on
the four worlds.  The first list (65-67) covered 100.000 % of the cells
every time, and the farthest listed neighbour was at least 1.19 h, so
the re-query path is never taken there.  A list of just the expected
count would have covered only 48-55 %.
`test_wendland_gathers_the_whole_support` drives the re-query path on a
cloud with a cap 20x denser than the mean and checks the weights against
a brute-force radius query.

Continuity (scratch `jumps.py`): bisecting a walk to the moment the
12-nearest set changes, to 1e-12, at 400 random events, and reading `c`
(the 0/1 kind blend) on either side.  The Gaussian jumped at 89 of them
on `small` seed 0 and 165 on `tiny` seed 1, median 0.021 and max 0.038 /
0.039.  The tapered and Wendland kernels changed by at most 2.7e-12
(`test_splat_kernel_weights_are_continuous`, which also crosses the
Wendland support radius).

### Measured

Tectonics-only bakes, `small` and `tiny`, seeds 0 and 1 (from `bake/`):

    P=../.venv/bin/python; S=../scratch/kernel
    for preset in small tiny; do for seed in 0 1; do for k in gaussian wendland tapered; do
      $P scripts/bake.py --world $S/$preset$seed-$k --preset $preset --seed $seed --only tectonics --set tectonics.splat_kernel=$k
    done; done; done
    $P scripts/coastline.py --equal-area $S/small0-gaussian $S/small0-wendland $S/small0-tapered   # and small1, tiny0, tiny1
    $P scripts/hypsometry.py --preset small $S/small0-gaussian $S/small0-wendland $S/small0-tapered
    $P $S/world_metrics.py --preset small $S/small0-gaussian $S/small0-wendland $S/small0-tapered

(`world_metrics.py` reads the belts as collision-zone land, the medians,
the scale from the manifest and the residual: bed minus a 1.5-spacing
Gaussian, std over land more than 2 spacings from the `crust_kind`
boundary.)  The in-process numbers match the bakes to the digit.

`small` (spacing 7.46 coarse cells, opening disc r = 3.7):

| | s0 gaussian | s0 **wendland** | s0 tapered | s1 gaussian | s1 **wendland** | s1 tapered |
|---|---|---|---|---|---|---|
| coast L/sqrt(A) at equal area | 12.917 | **12.273 (-5.0 %)** | 14.200 (+9.9 %) | 13.497 | **13.136 (-2.7 %)** | 15.071 (+11.7 %) |
| coast fingers / inlets at equal area | 1.25 / 0.95 % | **1.11 / 0.88 %** | 1.88 / 0.76 % | 1.11 / 0.75 % | **1.10 / 0.65 %** | 2.03 / 0.91 % |
| equal-area level | | 0.0 m | -0.1 m | | 0.0 m | -0.0 m |
| shelf edge (first world's level) | 7.787 (-435 m) | 7.734 | 8.727 | 6.045 (-521 m) | 6.014 | 3.802 |
| crust boundary L/sqrt(A) | 7.859 | 7.860 | 8.801 | 6.065 | 6.067 | 7.026 |
| `crust_kind` cells changed | | 0.28 % | 1.76 % | | 0.26 % | 1.62 % |
| land % | 29.67 | 29.67 | 29.61 | 30.86 | 30.86 | 30.84 |
| scale m/unit | 7131.3 | 7142.9 (+0.2 %) | 4438.6 (-37.8 %) | 10173.6 | 10198.5 (+0.2 %) | 5036.8 (-50.5 %) |
| land median m | 35.0 | 36.1 (+3.1 %) | 20.0 | 62.1 | 60.8 (-2.1 %) | 30.4 |
| sea median m | -516.4 | -512.6 (0.7 % shallower) | -356.1 | -731.4 | -728.5 (0.4 % shallower) | -396.4 |
| sea median, bedrock units x 1000 | -72.41 | -71.77 | -80.23 | -71.89 | -71.43 | -78.70 |
| belt median / p99 / p99.9 m | 68.5 / 528.6 / 578.2 | 68.5 / 532.3 / 589.5 | 41.1 / 484.2 / 599.1 | 84.8 / 489.3 / 572.3 | 82.7 / 488.7 / 573.8 | 39.9 / 428.9 / 622.0 |
| residual std, land > 2 sp from the step | 55.6 m | 54.2 m | 40.9 m | 64.2 m | 62.5 m | 46.8 m |
| hypsometric bands | all 0-1 km | unchanged | unchanged | all 0-1 km | unchanged | unchanged |
| tectonics stage (61 frames + finalise) | 5.1 s | 7.4 s | 4.9 s | 4.8 s | 8.4 s | 3.4 s |

`tiny` (spacing 4.17 coarse cells, opening disc r = 2.1; land is ~1850
cells, so 0.4 % of fingers is 7 cells):

| | t0 gaussian | t0 **wendland** | t0 tapered | t1 gaussian | t1 **wendland** | t1 tapered |
|---|---|---|---|---|---|---|
| coast L/sqrt(A) at equal area | 7.861 | **7.720 (-1.8 %)** | 9.386 (+19.4 %) | 7.961 | **7.360 (-7.5 %)** | 8.494 (+6.7 %) |
| coast fingers / inlets at equal area | 9.91 / 4.04 % | **8.76 / 3.90 %** | 13.22 / 4.09 % | 6.00 / 3.51 % | **6.37 / 3.02 % (worse)** | 6.00 / 3.76 % |
| crust boundary L/sqrt(A) | 5.549 | 5.511 | 6.485 | 6.096 | 5.987 | 6.592 |
| `crust_kind` cells changed | | 0.54 % | 3.48 % | | 0.67 % | 3.32 % |
| land % | 29.56 | 29.56 | 29.61 | 30.39 | 30.40 | 30.47 |
| scale m/unit | 2148.4 | 2151.0 (+0.1 %) | 1082.8 (-49.6 %) | 3354.1 | 3384.8 (+0.9 %) | 1862.5 (-44.5 %) |
| land median m | 17.45 | 17.49 (+0.2 %) | 6.39 | 28.8 | 31.1 (**+7.9 %**) | 12.7 |
| sea median m | -111.9 | -110.0 (1.7 % shallower) | -68.1 | -163.0 | -159.7 (2.0 % shallower) | -108.7 |
| sea median, bedrock units x 1000 | -52.09 | -51.14 | -62.91 | -48.59 | -47.19 | -58.35 |
| belt median / p99 / p99.9 m | 21.8 / 275.7 / 309.7 | 21.8 / 278.4 / 312.0 | 7.8 / 262.7 / 312.5 | 28.8 / 279.1 / 312.2 | 31.2 (**+8.4 %**) / 285.2 / 310.0 | 12.6 / 266.2 / 311.6 |
| residual std, land > 2 sp from the step | 29.2 m | 26.7 m | 23.4 m | 18.7 m | 16.6 m | 11.1 m |
| tectonics stage | 0.70 s | 1.35 s | 0.75 s | 0.68 s | 1.30 s | 0.73 s |

`small` in Earth's sea-level mode (`shelf_fraction` 0.275, seed 0,
in-process): wendland coast 9.955 -> 9.912 (-0.4 %), fingers 1.07 ->
1.13 %, scale +0.1 %, land median 128.9 -> 129.5 m, sea median -415.6 ->
-412.3 m, belt median 120.8 -> 119.6 m, p99 530.1 -> 531.5 m.

### Reading it

* **The truncation is real, and it is not all of the lace.**  At matched
  width the Wendland kernel gets section 5's knn-48 coast gain on `small`
  (-5.0 / -2.7 % against -5.6 / -2.8 %) without knn-48's shift in the
  vertical scale (+0.2 / +0.2 % against -0.5 / +3.7 %).  So the jumps
  were worth about that much.  The coast is still 12.3 / 13.1 against the
  crust boundary's 7.9 / 6.1, and the crust boundary itself hardly moved
  (7.859 -> 7.860): `c` is continuous now, but its isolines still follow
  where the individual boundary segments are.
* **The tapered Gaussian removes the jump and makes the coast worse.**
  Cutting the Gaussian to zero at the 12th neighbour (~1.95 sigma) halves
  the weight at 1 spacing (the taper is 0.54 there), so the kernel narrows
  a lot.  Belts sharpen (the top of land in units rises and the scale
  drops 38-50 %), the step's flank steepens, and the coast resolves the
  individual segments more, not less (+7..+19 %).  This is the
  `splat_sigma_factor` lesson again: a kernel narrower than its samples
  resolves the samples.  The residual away from the step drops in metres
  only because the scale did; in bedrock units it rises 18 / 47 %.
* **The sea.**  A kernel that reaches 3.45 spacings, even with little
  weight out there, averages continental base into the oceanic side
  within a couple of spacings of every margin, and on `small` and `tiny`
  nearly all of the ocean is that close.  The shallowing is 0.6-2.9 % in
  units: about a quarter of knn-48's on `small` seed 0 (0.9 % against
  ~3.5 %), but unlike knn-48's it points the same way on every world.
  On Earth, with 31 % of the oceanic crust within 3 spacings of a margin,
  expect the effect concentrated there.
* **`tiny` seed 1.**  The largest coast gain (-7.5 %) comes with 7 more
  cells of fingers and land / belt medians +8 %.  On a 300-segment world
  the land median is ~29 m against ~310 m of relief, and one coast
  segment's worth of lowland moves it.  It is a real miss of a stated
  condition, not one to explain away.

### Cost

Per splat build on `small` 31 ms against 5 ms, one blend 2.9 ms against
0.7 ms (37 columns).  The stage pays that in 61 builds, +2.3 / +3.5 s on
~5 s.  At Earth scale (synthetic 17286-point cloud, 6 x 256^2 cells, the
Earth defaults' design spacing): a build takes 0.35 s against 0.08 s and a
blend 45 ms against 18 ms, with `nb` + `w` at 226 MB against 75 MB (36
columns on the regular synthetic cloud; the real clouds on `small` needed
up to 51) and +350 MB of transient peak after chunking the gather.  Across
1 finalise + 60 frames that is about 20 s more on a ~170 s stage.
`tapered` costs what `gaussian` does.

### Measured at Earth scale, four seeds

The test this section asked for. Tectonics-only `earth` bakes with
`splat_kernel = wendland` at seeds 0-3, against the `gaussian` baselines
of docs/land-median.md (same code, same parameters otherwise):

    python scripts/bake.py --world scratch/floor/wendland_sS --preset earth --seed S --only tectonics --set tectonics.splat_kernel=wendland
    python scripts/coastline.py --equal-area scratch/floor/floor1200_sS scratch/floor/wendland_sS
    python scripts/hypsometry.py scratch/floor/floor1200_sS scratch/floor/wendland_sS

| seed | coast L/sqrt(A) at equal area | fingers | crust boundary | land median | ocean median | max | 2-3 / 3-4 / 4-5 km |
|---|---|---|---|---|---|---|---|
| 0 | 25.965 -> 25.277 (-2.6 %) | 0.27 -> 0.25 % | 25.358 -> 25.038 | 786 -> 771 m | -3227 -> -3211 m | 6747 -> 6638 m | 10.2 / 3.8 / 0.9 -> 10.2 / 3.6 / 0.9 |
| 1 | 25.145 -> 24.340 (-3.2 %) | 0.29 -> 0.25 % | 21.492 -> 21.287 | 264 -> 273 | -2682 -> -2661 | 7283 -> 6866 | 5.1 / 1.8 / 0.5 -> 4.9 / 1.8 / 0.4 |
| 2 | 26.528 -> 25.896 (-2.4 %) | 0.33 -> 0.34 % | 20.974 -> 20.799 | 252 -> 253 | -2875 -> -2862 | 7200 -> 6906 | 4.3 / 1.0 / 0.5 -> 4.2 / 1.0 / 0.5 |
| 3 | 24.315 -> 23.904 (-1.7 %) | 0.26 -> 0.25 % | 21.215 -> 21.115 | 340 -> 347 | -2730 -> -2711 | 4865 -> 4744 | 3.3 / 1.3 / 0.2 -> 3.2 / 1.3 / 0.1 |

In shelf mode at Earth scale the gain holds on every seed, unlike the
`small` shelf-mode arm (-0.4 %), and the fingers are not worse. It is
also small: 1.7-3.2 %, below anything the viewer shows. The sea is
shallower on all four seeds by 13-21 m (0.5-0.8 %), the same systematic
direction as on `small` and `tiny`, and every world's highest point
comes down 109-417 m (2-6 %) while the bands barely move. The default
stays `gaussian`: the kernel removes a real artefact whose visible share
of the lace turns out to be a few per cent, at the cost of the peaks and
a shallower sea.

### What would change the verdict

The same Earth-scale tectonics-only bake section 5 asks for, run against
`worlds/earth-v7`: coast L/sqrt(A) and fingers at equal area in shelf
mode (where `small` gained only -0.4 %), `scripts/ocean_depth.py` for the
margin-adjacent sea, and the coastal belt heights.  Without a gain that
survives shelf mode, the sea and `tiny` seed 1 decide it.

### Hashes

`splat_kernel` is a `TectonicsParams` field, so `content_hash` and
`group_hash('world', 'tectonics')` change for every existing world, as
with the fields before it.  At `'gaussian'` the bedrock is bit-identical
(the stage hashes above), so erosion checkpoints do not move.  The
erosion `KERNEL_VERSION` is not touched.
