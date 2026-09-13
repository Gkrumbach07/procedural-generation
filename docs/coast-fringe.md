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
> gated now). What survives is section 1's diagnosis, `scripts/coastline.py`
> as the instrument, and section 4: the lace at the *coastline* is the
> narrow splat resolving per-segment history, which a margin ramp never
> touched. The fix wants to be in how the cloud is rasterised near a type
> boundary, measured at equal area and on more than one seed.

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

## 4. Not fixed here: the coastline's own lace

The ramp works below sea level and leaves the coast where it was (coast
L/sqrt(A) 12.917 -> 12.913).  The lacy shoreline on earth-v5 is a different
thing: the segment-scale residual (bed minus a 1.5-spacing Gaussian) is
245 m std on the drowned shelf and 186 m on low land against an 11 m/cell
regional slope, so a one-sigma worm displaces the shoreline ~22 coarse
cells (1.4 spacings).  That is the narrow splat resolving per-segment
history -- arc accretion, the taper, `continental_spread` -- not the type
boundary, and it is a separate item.
