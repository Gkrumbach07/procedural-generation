# Why the terrain reads as bland, measured

The recurring complaint is that this world's features are smooth and samey.
This is the measurement behind it. Metrics live in
`bake/scripts/terrain_stats.py`; `bake/scripts/sweep_dissection.sh` reproduces
the parameter sweeps.

Read the history section at the end before trusting any number quoted in
conversation: three separate metrics gave confident, wrong answers here before
being fixed, and each fix reversed a conclusion.

## What is actually wrong

**The relief is concentrated at the longest wavelengths.** The valley network
is in the right place; it just is not cutting deep enough to put relief at the
scales you look at.

Radial power spectrum, on a 256×256 all-land window of the fine grid (3.2 km
at 12.5 m), with an fBm control of known slope measured on the *same* window
and cell size:

| band | ours | fBm control (true β = 2.0) |
|------|------|---------------------------|
| 100–800 m | **5.62** | 1.77 |
| 200–1600 m | **6.93** | 1.69 |
| 400–3200 m | **7.29** | 1.61 |

Real topography sits near β = 2. Ours is 5.6–7.3, and it gets *worse* toward
longer wavelengths, which means power is overwhelmingly concentrated in the
continental swell with little left for the 100 m–3 km valley-and-ridge relief
that gives a landscape its texture.

The same thing in units you can picture — local relief, max − min in a moving
window:

| window | our local relief |
|--------|------------------|
| 125 m | 1.9 m |
| 500 m | 9.7 m |
| 1 km | 21.2 m |
| 2 km | 41.0 m |

Local relief at 1 km is **0.27** of the total relief across 3.2 km — expressed
as a fraction so a gentle region is not penalised for being gentle. A landscape
with valleys nested at every scale sits at 0.4–0.7. Ours is a smooth swell with
shallow scratches on it.

## The cause: erosion inherits a spectrum it cannot fix

Measured on the coarse grid of `final512`, over 200-1600 m:

| field | β |
|-------|---|
| bedrock leaving **tectonics** (erosion's input) | **10.68** |
| after erosion | 4.65 |
| fractal noise (what McDonald's sims start from) | 3.83 |

Erosion is doing real work — it pulls 10.7 down to 4.6 — but it cannot
manufacture variance that was never in its input, which is exactly why every
erosion parameter measured inert. McDonald starts from fractal noise, which
has roughly the right slope by construction. We start from a tectonic field
that is almost pure long-wavelength swell.

Tested directly on the `small` preset, same tectonics, same climate, same 60
iterations, changing only the erosion input
(`scripts/inject_bedrock_detail.py`):

| erosion input | β (200-1600 m) | relief |
|---------------|----------------|--------|
| tectonics as-is | 3.71 | 324 m |
| tectonics, glaciers and lakes **off** | 3.61 | 322 m |
| tectonics **+ fBm detail** | **1.84** | 298 m |

Two results there. Dropping our own additions is a no-op — they were never
the problem. Giving the bedrock a realistic spectrum moves β onto the
real-topography target in one step, at unchanged relief.

This is also the cheapest possible place to fix it: a field operation on the
coarse grid, `O(C)`, feeding a stage that already costs 6 s
(docs/pipeline-cost.md). It is not an architecture problem — running erosion
per-basin, in chunks, or globally makes no difference to it.

Before it can become a stage: the noise must be generated seam-aware on the
sphere (the test is per-face and will show cube edges), the amplitude and
low-frequency cutoff need tuning (0.15 overshot to β 0.30 at 800-6400 m), and
it perturbs the mass budget `hold_datum` balances (land fraction moved
14.4 % -> 14.0 %).

## What is right, and must not be "fixed"

Measured with slope–area analysis, the standard threshold-free way to locate
channel heads in a DEM (gradient rises with contributing area on hillslopes,
then reverses to S ~ A^-θ once flow is channelised; the turnover is the channel
head):

* **Channel head at 6,589 m² → 81 m hillslopes** on the fine grid. Real
  landscapes: 10³–10⁴ m², 30–100 m. **Inside the real range.**
* **Channel concavity θ = 0.45.** Real: 0.4–0.5. Textbook.

So the drainage network's geometry is correct. The problem is the amplitude of
erosional relief, not where the channels are.

**Refinement helps, it does not hurt.** Comparing coarse (50 m,
pre-refinement) against fine (12.5 m, post-refinement) over identical physical
bands, β falls by about 2.3 in every band. `block_drift` removes
coarse-cell-scale drift, but the net effect of the refine pass on the spectrum
is clearly positive.

**Not a missing technique.** Every fix from McDonald's 2023 follow-up
(*Procedural Hydrology: Improvements and Meandering Rivers*) is already in
`globe/erosion/particle.py`: the one-cell dynamic step, the finite-difference
normal, the stream-momentum coupling (a faithful port of his force law,
dot-product scaling included), and the discharge factor in `c_eq`. On lakes we
are ahead of him — he deleted them and still calls dynamic lakes unsolved.

**Not compute starvation.** 4× the particles: β 4.86 → 4.90, network unchanged.
A faster machine buys more iterations of something that is not converging
toward the missing quantity.

## Erosion parameters are inert

Drainage density does not respond to any erosion parameter tried —
`particles_per_cell` over 4×, `disc_saturation` (channel initiation) over 16×,
and the full range of `thermal_rate`/`creep_rate`/talus. That is consistent
with the slope–area result: the network is already where it should be, so the
knobs that would move it have nothing left to do.

What has *not* been swept is anything targeting relief **amplitude** at
100 m–3 km, which is the quantity actually missing. That is the next
experiment, with the 1 km relief ratio (currently 0.27, target 0.4–0.7) and β
(currently 5.6, target ~2) as the objective, and the talus angle as the
constraint.

## A caution on diffusion

`thermal_rate` and `creep_rate` do move β — but downward via small-scale noise,
not via finer valleys. Turning both off reaches β 2.86 at **80° slopes**
(heights are in cell units with `height_unit == cell_size`, so those are true
tangents), exactly the jagged, deep-ridged failure McDonald's Criticism 1
describes and which avalanching exists to prevent. Low β obtained that way is
noise, not landscape. `terrain_stats.py` flags any row whose p99 slope exceeds
the talus angle for this reason.

## History: three metrics that lied

Recorded because each was confidently reported before being caught, and each
reversed a conclusion.

1. **Drainage density as a discharge percentile.** "Cells above the 90th
   percentile" marks 10 % of cells by construction, so it returned an identical
   2.00 km/km² for every configuration. The tell was the suspicious constant.
   Fixed to a support threshold — which then reversed the diffusion result:
   less diffusion gives *fewer*, larger channels, not more.
2. **Discharge treated as contributing area.** `discharge` is an EMA of
   particle volume, not an upstream-cell count, so "support × mean q" was never
   an area threshold and the densities were not comparable to published values.
   This produced the bogus "5× under-dissected" claim. Fixed by computing real
   D8 flow accumulation, which showed the dissection is correct.
3. **β fitted below Nyquist.** The original 60–600 m band is 1.2–12 cells at
   50 m, so the small end is grid roll-off rather than terrain. The same field
   measured β 5.44 over 60–600 m and 2.18 over 400–3200 m. Always fit over a
   band the grid resolves, and always run a synthetic control of known slope on
   the same window and cell size.

The lesson worth keeping: a terrain metric that returns a suspiciously stable
number across configurations that plainly differ is broken, not insensitive.
