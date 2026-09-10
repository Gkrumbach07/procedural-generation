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

### Does it survive erosion, and does it seed valleys?

Two follow-ups, both run at production length (400 iterations) on the `small`
preset, then through refine:

**It survives.** The improvement is not washed out by a long run, though
erosion does eat some of it — the gap narrows from 1.87 at 60 iterations to
1.24 at 400:

| coarse, 400 iterations | 200-1600 m | 400-3200 m | relief |
|---|---|---|---|
| tectonics | 3.88 | 3.44 | 819 m |
| tectonics + fBm | **2.64** | **2.61** | 813 m |

**But it does not create channels.** Slope-area on the *fine* grid (25 m
cells, so a finer channel head was resolvable — the bins run down to 1.3
cells):

| fine grid | β (200-1600 m) | channel head |
|---|---|---|
| tectonics | 5.49 | 14,821 m² → 122 m |
| tectonics + fBm | **3.69** | 14,821 m² → 122 m |

Identical channel head. The added octaves put texture *between* the existing
channels; they do not seed new tributaries, and the drainage network is
unchanged.

So this fixes one kind of blandness and not the other. If the complaint is
"the surfaces are featureless", it is addressed and it lasts. If the
complaint is "every basin drains the same boring way", it is not — that is
network topology, and nothing measured so far moves it.

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


## The input spectrum is not the constraint — erosion's attractor is

This section supersedes "The cause: erosion inherits a spectrum it cannot
fix" above, which is **wrong** as an account of the operative limit.

`tectonics.detail_amp` (`inject_detail` in `globe/tectonics/run.py`) is the
shippable version of `scripts/inject_bedrock_detail.py`: seam-free 3-D
noise on the sphere, amplitude following *local* relief so plains stay
flat, and sea level re-derived afterwards so the land fraction holds
(25.1 % → 25.0 %, against the prototype's 14.4 % → 14.0 % drift).

Measured end to end at `N_c = 256`, 300 erosion iterations, R = 4,
β over 200–1600 m at three points in the chain:

| `detail_amp` | bedrock | **after erosion** | after refine | relief | lr@1000 |
|---|---|---|---|---|---|
| 0.00 | 13.09 | **6.21** | 6.26 | 627 m | 0.488 |
| 0.25 | **4.70** | **6.03** | 6.04 | 605 m | 0.495 |
| 0.50 | 5.04 | 4.69 | 4.82 | 648 m | **0.168** |

The injection does exactly what it is meant to — bedrock β falls from 13.09
to 4.70 — **and erosion removes it again.** A 8.4-point improvement in the
input becomes a 0.18-point improvement in the output. Turning the amplitude
up to 0.5 finally moves the eroded field (4.69), but at the cost of the
valley structure: the 1 km local-relief ratio collapses from 0.488 to
0.168, i.e. the nesting that made it a landscape is gone.

So the erosion stage **converges to β ≈ 6 from either side** — down from
13.09, and back up from 4.70. That is an attractor of the kernel and its
parameters, not a deficit inherited from tectonics. Feeding it a better
spectrum cannot fix it, which is also why every erosion parameter measured
inert against drainage density: the knobs that were swept are not the ones
that set where the attractor sits.

`detail_amp` therefore defaults to **0**. It is kept, working and
documented, because it is the correct implementation of an idea worth
having on file, and because a *low* setting is free — 0.25 costs nothing
and leaves the bedrock in better shape for anything downstream that reads
it directly.

Caveat on the measurement: the analysis window is chosen per world as the
highest-relief all-land patch, so the three rows are not the same ground.
Land fraction and total relief agree closely across them (25.0–25.3 %,
605–648 m), so the comparison is fair, but a same-window version would be
better.

**Where to look next**, given the attractor: the quantities that could set
it are the ones never swept — `iter_erode` / `iter_deposit` (how much a
cell may change per iteration), `dt` and `friction` (the particle step),
and the number of iterations itself. Not `disc_saturation`, `thermal_rate`,
`creep_rate` or `particles_per_cell`, all of which are already measured
inert.


## Every erosion parameter is inert against the spectrum

The sweeps are now comprehensive enough to state this flatly. Nothing in
`ErosionParams` moves β meaningfully.

Previously measured inert: `particles_per_cell` (4x), `disc_saturation`
(16x), `thermal_rate`, `creep_rate`, talus angles. Now also measured, on a
**fixed** analysis window (chosen once on the baseline and reused, so every
row is literally the same ground):

| config | β 400–3200 | β 200–1600 | relief | lr@2000 |
|---|---|---|---|---|
| base | 7.30 | **6.24** | 639 m | 0.757 |
| `iter_erode` ×4 (100 m) | 7.27 | **6.25** | 639 m | 0.757 |
| `iter_erode` ×0.25 (6.25 m) | 7.53 | **6.27** | 645 m | 0.754 |
| `dt` 1.2 → 0.6 | 7.40 | 6.32 | 640 m | 0.756 |
| `friction` ×2.4 | 7.20 | 6.16 | 638 m | 0.758 |
| 900 iterations (3x) | 8.05 | 5.88 | **359 m** | 0.696 |

A **16x range on `iter_erode` moves β by 0.03**. Tripling the iterations
moves it by 0.36 and destroys 44 % of the relief on the way — it planes the
landscape rather than dissecting it. β ≈ 6.2 is a fixed point that no knob
reaches.

### What the run logs say instead

    mean particle steps before death   20
    particles dying at the ocean       99 %
    largest landmass                   137 cells across
    needed for an order 7-8 network    500-1000 cells

A particle crosses about 20 cells of land on a landmass 137 wide. There is
no room for a nested hierarchy: one or two Strahler orders where a real
landscape has seven. That is a coherent account of the whole inert-knob
history — **no parameter can manufacture scale range the domain does not
contain** — and of why the network *topology* measures correct (channel
head and concavity both in published ranges) while the spectrum does not: a
small network is still a correct small network.

It also supersedes the "erosion converges to β ≈ 6 regardless of input"
framing in the section above. That observation stands, but calling it an
attractor of the *kernel* was the wrong noun; it behaves like an attractor
of the *domain*.

**Domain size is real but far too weak to be the cause** — measured below,
with the relief confound controlled.

## `relief_spacings` makes world size and steepness the same knob

Doubling `N_c` at fixed `cell_size_m` doubles the planet radius, and with
the default `relief_spacings` relief is a fixed fraction of radius, so the
bigger world is also **twice as steep** over any given window. Measured:

| N_c | land across | relief | β 200–1600 | β 400–3200 |
|---|---|---|---|---|
| 256 | 137 cells | 639 m | 6.24 | 7.30 |
| 512 | 276 cells | **1301 m** | 6.35 | 5.89 |

β at 200–1600 m moved the *wrong* way and β at 400–3200 m improved, and
neither is interpretable: the two worlds differ in slope as well as extent.

docs/world-scale.md records this coupling as a scaling issue. It is also an
experimental hazard: **any experiment that varies `N_c` must pin
`tectonics.relief_m` and set `relief_spacings = 0`**, or it is comparing two
things at once.


## Domain size: measured, real, and not enough

With relief pinned (`relief_m = 700`, `relief_spacings = 0`) so the domain
is the only variable, and the same 50 m cells and 3.2 km analysis window:

| N_c | landmass across | relief | β 200–1600 | β 400–3200 | mean particle steps |
|---|---|---|---|---|---|
| 256 | 138 cells (6.9 km) | 794 m | 6.29 | 7.25 | 21 |
| 512 | 271 cells (13.6 km) | 816 m | **6.02** | **6.56** | **35** |

The effect is real: both bands improve, monotonically, and the mechanism is
visible in the particle statistics — flow paths lengthen from 21 steps to
35. It is also **far too small**. One doubling buys 0.27–0.69 of β and the
gap to real topography is 4.0. Extrapolating the two bands disagrees by a
factor of 500 (15,000 vs 7.8 million cells across), so a two-point
log-linear fit is not worth trusting.

What the exercise does establish is more useful than the extrapolation:

**Every β measurement in this project has been made on a toy planet.** The
presets are 4–33 km bodies and the largest landmass in these tests is
**13.6 km across** — an island. Real continents are thousands of km. The
400–3200 m band extrapolates to wanting ~750 km of landmass, which is not
an exotic requirement; it is an ordinary continent.

So "our terrain is bland" has been measured almost entirely in a regime no
real landscape occupies, and the β ≈ 6 floor may be as much an artifact of
island-sized test worlds as a property of the erosion kernel. The two
cannot be separated on a 16 km planet.

**This is the argument for the basin architecture**, arrived at from a
different direction than the cost analysis in docs/world-scale.md: take the
regional trend from an Earth-scale tectonic field, where landmasses are
genuinely continental, and run erosion per basin at 25 m where the local
domain is the 500–1000 cells a drainage network needs. Neither the global
coarse grid (9.8 km cells, no fluvial process) nor a toy planet (13.6 km
landmasses, no scale range) can produce a landscape on its own.
