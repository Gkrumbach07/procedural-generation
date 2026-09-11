# Erosion kernel tuning — results table

How the ★ defaults of `ErosionParams` (globe/config.py) were chosen: the sweep that
settled `creep_rate`, `disc_exponent`, `chunk` and `fan_room`, against PLAN 8.4's
acceptance criteria (dendritic networks, valleys that widen downstream, meanders).

The harness that produced these rows (`scripts/erosion_face_experiment.py`)
was removed with the rest of the one-off tooling; recover it from git at
`5daf256` if a row needs re-deriving. The table is kept because it is the
only record of *why* the starred defaults are what they are, and re-running
the sweep costs more than reading it.

Harness: single face, N=256, cell 50 m, mask 1 inside, 1-cell frozen rim, mask-0 halo.
Cases: **a** tilted plane (8 cells over the face) + noise ±3 cells, uniform uplift 5 cells/300 it, ocean strip i<6 (base level, no uplift);
**b** gaussian massif (33 cells) above a −3 sea, uplift ∝ dome (10 cells/300 it on land);
**c** floodplain slope 0.002 with a fixed inflow block (precip 150 on 12 cells at the far edge), ambient rain 0.02;
**d** floodplain slope 0.002 (i < 0.55 N, rain 0.25) fed by a noisy hinterland (slope 0.06 + noise ±1.5, rain 1, uplift 5/300 it).
Metrics at iteration 300 (97th-percentile channel mask): top1 = discharge share of the top 1 % land cells; big = number of
8-connected channel components > 100 cells (sizes); sinu = mean / max sinuosity of the 10 longest D8-traced channels
(7-cell smoothed path length / chord); HI = hypsometric integral; churn = fraction of active cells changed by > 1 cell unit
in the last iteration; s/it on 4 cores.

## Invalidated runs (earlier harness: tilt 25 cells, uniform uplift also on the ocean strip)
`a_base`, `a_kmom0`, `a_noinertia`, `a_talus_low`, `a_kdisc5`, `a_erod1` (runs/…): parallel drainage — the correct answer for a
3× steeper uniform slope; the coast also silted up and was uplifted above sea level (land 0.973 → 0.996, base level lost).
Not comparable with anything below.

## Valid runs (tilt 8, ocean strip fixed)
| tag | case | change vs ★ defaults | top1 | comps / big (sizes) | frac in big | sinu mean/max | HI | churn | relief | s/it | image | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_base2 | a | none | 0.386 | 89 / 6 (503,399,282,191,168,115) | 0.88 | 1.075/1.086 | 0.582 | 0 | 13.9 | 0.20 | runs/a_base2/iter0300.png | dendritic trunks with angled tributaries; hillslopes striated by parallel micro-rills; gentle sinuosity, no cutoffs |
| a_ssat05 | a | slope_saturation 0.05 | 0.301 | 80 / 4 (927,189,177,145) | 0.76 | 1.034/1.039 | 0.594 | 0 | 13.9 | 0.29 | runs/a_ssat05/iter0300.png | one dominant river (qmax 3×), 2223 particles hit max_steps; worse |
| a_ssat10 | a | slope_saturation 0.1 | 0.378 | 84 / 4 (1047,179,169,142) | 0.81 | 1.147/1.159 | 0.586 | 0 | 13.9 | 0.24 | runs/a_ssat10/iter0300.png | finer dendritic tributaries, more sinuous trunks; but one trunk aggrades into a wide smeared plain near the coast (496 particles hit max_steps) |
| a_kdisc10 | a | k_disc 10, disc_saturation 10 (McDonald entrainment) | 0.305 | 78 / 5 (691,483,180,125,104) | 0.83 | 1.009/1.013 | 0.429 | 0 | 11.0 | 0.21 | runs/a_kdisc10/iter0300.png | broad incised trunk valleys, less striation, but straight channels and the hypsometry collapses (too much incision everywhere) |
| a_ssat05_kdisc10 | a | both | 0.111 | 422 / 1 (105) | 0.06 | 1.000 | 0.499 | 0 | 12.5 | 0.38 | runs/a_ssat05_kdisc10/iter0300.png | sheet wash, no network at all |
| a_creep01 | a | creep_rate 0.1 | 0.528 | 57 / 6 (436,353,273,240,190,137) | 0.86 | 1.090/1.092 | 0.604 | 0 | 12.8 | 0.25 | runs/a_creep01/iter0300.png | soil-mantled look: rounded interfluves, convex-concave hillslopes, clear dendritic trunk network, channels 128 cells long; a few straight/L-shaped reaches remain |
| a_creep03 | a | creep_rate 0.3 | 0.575 | 33 / 5 (638,532,218,179,150) | 0.90 | 1.066/1.097 | 0.608 | 0 | 12.2 | 0.25 | runs/a_creep03/iter0300.png | over-diffused: smeared ridgelines, blobby, network too coarse |
| a_creep005 | a | creep 0.05 | 0.482 | 66 / 7 | 0.88 | 1.101/1.132 | 0.596 | 0 | 13.1 | 0.2 | runs/a_creep005/iter0300.png | between base2 and creep01: residual micro-rills more visible |
| a_creep01_fric005 | a | creep 0.1, friction 0.05 (PLAN ★) | 0.534 | 69 / 5 (569,554,…) | 0.86 | 1.054/1.093 | 0.605 | 0 | 12.8 | 0.2 | runs/a_creep01_fric005/iter0300.png | stronger inertia straightens channels (sinuosity down) |
| a_creep01_ssat10 | a | creep 0.1, slope_saturation 0.1 | 0.451 | 61 / 4 (1197,…) | 0.92 | 1.043/1.045 | 0.617 | 0 | 12.7 | 0.25 | runs/a_creep01_ssat10/iter0300.png | one giant basin (qmax 23k), straighter |
| a_creep01_kmom3 | a | creep 0.1, k_mom 3 | **0.548** | 72 / 6 (381,379,247,227,207,153) | 0.84 | **1.125/1.137** | 0.603 | 0 | 12.8 | 0.19 | runs/a_creep01_kmom3/iter0300.png | best (a) so far: sinuous trunks, angled tributaries, rounded divides, channels 134 cells; trunks still tilt-aligned, no cutoffs |
| b_base | b | none | 0.129 | 305 / 0 | 0 | 1.000 | 0.215 | 0 | 37.4 | 0.13 | runs/b_base/iter0300.png | perfectly radial parallel rills on the dome, nothing merges |
| c_base | c | none | 0.768 | 121 / 6 | 0.54 | 1.001/1.002 | 0.599 | 0 | 0.8 | 0.25 | runs/c_base/iter0300.png | inflow river cuts a dead-straight 0.8-cell trench (sediment-starved inflow); no lateral migration |
| c_ssat05 | c | slope_saturation 0.05 | 0.717 | 86 / 5 | 0.75 | 1.008/1.014 | 0.605 | 0 | 0.8 | 0.31 | runs/c_ssat05/iter0300.png | same |
| d_base | d | none | 0.160 | 120 / 9 | 0.65 | 1.015/1.022 | 0.305 | 0 | 12.5 | 0.24 | runs/d_base/iter0300.png | hill rivers dump alluvial fans at the mountain front (load gone in ~25 steps) then sheet-flow across the plain; no channel on the plain |
| d_ssat05 | d | slope_saturation 0.05 | 0.159 | 122 / 8 | 0.74 | 1.042/1.070 | 0.298 | 0 | 12.7 | 0.29 | runs/d_ssat05/iter0300.png | same |
| d_kdisc10 | d | k_disc 10, dsat 10 | 0.247 | 94 / 5 | 0.78 | 1.007/1.040 | 0.295 | 0 | 6.3 | 0.56 | runs/d_kdisc10/iter0300.png | pathological: hills planed to relief 6, 25 000 cell-units stuck in `pending` |
| d_kmom3 | d | k_mom 3 | 0.183 | 123 / 7 | 0.52 | 1.041/1.095 | 0.324 | 0 | 11.7 | 0.33 | runs/d_kmom3/iter0300.png | more coherent streaks on the plain, still a staircase of fans |
| d_fric005 | d | friction 0.05 | 0.186 | 112 / 5 | 0.48 | 1.019/1.023 | 0.306 | 0 | 12.4 | 0.44 | runs/d_fric005/iter0300.png | fans; pending 1243 |
| d_creep01 | d | creep 0.1 | 0.236 | 60 / 8 | 0.71 | 1.050/1.203 | 0.347 | 0 | 10.8 | 0.53 | runs/d_creep01/iter0300.png | hills good; plain still fans/sheet flow (the 1.2 is a 40-cell stub); creep diffused 1.6 % of coastal land into the sea → creep now skips submerged cells |

Diagnosis after sweep 4: with the erf law the transport capacity saturates at q = disc_saturation (8 cells of rain), so a loaded
trunk river needs the same slope as a rill to carry its load ⇒ no concave profiles, every plain becomes an alluvial fan, and
nothing can meander because no channel survives on a plain. Fix under test: `disc_exponent` (power-law entrainment).

## Power-law entrainment (`disc_exponent`) and combinations
| tag | case | change vs ★ defaults | top1 | comps / big (sizes) | frac in big | sinu mean/max | HI | churn | relief | s/it | image | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_creep01_kd3_ds300 | a | creep 0.1, k_disc 3, dsat 300 | 0.467 | 63 / 6 | 0.90 | 1.061/1.085 | 0.552 | 0 | 12.8 | 0.2 | runs/a_creep01_kd3_ds300/iter0300.png | no gain |
| a_creep01_kd3_ds1000 | a | creep 0.1, k_disc 3, dsat 1000 | 0.502 | 51 / 5 | 0.88 | 1.037/1.115 | 0.557 | 0 | 12.9 | 0.2 | runs/a_creep01_kd3_ds1000/iter0300.png | no gain |
| d_creep01_kd3_ds300 | d | creep 0.1, k_disc 3, dsat 300 | 0.264 | 82 / 7 | 0.67 | 1.022/1.038 | 0.326 | 0 | 10.5 | 0.36 | runs/d_creep01_kd3_ds300/iter0300.png | still fans |
| a_creep01_dep005 | a | creep 0.1, deposition_rate 0.05 | 0.557 | 45 / 5 | 0.87 | 1.091/1.103 | 0.609 | 0 | 13.2 | 0.19 | runs/a_creep01_dep005/iter0300.png | slightly more coherent network (147-cell channels), not more sinuous |
| a_creep01_pow05 | a | creep 0.1, disc_exponent 0.5 | 0.435 | 59 / 6 (837,232,217,175,171,113) | 0.92 | 1.116/1.121 | 0.463 | 0 | 12.2 | 0.22 | runs/a_creep01_pow05/iter0300.png | **valleys now widen downstream**: flat-floored trunk valleys near the coast, narrow V headwaters, rounded but distinct divides; convincing |
| d_creep01_pow05 | d | creep 0.1, disc_exponent 0.5 | 0.485 | 45 / 3 (888,454,438) | 0.93 | **1.136/1.412** | 0.274 | 0 | 9.9 | 0.22 | runs/d_creep01_pow05/iter0300.png | **a channel survives across the plain** (collects the hill outlets, sinuous course, shifts between it 200 and 300); fans still build an apron at the front |
| a_creep01_pow05_kd03 | a | + k_disc 0.3 | 0.525 | 41 / 4 (1094,…) | 0.85 | 1.051/1.061 | 0.557 | 0 | 12.8 | 0.28 | runs/a_creep01_pow05_kd03/iter0300.png | straighter, one giant basin |
| d_creep01_pow05_kd03 | d | + k_disc 0.3 | 0.313 | 51 / 9 | 0.75 | 1.041/1.127 | 0.334 | 0 | 10.9 | 0.30 | runs/d_creep01_pow05_kd03/iter0300.png | plain channel weaker |
| d1_creep01_pow05 | d (plain 1 %) | creep 0.1, exp 0.5 | 0.480 | 62 / 4 (993,383,…) | 0.87 | 1.122/1.151 | 0.296 | 0 | 10.7 | 0.25 | runs/d1_creep01_pow05/iter0300.png | coherent plain channel; the harness's 3-cell sea filled up (pending 24 780) → sea deepened to −20 for c/d from here on |
| d1_creep01 | d (plain 1 %) | creep 0.1 | 0.220 | 53 / 8 | 0.70 | 1.020/1.033 | 0.368 | 0 | 11.9 | 0.24 | runs/d1_creep01/iter0300.png | fans without the power law even at 1 % |
| a_cand | a | creep 0.1, exp 0.5, k_mom 3 | 0.496 | 63 / 5 | 0.84 | 1.037/1.083 | 0.447 | 0 | 12.3 | 0.27 | runs/a_cand/iter0300.png | k_mom 3 straightens channels under the power law |
| d_cand | d (sea −20) | creep 0.1, exp 0.5, k_mom 3 | 0.381 | 41 / 5 | 0.81 | 1.026/1.034 | 0.287 | 0 | 9.7 | 0.22 | runs/d_cand/iter0300.png | same: less sinuous than without k_mom 3 → k_mom stays 1 |
| b_cand | b | creep 0.1, exp 0.5, k_mom 3 | 0.263 | 88 / 3 | 0.30 | 1.037/1.159 | 0.165 | 0 | 31.6 | 0.14 | runs/b_cand/iter0300.png | radial-dendritic: tributaries merge into ~12 trunks, broad lower valleys, crisp summit ridge; no pit blob (top1 2× the baseline) |
| c_cand | c (sea −20) | creep 0.1, exp 0.5, k_mom 3 | 0.792 | 12 / 6 | 0.92 | 1.004/1.005 | 0.599 | 0 | 0.8 | 0.29 | runs/c_cand/iter0300.png | sediment-starved inflow still cuts a straight trench: case (c) as specified cannot meander in this model |

## Final defaults and their verification (creep 0.1, disc_exponent 0.5, k_mom 1 ★, chunk 2048, fan_room 1, offshore loss)
| tag | case | seed | top1 | comps / big (sizes) | frac in big | sinu mean/max | HI | churn | relief | image | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a_c2_s1 | a | 1 | 0.476 | 62 / 3 (731,609,132) | 0.78 | 1.288/1.295 | 0.483 | 0 | 11.9 | runs/a_c2_s1/iter0300.png | candidate, seed 1 |
| a_kmom3_s1 | a | 1 | 0.557 | 51 / 3 | 0.81 | 1.113/1.115 | 0.470 | 0 | 11.9 | runs/a_kmom3_s1/iter0300.png | k_mom 3 again less sinuous → k_mom stays 1 |
| d_c2 | d (sea −20, 4 cols) | 0 | 0.481 | 43 / 3 (842,423,407) | 0.90 | 1.017/1.056 | 0.288 | 0 | 9.7 | runs/d_c2/iter0300.png | plain channel kept; sinuosity on the plain is seed-noisy (1.02–1.14) |
| d_c2_s1 | d | 1 | 0.469 | 34 / 4 (1166,…) | 0.91 | 1.062/1.123 | 0.286 | 0 | 10.4 | runs/d_c2_s1/iter0300.png | the 3-column sea filled by it 250 (pending 10 482) → harness sea widened to 16 columns |
| a_final | a | 0 | 0.438 | 54 / 6 (815,237,212,174,149,126) | 0.91 | 1.148/1.151 | 0.463 | 0 | 12.2 | runs/a_final/iter0300.png | sqrt instead of pow (rounding-level change vs a_creep01_pow05) |
| a_final_ch2048 | a | 0 | 0.417 | 43 / 6 | 0.93 | 1.050/1.058 | 0.469 | 0 | 12.3 | runs/a_final_ch2048/iter0300.png | chunk 2048 |
| a_final_s1 | a | 1 | 0.474 | 62 / 4 | 0.84 | 1.246/1.280 | 0.483 | 0 | 11.9 | runs/a_final_s1/iter0300.png | chunk 512 |
| a_final_ch2048_s1 | a | 1 | 0.482 | 61 / 3 | 0.77 | 1.301/1.308 | 0.487 | 0 | 12.0 | runs/a_final_ch2048_s1/iter0300.png | chunk 2048: within seed noise of 512 on every metric → default 2048 (−21 % trace time) |
| d_final_ch2048 | d (sea −20, 16 cols) | 0 | 0.467 | 45 / 3 (787,469,354) | 0.92 | 1.208/1.277 | 0.270 | 0 | 9.4 | runs/d_final_ch2048/iter0300.png | before the offshore-loss fix: pending 34 915 (a submarine fan can only descend 64 cells × fan_slope from a mouth, then deadlocks) |

Kernel fixes found by the global test (stub small world, 30 iterations): with the power law the trunks deliver ~160 cell-units of
sediment per iteration to shelves that were never deeper than 0.07 cells; a shelf filled to −DEP_FLOOR can take nothing more
(fan_slope descent + sea-level ceiling) and never becomes land, so `pending` grew without bound (4 916 after 30 iterations,
> 1 % of the mass). Now a load that a seafloor walk cannot place is *lost offshore* (deep ocean, as PLAN 8.2's
`into ocean: break`), reported as `lost_offshore`; `pending` remains for land pits (conservation on a closed window unchanged).
`fan_room` (1.0) lets a seafloor step settle up to 1 cell unit where there is depth (was DEP_FLOOR = 0.02).


## Glacial parameters (`glacial_from`, `glacial_every`, `ice_evap`)

Sweeps on the small preset at `N_c=256`, 200 iterations, each config reusing
one template's tectonics + climate so only the glaciation differs.

**`glacial_rate` saturates.** Across 0.5/0.75/0.9 x 1.0/3.0/8.0 (rate in the
cell units it was then consumed in: 50, 150 and 400 m a pass at the small
preset's 50 m cells; it is declared in metres now) there is no
monotonic gain from a harder cut — rate 8.0 gave shallower lakes than rate
1.0 (the carve hits the sea-level floor and the moraine cap). Rate is not
the lever.

**Later and more frequent is better, on one seed.** Sweeping
`glacial_from` x `glacial_every` on seed 0, the best was `from 0.9,
every 1` at 226 lake cells / 0.206 % of land / 2.9 m mean depth, against
158 / 0.144 % / 1.7 m for the shipped `from 0.75, every 10`.

**But the seed dominates the parameters, so those defaults were not
changed.** The same three configs on two fresh seeds:

| seed | control | best-area | best-depth | lake area |
|---|---|---|---|---|
| 0  | 158 cells | 226 | 203 | 0.144 % |
| 11 | 817 cells | 773 | 820 | 0.736 % |
| 22 | 348 cells | 348 | 348 | 0.314 % |

Seed swings lake area 5x; the parameters swing it at most 43 %, and on seed
11 the three configs are within noise of each other. Seed 22's three runs
are *byte-identical* because its coldest land is +1.64 C: it has no ice at
all, so no setting of `glacial_from` or `glacial_every` can give it a lake.

**`ice_evap` is the first-order control.** It is the equilibrium-line
altitude: `evap` is `k_evap * max(T, 0)`, so 0 is the freezing line and a
positive value glaciates a warmer world. On seed 22 (0 % ice at the
default) raising it to 0.35 glaciates 3.3 % of land and gives 139 coarse
lakes / 0.359 % area / 27 m deepest, against 116 / 0.314 % / 9.4 m — with
rivers, channel fraction and land fraction unchanged. If a world looks
lake-poor, this is the knob, not the cadence.
