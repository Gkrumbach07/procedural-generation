# The ceiling on uplift: why 2 % of the land stood above 5 km

> **Superseded (docs/uplift-replay.md).** This note found the cause and
> then clipped it. "Applied undamped for all iterations" is only half of
> it: the `uplift` window is already in `bedrock`, so the stage was
> counting the last 100 tectonic steps twice. That is why surface minus
> applied uplift landed on Earth's 0.3 %. `erosion.uplift_mode = 'replay'`
> (now the default) starts erosion from the reference-step crust and ends
> at `bedrock` when nothing erodes. On the small preset it keeps every
> cell under the bedrock's 99.9th percentile without a cap, so
> **`uplift_max_m` now defaults to 0**. The 2 m/it default and the "halves
> the excess" framing below hold only for `uplift_mode = 'stack'`, which
> reproduces earth-v7 with `uplift_max_m = 2`. The mechanism analysis, the
> tables and the datum-hold section still stand. Under replay the Earth
> ocean shallows by ~450 m, because the window's sea-floor subsidence was
> double-counted too (docs/uplift-replay.md, "What to expect at Earth
> scale").

After erosion, earth-v5 has 2.2 % of its land above 5 km (Earth: 0.3 %)
and a highest point of 9862 m; sweep-od006 (same, `orogen_decay` 0.006)
has 1.8 % and 14,780 m; sweep-od008 has 0.2 % and 6674 m. Before erosion
the bedrock maxima are 6747 / 8612 / 3839 m, so **erosion raises the
highest ground by 3-6 km**. This note separates the four candidate
mechanisms with the baked worlds' data, finds it is one of them, and adds
the fix: `erosion.uplift_max_m`, a per-iteration ceiling on the tectonic
uplift a cell may receive (default 2 m per iteration, `LENGTH_PARAMS_M`,
0 = off; see the verdict below for why 2 and not 1). No kernel change: `KERNEL_VERSION` stays 8, the new
`ErosionParams` field changes the erosion params hash.

Everything at Earth scale is read off the read-only reference worlds
(`worlds/earth-v5`, `sweep-od006`, `sweep-od008`: `coarse/*.f{0..5}.npy`,
`frames/erosion/*.npz` every 10 iterations, `checkpoints/erosion_iter*.npz`)
with scratch scripts under `scratch/peaks/` (untracked working files, not
in the tree); nothing at Earth scale was re-baked. The fix is measured on the small preset with the real code.

> **Verdict on the default (review).** The change was implemented and
> measured with a 1 m/it cap and reviewed adversarially. The mechanism
> and every load-bearing number reproduce (uplift max 4.510 m/it, 9.29 %
> of land above 1 m/it, the 750 -> 800 checkpoint rise at the > 5 km
> cells 129 m against 50 x uplift = 133 m). What the review changed is
> the framing of 1 m/it: those 9.3 % of land cells are 52 % of the land
> above 2 km, 35 % of the 2-4 km band and 83 % of the 4-5 km band, and
> the median capped cell loses ~700 m of uplift over the run, so a 1 m/it
> cap clips ordinary mountain growth, not just the runaway crests; on
> the small preset it clips 98 % of the land above half the bedrock
> maximum and the world's maximum falls below its own bedrock maximum.
> **The default is 2 m/it**: 4.1 % of land, and by the static estimate on
> earth-v5's fields the share above 5 km 2.2 -> ~1.6 % and the maximum
> 9862 -> ~9070 m, without reaching into the 2-4 km bands. That halves the
> excess rather than removing it; the rest is the belt-construction jumps
> the last window of tectonics contains, which is a question about how
> `uplift` is derived, for another pass. The tables below are the 1 m/it
> measurements as taken.

> **Measured at Earth scale** (`worlds/earth-v7`, the 2 m/it default on
> earth-v5's tectonics, with kernel 8, graph rivers and the cross-face
> basins of the same commit): 77,451 cells capped; land above 5 km
> 2.2 -> **1.6 %**, highest point 9862 -> **9195 m**, the 2-3 / 3-4 / 4-5
> km bands 6.4 / 2.8 / 2.3 -> 6.9 / 3.1 / 2.5 % (Earth 7.5 / 3.8 / 1.7),
> land median 655 -> 695 m, ocean median -3247 -> -3195 m. The static
> estimate (1.6 %, ~9070 m) was right to the digit on the share and 1 %
> on the maximum.

## The answer in one paragraph

The `uplift` field tectonics writes is the height a segment gained over
the last `tectonics.uplift_window = 100` steps divided by
`erosion.iterations`, so the erosion stage applying it every iteration
reproduces those 100 steps of orogeny a second time -- and nothing bounds
it. earth-v5's fastest cell carries 4.51 m/it, 3.6 km over 800 iterations;
sweep-od006's 8.52 m/it, 6.8 km. The cells that end above 5 km are ridge
crests with ~0 discharge (no particle passes), where thermal erosion never
triggers at Earth's cell size (talus 1.2 rise/run = 11.7 km per 9.8 km
cell) and the glacial carve tapers to zero (it scales with discharge and
ramps down to the ice margin), so creep (0.1) is their only counterweight.
Isostasy, the glacial pass and stacked orogen profiles are ruled out below.
A second, smaller, by-design contributor is the datum hold: bedrock has
24.6 % land against a forced `world.land_fraction` of 0.30, so iteration 1
lifts the whole surface 634 m (866 m over the run).

## When: the maximum climbs linearly through the whole stage

`scratch/peaks/frames_max.py` on the erosion frames (block-mean 256 per
face surface, land = h >= 0):

| iteration | earth-v5 max / >5 km | sweep-od006 | sweep-od008 |
|---|---|---|---|
| 0 (bedrock) | 6736 m / 0.39 % | 8544 / 1.00 % | 3834 / 0.00 % |
| 10 | 7388 / 0.72 % | 8880 | -- |
| 100 | 7580 / 0.76 % | 9736 / 1.16 % | -- |
| 300 | 8096 / 1.09 % | 11,272 / 1.47 % | 4876 |
| 500 | 8776 / 1.59 % | -- | 5420 / 0.06 % |
| 600 (glacial starts) | 9096 / 1.82 % | 13,296 / 1.74 % | 5816 / 0.11 % |
| 700 | 9456 / 2.01 % | -- | -- |
| 800 | 9784 / 2.19 % | 14,664 / 1.86 % | 6652 / 0.20 % |

~3.7 m/it (v5), ~7.6 m/it (od006), ~3.5 m/it (od008), with no change of
slope at iteration 600 where the glacial pass starts (`glacial_from`
0.75). The jump between iterations 0 and 10 (+650 m v5, +336 m od006) is
the datum hold at iteration 1 (below).

## Where, and what the cells carry

`scratch/peaks/final_state.py`; "applied uplift" is the mean-free total,
`(uplift - mean) x 800`.

earth-v5: uplift mean -0.082 m/it, land p50 / p99 / max 0.066 / 3.065 /
4.510 m/it. The 41,515 cells above 5 km (2.20 % of land):

| | median |
|---|---|
| bedrock | 3582 m (max 6747) |
| uplift | 2.653 m/it -> 2188 m applied (p10 918, max 3674) |
| surface - bedrock | 2329 m |
| sediment | 13 m (p90 30) |
| ice (evap <= 0) | 97 % |
| in a collision zone | 81 % |

Uplift is concentrated exactly where the bedrock is already high: by
bedrock band, 0-1 km cells carry 0.07 m/it, 3-4 km 1.32, >4 km 1.81; by
final surface, 2-4 km cells 0.516 m/it (478 m over the run), 4-5 km 1.95
(1627 m), >5 km 2.65 (2188 m). Only 14 % of the final >5 km cells had
bedrock above 5 km (34 % above 4 km). The top cell (face 2, 87, 68):
9862 m = 5798 bedrock + 3162 applied uplift (3.871 m/it) + ~866 datum
- 36 m of net erosion and rebound over the run,
sediment 0. **Surface minus applied uplift: max 7398 m, 0.34 % of land
above 5 km** -- Earth's 0.3 %.

sweep-od006: uplift max 8.516 m/it; the >5 km cells (34,883) carry
bedrock 4852 m, uplift 4.375 m/it (3560 m applied, max 6873), sediment
21 m, 99 % in collision zones; top cell 14,780 = 8544 bedrock + 6855
uplift. Surface minus uplift: max 7925, 0.61 % above 5 km. Its bedrock
alone already has 1.00 % of land above 5 km and a maximum of 8612 m: that
part is stacked belt profiles at a reassembly (`tectonics/orogeny.py
shape_belt`), tectonics' and not erosion's, and this note does not fix it.

sweep-od008: no bedrock above 4 km anywhere (max 3839); the 3861 cells
above 5 km carry 3.413 m/it (2816 m); top 6674 = 3434 + 3092. Surface
minus uplift: max 3831, 0.00 % above 5 km.

## The rate, attributed

`scratch/peaks/frame_rate.py` regresses the per-cell rise between frames on
the block-mean uplift, with the datum estimated from the |u| < 0.05 cells.
earth-v5, iterations 100 -> 600: the >5 km cells rise +2.42 m/it, their
uplift is +2.76 m/it, so the residual -- erosion + isostasy + glacial
together -- is **-0.16 m/it** (p10 -1.34, p90 +0.41; correlation 0.73;
land-wide fit rise = 0.74 x uplift - 0.54). 600 -> 800 with the glacial
pass on: rise +1.99, uplift +2.74, residual -0.58. The single top frame
cell rises +4.50 against +3.91 of uplift: flexural rebound from the
valleys around it adds ~0.6 m/it at that one cell, but not to the
population. od006 100 -> 600: +3.85 vs +4.60 (residual -0.48); od008:
+3.75 vs +3.64.

`scratch/peaks/ckpt_delta.py` on the exact checkpoint arrays, 750 -> 800:
earth-v5 >5 km cells d(height) median **+134 m against 50 x uplift =
+137 m**, residual +2 m (p10 -48, p90 +39), d(sediment) 0; with ice
(n = 40,176) residual +1, without ice (n = 1339) +12, so the glacial pass
takes ~10 m per 50 iterations off the peaks it reaches. od006: +208 vs
+222 (-4); od008: +172 vs +176 (+9). `iso_acc` is zero at 800 (just
applied). The last 50 iterations at the peaks are uplift and nothing else.

## The four mechanisms

1. **The uplift field applied undamped for all iterations.** Confirmed:
   the linear climb, the per-cell decomposition, the 750 -> 800 delta and
   "surface minus applied uplift lands on Earth's 0.3 %" all say the same
   thing.
2. **Isostatic rebound (`isostasy` 0.8) lifting eroding belts.** Ruled
   out for the population: the residual after uplift is negative
   (-0.16 m/it); rebound is visible only at the single top cell (+0.6 m/it),
   which sits between carved valleys.
3. **Stacked orogen profiles at a reassembly.** Real but separate and
   smaller: od006's bedrock has 1.0 % of land above 5 km and an 8612 m
   maximum before erosion; earth-v5's 0.39 % and 6747 m. It is what the
   cap leaves behind (below).
4. **The glacial pass sharpening peaks.** Ruled out: no kink at iteration
   600, and the pass makes the residual *more* negative (-0.16 -> -0.58
   m/it; ~10 m per 50 iterations off the iced peaks).

Why nothing erodes the crests: they have ~0 discharge (hardness 0.72,
no particle passes), thermal erosion needs a 1.2 rise/run talus that no
mountain cross-section reaches at a 9.8 km cell, and the glacial carve
scales with sqrt(discharge/32) and tapers to zero at the ice margin. Only
creep touches them. docs/missing-relief.md's finding that uplift is "an
order of magnitude short of holding a range up" (952 m to the >2 km land)
is about the belts' flanks and interiors, where particles erode; at the
crests the same field is unopposed. Both are true: the field is right in
bulk and has no ceiling.

## The datum hold's share

`hold_datum` forces `world.land_fraction` = 0.30 from iteration 1, and the
bedrock has 24.6 % (v5) / 26.5 % (od006) of cells above zero, so the first
iteration shifts the whole surface up 634 m / 310 m (median shift of
deep-ocean frame cells); erosion info `datum_drift_m` = -866 m for v5 over
the run. That is 0.6-0.9 km of every maximum, it is by design
(`land_fraction` is a hard guarantee; moving it to tectonics would be a
`shelf_fraction` change), and the cap does not and should not touch it.

## The fix: `erosion.uplift_max_m`

`maps.apply_uplift(state, cap)` clips the field at `cap` (cell units)
*before* taking the interior mean, so the applied change is still exactly
mass-free; `maps.uplift_cap` turns `erosion.uplift_max_m` into cell units
for the global pass and returns `None` for a window (the window tests hand
in 2.7-5 m/it synthetic fields and must keep applying them as given;
refine passes `uplift = 0` anyway) or when the knob is 0. The stage info
gains `uplift_cap_m_per_iter` and `uplift_capped_cells` (active interior
cells whose field exceeds the cap), and the stage logs them at the start.

### Alternatives measured on the small preset

`scratch/peaks/small_arms.py` swaps `apply_uplift` for each rule and drives
`maps.step` exactly as `run.py` does, from bedrock, on the small preset
(N_c 128, 50 m cells, seed 0, 300 tectonic steps, 60 erosion iterations;
bedrock max 584.6 m, uplift max 8.23 m/it = 494 m over the run, 85 % of the
relief). "Above 0.75 x bedrock max" is the small preset's analogue of
"above 5 km".

| rule | max | land > 0.75 x bedmax | > 0.5 x bedmax | land median |
|---|---|---|---|---|
| baseline (HEAD) | 744.2 m | 1.231 % | 2.75 % | 23.9 m |
| uplift off | 444.9 | 0.010 | 0.65 | 22.5 |
| **cap 1 m/it** | **495.6** | **0.197** | **1.03** | 21.8 |
| cap 2 m/it | 556.4 | 0.414 | 1.56 | 22.9 |
| exponential decay, tau = N/4 | 507.7 | 0.193 | 1.03 | 20.6 |
| uplift for the first quarter only | 505.2 | 0.180 | 0.99 | 21.0 |
| uplift x 0.5 (`uplift_scale`) | 588.3 | 0.488 | -- | -- |

The cap is preferred over decay, first-N and `uplift_scale` because it
removes only the run-away cores -- 9.3 % of earth-v5's land cells at
1 m/it -- and leaves every other belt's uplift intact, whereas the
proportional rules take the same fraction off every belt, the direction
docs/missing-relief.md already found wrong (the flanks need *more* uplift
against fluvial erosion, not less). `uplift_scale` would also re-bake
tectonics.

### Measured with the real code

```
cd bake
python scripts/bake.py --world scratch/peaks/small-w0    --preset small --to erosion --set erosion.uplift_max_m=0
python scripts/bake.py --world scratch/peaks/small-after --preset small --to erosion
python scripts/fork_erosion.py scratch/peaks/small-after --fresh --preset small --iterations 60 --label cap2 --set erosion.uplift_max_m=2
```

against `scratch/peaks/small-before` baked from HEAD b45422e:

| world | max | land > 0.75 x bedmax | > 0.5 x bedmax | > bedrock p99.9 | land p99.9 | median | capped cells |
|---|---|---|---|---|---|---|---|
| small-before (HEAD) | 744.2 m | 1.231 % | 2.75 % | 0.573 % | 686.7 m | 23.9 m | -- |
| small-w0 (`uplift_max_m` 0) | 744.2 | 1.231 | 2.75 | 0.573 | 686.7 | 23.9 | 0 |
| **small-after (default 1 m/it)** | **495.6** | **0.197** | **1.03** | **0.000** | 463.0 | 21.8 | 5916 of 98,304 |
| fork cap2 (2 m/it) | 556.4 | -- | -- | -- | -- | 22.9 | -- |

small-w0's `height`, `sediment`, `discharge` and `momentum` are
byte-identical to small-before (`np.array_equal`), and its erosion stage
output hash is the same `43aca3e0aec2ba44`; only the manifest
`params_hash` differs (`1e618b2b7fc9d1b9` -> `1cec3fb1cca6b952`). The
default lowers the small world's maximum below its own bedrock maximum
(496 against 585): on a 50 m preset whose relief is 85 % uplift, 1 m/it
is a tight cap (60 m of 494), acceptable for a test preset. The datum
drift moves 23.4 -> 28.7 m: less uplift on the cores means more of the
land fraction is held by the shift.

### What to expect at Earth scale

`scratch/peaks/cap_estimate.py`: surface minus the sum of (uplift - cap)+
over 800 iterations, datum-corrected, on the baked fields. It ignores the
erosion the extra height would have caused, which at the crests is ~0
(residual +2 m per 50 iterations), so it should be close; the flanks may
differ slightly.

| cap (m/it) | land cells above it | earth-v5 max | >5 km | >4 km |
|---|---|---|---|---|
| none (baked) | -- | 9862 m | 2.20 % | -- |
| 0.5 | 15.0 % | 7900 | 0.55 | 1.74 |
| **1.0** | **9.3 %** | **8286** | **0.78** | **2.49** |
| 1.5 | 6.4 % | 8676 | 1.14 | -- |
| 2.0 | 4.1 % | 9070 | 1.56 | -- |
| Earth | | 8849 | 0.3 | |

At 1.0 the 2-4 km band's median applied uplift is unchanged (501 vs 478 m)
and the band grows 9.5 -> 10.2 % of land. sweep-od006 at 1.0: ~8810 m,
0.91 % above 5 km (the remaining 0.9 % is the bedrock's own 1.0 % plus
the datum lift); sweep-od008: ~4722 m, 0.00 %. Read the real numbers off
the next Earth bake with `scripts/hypsometry.py <world> --checkpoints` and
the erosion info's `uplift_capped_cells`; if the maximum is still above
8849 m, the remainder is the bedrock's own peaks plus the datum lift.

## Invalidation

`uplift_max_m` is a new `ErosionParams` field, so `group_hash("world",
"erosion")`, every world's manifest `params_hash` and every erosion
checkpoint's hash change; `bake` refuses an existing world without
`--force`, and `--force --from erosion` re-runs erosion and everything
downstream. Tectonics and climate outputs are untouched. `KERNEL_VERSION`
stays 8 (no kernel code moved; the rule is docs/sea-death-stockpile.md's).
YAML written with the new field cannot be read by older source
(`from_dict` rejects unknown keys); older files load with the default.

## Tests

`tests/test_erosion.py::test_uplift_cap_bounds_the_rise_and_stays_mean_free`:
a stub world with a 5 m/it belt core over a ~0.3 m/it background; with the
cap every interior rise is <= cap - mean(min(u, cap)), total mass is
unchanged to 1e-9, cells under the cap rise by exactly their own uplift
less the datum, the core is pinned at the cap, `cap=None` /
`uplift_max_m=0` reproduce the old rule bit for bit, and a window state
(`make_window "dome"`, uplift above the cap) is not capped by `step`.
