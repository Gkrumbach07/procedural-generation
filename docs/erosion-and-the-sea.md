# Erosion's sea, and the level a closed basin settles at

Two of the three questions docs/plates-rifts-and-water.md ended on, both of
them about the same confusion: *below the waterline* is not *sea*, and
*full to the rim* is not *the level a lake sits at*.

All numbers are the shipped `earth` preset (1024² per face, 9773 m cells,
1500 tectonic steps, 800 erosion iterations), seed 0. The worlds:

| world | what it is |
|---|---|
| `earth-full` | the bake of docs/earth-bake.md; tectonics before the fixes |
| `earth-full-water` | `earth-full`'s erosion, hydro onward re-run on the connected-ocean mask |
| `earth-full-lakes` | `earth-full`'s erosion, hydro onward re-run with `lake_evap = 6` |
| `earth-v2` | full bake, fixed tectonics, **erosion still reading the sign** |
| `earth-v3` | full bake, fixed tectonics **and** both fixes here |

`earth-v2` and `earth-v3` carry identical tectonics and climate hashes
(`d0e3d0b55e0413ad` / `963243bb398f1295`), so every difference between them
is this change. That pair is the measurement; `earth-full` is history.

## 1. Erosion decided the sea by the sign of the height

`hydro.open_ocean` has taken the sea to be the *connected* body of water
below sea level since docs/plates-rifts-and-water.md, but erosion runs
before hydro and kept reading `surface < 0`. Every landlocked basin under
the waterline was therefore a marine sink for the whole erosion stage:
particles entering it switched to the deposit-only seafloor walk, its floor
could not be eroded, hillslope creep was frozen on it, and the load a
submarine fan could not place inside it was written off as `lost_offshore`.

### What the census says, and where the old evidence was wrong

Measured on `worlds/earth-full` at iteration 800 — 504 closed basins,
91,583 cells, 1.361 % of the globe and 4.329 % of the land, the largest
78,144 cells and 5.84 M km². Those figures reproduce
docs/plates-rifts-and-water.md exactly.

> **Retracted.** That document reads the largest basin's floor at a median
> −165 m against a bedrock −433 m and concludes it is "filled with
> 'offshore' sediment it should never have received". The **median
> sediment in that basin is 0 m**. The 263 m between the floor and the
> bedrock is the erosion stage's `height` field having risen — uplift,
> isostatic rebound and the `hold_datum` shift — and it is not local:
> the open ocean's median `height − bedrock` is +156 m and the land's
> +46 m over the same run. The inference was drawn from a statistic that
> does not measure deposition.

The defect is real; that is not how to show it. What does show it:

| zone at iteration 800 | area | sediment | concentration |
|---|---|---|---|
| open ocean | 68.57 % | 73.82 % | 1.08× |
| **closed basins** | **1.36 %** | **2.33 %** | **1.71×** |
| land (surface ≥ 0) | 30.07 % | 23.85 % | 0.79× |

Closed basins are the most concentrated sediment sink on the planet — more
so than the deep ocean. And the *distribution* inside the largest one is
the giveaway: median 0 m, mean 127 m, 90th percentile 503 m, 34.1 % of its
cells carrying more than a metre, 744,292 km³ in total. That is a fan
prograding into a sea, not a basin filling as a lake. Three smaller basins
are buried outright: 3,650 cells at a median 1909 m of sediment over
continental crust at −1016 m, 818 cells at 1190 m, 379 cells at 1016 m,
each with sediment on 100 % of its cells.

### The mask, and how often it has to be recomputed

`maps.refresh_base` gives every cell a **base level** in
`particle.S_BASE`, the one free channel of the packed sample array (no
extra array, no extra cache line): 0 on the open ocean and on ordinary
land, and the surface of its own basin's deepest cell inside a closed
depression. The kernel's sea is then `surface < base` rather than
`surface < 0`, which is bit-identical to the old rule wherever the base is
0 — so `erosion.sea_mask_every = 0` is a true no-op and an A/B measures the
mask and nothing else. A closed basin becomes land with an internal base
level: particles cross it and deposit as in any pit, its floor may be
eroded down to (never below) its own low point, the glacial pass may deepen
it past the waterline, hillslope creep runs on it, rain spawns on it, and
its mass stays in the model. The routing flood is seeded the same way, so a
basin's sub-depressions fill towards its floor instead of every cell of it
being an outlet.

The coastline moves while erosion runs, so the mask is recomputed on a
stride, and **the stride is a measurement**. `open_ocean` is a connected-
component labelling and therefore discontinuous: one cell rising out of the
water can sever a strait and flip a whole sea. Driving the real iteration
and recomputing the mask every step:

| 30 iterations from | median | mean | **max** | cells stale after 30 |
|---|---|---|---|---|
| 400 (fluvial) | 58 | 91 | **1,060** | 1,691 (0.027 % of the grid) |
| 750 (glacial) | 121 | 1,920 | **18,682** | 20,730 (0.354 %) |

The means are worthless and the maxima are the whole story. The bursts are
not scattered: in the glacial window they land on iterations **760, 770 and
780** and nowhere else, and in the fluvial window on **420** — exactly the
iterations `glacial_every = 10` and `isostasy_every = 20` fire on. Those
are the two passes that move the surface in a lump, and they are what opens
and severs a strait; a ~1 M km² body toggles between open ocean and closed
basin on each of them and back again the next iteration.

So the stride has to divide both, and `erosion.sea_mask_every = 10` also
keeps it in step with `flood_every = 10`, which matters because the routing
flood is seeded on this mask and must never be handed a stale one. The mask
costs **0.40 s** at N = 1024 against a 5.4 s iteration: **0.74 % of the
stage**, about 32 s over an 800-iteration run.

### What it changes

Forked from `earth-full` at iteration 800, 25 iterations, the two arms
differing only in `sea_mask_every` (0 against 10) — sediment gained per
depth zone, metres summed over cells:

| zone | sea mask off | sea mask on |
|---|---|---|
| land | +2,650,367 | **+11,103,465** |
| coast (0..−50 m) | **+0** | **+1,264,352** |
| shelf (−50..−200) | +209,596 | **+3,390,019** |
| slope (−200..−1000) | +6,790,786 | +7,810,280 |
| deep (< −1000) | +3,987,410 | +3,149,280 |

and `lost_offshore` — mass deleted outright because a seafloor walk could
not place it — falls from 77,066,804 m to 57,710,139 m, a quarter. Deaths
move with it: particles dying on land 7.55 % → 17.30 %, ocean deaths
36.34 M → 29.92 M, pit deaths 3.00 M → 9.41 M over the fork.

The `+0` in the coast row of the off arm is the *other* known defect
showing through: `DEP_FLOOR` is a length in cell units, so at 9773 m cells
nothing may be deposited in the top 195 m of the water column
(`particle.DEP_FLOOR`, docs/crust-audit.md). With the mask on, a closed
basin's cells are land and the waterline ceiling does not apply to them, so
that band can receive sediment again inside a basin — it still cannot in
the open sea.

From a fresh start the change bites on the first iteration: at iteration 1
of the `earth` preset, 1,016,819 particles died in the sea and 556,045 in a
pit before; after, 659,777 and 913,087. A fifth of every iteration's
particles were dying in water that is not there.

### Through a full bake (`worlds/earth-v3`)

`earth-v3` is the `earth` preset baked from scratch with both fixes on. It
carries the **same tectonics and climate hashes as `worlds/earth-v2`**, so
the pair isolates the erosion and hydro change at Earth scale and nothing
else. 122 minutes on 20 threads (erosion 6553 s, refine 513 s, the rest
seconds).

**Mass stops leaking.**

| | earth-v2 | earth-v3 |
|---|---|---|
| `lost_offshore` (mass deleted outright) | 1706.8 Mm | **1141.9 Mm** (−33 %) |
| sediment in the world | 38.80 M km³ | **63.63 M km³** (+64 %) |
| mean sediment thickness | 79.0 m | **129.0 m** |
| mean sediment on the open ocean floor | 38 m | **76 m** |

**Where it goes**, area-normalised, zoned by the bedrock under the pile:

| | earth-v2 area / share / conc | earth-v3 area / share / conc |
|---|---|---|
| open ocean | 68.00 % / 33.75 % / 0.50× | 69.73 % / 42.25 % / **0.61×** |
| closed basins | 2.37 % / 29.42 % / 12.43× | **0.65 %** / **18.07 %** / 27.69× |
| land | 29.63 % / 36.82 % / 1.24× | 29.62 % / 39.68 % / 1.34× |

> Read the area column before the concentration. The closed basins' 12.43×
> → 27.69× is **not** the fix making them a worse sink: their *share* of the
> world's sediment falls, 29.42 % → 18.07 %, while the area they cover falls
> faster, 2.37 % → 0.65 %. They shrank because they now fill — closed-basin
> cells below sea level 154,474 → 42,153, median sediment inside one 6 m →
> 2703 m. Before the fix they were a wide, shallow, permanent marine sink;
> after it they are a few deep basins that are filling up.

**The offshore table, re-measured — and a null result.** The concentration
by depth zone of the bedrock, against the table docs/earth-bake.md opened
this question with:

| zone | first Earth bake | earth-full | earth-v2 | **earth-v3** |
|---|---|---|---|---|
| deep (< −200 m) | 1.33× | 1.26× | 1.21× | **1.21×** |
| shelf (−200..0) | **0.59×** | 1.40× | 1.69× | **1.39×** |
| coast (0..50) | 0.26× | 0.35× | 0.40× | 0.33× |
| land (> 50) | 0.33× | 0.31× | 0.30× | 0.22× |

**The deep/shelf inversion is gone, and this change is not what fixed it.**
The shelf has been the more concentrated of the two since `earth-full` — the
streaks-and-flats work did it — and the sea mask leaves the ratio alone
(1.21 / 1.69 → 1.21 / 1.39, if anything slightly worse on the shelf). What
the sea mask moves is the *mass*: a third less deleted, twice as much on the
ocean floor. Anyone chasing the offshore-sediment bias should stop citing
the 1.33× / 0.59× table; it describes a world two fixes ago.

**The hypsometry barely moves**, which is the right answer for a change
about where sediment goes rather than how fast land is planed:

| | Earth | earth-v2 @800 | earth-v3 @800 |
|---|---|---|---|
| 0–1 km | 71.6 | 82.2 | 81.5 |
| 1–2 | 15.4 | 7.8 | 8.1 |
| >5 | 0.3 | 2.2 | 2.3 |
| land median | ~350 m | 197 | 216 |
| max | 8849 m | 12 769 | 12 761 |
| ocean median | −3700 m | −3571 | −3572 |

### Two things this turned up that are not fixed

**A basin can load itself down and keep accepting sediment.** `earth-v3`'s
second-largest closed basin is 5211 cells with its bedrock at −1392 m, its
`height` driven to **−7232 m**, a mean 6564 m of sediment on top and a
surface at −118 m. `erosion.isostasy = 0.8` subsides a loaded column by
four fifths of what it gains, which makes room for more: a positive feedback
with nothing opposing it. In *kind* that is the South Caspian Basin, whose
basement lies ~20 km down under ~20 km of fill, so it is not obviously
wrong — and it is in both arms (`earth-v2`'s basin 9 has bedrock at
−1077 m under a −2759 m floor), so the sea mask did not create it. It wants
a measurement of its own before anyone calls it correct.

**The stage census is taken at the worst moment, on purpose.** The mask is
refreshed on iterations ≡ 0 (mod 10) and the glacial pass fires at the end
of iterations ≡ 9 (mod 10), so every refresh happens immediately after a
glacial carve. That is the right moment to look — those carves are what move
the coastline, which is the whole reason for the stride — but it means the
`erosion.sea` census in `manifest.json` is a **post-glacial maximum**:
153,707 closed cells at iteration 790, against 57,437 at the iteration-750
checkpoint and 42,153 at 800. The fluvial pass refills the carve inside ten
iterations. Read the checkpoints for the typical state, not the manifest.

## 2. A lake was filled to its rim, with nothing taking water out

The priority flood fills every closed depression to its **spill point**,
which is the level a lake reaches when nothing removes water from it. Real
closed basins settle where inflow balances evaporation off the water
surface, which can be far below the rim — the Caspian is the largest lake
on Earth and its surface stands 28 m *below* sea level.

`hydro/balance.py` solves that balance per depression, on the hypsometric
curve of the basin, inside the accumulation pass:

    inflow  = the flow accumulation arriving at the depression
    loss(L) = lake_evap · Σ over the cells under L of evap[c]·area[c]/cell²

`precip` is a volume per cell normalised to a land mean of 1 and `evap` is
the dimensionless `k_evap·max(T, 0)` (≈ 1 at a warm sea-level cell), so the
two are commensurable and `hydro.lake_evap` is a single number. `loss` is
non-decreasing in `L`, so the level is found by sorting the depression's
cells by elevation and walking the cumulative curve. A basin whose balance
level is above its spill point overflows and keeps its outlet — **the
current behaviour, and the common case.**

Cascades come out right for free. The whole thing rides on *one*
accumulation pass in upstream-first order, and a depression's outlet is the
first of its cells the flood popped and therefore the last of them in that
order: when the pass reaches it, every drop the basin receives has arrived,
and what it writes back is the outflow — zero for a closed lake. A lake
that stops overflowing stops supplying everything below it in the same
pass, with no iteration.

### `lake_evap` is derived, not tuned

The number has to absorb something the model does not have: `flow_acc`
routes **precipitation** as if all of it were runoff, where Earth's land
yields only 0.31–0.36 of its precipitation to rivers. So

    lake_evap = (open-water evaporation / land-mean precipitation) / runoff ratio

With Earth's land precipitation ~800 mm/yr, open-water evaporation where
closed basins actually sit of 1000–2000 mm/yr (Caspian ~1000, Chad ~2200)
and a runoff ratio of 0.31–0.36, that is **3.5 to 8.1, centre 5.8**. The
shipped value is **6.0**.

### What the sweep says

On `earth-full`, 3,017 depressions, re-solving the balance at each value
against one flood:

| `lake_evap` | overflowing | closed | dry | lake cells | % of globe | land draining internally |
|---|---|---|---|---|---|---|
| 0 (spill point) | — | — | — | 204,806 | 3.185 % | 0 % |
| 2 | 2,966 | 27 | 24 | 191,831 | 2.989 % | 0.36 % |
| 3 | 2,946 | 34 | 37 | 188,902 | 2.938 % | 6.11 % |
| 5 | 2,907 | 56 | 54 | 177,848 | 2.745 % | 9.28 % |
| **6** | **2,896** | **63** | **58** | **169,902** | **2.619 %** | **35.00 %** |
| 8 | 2,869 | 75 | 73 | 135,957 | 2.106 % | 35.00 % |
| 10 | 2,851 | 76 | 90 | 115,065 | 1.798 % | 35.00 % |

Earth drains about 18 % of its land internally, and **no value of
`lake_evap` puts this world there.** One basin takes 29 % of all the land's
precipitation, and it flips between 5 and 6: below that it overflows and
the internally-drained share is under 10 %, above it the share jumps
straight to 35 %. That is a fact about this world's drainage topology — a
single continental sink of 7.43 M km² with a rim only 22 m above sea level
— and not a calibration error, so `lake_evap` is left where the derivation
puts it rather than bent to reach 18 %.

Two more things the sweep settles. **Most depressions can never close**:
1,168 of 3,017 (38.7 %) have `evap = 0` over every one of their cells, so
their evaporative term is identically zero whatever `lake_evap` says. They
are the cold ones, and that is the right answer — a frozen lake does not
evaporate, and it is also why Earth's lake-rich terrain is its recently
glaciated terrain. **And the median depression by area needs `lake_evap`
7.8 to close**, against 4.5 at the 10th percentile and 15.3 at the 75th:
the basins are close enough to the balance that the value matters, which is
the useful kind of sensitivity.

### Carried through the stage (`worlds/earth-full-lakes`)

Hydro onward re-run on `earth-full`'s erosion output, which is exactly the
`earth-full-water` recipe with the balance switched on, so the only thing
that differs between the two columns is the lake level:

| | spill point (`earth-full-water`) | balanced, `lake_evap = 6` |
|---|---|---|
| depressions | 3,017 | 2,896 overflowing / 63 closed / 58 dry |
| cells under water | 208,651 | **173,440** |
| lakes | 2,362 | 2,336 |
| lake cells | 204,806 | **169,902** (−17.0 %) |
| channel cells | 91,154 | 90,465 |
| reaches | 8,401 | 8,223 |
| coarse biome: lake | 204,806 | **169,902** (−17.0 %) |
| coarse biome: wetland | 156,691 | **148,856** (−5.0 %) |
| coarse biome: ocean | 4,312,436 | 4,312,436 (unchanged) |
| coarse biome: savanna | 131,136 | 144,555 (+10.2 %) |
| coarse biome: riparian | 370,520 | 385,568 (+4.1 %) |

The drawdown is a **median of 0 m and a maximum of 1887 m**: almost every
depression still overflows and keeps the level it had, and the ones that do
not are drawn down a long way. The ocean count not moving by a cell is the
check that the balance touches lakes and nothing else. What was lake bed
comes back as savanna, shrubland and riparian margin, which is what the
shore of a shrinking closed sea looks like.

For the whole sequence on the same erosion output: lake cells were
**51,419** when below-sea-level meant ocean, **204,806** once the sea was
the connected sea, and **169,902** once a closed basin settles where its
inflow can hold it.

### And through the same full bake

`earth-v3` again, against `earth-v2` — the same tectonics, the erosion fix
as well as the balance, so this column is the shipped world rather than an
isolated A/B:

| | earth-v2 | earth-v3 |
|---|---|---|
| closed-basin cells below sea level | 154,474 | **42,153** |
| depressions | — | 3,480 → 3,283 overflowing / 110 closed / 87 dry |
| cells under water | 214,423 (spill point) | **131,249** |
| lakes | 2,571 | 2,219 |
| coarse lake cells | 437,090 | **124,276** |
| coarse biome: lake | — | 124,276 |
| coarse biome: wetland | — | 155,475 |
| channel cells / reaches | 101,729 / 8,523 | 76,577 / 7,259 |
| drawdown | — | median 0 m, max **134 m** |

Most of the fall from 437,090 to 124,276 is the *erosion* fix, not the
balance: closed basins that spend the run filling with sediment are shallow
by the time hydro sees them, so there is much less to flood. The balance
takes the last 83,174 cells off (214,423 → 131,249). The maximum drawdown
is 134 m here against 1887 m on `earth-full-lakes`, for the same reason —
these basins no longer have kilometres of empty depth to draw down.

### What it does not do

`refine` re-floods each basin on the fine grid and writes the spill-point
fill back into the fine `water_surface`, so the balance is a coarse-grid
result only. `earth-v3` shows the size of the gap: **510,341 fine lake
cells against 124,276 coarse ones**, four times as many, because every
depression is a full pool again at fine resolution. And `flow_dir` still
points out of a closed lake's outlet, because it is computed on the filled
DEM and has to stay acyclic — `flow_acc`, which is 0 below such an outlet,
is what says the river is not there.
