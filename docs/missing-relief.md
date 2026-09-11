# Where the 3.5 km of missing relief went

`docs/earth-bake.md` ended the first complete Earth-scale bake with one big
defect: after erosion the >5 km band holds 0.0 % of the land, the highest
point on the planet is 5355 m against Earth's 8849, and every band above
2 km is about half of Earth's. Two contributors were named and not
separated — `tectonics.orogen_decay = 0.008`, which left tectonics at a
6619 m maximum, and the glacial pass, which took the maximum from 7300 m to
4900 m in fifty iterations.

They are separated here, one at a time. Neither is the largest term.

## Tectonics is not the problem: it is already on Earth's curve

Three tectonics-only runs of the `earth` preset, identical but for
`orogen_decay`, ~3 minutes each (`scripts/hypsometry.py` reads them):

| band, % of land | Earth | **0.008** (shipped) | 0.004 | 0.002 |
|---|---|---|---|---|
| 0–1 km | 71.6 | 59.7 | 58.0 | 56.0 |
| 1–2 | 15.4 | 27.1 | 25.1 | 20.1 |
| 2–3 | 7.5 | 7.6 | 8.9 | 10.2 |
| 3–4 | 3.8 | 3.5 | 4.4 | 7.5 |
| 4–5 | 1.7 | 1.5 | 2.2 | 4.0 |
| >5 | 0.3 | 0.7 | 1.3 | 2.0 |
| **above 2 km** | **13.3** | **13.2** | 16.9 | 23.9 |
| above 3 km | 5.8 | 5.7 | 7.9 | 13.6 |
| above 4 km | 2.0 | 2.2 | 3.5 | 6.1 |
| max | 8849 m | 6619 | 8003 | 7350 |
| land % | 29.2 | 27.8 | 28.4 | 27.9 |

Read the bold column against the Earth column. **At the shipped decay rate
the pre-erosion high ground is already exactly Earth's post-erosion
distribution** — 13.2 % of land above 2 km against 13.3, 5.7 above 3 km
against 5.8, 2.2 above 4 km against 2.0. Three bands, all within 0.2 points.

That is the finding, and it inverts the argument the value was chosen on.
`docs/crust-types.md` picked 0.008 as "modestly above Earth's curve, on the
argument that erosion closes the rest". It is not modestly above the curve;
it is *on* the curve, before erosion has run — so there was nothing left for
erosion to close and it could only overshoot. After erosion the same bands
come out at 3.2 / 0.7 / 0.2: **erosion removes 69 % of the high ground it is
handed.**

Two consequences.

**Lowering `orogen_decay` cannot fix this on its own.** To end at Earth's
13.3 % after losing 69 %, tectonics would have to deliver ~43 %. The sweep
reaches 23.9 % at 0.002, and `docs/crust-types.md` measured 31.1 % with the
decay switched off entirely. Even a world where no orogen ever came down is
short by a third.

**`max` is not the statistic to tune on.** It is one segment, and it is not
monotone in the knob: 6619 m at 0.008, 8003 at 0.004, 7350 at 0.002. The
bands are monotone and the peak is noise on top of them. (Earth's 8849 m is
also a post-erosion number for a range that is still actively rising, so
matching it pre-erosion would be wrong anyway.)

So the question is not "how much more relief should tectonics build" but
"why does erosion remove 69 % of it", and that is the second contributor.

## The glacial pass is carving in the wrong units

Before running anything, the arithmetic. `erosion/glacial.py` lowers the bed
by

```python
dz = rate * np.sqrt(q) * (1.0 - 0.5 * state.hardness) * taper
np.clip(dz, 0.0, float(ep.glacial_max), out=dz)
state.height -= dz
```

`q` is `discharge / disc_saturation` and is dimensionless; `taper` and the
hardness factor are in [0, 1]. So `glacial_rate` and `glacial_max` are
**lengths**, in the kernel's height unit, which is the grid's cell size.

`config.py` already knows that this is a trap. `LENGTH_PARAMS_M` exists
precisely so that a length is declared in metres and divided by the cell
size at use, with its docstring saying why:

> Their defaults are the values they were tuned at (the old cell-unit number
> × 50 m), so a 50 m world is unchanged and every other cell size now means
> the same physical thing.

`cover_depth`, `max_erode`, `iter_erode`, `iter_deposit`, `thermal_max`,
`fan_room` and `min_volume` are all in that list. **`glacial_rate` and
`glacial_max` are not.** They are consumed raw, and
`docs/erosion-tuning.md` records where their values came from: "Sweeps on
the small preset at `N_c = 256`" — and the small preset's cells are
**50 m**, whatever `N_c` is set to, because `cell_size_m` is its own
parameter.

> **Fixed since.** Both are now in `LENGTH_PARAMS_M`, declared in metres
> at the values they were tuned at — 50 m and 20 m a pass — so a 50 m-cell
> world is bit-identical and the `earth` preset carves 50 m / 20 m instead
> of 9773 m / 3909 m. The fix is coupled to isostasy: correcting the scale
> alone makes the tidal flat worse (below), and isostasy with the old scale
> runs away to 22.8 km peaks and −17.5 km pits (docs/streaks-and-flats.md).
> What follows is the defect as it was measured.

So at the `earth` preset's 9773 m cells the shipped values mean:

| | cell units | at 50 m cells (where tuned) | at 9773 m cells (as shipped) |
|---|---|---|---|
| `glacial_rate` (at the reference ice flux) | 1.0 | 50 m per pass | **9773 m per pass** |
| `glacial_max` (the per-pass cap) | 0.4 | 20 m per pass | **3909 m per pass** |

a factor of **195**. The cap alone permits a cell to lose four kilometres of
bed in a single pass, and `glacial_every = 10` fires 20 passes between
iteration 600 and 800. On `tiny`, where the values were tuned, the deepest
cell of a pass carves 15 m against the 20 m cap — nearly binding, which is
what a calibrated cap looks like.

That is arithmetic on the configuration, not a measurement. The
measurement costs seconds and nobody had taken it: `glacial.carve` is
mass-conserving and touches nothing but `height`, so **one** call on a copy
of a real Earth-scale state *is* a pass. Loading the iteration-200
checkpoint and carving once at each setting:

| `glacial_rate` / `max` | ice cells | deepest cell | mean over carved cells |
|---|---|---|---|
| **1.0 / 0.4** (shipped) | 508,938 | **3909 m** (= the cap) | **848 m** |
| 0.2 / 0.08 | 508,938 | 782 m | 373 m |
| 0.05 / 0.02 | 508,938 | 195 m | 131 m |
| 0.015 / 0.006 | 508,938 | 59 m | 39 m |
| 0.005116 / 0.002046 (= 50 m / 20 m) | 508,938 | 20 m | 13 m |

The same 508,938 cells glaciate at every setting, which is `ice_evap`'s
doing and not this knob's: 8.1 % of the globe, about 27 % of the land —
Last-Glacial-Maximum coverage rather than today's.

**One pass at the shipped settings takes 848 m off the average glaciated
cell and the full 3909 m cap off the deepest, and there are 20 passes.**
Earth's entire Quaternary — 2.5 My of it — deepened valleys by a few hundred
metres, with the deepest fjords around a kilometre.

It also explains the one previous result in this area.
`docs/erosion-tuning.md` reports that `glacial_rate` *saturates*: rate 8.0
gave shallower lakes than rate 1.0, "the carve hits the sea-level floor and
the moraine cap". That sweep only ever went *up* from 1.0. The deepest cell
at rate 1.0 is exactly `glacial_max`, so the pass is already pinned against
its own cap before the sweep starts — saturation was the measurement telling
us the scale was wrong, read as a property of the knob.

## The fluvial half, measured rather than extrapolated

`docs/earth-bake.md` is emphatic that an erosion trajectory must not be
extrapolated across iteration 600, because the glacial pass reverses every
trend. The corollary is that the *pre*-glacial half has to be measured on
its own, which means keeping checkpoints that `_prune_checkpoints` would
otherwise delete — it retains only the two newest per parameter hash, so
iteration 200 is gone by iteration 300.

Share of land above 2 km, all from checkpoints of the one baseline run:

| | tectonics | iter 150 | iter 200 | iter 350 | iter 400 | iter 500¹ | iter 800¹ |
|---|---|---|---|---|---|---|---|
| above 2 km | **13.2** | 10.1 | **9.0** | 7.3 | **6.9** | 6.4 | **4.1** |
| land median | 741 m | 356 | 316 | 267 | 254 | 226 | 585² |
| max | 6619 m | 6831 | 6942 | 7230 | 7312 | ~7300 | 5355² |
| 0–1 km band | 59.7 | 72.3 | 75.2 | 80.8 | 82.0 | — | 76.2² |

¹ from `docs/earth-bake.md`'s table of the same configuration.
² after the glacial pass; the median *rises* and the max falls.

Iteration 400 is the overlap between this run and the one
`docs/earth-bake.md` measured, and it agrees to the digit — 6.9 % above
2 km, 254 m median, 7312 m maximum — so the baseline reproduces and every
number here can be read against that document's.

Two things worth separating.

**The lowlands settle quickly and then overshoot.** Land median crosses
Earth's ~350 m at about iteration 155 and keeps going, reaching 254 m by
400; the 0–1 km band crosses Earth's 71.6 % just before iteration 150 and
reaches 82.0 % by 400. So the fluvial pass does not stop where Earth is —
it is the glacial pass, reversing the trend after 600, that puts the median
back up to 585 m and the 0–1 km band back to 76.2 %. Two errors of opposite
sign, and the shipped configuration is the point where they happen to
cross.

That makes a prediction worth writing down before the glacial arms are
measured: **weakening the glacial pass should make the lowland bands
worse** even as it makes the mountains better, because it is the glacial
pass that is currently carrying the lowlands back up from where the fluvial
pass over-planed them. If the arms show the land median falling below 585 m
and the 0–1 km band climbing above 76 % as `glacial_rate` comes down, the
two-errors reading holds and the fluvial/uplift balance has to be fixed
too. If they do not, it does not.

**The mountains never stop coming down.** The above-2 km share falls
monotonically for the whole pre-glacial run — 13.2, 10.1, 9.0, 7.3, 6.9,
6.4 —
while the single highest peak *rises* (6619 → 6942 → 7312) because uplift is
concentrated on the collision zones that are still active. So uplift holds a
handful of peaks up and loses the rest of the high ground: by iteration 500
the fluvial pass alone has removed **52 %** of it, before the glacial pass
has run at all.

## Uplift is not holding the mountains up

The `uplift` field is metres per erosion iteration, so what a cell receives
over the run is `uplift × erosion.iterations` — less the global mean, which
`apply_uplift` removes every iteration so that uplift redistributes relief
instead of inflating the planet. Area-weighted, on the baseline world:

| where | cells | over the 800-iteration run, mean-free | p90 |
|---|---|---|---|
| land above 2 km | 241,866 | **952 m** | 2090 m |
| land 0–2 km | 1,504,064 | 158 m | 520 m |
| ocean | 4,545,526 | −101 m | — |

So the high ground is fed 0.95 km over the whole run — 2.1 km on the most
active tenth of it — and still loses two thirds of its area above 2 km. The
uplift is correctly *placed* (it is 8× larger on the mountains than on the
plains, and negative offshore, which is the mean-free construction working),
and it is an order of magnitude short of holding a range up against this
erosion.

That is a third contributor, independent of both knobs this document was
asked to separate, and it is the one that decides where the bands end up:
`orogen_decay` sets what tectonics hands over, the glacial pass takes a
slice off the top at the end, and the ratio of fluvial erosion to uplift
decides everything in between. It is also the one that cannot be settled by
a fork — it acts from iteration 1 — so it needs a full re-bake to test and
is out of scope here.

## The glacial half, measured

Five arms forked from the one iteration-600 checkpoint and run 50 iterations
(five glacial passes) each, differing only in `glacial_rate` / `glacial_max`.
Everything before the fork is shared, so nothing but the glacial pass can
account for a difference. `scripts/fork_erosion.py`; the `base` arm is the
shipped configuration and reproduces `docs/earth-bake.md`'s iteration-650
row exactly — **4900 m** maximum, 3.0 % of land above 2 km — which is what
says the harness is measuring the real run.

At iteration 650:

| arm | deepest carve per pass | max | above 2 km | land median | ±50 m of sea |
|---|---|---|---|---|---|
| `off` (rate 0) | — | **7163 m** | 4.2 % | 280 m | 9 % |
| `metres` (0.005116) | 20 m | 7124 | 4.4 | 309 | 9 |
| `earthlike` (0.015) | 59 m | 7055 | 4.4 | 320 | 9 |
| `rate005` (0.05) | 195 m | 6833 | 4.2 | 346 | 5 |
| **`base`** (1.0, shipped) | **3909 m** | **4900** | **2.9** | 463 | 4 |
| Earth | | 8849 | 13.3 | ~350 | 1–2 |

**The glacial pass costs 2263 m of peak height in fifty iterations** — 7163
against 4900 — and 1.3 points of the band above 2 km. Every weaker setting
recovers essentially all of it: the difference between carving 20 m a pass
and carving 195 m a pass is 291 m of peak and nothing at all in the bands,
while the difference between 195 m and 3909 m is 1933 m of peak.

### So the split is

| | above 2 km, % of land |
|---|---|
| tectonics hands over | **13.2** |
| after fluvial erosion and uplift, no glaciation (iter 650) | **4.2** |
| after the shipped glacial pass (iter 650) | 2.9 |
| Earth | 13.3 |

Of the 10.3 points of high ground that erosion removes, **the fluvial pass
and uplift account for 9.0 and the glacial pass for 1.3.** `orogen_decay`
accounts for none of it: it hands over exactly Earth's figure.

For the *highest point* the weighting is the other way round — uplift keeps
raising the peaks through the fluvial half (6619 → 7312 by iteration 400)
and the glacial pass takes 2263 m off them in fifty iterations. Which
statistic you look at decides which contributor looks dominant, and both
are worth fixing.

### The prediction held

Written down before these ran: weakening the glacial pass should make the
lowland bands *worse* while it makes the mountains better, because the
glacial pass is what carries the lowlands back up from where the fluvial
pass over-planed them.

It does. Land median runs 280 m at `off`, 309, 320, 346, and 463 at the
shipped setting — monotone in `glacial_rate` — against Earth's ~350. The
land within ±50 m of sea level runs 9 % at `off` down to 4 % at the shipped
setting, against Earth's 1–2 %. **On the lowland statistics the shipped,
mis-scaled glacial pass is the best of the five arms**, and `rate005` lands
the median almost exactly on Earth's.

That is not a reason to keep it. It is two errors of opposite sign, and the
knob that happens to cancel them is a unit bug carving four kilometres a
pass. Correcting the units exposes the lowland error rather than creating
it — which is the point of correcting it.

## Correcting the glacial scale alone makes the world worse

The glacial pass only ever runs after iteration 600, so nothing before that
depends on `glacial_rate`: **a fork from 600 to 800 with different glacial
values is bit-for-bit what a full re-bake with those values would produce.**
The complete corrected world therefore costs 200 iterations rather than 800.

> **Correction — off by exactly the pass that matters.** The gate fires on
> the step that *produces* iteration 600, so the iteration-600 checkpoint
> already contains one glacial pass at the shipped settings. The fork below
> therefore carried that pass into its "corrected" world, and that one pass
> is not a detail: measured on the re-baked world in
> docs/streaks-and-flats.md, it flattened **4.94 % of the globe** — every
> cold cell it reached — from more than 50 m to within ±1 m of sea level,
> onto a single plane at exactly 0.000 m. A continent-sized plane at
> 0 m drags the land median down and fills the ±50 m band, which is
> precisely the direction of the "every band statistic gets worse" result
> below. A clean corrected-glacial world has to fork from a checkpoint
> *before* 600 (any ≤ 590; 550 is used there), and that result supersedes
> the table in this section.
>
> **The clean fork, on the re-baked world** (`worlds/w-base`, forked from
> iteration 550 so no shipped pass ever runs; only the glacial values
> differ), at iteration 800:
>
> | | shipped | glacial units corrected | Earth |
> |---|---|---|---|
> | highest point | 5192 m | **10,096 m** | 8849 m |
> | land above 2 km (bedrock) | 5.2 % | 6.1 % | 13.3 % |
> | 0–1 km band | 74.5 % | 85.0 % | 71.6 % |
> | land median (bedrock) | 575 m | 226 m | ~350 m |
> | globe within ±50 m of sea level | 6.40 % | 11.45 % | 1–2 % |
> | sea-level change per 1 % of land | 37 m | 2.6 m | — |
> | ocean median | −1927 m | −2525 m | −3700 m |
>
> The conclusion survives the correction and its numbers change: the
> highest point recovers completely — and now *overshoots* Earth, because
> once nothing carves four kilometres a pass the active belts keep building
> — while the lowland distribution gets worse, exactly the two-errors
> result below. Two things are new. The ocean is 600 m deeper, so part of
> the shallow-ocean defect was the shipped glacial pass's spoil going
> offshore. And the tidal flat is *worse* with the units corrected, which is
> what the missing isostatic response predicts: docs/streaks-and-flats.md.

`earthlike` (0.015 / 0.006 — 147 m and 59 m per pass at this grid) run out
to iteration 800, against the shipped run and Earth:

| band, % of land | Earth | shipped | **glacial scale corrected** |
|---|---|---|---|
| 0–1 km | 71.6 | 76.2 | **87.9** |
| 1–2 | 15.4 | 19.7 | **8.5** |
| 2–3 | 7.5 | 3.2 | 2.6 |
| 3–4 | 3.8 | 0.7 | 0.7 |
| 4–5 | 1.7 | 0.2 | 0.2 |
| >5 | 0.3 | 0.0 | **0.1** |
| **above 2 km** | **13.3** | 4.1 | **3.5** |
| land % of globe | 29.2 | 27 | 25 |
| land mean | 840 m | 696 | 445 |
| land median | ~350 m | 585 | **241** |
| **max** | **8849 m** | 5355 | **6795** |
| ocean median | −3700 m | −2142 | −2661 |
| within ±50 m of sea | 1–2 % | 5 | **11** |

The maximum improves by 1440 m and **every band statistic gets worse.**
Above 2 km falls 4.1 → 3.5. The land median goes from 585 m — an overshoot
of Earth's ~350 — to 241 m, an undershoot. The land within ±50 m of sea
level doubles to 11 % against Earth's 1–2 %.

That is the two-errors result stated at full strength, and it is worth being
blunt about what it means for the previous document.
`docs/earth-bake.md`'s headline was that "the continental interior behaves"
— that erosion moved the 0–1 km and 1–2 km bands almost exactly onto
Earth's. **That agreement was manufactured by the unit bug.** A glacial
pass carving four kilometres a pass was hauling the lowlands back up from
where the fluvial pass had over-planed them, and it happened to stop in
about the right place. Take the bug away and the fluvial error is visible
underneath it: with a correctly-scaled glacial pass the landscape planes
straight through Earth's median and keeps going — 320 m at iteration 650,
241 m at 800, still falling.

So the ordering of the work is fixed by this, and it is not the ordering
that looked obvious at the start:

1. **The fluvial/uplift balance is the first problem**, not the third. It
   removes 9.0 of the 10.3 points of high ground, it over-planes the
   lowlands, and it does not reach a steady state — the median is still
   falling at iteration 800. Uplift delivers 952 m to the high ground over
   the run; that is the number to move, and `uplift_scale` is the knob.
   It acts from iteration 1, so it needs a full re-bake to test: ~2.5 hours.
2. **Then the glacial units.** The fix is mechanical (the
   `LENGTH_PARAMS_M` pattern) but it should land *after* (1), because
   correcting it today trades a 1440 m improvement in the maximum for a
   worse distribution everywhere else.
3. **Only then `orogen_decay`.** It hands over exactly Earth's curve today.
   What it *should* hand over depends entirely on what (1) and (2) take
   away, and that is not known until they are settled. Tuning it now would
   be fitting the last free parameter to two known defects.
