# Why the terrain reads as bland, measured

The recurring complaint about this world is that its features are smooth and
samey — "they all follow a pretty standard river system down". This is the
measurement behind that, and what it rules in and out.

Run the metrics with `bake/scripts/terrain_stats.py`.

## The numbers

Measured on the shipped `final512` world (fine grid, 12.5 m cells):

| metric | ours | real landscapes |
|--------|------|-----------------|
| spectral slope β (P ~ k⁻ᵝ, 60–600 m) | **4.7** | ~2 |
| drainage density | **1.02 km/km²** | 2–20; humid temperate 4–8 |
| median hillslope length | **475 m** | 50–200 m |

p99 slope is 0.65, well under the talus angle, so we are *not* slope-limited —
there is headroom to cut valleys, and we are not using it.

β is fitted over the band where valleys, ridges and tributary junctions live.
The estimator was checked against synthetic fBm of known slope and is accurate
to ±0.1 over β = 1..5, so the reading is trustworthy.

The three agree on one story: **the valley network is too coarse.** Relief and
slopes are fine — it is the space *between* channels that is wrong, and 400 m
of undissected hillslope between one stream and the next is exactly what
"smooth blob" looks like from the air. The missing spectral variance at
60–600 m is the missing first- and second-order tributaries.

## What it is not

**Not a missing technique.** Every fix from McDonald's 2023 follow-up article
(*Procedural Hydrology: Improvements and Meandering Rivers*) is already in
`globe/erosion/particle.py`, verified against the article:

| his criticism | his fix | ours |
|---------------|---------|------|
| 1. natural smoothing | thermal erosion / avalanching at the angle of repose | `thermal_erosion`, talus 0.6/1.2 |
| 2. surface normal | normal from the finite-difference gradient, not mesh triangles | gradient-based, sub-cell |
| 3. constant time-step | normalise step to one cell so particles cannot tunnel | "the step is always exactly one cell" |
| 4. flood-fill pooling | (he removed lakes entirely) | priority flood + glacial carving |
| 5. directionless streams | stream momentum map coupling particles | `S_MOMA/S_MOMB`, same force law |

The momentum coupling is a faithful port, dot-product scaling and all, and
`c_eq` already carries his discharge factor `(1 + k_disc·q)`. On lakes we are
ahead of him: he deleted them and still calls dynamic lakes unsolved, while we
have glacial carving producing real ones.

**Not compute starvation.** Quadrupling `particles_per_cell` (0.25 → 1.0)
changed nothing: β 4.86 → 4.90, drainage density 2.00 → 2.00 km/km²,
hillslope 200 → 180 m. The network is not particle-limited, so throwing more
work at it will not help.

## What actually sets it

Dissection wavelength in a landscape-evolution model scales as √(D/K) —
hillslope diffusivity against fluvial erodibility. Diffusion sets how far a
channel head can be from its neighbour, because it suppresses channel
initiation. We have two diffusive passes, both applied every iteration:

* `thermal_rate` (0.5) — mass wasting above the talus angle. Our slopes ride
  that angle, so it is active over much of the map, not just on cliffs.
* `creep_rate` (0.1) — the same pass with **talus 0**, i.e. unconditional
  linear diffusion on every cell.

Sweeping them (case b, 120 iterations, everything else fixed) shows they are
redundant smoothers — removing either one alone barely moves β:

| thermal | creep | talus | β | p99 slope |
|---------|-------|-------|---|-----------|
| 0.5 | 0.1 | 0.6/1.2 | 5.43 | 1.14 (49°) |
| 0.0 | 0.1 | 0.6/1.2 | 5.13 | 1.44 |
| 0.5 | 0.0 | 0.6/1.2 | 4.83 | 1.29 |
| 0.5 | 0.0 | 1.2/2.0 | 4.40 | 1.91 (62°) |
| **0.0** | **0.0** | — | **2.86** | **5.68 (80°)** |

## The trap, and a correction

Turning both off gets β to 2.86 — but at 80° slopes, precisely the jagged,
deep-ridged failure McDonald describes in his Criticism 1 and which
avalanching exists to prevent. Heights are stored in cell units and
`height_unit == cell_size`, so those slope figures are true tangents; 5.68 is
vertical rock.

**An earlier draft of this file concluded from that table that the fix was to
lower diffusion and raise erodibility together, moving √(D/K). That was
wrong, and a better metric refuted it.** The drainage-density figure it rested
on was computed as "cells above the 90th percentile of discharge", which marks
10 % of cells by construction and therefore reported an identical 2.00 km/km²
for every configuration — it could not see the quantity it was named after.
With a support-area threshold instead, the direction reverses: **less
diffusion gives *fewer*, larger channels and *longer* hillslopes** (0.53 →
0.32 km/km², 391 → 502 m). The β gain from removing diffusion is small-scale
noise, not finer dissection.

## Every erosion knob is a null result

With the corrected metric, drainage density does not respond to erosion
parameters at all. `scripts/sweep_dissection.sh` reproduces these:

| knob | range tried | drainage density |
|------|-------------|------------------|
| `particles_per_cell` | 0.25 → 1.0 (4×) | 0.42 → 0.42 |
| `disc_saturation` (channel initiation) | 32 → 2 (16×) | 0.42 → 0.42 |
| `thermal_rate` / `creep_rate` / talus | full off → default | 0.32 – 0.53 |
| initial noise amplitude | 0.5 → 12 (24×) | 0.52 → 0.31 (*wrong way*) |

Nothing reaches even 1 km/km², against a 4–8 target. The only knob that moves
the network is the amplitude of the *initial* terrain, and it moves it the
wrong way: more noise concentrates drainage into fewer, larger paths.

## What that means

The deficit is **structural, not a tuning problem**. The pattern — erosion
parameters inert, initial-terrain spectrum decisive — is what you would see if
the kernel only ever *deepens* the valley network implied by its input and
never subdivides it. Our input is a smooth, low-frequency tectonic field, so
the network it implies is coarse, and no amount of erosion refines it.

That is the thing to test next, and it is a question about the kernel rather
than about parameters: does a channel head ever migrate or a new tributary
ever appear, or is the network topology fixed from iteration 1? Instrument the
drainage network's topology over iterations and count new channel heads. If
the count is zero, no parameter will ever fix this and the work belongs in
channel initiation.

A second, independent ceiling sits downstream: `refine/basin_job.py`'s
`block_drift` subtracts every per-coarse-cell mean from the refined surface,
so the fine pass "agrees with the coarse one at coarse-cell scale by
construction and only sub-coarse detail remains". Whatever the erosion learns
to do at 50 m and above, refinement is structurally forbidden from adding it.
