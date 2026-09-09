# Why the terrain reads as bland, measured

The recurring complaint about this world is that its features are smooth and
samey — "they all follow a pretty standard river system down". This is the
measurement behind that, and what it rules in and out.

Run the metrics with `bake/scripts/terrain_stats.py`.

## The numbers

Measured on the shipped `final512` world (fine grid, 12.5 m cells):

| metric | ours | real landscapes |
|--------|------|-----------------|
| spectral slope β (P ~ k⁻ᵝ, 60–600 m) | **4–6** | ~2 |
| drainage density | **1.89 km/km²** | 2–20; humid temperate 4–8 |
| median hillslope length | **403 m** (p90 898 m) | 50–200 m |

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

## The trap

Turning both off gets β to 2.86 — but at 80° slopes, which is precisely the
jagged, deep-ridged failure McDonald describes in his Criticism 1 and which
avalanching exists to prevent. Heights are stored in cell units and
`height_unit == cell_size`, so those slope figures are true tangents; 5.68 is
vertical rock.

So the mass-wasting rate is a **symptom knob**, not the cause. It trades β
against slope realism along a single axis, and neither end of that axis is
right. Real terrain achieves β ≈ 2 *while* keeping slopes below the angle of
repose, because its variance comes from a branching valley network rather than
from steepening individual cells.

The prescription that follows is to move D and K *together* — lower the
diffusion and raise `erodibility` in step — so hillslopes shorten and new
channel heads form, instead of the existing slopes simply getting steeper.
That is the experiment to run next, with the three metrics above as the
objective and `p99 slope < 1.2` (the talus angle) as the constraint.

A second, independent ceiling sits downstream: `refine/basin_job.py`'s
`block_drift` subtracts every per-coarse-cell mean from the refined surface,
so the fine pass "agrees with the coarse one at coarse-cell scale by
construction and only sub-coarse detail remains". Whatever the erosion learns
to do at 50 m and above, refinement is structurally forbidden from adding it.
