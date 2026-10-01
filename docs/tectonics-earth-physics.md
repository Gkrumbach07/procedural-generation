# Tectonics closer to Earth: the start, island arcs, and the crust that went missing

This round (branch `tectonics-earth-physics`, 2026-09-30 to 10-01) set out to answer three complaints:

1. The planet started as one supercontinent plus a few ocean plates. They broke up and moved in odd directions.
2. Island arcs never showed up.
3. The model kept "losing mass".

Every claim below was measured on the Earth preset over several seeds (0-4 and 1423). Each finding was then re-measured by an independent reviewer on seeds the first agent had not used. Time is reported at **0.15 My per step** (`tectonics.myr_per_step`), the value that plate speeds, sea-floor ages and `ridge_age` all point to. A 4000-step run is therefore about 600 My.

## 1. What was wrong

### The start and the "weird directions"
The random initial Euler poles were not the cause. Damping 0.05 erases them within about 50 steps. The lasting causes were these:

- **An anti-girdle.**
  - Plates move toward *high* heat, so in this code "heat" behaves as negative mantle temperature.
  - The insulation rule `bg -= ins * (2*frac - 1)` drove every ocean cell toward 1 within 250-500 steps.
  - That pulled sea floor *away* from the continent. 86-95 % of the supercontinent's coast became passive margin, i.e. no ring of subduction around it.
  - Pangaea had about 65,500 km of trench. The model had about 7,000 km of convergent boundary.
- **A plate bigger than a hemisphere.**
  - The supercontinent started at 0.75 of the sphere and grew to 0.77-0.92 by absorbing new ridge crust.
- **The first rift had to converge somewhere.**
  - It cut a great circle through that plate.
  - Along a cut longer than 180°, any rigid relative rotation closes on part of the cut.
  - 69-76 % of the two halves' collisions happened more than 90° from the rift. From then on, continent-continent contacts were 80-96 % of all collisions.
- **Flicker and teleporting crust.**
  - `snap_cratons` undid 68-74 % of collision relabels.
  - `split_disconnected` made 450-2,100 plates per run with a median life of 2-9 steps.
  - Drag was inertia (area × column mass), so small plates raced and continents were sticky for the wrong reason.
  - The shipped `slab_pull` was proportional to this step's subduction, which acts as negative drag.

### Island arcs
- **Born as specks.** An arc was a 5 % coin flip that turned one trench segment into continental crust about 1.5 km below sea level.
- **Blurred away.** The 160 km splat gives a lone segment about a quarter of its height, so arcs never emerged.
- **More segments would not help.** The splat is sized in segment spacings, so the effect is resolution-invariant. 4x the segments would cost 6-8x the time and lose crust faster.
- **They piled up.** Arc crust could never subduct, so 28-56 % of it ended stranded mid-plate. It covered 3.4-5.1 % of the planet, against Earth's ~1 %.
- **Trench froth.** 32 % of new sea floor was spawned into the holes trenches leave, and consumed again at once.

### The missing mass
- **The log number.**
  - `mass=` in the log, and the stage's `final_mass`, were `total_mass`, a column sum that variable extent does not conserve.
  - It fell 25 % by step 4000 while the conserved crust fell 7 %.
- **Real sinks.** Collision-thickened crust was sent straight to the mantle: `orogen_decay` and delamination.
- **Hidden by bookkeeping.** About 64 % of that loss was offset by artefacts filed under 'subducted':
  - column-for-column transfers in `shape_belt`/relax;
  - an `extent_min` merge that created 19 % of the pair's crust;
  - the extent closure.
- **What the viewer drew.** Continental area followed the *count* of continental segments, which only fell (19-20k merges against about 9k arc births). Extent was invisible.
- **The 75 % start was a calibration of a transient.** Every seed ended near 23 % continental by step 8000.

## 2. What changed

**Bookkeeping (Phase I)**
- Per-kind crust books (`sim.books`). Continental and oceanic each close to about 1e-14 every step.
- Every lateral transfer conserves Σext·th. The merge fix.
- The log reports `crust=`, `cont=` and `vol_c=`.
- Plates stop flickering:
  - cratons snap only where a boundary is drawn;
  - relabels happen only when ground is spent;
  - plate minimums are areas;
  - fragments weld whole.
- `finalise()` runs under 2 GB, and the step is about 1.5x faster.
- `scripts/tect_scorecard.py` measures all of this against Earth targets.

**Dynamics** (`forces.py`, `intraplate.py`)
- **A Pangaea-era start.**
  - A supercontinent of 0.40 on one plate.
  - 8 ocean plates with Zipf sizes.
  - Ocean floor aged from the ridges the initial forces open.
  - A subduction girdle under the continental margin.
  - Omegas from the force balance.
- **One-sided insulation.** Continents become an upwelling; the ocean relaxes to neutral.
- **Rate-independent slab pull** from a mantle-frame slab field, solved per plate with boundary, basal (keel) and continent-continent collisional drag.
- **Rifts.**
  - Triggered by insulation under a large continent.
  - Cut where the released halves actually open: no far side, no kick.
  - Slow then fast, by necking. Failed rifts heal.
- **Life cycle of plates and margins.**
  - Old passive margins founder into new trenches.
  - Microplates are captured.
  - Quiet sutures weld.
- **Spawning.** A two-plate gap spawns sea floor only where the plates separate.
- **`CLASSIC_DYNAMICS`.** The `small` and `tiny` presets keep the old model, bit-identical.

**Island arcs**
- No coin flip.
- Arc crust is added at the volcanic front, about 180 km behind the trench, from the slab's share.
- Thin arcs partly subduct. Thick ones dock as terranes, with the change of kind booked. Roots founder above 35 km.
- The arc ridge is redrawn at its own width.
- Volcanic edifices grow only on arc crust at persistent trenches. They are stamped at finalise and re-added after erosion while active.
- Hotspot chains.

**Continental mass**
- Collapse and the thickness cap return 85 % / 80 % of what they take as ground, at constant volume.
- Belts collapse with a 200 My e-fold toward a 0.15-column floor.
- Margin erosion at ocean-under-continent contacts: 0.9 km per km of slab.
- The extent closure acts on sea floor only.
- A visibility-only extent split, so the map shows the ground the crust owns.
- Rifted margins stretch at the divergence's rate over 150 km.

## 3. Where it lands (Earth preset, 600 My unless noted)

| | shipped (e2870bd) | now | Earth |
|---|---|---|---|
| continental extent | 0.32-0.59 (draining) | 0.34-0.38 | ~0.40 |
| continental volume vs start | 0.63-0.91 | 1.03-1.07 | ~steady |
| subduction at the start | ~7k km | 55-75k km | 65.5k (Pangaea) |
| coast with a trench within 400 km, step 250 | 0.10-0.14 | 0.78-0.91 | girdle |
| rift far-side convergence | 0.69-0.76 (first rift) | 0.00 | 0 |
| plates ≥ 1 % / largest plate | 8-13 / 0.18-0.48 | 8-18 / 0.15-0.30 | ~12 / 0.20 |
| median plate life | 0.5-0.9 My | 11-14 My | small plates 10-20 My |
| plates at the speed cap | up to 33 % | 0 | 0 |
| slab-attached / continental speed | noisy, capped | 5-10 / 1-5 cm/yr | 8 / 2.8 |
| supercontinent cycle | one break-up, no reassembly | break-up at 40-72 My, reassembly to 0.8-1.0 at 300-500 My | 400-800 My cycle |
| arc crust / arc crest p50 | 3.4-5 % specks, never emerge | 0.4-2.4 % / -2.1 to -2.6 km | ~1 % / -1 to -3 km |
| arc islands per km of ocean-ocean trench | 0 | 0.9-4.8 km² | 0.4-9 |
| land 0-1 / 1-2 / 2-3 / 3-4 km (finalise bedrock) | 38-73 / 25-45 / 2-15 / 0-3 % | 59-77 / 18-23 / 4-8 / 1-6 % | 71 / 15 / 7.5 / 3.8 |
| gross continental recycling | ~25 km³/yr | 2.3-3.3 km³/yr | 3.2-4.9 |

**Still open**
- **Continental extent settles a little low** (0.34-0.38, so land is 23-25 %).
  - Slower collapse keeps the crust thick rather than spreading it.
  - The lever is `orogen_return` or a start above 0.40. Neither has been tried.
- **The ocean-age tail is long.** max/mean is 7-15 against Earth's ~3.
- **Rifts open slower than Earth's fast phase.** The peak is 2.1-3.1 cm/yr against >3.5.
- **Collision count is misleading.** Counted per collision, continent-continent contacts are 0.4-0.7 of collisions after assembly. Measured by convergence (`cc_kin_share`) they are 0.02-0.05, so read that number instead.

## Where the evidence is
Scratchpad of session 226075d6 (not in the repo):
- `BRIEF.md` (the verified diagnosis);
- `synth/`, `synth2/`, `synth3/` (every investigation, design, judge and review);
- `foundation/` and `baseline/` scorecards;
- `tune/` (the collapse sweep).

The scorecard reproduces any row above: `python bake/scripts/tect_scorecard.py --preset earth --seeds 0 1 2 3 --steps 4000`.
