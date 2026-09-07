# Developer guide: cross-stage contract

PLAN.md is the design.  This file pins the details the plan leaves open so
that stages written independently fit together.  Change it here first if
you must change a contract.

## Grids

| name | N | cell size | where |
|---|---|---|---|
| coarse | `N_c` | `cell_size_m` | `params.coarse_grid()` — all `coarse/*` fields |
| tect | `N_tect` (≤ N_c) | `cell_size_m·N_c/N_tect` | `params.tect_grid()` — tectonics internal only |
| fine | `N_c·R` | `cell_size_m/R` | `params.fine_grid()` — `fine/*` arrays and tiles |

All `Grid`s share `R_planet` (derived from the coarse grid).

## Coarse fields (`coarse/<name>.f{0..5}.npy`, interior `N_c×N_c`, `[i, j]`)

| stage | field | dtype | units / meaning |
|---|---|---|---|
| tectonics | `bedrock` | f32 | metres; shifted so `land_fraction` of cells are ≥ 0 |
| | `uplift` | f32 | metres per **erosion iteration** (already divided by `erosion.iterations`, scaled by `uplift_scale`) |
| | `hardness` | f32 | [0,1], 1 = hardest |
| | `plate_id` | i16 | plate index; every cell has one |
| | `plate_vel` | f32 (2) | contravariant coarse cells per tectonic step (vector field) |
| climate | `temperature` | f32 | °C |
| | `wind` | f32 (2) | contravariant coarse cells per advection step (vector field) |
| | `precip` | f32 | *volume* per cell per erosion iteration: rain × `cell_area/cell_size_m²`; mean over land = `precip_mean` (nondimensional particle-volume units). Ocean cells hold the (small, unfloored) rain of the ocean source on the same scale — consumers mask to land |
| | `evap` | f32 | dimensionless evaporation multiplier `k_evap·max(T,0)` (~1 at `T_eq`, 0 where `T ≤ 0`); erosion decays particles by `volume *= 1 − dt·erosion.evap_rate·evap` |
| erosion | `height` | f32 | bedrock surface after erosion, metres, sea level 0 |
| | `sediment` | f32 | metres of loose sediment on top of `height`; the terrain surface is `height + sediment` |
| | `discharge` | f32 | EMA of particle volume passing through the cell per iteration |
| | `momentum` | f32 (2) | EMA of volume-weighted particle velocity (contravariant cells/step) |
| hydro | `water_surface` | f32 | metres; `≥ surface`; equal to surface except in lakes; ocean cells = 0 |
| | `flow_dir` | u8 | D8 code 0..7 (below), 255 = ocean / no outflow (endorheic pit is impossible after priority flood, so 255 ⇒ ocean) |
| | `flow_acc` | f32 | accumulated `precip` volume (includes the cell's own) |
| watersheds | `basin_id` | i32 | −1 = ocean; otherwise a basin id from `graph/basins.json` |
| derive | `biome` | u8 | biome code (see `derive/biomes.py`); 0 = ocean |

`surface = height + sediment` everywhere; `ocean = surface < 0` (hydro may
re-quantile `height` so that exactly `land_fraction` is land).

## D8

```
D8_OFFSETS = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]  # (di, dj) for code 0..7
```
The downstream cell of `(f, i, j)` with code `k` is the extended cell
`(f, i+H+di, j+H+dj)` mapped through `grid.owner` (a flat index into the
`(6, NE, NE)` extended array; use `grid.unflat_index` to get `(f', ei', ej')`
and subtract `H`).  Codes are defined in `globe/hydro/d8.py` together with
`downstream_flat(owner, N, H, f, i, j, code)`.

## Fine data (`fine/<name>.f{0..5}.npy`, `N_fine×N_fine`, `[i, j]`, no halo)

Written with `np.lib.format.open_memmap` so a face is never fully resident
when it is large.  Producers and consumers:

| field | dtype | producer | consumers |
|---|---|---|---|
| `height`, `sediment`, `water_surface`, `discharge` | f32 | refine | derive, tiles |
| `hardness` | f32 | refine (upsampled) | tiles |
| `basin_id` | i32 | refine (upsampled, −1 ocean) | derive, tiles |
| `biome`, `vegetation`, `river_mask` | u8 | derive | tiles |

`tiles` treats a missing fine field as zeros.  The stub `refine` writes
plain upsampled coarse fields so `derive`/`tiles` can be developed alone.

## Basins

* Basins never cross a cube-face edge: a channel crossing an edge is cut
  there and the upstream part becomes a child basin whose outlet is the
  first cell on the other face.  A refinement window is therefore a
  rectangle on one face (plus a halo that may extend past the edge; sample
  it with `FaceField.sample_window`).
* `graph/basins.json`: `{"basins": [{"id", "parent" (−1 root), "face",
  "outlet": [face, i, j], "downstream_basin" (id or −1 ocean), "area_cells",
  "bbox": [i0, j0, i1, j1] (exclusive), "order" (Strahler at outlet),
  "tiles": [[lod0_x, lod0_y], ...]}]}`
* Each basin record also carries `"exits": [[f,i,j], ...]` — every cell of
  the basin whose downstream cell lies outside it (ocean, another face or
  another basin), `outlet` first — with a parallel `"exit_kinds"`
  (`"ocean" | "face" | "basin"`), plus `"outlet_downstream"` ([f,i,j] or
  null).  Merged coastal strips and face-edge slivers have many exits, so
  **refine seeds its basin-local priority flood with every exit cell and
  treats every exit as a particle sink**; using `outlet` alone would dam the
  other exits.  Basins smaller than `basin_min_cells` carry
  `"undersized_reason"`: `"max"` (every same-face neighbour would exceed
  the maximum), `"edge"` (no same-face neighbour but land across a cube
  edge — refine may process such a sliver together with the basin across
  the edge) or `"island"` (no land neighbour at all).

## Graph JSON

* `graph/drainage.json`: `{"nodes": [{"id", "cell": [f,i,j], "kind": "source|junction|outlet|lake_in|lake_out", "acc"}],
  "edges": [{"id", "from", "to", "order", "length_m", "mean_discharge", "cells": [[f,i,j], ...]}]}`
* `graph/lakes_coarse.json` (hydro) and `graph/lakes.json` (derive, fine):
  `{"lakes": [{"id", "surface_m", "area_m2", "outlet": [f,i,j] or null,
  "polygon": [[f,u,v], ...]}]}`
* `graph/rivers.json` (derive): `{"rivers": [{"id", "edge_id", "order",
  "points": [[f,u,v,width_m,height_m], ...]}]}` (`height_m` = water surface,
  non-increasing downstream; polylines are per face and meet at cube edges)

## Erosion kernel contract (`globe/erosion/particle.py`)

The same kernel runs globally (6 faces, particles transfer across edges
with `cubesphere.transfer_velocity`) and inside a basin window (1 face
array, particles leaving the window or the mask die).  Inputs are plain
arrays shaped `(F, NE, NE[, 2])` plus `N, H`, `metric_inv`, `mask`
(uint8: 0 outside, 1 active, 2 active-but-frozen divide), and a flag
`spherical`.  Heights inside the kernel are in **cell units**
(`metres / cell_size_m`) so the ★ parameters apply as published.

Evaporation: the climate field `evap` is a dimensionless multiplier
(`k_evap·max(T,0)`, ~1 at `T_eq`, 0 where `T ≤ 0`); the kernel decays a
particle by `volume *= 1 − dt·erosion.evap_rate·evap[cell]`.  PLAN 8.2's
`evap_at_pos` reads as `evap_rate·evap(pos)`, so the ★ `evap_rate` keeps
its published meaning.

## Tiles (`tiles/L{lod}/f{face}/{x}_{y}/`, see `globe/io/tiles.py`)

* A tile holds `(T+1)×(T+1)` **vertex** samples.  Sample `(k, l)` of tile
  `(lod, face, x, y)` sits at face-local
  `u = (x·T + k)·2^lod / N_fine`, `v = (y·T + l)·2^lod / N_fine`,
  `k, l = 0..T` — on fine-cell corners.  The `k = T` column is the next
  tile's `k = 0` column; on the last tile of a face it lies exactly on the
  cube edge (`u = 1`), where the neighbouring face's tile places its own
  edge column, so meshes share vertices across face edges too.
* Values reduce the fine cells around the corner (2×2; 2 + 2 across a
  face edge, 3 at a cube corner; `globe/refine/lod.py`): mean for the
  float channels, max for the lake surface and river mask, mode for ids and
  biome (ties → largest value).  The reduction is permutation invariant and
  uses the same cells on both faces, so a shared vertex is **bit-identical**
  on both tiles before quantisation (after it: within the two tiles'
  16-bit steps).  Skirts still hide LOD/quantisation cracks.
* `height.png` is the terrain **surface** `height + sediment` (what the
  mesh displaces and what `water_surface` is measured against); the
  sediment depth is `layers.R`, so bedrock = surface − sediment.
* LOD `l+1` vertex `(k, m)` reduces the 3×3 LOD-`l` vertices around LOD-`l`
  vertex `(2k, 2m)` — across face edges (6 own + 3 neighbour vertices; the
  7 distinct vertices at a cube corner).  Height takes the centre, so LOD
  `l+1` heights **are** the strided LOD-`l` (and LOD-0) vertices; water and
  river mask max; basin id and biome mode; the other byte channels mean
  (exact, round half up).  `max_lod` has one tile per face.
* Arrays are `[i, j]` (`i` along `u`); images are row = `j`/`v`,
  column = `i`/`u` (the transpose).
* Ocean is implicit: `water.png` is 0 wherever the fine `water_surface` is
  ≤ 0 or less than 0.05 m above the surface; render sea level 0 where
  height < 0 and use `water.png` for lakes only.
* `flow.G` is the index into `meta["basins"]` (basin ids present in the
  tile, −1 excluded, most frequent first, capped at 255 entries); 255 =
  ocean / not listed (basins beyond the first 255 in a tile), so index
  255 never names a basin.
* `meta["neighbors"]` = `[[lod, face, x, y] × 4]` of the edge-adjacent
  tiles across the tile's sides `+i (u = 1), −i, +j (v = 1), −j`, crossing
  cube edges through `cubesphere` (`lod.tile_neighbors`; the neighbour's
  along-index is reversed on 4 of the 12 cube edges, i.e. 8 of the 24
  `(face, side)` links, see `lod.edge_links`).
* `tiles/index.json`: `{"lods": [{"lod", "tiles_per_face", "size"}], "T",
  "N_fine", "R_planet", "max_lod", "faces": 6, "sides": ["+i","-i","+j","-j"]}`
  (`size` = `T + 1` samples per tile edge).
* Deviations from PLAN §3: `flow.png` is RGB8 (R = log discharge,
  G = basin-local id, B = river mask); `flow_dir` is `uint8`; `meta.json`
  carries the `neighbors` entry above; height is the surface.

## Quicklooks

Each stage module defines `quicklook(store, params, path)`.  Use
`globe.viz.quicklook` helpers; write the unfolded net.

## Tests

`bake/tests/test_<stage>.py`, runnable in < 60 s on the `tiny`/`small`
presets; never depend on another stage's real implementation when its stub
suffices.

## Biome codes (`derive/biomes.py`, coarse `biome` and `fine/biome`)

`uint8`; Whittaker-style lookup on temperature `T` (°C) and annual
precipitation `P` (cm) with overrides.  `P = min(derive.precip_scale_cm ·
wetness^derive.precip_gamma, derive.precip_max_cm)`, `wetness = precip /
(cell_area/cell_size_m²) / mean`, the rain rate over its land mean
(`climate.precip_mean` by contract; derive measures it).  `precip_gamma =
0.5` compresses the skewed climate output: the land mean maps to 100 cm,
a quarter of it to 50 cm, a sixteenth to 25 cm (desert), 4× to 200 cm.
Override precedence, lowest to highest: riparian < alpine < cliff <
wetland < lake < ocean.

| code | name | rule |
|---:|---|---|
| 0 | ocean | `surface < 0` |
| 1 | ice | T < −12 |
| 2 | tundra | T < −2, or T < 5 and P < 20 |
| 3 | boreal_forest | T < 5, P ≥ 20 |
| 4 | temperate_grassland | 5 ≤ T < 20, 25 ≤ P < 60, T < 13 |
| 5 | temperate_forest | 5 ≤ T < 20, 60 ≤ P < 180 |
| 6 | temperate_rainforest | 5 ≤ T < 20, P ≥ 180 |
| 7 | desert | T ≥ 5 and P < 25 (P < 30 when T ≥ 20) |
| 8 | shrubland | 5 ≤ T < 20, 25 ≤ P < 60, T ≥ 13 |
| 9 | savanna | T ≥ 20, 30 ≤ P < 100 |
| 10 | tropical_seasonal_forest | T ≥ 20, 100 ≤ P < 220 |
| 11 | tropical_rainforest | T ≥ 20, P ≥ 220 |
| 12 | alpine | `surface ≥ derive.alpine_min_m` and T < `derive.alpine_T` |
| 13 | cliff | slope > `derive.cliff_slope` (rise/run) |
| 14 | riparian | within `derive.riparian_cells` coarse cells (× R at fine) of a channel / river-mask cell |
| 15 | wetland | within `derive.wetland_cells` of a lake cell |
| 16 | lake | `water_surface − surface > hydro.lake_min_depth` on land |

`fine/vegetation` is `255 · sqrt(min(P/100, 1)) · sqrt(1 − min(slope/cliff_slope, 1))
· biome_factor · soil_factor` (biome factors in `biomes.BIOMES`, soil factor
`0.7 + 0.3·min(sediment/derive.soil_full_depth_m, 1)` from `derive/soil.py`).
`fine/river_mask` is 255 within `w/2` of a river centreline, `w = a·(Q/Q_thr)^b`
fine cells (`Q_thr` = the discharge threshold recorded in `graph/rivers.json`).
`graph/rivers.json` polylines use cell-centre `u = (i+0.5)/N_fine`;
`graph/lakes.json` rings use the corner lattice `u = i/N_fine` (closed,
counter-clockwise outer rings, clockwise holes; `rings` lists every face
piece of a lake, `polygon` the largest; pieces touching across a cube edge
at one level are one lake; `coarse_id` / `outlet` are the `id` / `outlet`
of the `lakes_coarse.json` lake under it, `-1` / `null` when none).
The river mask's hysteresis / minimum-size decision is global (blobs are
joined across cube edges), the fine slope stencil reads the neighbouring
faces, so `fine/biome`, `fine/vegetation` and `fine/river_mask` are
seamless up to the mask discs (`w/2` cells) at an edge.
