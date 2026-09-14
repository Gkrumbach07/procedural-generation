# Watersheds that cross cube-face edges

> **Verdict: the partition half ships; the fine raster at the seams is
> unchanged.**  Basins are now drainage basins of the cross-face flow
> graph (small preset: 160 -> 149 basins, 0 -> 24 of them on two faces,
> every one of the 210 cross-face drainage links inside a basin; earth-v5's
> flow_dir: 477 -> 450 basins, 46 on two or more faces holding 51 % of the
> land), and the basins layer no longer shows the cube edges as straight
> meridians through the continents.  Refine runs one job per (basin, face)
> piece with the erosion mask following the basin across the edge, but the
> seam row of every piece stays pinned to the plain upsample exactly as a
> divide is, so the fine surface across the 12 edges is *identical to the
> digit* before and after (mean |step| 2.2997 m both, 0.87x the first step
> inside the face) and the detail-deficit strip along the edges remains
> (0 -> 0.05 m over 9 fine cells against 0.11 m in the interior on small;
> 0 -> 0.85 m against 2.5 m on Earth).  Removing that strip needs a
> seam-blend pass (section 5), which this change does not attempt.
>
> **Update: the seam blend (section 5) removes the strip.**  Each piece of
> a multi-face basin returns its refined off-face strip resampled onto the
> neighbouring face, and the two pieces are blended with complementary
> weights over `feather_cells` on either side of the edge.  Small preset,
> seeds 0 / 1: detail |refined - plain| on the seam row where the basin
> continues 0 -> 0.024 / 0.045 m (the same basins 4..9 cells in: 0.030 /
> 0.048 m), seam / first-step-inside ratio 0.926 -> 0.912 / 0.888 -> 0.867
> (no crease), only the 4,129 / 7,075 blended cells change, fine
> discharge and basin ids byte-identical, the blend itself 0.01 s.

`bake/globe/hydro/watersheds.py` partitioned the drainage into basins
with a cut at every cube-face edge: a land cell whose downstream cell lay
on another face was an outlet, so no basin ever spanned two faces and
`refine/basin_job.py` could treat every basin as a window on one face.
Hydro's flow itself crosses faces (`downstream_table(fd, grid.owner, H)`,
D8 through the halo/owner tables), so the cut was purely an artefact of
the refine design, and it showed: the viewer's basins layer drew the six
face boundaries as straight lines through every continent, and a river
crossing an edge was refined in two jobs that met at the seam.

## 1. What the cut cost (earth-v5, read-only)

`python scripts/face_seams.py worlds/earth-v5`:

| | earth-v5 |
|---|---|
| basins / on more than one face | 477 / 0 |
| land cells | 1,903,249 |
| land cells whose downstream cell is on another face | 3,319 |
| of those inside one basin | 0 |
| exits of kind `face` (in basins) | 3,319 (66) |
| undersized reasons | island 296, **edge 11**, max 0 |

19.4 % of the land (369,639 cells) drains through one of those 3,319
links; the largest single crossing carries 68,092 upstream cells.  Labelling
every land cell by the coastal outlet it reaches gives 38,214 true drainage
basins of which 411 span faces (4 span three) and 738,698 cells (38.8 % of
the land) lie in one of them.

## 2. The change

*Partition* (`hydro/watersheds.py`):

* the cut rule is `cut = land & ((down < 0) | ~land[dn])` — outlets are
  the last land cells before the sea, on whatever face.  The labelling,
  the Pfafstetter split kernels and the topological order already worked
  on cell ids of the cross-face graph and needed nothing;
* the merge table gains the cross-face contacts: a numba kernel over the
  face-border cells (`_cross_face_boundary_pairs`, `neighbor_cid` through
  `grid.owner`, weights 2 / 1 for edge / diagonal contact like the
  same-face pairs, each contact counted once from its lower cell id — the
  cross-face neighbour map is symmetric at N = 16, 32, 64, checked), so a
  coastal sliver on one face merges into the basin across the edge like
  one beside it.  The `edge` undersized reason ("no same-face neighbour
  but land across the edge") becomes unreachable and is retired;
* a cell whose downstream cell is on another face stays an `exit` of kind
  `face` — now *inside* its basin — because the per-face refine job still
  floods and drains there;
* records: `pieces: [{face, bbox, area_cells, tiles}]` per face in face
  order and `faces`; the top-level `face` / `bbox` / `tiles` are the outlet
  face's piece so old readers see one piece.  `info` reports
  `n_multi_face`, `n_pieces`, `cells_in_multi_face_basins`.

*Refine* (`refine/upsample.py`, `basin_job.py`, `rasterize.py`, `run.py`):

* one job per (basin, face) piece (`run.basin_pieces`, largest piece
  first, ties by id then face); `run_basin(..., face=)` takes the piece's
  window (`basin_window(basin, R, halo, face=, N=)`) and its own stream
  `params.rng("refine", bid, face)`;
* `build_mask` needed no change: `sample_window` looks the nearest
  `basin_id` up on the owning face beyond the edge, so the cells of the
  same basin past the edge come out mask 1 and the frozen ring moves from
  the face edge to the true divide.  Particles now cross the edge into the
  strip of the window that lies on the neighbouring face (they used to die
  on the first mask-0 cell there);
* the off-face frozen ring is added to the flood drains: without it the
  strip beyond the edge, lying below the face exits, ponded up to their
  level;
* `blend_result` computes the feather weight on `own & on_face`, so the
  seam row of a piece has weight 0 like a divide and the two independently
  eroded pieces meet on the plain upsample.  Without this the weight at
  the seam would be 1 and two erosion runs would meet with a crease
  (estimated ~sqrt(2) x 2.5 m mean mismatch on a 13 m natural step at
  Earth);
* the square padding of a window now shifts towards the face interior by
  as much as the padding allows (bbox + halo never moves): a thin piece
  hugging an edge used to put up to half its long side beyond the edge,
  where the gnomonic extension is increasingly distorted.  No window
  extends past an edge by more than `halo_cells` any more (unless the
padded square is wider than the face, which happens only on the tiny preset).

`stubs.stub_watersheds` emits `pieces` so a real refine runs on stub
watersheds; `docs/DEVELOPING.md` carries the new schema.  No parameter
changes; `KERNEL_VERSION` and the erosion checkpoints are untouched
(nothing here is upstream of hydro).

## 3. Measured: small preset, seed 0

`scripts/bake.py --world scratch/xface/small-{before,after} --preset small
--set refine.workers=2` (25 s each), then `python scripts/face_seams.py
<world>`:

| | before | after |
|---|---|---|
| basins | 160 | 149 |
| on more than one face | 0 | 24 (0 on three) |
| cells in multi-face basins | 0 | 7,835 of 29,494 |
| largest basin (cells; `basin_max_cells` 2,304) | 1,743 | 2,276 |
| cross-face drainage links inside one basin | 0 / 210 | 210 / 210 |
| exits of kind `face` (basins) | 210 (26) | 210 (24) |
| undersized | island 2, edge 1 | island 2 |
| initial outlets / merges | 2,510 / 2,351 | 2,300 / 2,152 |
| refine jobs | 160 | 173 pieces |
| fine cells written | 117,976 | 117,976 |
| refine deaths `exit` | 406,645 | 448,241 |
| refine wall / cpu (2 workers) | 2.2 s / 2.7 s | 2.4 s / 3.1 s |
| mean \|step\| across the seams (m) | 2.2997 | 2.2997 |
| mean \|step\| first cell inside (m) | 2.656 | 2.656 |
| seam / inside ratio | 0.866 | 0.866 |
| detail \|refined - plain\| at 0, 1, 2 ... 9 cells from the seam (m) | 0, .007, .033, .047, .043, .051, .043, .051, .055, .057 | 0, .010, .025, .034, .044, .049, .044, .049, .052, .054 |
| detail in the interior (m) | 0.116 | 0.114 |
| fine discharge ratio across the 210 face links, median (< 0.3) | 0.90 (1.4 %) | 0.89 (1.4 %) |

Reading it:

* the partition is what it should be.  Every land cell still belongs to
  exactly one basin, every exit is reached without leaving the basin (the
  `_check_partition` invariants), and the only remaining cuts are the
  coast and the Pfafstetter splits (the test asserts that every cross-face
  link is inside a basin or at a split outlet).  The viewer's basin layer
  (`scripts/viewer_shot.py <world> --layer basin --view flat`) shows the
  basins running across the ±45° face boundaries that used to slice them;
* (history) at this change the fine raster at the seams did not move, to
  the digit: both seam rows were the plain upsample (before: the frozen
  ring at the face edge; after: the feather weight restricted to the
  face), so the detail strip along the edges stayed — "relabelled, not
  fixed".  The seam blend of section 5 replaces that restriction; on the
  small preset the seam-row detail where a basin continues across is now
  0.024 m (seed 0) and 0.045 m (seed 1) instead of 0, close to the
  detail a few cells in, with the seam / inside step ratio 0.91 and 0.87
  (section 5.3);
* the rivers across the seam are as continuous as before (0.90 -> 0.89):
  that continuity was already carried by the coarse-initialised discharge,
  and the on-face cells within `feather_cells` of the seam are blended back
  to that discharge in both worlds;
* the cost: 13 more jobs and 10 % more `exit` deaths — the strip past the
  edge is eroded by both pieces that reach it (particles spawn there and
  leave through the window border) and written by neither.  The strip is
  at most `halo_cells` + padding wide.

## 4. Earth (partition only; no bake)

Running the new `partition` on earth-v5's baked `flow_dir` (read-only,
in-process, warm numba):

| | earth-v5 | new partition |
|---|---|---|
| basins | 477 | 450 |
| on more than one face / on three | 0 / 0 | 46 / 5 |
| cells in multi-face basins | 0 | 963,981 (50.6 % of the land) |
| largest basin | 107,069 | 131,875 (`basin_max_cells` 262,144, no split forced) |
| initial outlets / merges | 41,533 / 41,056 | 38,214 / 37,764 |
| undersized | island 296, edge 11 | island 296 |
| refine jobs / largest piece window (fine cells) | 477 / 1136² | 501 pieces / 986² |
| partition time | 1.07 s | 0.87 s |

The largest piece window is smaller than today's largest basin window, so
the per-worker memory estimate does not grow.  earth-v5's fine raster at
the seams, for the record (`scripts/face_seams.py worlds/earth-v5`): 19,116
land pairs across the seams, mean |step| 12.85 m against 13.43 m for the
first step inside the face (ratio 0.96), detail |refined - plain| 0, 0.02,
0.12, 0.23, 0.30, 0.42, 0.57, 0.66, 0.76, 0.85 m by depth against 2.50 m
in the interior, fine discharge ratio across the 3,319 face links median
0.81 (7.8 % below 0.3).  After this change (before the seam blend of
section 5) those numbers stay where they are, for the reason of section 3.

The stages have no code-version stamp and `stage_done` compares only the
parameter hash, so existing worlds keep the old partition until `bake
--world <w> --force --from watersheds` (watersheds 1 s + refine ~330 s +
derive + tiles + viewer, about 8 minutes at Earth; not the erosion).
Basin ids renumber (the face-cut children vanish) and every fine raster
changes (the refine stream is keyed by (basin, face) now, and edge windows
shift into the face), so comparisons to earlier worlds go through the
`basin_id` rasters, never through ids.

## 5. The strip, and the seam blend that removes it

### 5.1 The problem (as written with the partition change)

The detail-deficit strip along the cube edges is the same artefact every
basin divide has (frozen ring + `feather_cells` ramp), only straight and
1024 cells long at Earth (~10 fine cells = ~50 km per side).  With the
mask now active across the edge the erosion result *exists* on both sides
of the seam; what is missing is a way to write it.  The extended fine
lattice of a face beyond its edge is the gnomonic extension and does not
coincide with the neighbour face's lattice, so restoring detail at the
seam means a seam-blend pass: resample each piece's cross-edge strip onto
the neighbour face's lattice (`project_to_face` + bilinear) and average
the two pieces' results over `feather_cells` on each side with the seam
row at weight 1.  It is a separate change with its own measurement (the
seam / inside ratio must stay near 1 while the detail at depth 0..9 rises
to the interior's), and whether it is worth its cost should be decided on
that measurement, not here.

### 5.2 The seam blend (`refine/rasterize.py`, `basin_job.job`, `run.py`)

* `blend_result` computes the divide feather on `own` over the whole
  window, off-face strip included.  Where the basin continues across the
  edge the seam is no longer a divide; where it ends there (a real divide,
  the coast, a basin with no piece on that face) the cells across are not
  `own`, the distance is 1 on the seam row and the result is today's to
  the bit (a basin on one face writes exactly what it wrote before).  The
  *discharge* keeps the face-restricted feather: a river position cannot be
  averaged between two erosion runs, and letting each face's refined
  discharge meet at the seam dropped the fine discharge ratio across the
  face links from 0.84 / 0.86 to 0.78 / 0.77 (seeds 0 / 1) in a first
  bake; with the restriction it is unchanged to the digit.
* a piece of a multi-face basin returns `seam_records` with its stats
  (pickled back from the worker): its own detail `surface - plain` and
  `sediment - plain sediment` on its face's cells within `feather_cells` of
  every edge towards a face the basin has a piece on (with the divide
  feather `w` and the surface / water surface it wrote), and, per such
  edge, its refined off-face strip resampled onto the neighbouring face's
  fine lattice: each neighbour cell centre of the basin (coarse
  `basin_id`) within `feather_cells` of the edge goes `to_sphere_v` ->
  `project_to_face_v(window face)` -> fractional window index -> bilinear;
* weights: `a(dist) = smoothstep((dist + F) / 2F)` for the piece whose face
  the cell is on, `1 - a` for the piece across (`dist` = the cell centre's
  distance from the edge in cells, so the two are 0.59 / 0.41 on the seam
  row at F = 4 and 1/2 on the edge line itself); the crossing weight is
  also ramped to 0 over F cells at that piece's window border, so a window
  ending along the edge fades out; near a cube corner the owning weight is
  the smaller of the two edges'.  Per cell the weights are normalised, so
  they sum to 1;
* after every job has written, the driver sorts all records by (face,
  basin, kind, source face), sums them in that order and rewrites each cell
  that has the owning piece's record, a crossing record and `w > 0`:
  `surface = plain + w * sum(beta dsurf) / sum(beta)`, `sediment =
  max(plain sediment + w * sum(beta dsed) / sum(beta), 0)`, `height =
  surface - sediment`.  The detail is blended, not the absolute surface, so
  the two gnomonic samplings of the bicubic plain surface never meet;
  `w` is the owning piece's, so divides stay pinned;
* categorical fields are the owning piece's: `basin_id`, `hardness`,
  `discharge` untouched; the water surface keeps the owning piece's level
  on its lake cells (`max(level, surface)`, which never lifts it above
  refine's lake-balance cap) and is the blended surface on dry cells;
* order independence: pass 1 still writes disjoint cells, pass 2 runs once
  on the sorted records (`test_seam_blend_job_order_independent` blends
  the tiny world's records in three orders, byte-identical, and checks the
  raster holds exactly those cells);
* memory: the records live in the driver until the blend.  They are
  bounded by the 24 face sides x F rows x N_fine cells with one owning and
  at most two crossing records each: at Earth (F = 8, N_fine = 2048) at
  most 393k target cells x ~80 B = ~32 MB whatever the basin count; small
  measured 0.3 / 0.5 MB.

No parameter changes (`feather_cells` sets the blend width); the refine
outputs change only in the blended cells.

### 5.3 Measured: small preset, seeds 0 and 1

`scripts/bake.py --world scratch/seam/a8cf/small-s{0,1}-{before,after}
--preset small --seed {0,1} --set refine.workers=2` (before = commit
af58d28), then `python scripts/face_seams.py <world>`.  `face_seams.py`
now also reports the `*_same_basin` numbers: the seam positions where the
seam-row cells on both faces carry the same basin id (the cells the blend
acts on; 1,196 of 1,248 land pairs on seed 0, 2,112 of 2,368 on seed 1).

| | s0 before | s0 after | s1 before | s1 after |
|---|---|---|---|---|
| basins / on more than one face / pieces | 146 / 28 / 174 | same | 166 / 48 / 216 | same |
| fine cells written (pass 1) | 118,012 | 118,012 | 117,964 | 117,964 |
| seam cells blended (pass 2) | — | 4,129 | — | 7,075 |
| cells that differ from before (height / sediment / water surface) | | 4,121 / 4,129 / 4,124 | | 7,059 / 7,075 / 7,063 |
| ... of those further than F = 4 from an edge | | 0 | | 0 |
| discharge, basin_id, hardness cells that differ | | 0 | | 0 |
| detail at 0, 1, 2 ... 9 cells from the seam, same basin (m) | 0, .008, .016, .022, .031, .028, .034, .029, .027, .029 | .024, .024, .024, .024, .031, .028, .034, .029, .027, .029 | 0, .011, .027, .042, .046, .044, .051, .050, .047, .048 | .045, .044, .044, .047, .046, .044, .051, .050, .047, .048 |
| detail by depth, all land (m) | 0, .008, .016, .023, .030, .033 ... | .023, .023, .022, .024, .030, .033 ... | 0, .010, .024, .037, .041, .041 ... | .039, .038, .036, .041, .041, .041 ... |
| detail in the interior (m) | 0.109 | 0.109 | 0.069 | 0.069 |
| mean \|step\| across the seam, same basin (m) | 1.811 | 1.796 | 1.911 | 1.882 |
| mean \|step\| first cell inside, same basin (m) | 1.956 | 1.969 | 2.153 | 2.171 |
| seam / inside ratio, same basin | 0.926 | 0.912 | 0.888 | 0.867 |
| seam / inside ratio, all land pairs | 0.891 | 0.879 | 0.887 | 0.868 |
| p99 \|step\| across the seam (m) | 14.95 | 14.96 | 7.86 | 7.77 |
| fine discharge ratio across face links, median (< 0.3) | 0.84 (4.1 %) | 0.84 (4.1 %) | 0.86 (5.7 %) | 0.86 (5.7 %) |
| water surface < surface - 1 mm / sediment < 0 | | 0 / 0 | | 0 / 0 |
| refine basin jobs wall / cpu (s) | 3.6 / 5.2 | 2.3 / 2.9 | 3.1 / 4.2 | 2.6 / 3.3 |
| seam blend (s) / records (MB) | — | 0.01 / 0.3 | — | 0.01 / 0.5 |

Reading it:

* the strip is gone where it should be.  The seam row of a basin that
  continues across an edge carries 0.024 m / 0.045 m of detail instead of
  0, flat across rows 0..3 and close to rows 4..9 (0.027-0.034 m /
  0.044-0.051 m).  It stays somewhat below the far interior's mean (0.109
  / 0.069 m), which is the land far from any edge and was never the
  near-edge value (rows 4..9 were at the same 0.03 / 0.05 m before);
  seed 0's rows 0..3 are ~20 % below its rows 4..9 — the average of two
  independently eroded detail fields at weights near 1/2 has less
  variance than either (1/sqrt(2) for uncorrelated fields), and part of
  the refined detail is shared (both runs start from the same plain
  surface and coarse discharge), so the loss is smaller than that bound;
* no crease: the step across the seam is slightly *smaller* than before
  relative to the first step inside (0.926 -> 0.912, 0.888 -> 0.867),
  because the two seam-row cells are both blends of the same two pieces
  and differ by less than two independent refinements would; the before
  worlds, whose seam rows were the plain upsample on both faces, sat at
  0.89-0.93 already, and p99 is unchanged;
* only the blended cells differ from the before world, and only in height,
  sediment and water surface.  The water surface change flips fine cells
  in the strip between "lake" (water > 5 cm above the surface) and dry: 8
  lake -> dry on seed 0 (of 19 strip lake cells), 47 lake -> dry and 3 dry
  -> lake on seed 1 (of 81), in shallow margins (e.g. seed 1 basin 65 on
  face 4, 0.05-1.35 m deep before).  Before, those seam rows were pinned
  to the plain upsample, whose lake depth is the bilinearly smeared coarse
  one the refine job already refuses to use as a level; the owning
  piece's refined flood has depth 0 there (checked on every one of the 47
  cells), as it does a few cells in;
* refine time is within run-to-run noise at this scale: the after runs
  were faster than the before runs, which the blend (it only adds work)
  cannot cause.  The pass itself is 0.01 s; the record extraction is a
  distance transform and a bilinear strip per multi-face piece, inside
  the job times.  At Earth the pass reads nothing back from disk and
  writes at most ~393k cells.

### 5.4 Review

An adversarial review verified the load-bearing claims and found no
blocker. It resampled the plain window surface of every multi-face piece
onto the neighbouring face with the blend's own geometry and compared it
with that face's own plain upsample: 4,943 cells, mean residual 0.018 m
against 0.94 m for a half-cell shift and 2.10 m for the step between
adjacent cells. It also found that no test guarded that geometry (a
two-cell shift passed all 20 refine tests), so the coordinates now live in
`rasterize.strip_window_coords` and
`test_seam_strip_is_sampled_where_the_neighbour_face_is` checks them the
same way (a two-cell shift fails it). The 8 / 47 lake flips come from the
pass-1 feather, not the blend: those cells had 0.05-0.5 m of smeared
coarse water depth under `lake_min_depth`, and the refined flood is dry
there before and after. Two things stay open and want an Earth hillshade
at a seam: the averaged detail is 7-20 % lower on rows 0..3 than a few
cells in, which may show as a faint band, and the discharge keeps the
face-restricted feather, so within `feather_cells` of a crossed edge a
river drawn from discharge can sit slightly off the refined valley.

### 5.5 Measured at Earth scale

`scripts/face_seams.py` on `worlds/earth-v7` (before the blend) and
`worlds/earth-v8` (with it; replay erosion as well, same tectonics):

| | earth-v7 | earth-v8 |
|---|---|---|
| detail at 0, 1, 2 ... 9 fine cells from the seam, same basin (m) | 0, .023, .084, .171, .292, .431, .613, .796, .920, .998 | **.429**, .443, .435, .480, .568, .573, .541, .554, .648, .661 |
| seam / first-step-inside ratio, same basin | 0.974 | 0.983 |
| same-basin pairs across seams | 18,836 | 18,692 |
| seam cells blended / seam pass | -- | 146,232 / 0.09 s |
| refine basin jobs wall | 287 s | 303 s |

The zero-detail seam row is gone: 0.43 m on the row that touches the edge,
flat across the first ten rows, and the step across the seam is no larger
than the step just inside (0.983). Both worlds' near-edge rows sit well
below their interior averages (earth-v7 1.0 m at row 9 against 3.3 m;
earth-v8 0.66 against 2.5), so the interior average is not the reference
for the edge rows at this scale: the cube edges run mostly through
lowland and ocean margin while the interior average carries the belts.
The faint band the review worried about wants a hillshade look at a seam
in the viewer; the numbers do not show one.

## 6. Notes

* `scripts/viewer_shot.py` takes `--layer basin` (the viewer's key; there
  is no `basins` layer, and an unknown key silently shows the terrain).
  The Playwright browsers on this machine are in `~/.cache/ms-playwright`;
  `PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers` no longer exists.
* The reader's Python prototype of the partition predicted 151 basins / 25
  multi-face on small; the shipped kernel gives 149 / 24 because it counts
  each cross-face contact once (the prototype counted it from both sides,
  doubling the cross-face boundary weights).

## Measured at Earth scale

`worlds/earth-v7` (the first full bake with this partition): **422
basins, 48 of them spanning two or more cube faces**, against 477 and
none on `earth-v5`; the viewer's basin layer no longer has a straight
meridian through every continent. Refine ran its per-piece jobs in 295 s
against 286 s on `earth-v6` (same tectonics and erosion kernel): the
redundant off-face erosion is a few percent, as the small-preset estimate
said. Lakes (1536), rivers (4614) and the hypsometry are those of
`earth-v6` to within noise, which is the expected null for a change to
the partition alone.

