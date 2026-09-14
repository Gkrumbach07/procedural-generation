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
* the fine raster at the seams did not move, to the digit.  Both seam rows
  were and are the plain upsample (before: the frozen ring at the face
  edge; after: the feather weight restricted to the face), so the step
  across the seam is the plain upsample's own step in both worlds and the
  detail strip along the edges stays.  This is the least-change design
  and it is a null result for the fine seam;
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
0.81 (7.8 % below 0.3).  After this change those numbers stay where they
are, for the reason of section 3.

The stages have no code-version stamp and `stage_done` compares only the
parameter hash, so existing worlds keep the old partition until `bake
--world <w> --force --from watersheds` (watersheds 1 s + refine ~330 s +
derive + tiles + viewer, about 8 minutes at Earth; not the erosion).
Basin ids renumber (the face-cut children vanish) and every fine raster
changes (the refine stream is keyed by (basin, face) now, and edge windows
shift into the face), so comparisons to earlier worlds go through the
`basin_id` rasters, never through ids.

## 5. What is left: the strip

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

