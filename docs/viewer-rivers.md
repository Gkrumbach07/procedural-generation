# Rivers, lakes and coasts in the viewer

The user's review of earth-v9 (2026-09-14): rivers drawn as staircases and
dotted lines, blocky lakes, sawtooth coasts. At a 9.8 km cell no river is a
cell wide, so the grid can't show rivers as terrain. The drawing made that
worse in three ways, all fixed here:

| | was | now |
|---|---|---|
| rivers | hydro's D8 flow accumulation, thresholded: one cell wide by construction, 45-degree steps | erosion's particle `discharge` (McDonald's stream map), interpolated and eased in from the 85th to the 99.5th percentile of land discharge |
| lakes | the per-cell water code, nearest sampled: whole cells | a signed lake depth (lake level above the ground, reaching into the dry neighbours), interpolated; the shore is its 0 crossing |
| coasts | the per-cell water code | the 0.5 contour of the ocean mask after two 3x3 binomial passes, each cell centre held on its own side |
| height | 16-bit codes that could round a coastal plain 0.05 m above the water below it | sea level on a code, every cell's sign kept |
| zoom | rivers equally strong at every scale | the fade-in moves up 10 bytes per doubling of cells per screen pixel, so zoomed out only the large rivers show |

Before / after at three zooms of earth-v9 (`scratch/shots/cmp_z3.png`,
`cmp_z10.png`, `cmp_z30.png`, `cmp_coast.png`).

## Two things that were not the drawing

**The coast staircase was in the terrain.** Rounding was the first suspect
and fixing it changed nothing. The cells showed why: erosion leaves the
coastal plain at +0.1 to +2 m next to a shelf filled to 3-5 m below the
water, so the interpolated 0 m line passes almost through the land cells'
centres and traces the cell pattern. The coast is therefore drawn from the
smoothed ocean mask, not from the height.

**The refined grid is the final frame now** (`render.viewer_refined`, on
by default since the fixes below). Drawing it from `fine/` first showed two
refine defects the coarse frame hid:

* land specks offshore: the bicubic upsample of a shallow shelf (erosion
  fills it to a few metres below the water) overshoots above sea level next
  to high ground -- 45k fine sea cells up to 245 m on earth-v9, 4,680
  specks of at most four cells in the coast zone;
* discharge that doesn't line up: every basin window feathered its
  discharge back into the upsampled coarse discharge along *all* its edges,
  the coastline included, so the last cells of a river became a blurred
  coarse channel beside the sharp one, and rivers ended short of the sea.

Fixed in the refine stage (`scratch/shots/cmp_mouths*.png`, before / after):

* a coast pass after every other writer (`rasterize.write_coast`): in the
  coarse cells that are ocean or touch it, a fine cell inside the 0.5
  contour of the smoothed ocean mask is sea at least 1 m deep and any other
  is land at least 1 m high -- specks 4,680 -> 507 (the rest are the coarse
  grid's own one-cell islands), and a coastline that rounds the coarse
  cells; a former sea cell it gives to land takes the refined land's
  discharge beside it;
* discharge is feathered only towards cube edges (the seam blend's
  business), and the frozen ring takes its active neighbours' discharge, so
  a channel runs into the sea;
* lakes pass their water on (`basin_job.lake_outflow`): a particle that
  reaches a lake dies in it, which left the river below every lake dry once
  the coarse blur was gone; an overflowing lake now spawns the flow through
  it at its spill cell.

Left: a lake right on the coast spills along the shore, which draws a short
river strip beside the sea.

The refined frame costs a 40 MB final frame against 14 MB and 58 s
against 33 s to export.
