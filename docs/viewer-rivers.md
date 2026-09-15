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

A lake right on the coast used to spill along the shore, drawing a short
river strip beside the sea: its spill was searched among land cells only,
so the outflow started at the lowest land cell of its shore and ran along
the coast. A lake whose lowest rim cell is the sea (or a window exit) now
sends nothing -- its water is already out (`lake_outflow(..., drain=)`;
earth-v9's refined grid predates the fix).

The refined frame costs a 40 MB final frame against 14 MB and 58 s
against 33 s to export.

## River lines

The discharge texture is the particles' stream map interpolated from its
texels, so a river is as wide as the channel of cells it filled: fine at a
cell per pixel, but closer than that every river is a 10-20 km blue band
with a fainter ghost beside it (earth-v9 at zoom 40 and 120,
`scratch/shots/a6/keep/lines_cmp_*.png`: the texture left, graph lines
middle, traced lines right). The final frame now draws its rivers as lines
(`viewer.river_lines`, `data/rivers.js`), at a width that follows their
discharge:

* **where they come from** (`export_viewer.py --river-source`, default
  `auto`): derive's `graph/rivers.json` -- the drainage graph's reaches,
  traced through the refined channel and Catmull-Rom smoothed, each with a
  width `30 m · (Q / Q_threshold)^0.5` (Leopold & Maddock) -- or, on a
  planet level's frame, which derive never traced, lines **traced** from the
  frame itself (`globe/viz/river_lines.py`): each face's surface
  priority-flooded towards its water and its edge, every cell draining to
  the cell that flooded it, carrying the largest particle discharge upstream
  of it; a river where that is over the 97th percentile of land discharge,
  spurs of under 8 cells dropped, reaches smoothed and simplified. Tracing
  on the flood tree is what keeps them joined -- following the discharge
  ridge to the neighbour with more left 25 k of 58 k lines ending in the
  middle of nowhere on earth-v9, where the time-averaged discharge dips for
  a cell;
* **drawing**: every segment is a quad around its projected ends shaded as
  a capsule, drawn with MAX blending into a canvas-sized texture that the
  main shader mixes in on land only, so joins do not darken and no line
  paints over a lake or the sea. A river is its own width, but never under
  1.3 pixels to 3.2 by discharge -- a map's rule; the Amazon-sized trunk of
  earth-v9 is 720 m, under a pixel until well past zoom 100 -- and zooming
  out hides the smaller ones as the texture did (the cut-off rises 10
  discharge bytes per doubling of cells per pixel).

Graph rivers are the cleanest and agree with hydro's lakes and basins, but
there are fewer of them: 3,766 reaches against the texture's dense
network. Traced lines at the texture's own 85th-percentile threshold turn a
plain's sheet flow into parallel hatching (188 k lines); at the 97th they
follow the texture's channels with some hatching left on flats (22 k
lines, 62 k vertices, 0.9 MB). The discharge texture still draws the
timeline's frames.
