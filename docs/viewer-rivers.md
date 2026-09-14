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

**The refined grid is not ready to be the final frame.** Drawing the final
frame from `fine/` (4.9 km, `render.viewer_refined` / `export_viewer.py
--refined`) gives visibly crisper channels, but it also shows two refine
defects the coarse frame hides:

* land specks offshore, where the refined surface stands above 0 inside a
  sea cell;
* discharge that doesn't line up across basin windows: a river can stop
  short of the coast while a separate channel reaches it, and a sharp
  channel can run beside a blurred copy of itself where a window's feather
  blends in the upsampled coarse discharge.

So the refined frame is opt-in until refine's discharge and coast are
fixed. It costs a 40 MB final frame against 14 MB and 58 s against 33 s to
export.
