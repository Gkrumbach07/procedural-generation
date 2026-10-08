# Rivers, lakes and coasts in the viewer

The user's review of earth-v9 (2026-09-14): rivers drawn as staircases and
dotted lines, blocky lakes, sawtooth coasts. At a 9.8 km cell no river is a
cell wide, so the grid can't show rivers as terrain. The drawing made that
worse in three ways, all fixed here:

| | was | now |
|---|---|---|
| rivers | hydro's D8 flow accumulation, thresholded: one cell wide by construction, 45-degree steps | erosion's particle `discharge` (McDonald's stream map), interpolated and eased in from the 85th to the 99.5th percentile of land discharge |
| | | (2026-10-07: zoomed in, the line on that map's ridge; and on the final frame the routed flow again, smoothed and drawn as lines -- the two sections before *Detail tiles*) |
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

## Lines on the discharge's ridge

Zoomed in until a texel is several pixels, the stream map drawn as the band
it fills is the user's 2026-10-07 screenshots of earth-v32: every river one
4.9 km cell wide with soft edges, fainter strands beside it, pale fans where
the flow spreads. The discharge texture was right as data and wrong as a
drawing: a channel is a *ridge* in it, and the river is the ridge's line.

The shader finds that line from the read it already makes. The cubic
B-spline is smooth in its second derivative, so the sixteen texels that give
the discharge at a pixel give its slope and curvature too (`detailJet`,
`zJet`, `byteJet`: one for each place the stream map comes from -- a detail
tile, a zoom level, the atlas). The ridge runs where the slope across the
direction of sharpest downward curvature is zero; one Newton step along that
direction is the distance to it (`riverRidge`; within 0.05 texel up to 0.2
texel from the line). Then, once a texel is more than a pixel:

* a channel one or two texels wide -- its curvature is twice, or once, its
  height over its banks -- is a line 1.4 to 3.4 pixels wide by what it
  carries, the river-lines' own map rule;
* water wider than that has no ridge (a flat top; its ripples are a few per
  cent of its height and are not channels) and is still the band, drawn to a
  crisp edge;
* between a texel a pixel and two, the old band fades out as the line fades
  in; zoomed out further nothing has changed.

It is a change to the page alone: any world shows it on its next export, on
the final frame, the timeline's frames, the 1.2 km tiles and the zoom
windows, in the flat, globe and 3-D views (`scratch/` shots of earth-v32,
earth-v26 and earth-v18's window, 2026-10-07).

What it does not do, and what was tried for it:

* **Braids stay.** The stream map is many particle paths averaged, and on
  low ground they run side by side; they are now thin strands instead of a
  smear, but they are strands.
* **A big river is still a band.** On the 1.2 km tiles earth-v26's trunk
  rivers are two to four texels across. Looking for their ridge on every
  second texel finds a centre line, but the texels themselves then show the
  channel's two shoulders as lines of their own -- a texel in from each bank
  the surface curves down as a ridge does -- and a river drawn as three
  parallel lines is worse than the band. Reading the map again either side
  of a candidate line tells a shoulder from a channel, and with it the
  software renderer the checks here run on drops its context, so that
  version could not be looked at and is not in.
* **Hydro's flow accumulation instead of the discharge** is one thread and
  runs through the lakes, but a D8 path is a staircase and its ridge comes
  out as dashes. One thread needs the river lines above (`--river-source`),
  which draw over the band rather than instead of it.

## One water: the final frame's rivers are the lakes' own

Lines made the rivers thin; they did not make them the lakes' rivers. The
erosion's discharge is where its particles ran while the ground was still
moving, and the lakes are hydro's -- the finished surface filled to its
spills. Drawn from the first, a river passes beside a lake or stops short of
it (the user, 2026-10-07: "can we make the rivers and lakes look more
cohesive").

The final frame now draws the routing that fills the lakes:

* **which water**: a planet level's own accumulated flow where the frame is
  one (it always was, widened into bands), else hydro's `flow_acc`, from an
  eighth of hydro's river threshold up (`viewer.hydro_rivers`,
  `HYDRO_RIVER_SHARE`: the drainage graph's reaches are the rivers a basin is
  named for, and a map draws their tributaries too). The routing runs across
  a lake at its level and out at its spill, and the rivers are drawn on land
  only, so each one meets the shore where the water does. One thread: no
  braids. The timeline's frames are before hydro and keep the erosion's
  discharge, on a scale read off the last erosion frame.
* **as what**: a routed river is a path one cell wide, so it is handed to the
  page as a channel to find the line of (`detail.river_strength`): each
  channel cell worth 0.4 at the threshold to 1 at full strength, and its path
  laid down as a ridge that high (`detail.path_ridges`). The frame says so
  (`river_lines`), and the page then draws every channel as a line, 1.2 to
  5.2 pixels by its strength, and none as a band.
* **rounded along the path, not across it**: a D8 path is a staircase, and
  the line on a staircase's ridge is a row of hooks. The first version
  rounded it with a blur -- a Gaussian of 1.3 cells; 0.8 leaves the hooks --
  and a blur that wide cannot tell two channels two or three cells apart:
  they are one ridge, somewhere between them. So a creek running beside its
  river, or coming in to it at a shallow angle, lost its line cells short of
  the junction, and the page showed faint creeks that joined nothing (the
  user's screenshots, 2026-10-08; replayed on earth-v32's 1.2 km flow, a
  third of one view's channel cells had no line on them and a line ran
  between two parallel creeks where neither was). Now each channel cell is a
  point of its path -- joined to the neighbour carrying the least more water
  and the one carrying the most less -- the points are moved a quarter of the
  way to each neighbour, twice, and each stretch between them is laid down
  as a Gaussian ridge 0.8 cells wide. The steps are gone along the path and
  the ridge is as narrow as the cells allow: the same view, 66 -> 84 % of
  channel cells with a line and both creeks drawn down to the river. A routed
  creek is also drawn whole now where the erosion's faint paths were eased
  in: it is a river or it is not.
* **in what colour**: the satellite's river teal was already its lakes' at a
  few metres; on the other layers a river is now the lakes' own blue.

* **and not its kinks**: a routed path is not straight at the scale of its
  cells -- it steps aside for one and back -- and each kink, smoothed, is a
  short spur on the channel's side that the ridge rule drew as a blob or a
  comma beside the river (the user's screenshot at -16.2, 11.0 on earth-v32's
  1.2 km tiles; replayed in numpy on the level's flow, 10 % of the lit pixels
  there and 17 % at 55, 128 were more than 0.9 cell from any channel cell).
  Pruning channel heads does nothing for it: the cells are on the path, not
  stubs of it. What tells a spur from a channel is the slope *along* its own
  line: a spur climbs the side it sits on (0.72 of its height a texel, the
  median), a channel carries the same water from one texel to the next (0.05;
  under 0.15 on nine pixels in ten). A line that climbs more than 0.3 to 0.5
  of its height a texel is not drawn (`riverRidge`'s `climb`): 1.4 and 0.9 %
  stray pixels left at the two places, and the tributaries there still reach
  their rivers.

Earth-v32, refined frame (`scratch` shots, 2026-10-07): a river into the
head of each valley lake and out of its foot, lake chains strung on one
line, trunks visibly heavier than their tributaries.

What it costs: on a frame finer than the routing (the refined grid, where
hydro routes the coarse one) a path is the coarse cell's, up to a cell from
the fine valley floor the erosion cut; a planet level routes its own grid
and has no such offset. And a zoom window's rivers are untouched: down to
5 m a cell a river is many cells wide and is still widened water
(`detail.widen_rivers`).

## Detail tiles

A browser texture holds one atlas of 2048^2 a face at most -- the refined
grid, or the 1.2 km planet level reduced by four. `export_viewer.py
--detail` also writes the final frame's terrain beyond its atlas as tiles
(`globe/viz/detail.py`, `viewer/tiles/L{L}/`), levels of 2x, 4x ... the
atlas up to the source's own resolution, 256 cells a tile, one lossless
RGBA WebP each (height 16-bit, signed lake depth, 255 - ocean mask; a cell
of pad so a tile samples bilinearly by itself; all-sea tiles not written).

The viewer picks the level whose cells are about a pixel, loads the tiles
the view covers into a 225-slot cache texture (least recently seen out),
and the shader looks each pixel up in the level's page table; a tile not
loaded yet draws from the atlas. Heights, shading, coasts and lake shores
come from the tiles, rivers are the lines. Earth-v9 exported at a 1024^2
atlas with tiles from its 2048^2 refined grid (230 tiles, 17 MB, 8 s)
matches the 2048^2 atlas at zoom 12 to 120
(`scratch/shots/e1/keep_detail_cmp.png`: atlas 1024^2, with tiles, atlas
2048^2).
