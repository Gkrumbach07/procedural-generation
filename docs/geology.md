# Rock types and cross-sections

A geologic map of a baked world, and sections through its crust. Nothing new
is simulated: the map is read off what the stages already know
(`bake/globe/derive/geology.py`), the way a geologist reads a landscape.

## What each stage knows

| stage | fact | field |
|---|---|---|
| tectonics | continental or oceanic crust | `diagnostics/crust_kind` |
| | age: the sea floor's since its ridge, a continent's since it was last assembled | `diagnostics/crust_age` |
| | caught in a collision | `diagnostics/collision_zone` |
| | craton, docked island arc, island arc at sea | `diagnostics/crust_province` (new) |
| | thickness of the crust, km | `diagnostics/crust_thickness` (new) |
| | volcanic edifices | `diagnostics/volcano_cone` |
| erosion | how deep it cut into the rock it was handed | `bedrock - height` |
| | sediment it laid | `sediment` |
| climate, hydro | temperature, rain, ocean, lakes | |

## The rules

**Basement** (`coarse/basement`), the crystalline rock under any cover:

* oceanic crust is **basalt**; thickened into an island arc at a trench, **andesite**;
* continental crust is **granite**; a craton is **gneiss**; a collision belt
  **schist**; a docked island arc **greenstone**; a volcano **andesite**.

**At the surface** (`coarse/rock`), the unit a map colours:

* 50 m of sediment or more is a basin: **sandstone**, or **shale** below 200 m
  and on lake floors, **evaporite** in a desert where the fill is over 500 m
  or under a desert lake, **glacial till** where the ground is frozen;
* 5-50 m of sediment is **alluvium** (till in the cold);
* bare rock that erosion cut 300 m or more into shows its basement, and so
  does a volcano;
* bare rock it barely touched keeps the platform cover: **limestone** where
  it is warm and below 500 m, **shale** on other lowland, **sandstone** above;
* a continental shelf is limestone (warm) or sandstone above -200 m, shale
  down the slope; the deep sea floor is basalt while young, **pelagic
  sediment** after 20 My, shale under a continent's fans.

## earth-v19

| land | share | | sea floor | share |
|---|---|---|---|---|
| sandstone | 26.9 % | | pelagic sediment | 62.4 % |
| shale | 15.9 % | | shale | 15.1 % |
| gneiss | 13.2 % | | basalt | 13.7 % |
| granite | 10.7 % | | limestone | 4.5 % |
| alluvium | 10.3 % | | sandstone | 3.5 % |
| limestone | 8.3 % | | andesite | 0.8 % |
| schist | 6.7 % | | | |
| glacial till | 6.3 % | | | |
| evaporite | 1.7 % | | | |

69 % of the land is sedimentary and 31 % crystalline; Earth's exposed rock is
about two thirds sedimentary. Crust thickness: continents 27.9 / 44.5 / 60.9 km
at the 5th percentile / median / 95th, 76 km at the thickest; sea floor 7 km,
arcs to 31 km. The median continent is about 8 km thicker than Earth's.

## Below the coarse grid

A contact between two kinds of crust is known to one coarse cell, 9.8 km on
the earth preset. Below that the map is drawn again on the finer grid's own
ground, sediment, water and climate, with the tectonic classes (kind, belt,
province) read from the coarse cell at a *warped* place: the point's own
position moved by a smooth vector field, 0.3 coarse cells at the median and
1.2 at the most, with detail down to two cells of whatever grid is asking
(`geology.warped_cells`, `classify_at`). The field is noise on the sphere
itself, so a contact crosses a cube edge as it crosses anywhere else, and it
is the same line at every refinement: the 1.2 km and the 305 m readings of it
differ by 0.07 of a 1.2 km cell at the median.

* The derive stage writes `fine/rock` and `fine/basement` on the refined grid
  (4.9 km), and the viewer's Geology layer draws those.
* A zoom level saves its own `rock` and `basement` in its `.npz`, mapped
  after it eroded, so its valley fills and lake floors are on it. The viewer
  does not draw a level's own map yet: inside a zoom window the Geology layer
  is still the planet's.

## Hardness for the zoom levels

A zoom level's bedrock is as hard as the rock it is
(`refine.zoom.ROCK_HARDNESS`, `geology.bed_hardness_at`): the basement's
hardness where erosion has cut through the cover (300 m, fading in over
200 m), where a volcano stands and on oceanic crust; 0.5 where the cover is
still there; capped at 0.85 as the field it replaces was. The active cones
stay fresh lava on top of that.

Only tectonics' classes and the erosion's own depth go into it. The map's
cover classes turn on the ground's height and the climate -- limestone below
500 m where it is warm, shale below 200 m -- which is how a map is drawn and
not where beds lie: a hardness read off them would step along those contours
and cut a terrace at 200 m and at 500 m around the whole planet.

| land of earth-v19 | tectonics' field (low-passed) | from the rock |
|---|---|---|
| mean | 0.74 | 0.67 |
| 10th / 50th / 90th percentile | 0.58 / 0.77 / 0.85 | 0.50 / 0.62 / 0.85 |

Two windows of earth-v19 baked both ways (a schist belt, face 2 cell 255 225;
gneiss against cover across a 3.9 km range, face 1 cell 488 165): no step,
terrace or cell edge along a contact at 1.2 km, 305 m or 76 m, and the relief
is of the same kind. The lake share moved both ways with it (1.2 km: 0.85 ->
0.33 % and 0.16 -> 0.14 %; 305 m: 3.1 -> 4.0 % and 1.5 -> 2.2 %; 76 m: 1.6 ->
2.6 %), which two windows cannot tell from noise.

## Using it

* A bake writes the map in its derive stage and the viewer carries a
  **Geology** layer, a **Basement rock** layer and **Crust thickness**.
* A world baked before this:
  `python bake/scripts/tect_diagnostics.py --world worlds/X` (reruns
  tectonics, four minutes on the earth preset, checks the bedrock comes out
  bit for bit and saves the two new diagnostics), then
  `python bake/scripts/geology.py --world worlds/X` (the map and the viewer).
* `geology.HARDNESS` gives each rock a resistance to erosion; the zoom
  levels read it (above). The coarse erosion and the refine stage still use
  tectonics' field: the map is drawn after them.

## In the viewer

* **Geology** and **Basement rock** are styles in the Layers panel, with a
  key grouped by family in place of the colour bar; the sea floor's rock is
  drawn darker. Clicking a place lists its rock, basement, sediment and crust
  thickness.
* **Section** (beside the style) takes the next two taps on the map as the
  ends A and B and draws what the great circle between them cuts through, in
  a panel along the bottom: on top the ground, the sea and lakes, the
  sediment and the basement, with the map unit as a strip above; below it
  the whole crust down to the Moho, on the mantle. Hovering reads out the
  place under the cursor and marks it on the map. The section is in the
  link (`sec=lat1,lon1,lat2,lon2`), so `scripts/viewer_shot.py --hash`
  draws it too. It is read from the final frame, 501 samples.
* The Moho steps where the line crosses from one kind of crust to the other:
  each kind keeps its own thickness in `crust_thickness`, so a continent's
  does not bleed into the sea floor's, and a rifted margin's taper is no
  wider than the continental cells that carry it.

## What it is not

* Not stratigraphy. The sediment thickness and the crust thickness a section
  draws are the model's; the rock under the sediment is the mapped basement
  class. The platform cover (limestone, shale, sandstone on bare rock) has no
  thickness in the model at all.
* Not history. Limestone is where it is warm and low now, not where a
  shallow sea once stood.
* The contacts are warped, not mapped: a line bent by noise, with no fold,
  fault or dip behind its shape.
* Volcanic rock is 0.01 % of the land: the edifices are a few cells each.
