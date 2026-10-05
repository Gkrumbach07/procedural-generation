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

## Using it

* A bake writes the map in its derive stage and the viewer carries a
  **Geology** layer, a **Basement rock** layer and **Crust thickness**.
* A world baked before this:
  `python bake/scripts/tect_diagnostics.py --world worlds/X` (reruns
  tectonics, four minutes on the earth preset, checks the bedrock comes out
  bit for bit and saves the two new diagnostics), then
  `python bake/scripts/geology.py --world worlds/X` (the map and the viewer).
* `geology.HARDNESS` gives each rock a resistance to erosion. Nothing reads
  it yet.

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
* At the coarse grid: 9.8 km cells on the earth preset, so contacts are
  blocky when zoomed in.
* Volcanic rock is 0.01 % of the land: the edifices are a few cells each.
