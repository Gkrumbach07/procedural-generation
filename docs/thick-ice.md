# Thick ice: a sheet in balance with its climate

`erosion.ice_sheet` (off until a world baked with it has been reviewed),
`bake/globe/erosion/icesheet.py`, and the climate's `temp_range`.

## What was there, and what it could not be

The ice was a line: ground whose year averages under freezing
(`ErosionState.cold`), read `ice_age_c` colder for the last maximum, with a
snow test (`ice_aridity`) to keep it off cold deserts. The threshold of that
test is what set how much ice there was: 1.5 gave Earth's quarter of the
land before the storm rules and 35 % after them, 0.7 gives it back
(docs/lakes-in-erosion.md, "The ice follows a temperature").

A line cannot be an ice cap. It has no thickness, so the viewer whitens
bedrock; nothing of it flows; it melts by its yearly mean, where ice melts
by its summer; and its extent is a threshold's.

## Seasons first (2026-10-08, a prototype)

The climate is one yearly mean: latitude and height. A summer and a winter
were prototyped on earth-v32's bedrock (seasonal range by the sun and the
distance from the sea; the wind belts moved 7 degrees with the sun and a
flow into the heated continent; the rain swept on each wind):

* the range matches Earth (land median 2 C at 0-10 degrees, 22 at 40-50, 29
  at 50-60; 34-40 C at the 90th percentile of the higher bands);
* a rule of the summer's warmth against the year's snow (Ohmura's line)
  alone leaves ice on 4.9 % of the land today and 13.7 % at 8 C colder:
  right about the big continental interiors, which are bare on Earth too,
  and wrong about the total, because an ice sheet is sustained by its own
  height and by flow;
* seasonal rain appears (winter rain at 20-40 degrees, summer rain over 43 %
  of the land with the monsoon flow) but the land under a quarter of the
  mean rain falls from 13.2 to 8.0 %: the model's dry belt is 6 degrees
  wide, and a 7 degree shift leaves nowhere dry all year. Not built.

So the seasonal *temperature* is in (`climate.temperature.seasonal_range`,
the `temp_range` field: every world has it), and the ice that needs it is
the sheet below. Seasonal rain waits for a subsidence belt as wide as
Earth's.

## The sheet

* **Balance** (`icesheet.balance`): metres of water a year at a surface.
  Snow is the precipitation of the part of the year under `ice_snow_c`
  (1 C); melt is `ice_ddf_mm` (4 mm) for every degree-day above freezing; the
  year is a sine about its mean, `temp_range` wide. The land's mean rain is
  `climate.land_rain_mm` (750). At 4 mm the balance is nothing where
  Ohmura (1992) found glaciers' own to be.
* **Shape** (`icesheet.geometry`): plastic ice, `sqrt(2 h0 d)` over the
  ground (smoothed over 50 km) a distance `d` inside the margin, with `h0`
  = `ice_yield_m` (8 m: 72 kPa). A mountain higher than that stands through.
* **Extent** (`icesheet.settle`): the balance is summed down the ice's own
  surface, each cell to all its lower neighbours by slope, never below
  nothing. Bare ground it reaches becomes ice, a ring a round; ice it no
  longer reaches melts (after two rounds: the last cell would flicker).
  Summed down single lines of steepest descent instead, an outlet's whole
  flux ran a thousand kilometres down a river valley one cell wide.

Grown from nothing it is an advance. Started from a bigger sheet it is a
thaw, and what is left is more than would have grown.

**On earth-v33's last ground, before it was in the bake** (constants as
above, none fitted):

| | the model | Earth |
|---|---|---|
| ice today (thawed back from the last sheet), % of land | 10.0 | 10 |
| ...its mean thickness | 1,982 m | about 2,000 m |
| ...the sea level in it | 65 m | 66 m |
| ice today if grown from nothing | 9.2 %, 59 m | |
| at 8 C colder, % of land | 25.3 | about 25 |
| ...the sea lower than today by | 111 m | 120-130 m |
| at 10 C colder | 33.7 %, 182 m lower | |
| thickest ice | 4.7-4.9 km | 4.8 km |

Today's ice by latitude: all the land past 80 degrees, 84 % of 70-80, a
quarter of 60-70 (Earth: about 15 %), none below 50.

## In the bake

* `maps.step` brings the sheet up to the iteration's climate
  (`glacial.ice_cooling`) every `glacial_every` iterations, `ice_rounds` (15)
  rounds at most: it follows the cooling a little behind, as ice does.
  `ErosionState.cold` is then the sheet, so the glacial pass carves under it
  and a lake takes no fill under it; `ice_evap` and `ice_aridity` are unused.
* The frames carry the thickness (`ice_h`) and the viewer draws the ice's
  surface: domes, not white ground.
* The stage ends on the last maximum, settled. `erosion.run.thaw` then
  thaws it to today's in `render.thaw_frames` steps and writes them as frames
  of their own (stage `thaw`; the viewer takes these and not its rule), with
  the fields `ice_max` and `ice_now` (thickness, m).
* The planet level's ice (`zoom/ice.py`) scours the ground `ice_max` stood
  on. `derive` gives the ice class to `ice_now` and takes it from ground that
  is only cold (`biomes.with_ice`).

## Not yet

* the sea does not fall with the ice (the volume is known: logged and in the
  stage's info);
* the crust does not sag under the ice or rebound after it;
* the carve is still the glacial pass's own rate, not the ice's flux;
* the final frame's ground is the ground: today's cap is its biome's white,
  without its dome, and has no ice layer in the elevation view;
* sea ice, ice shelves.
