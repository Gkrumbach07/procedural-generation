"""Rock types: the geologic map of a baked world, read off what the bake
already knows.

Nothing here is simulated.  Tectonics knows what kind of crust a cell stands
on, how old it is, whether it was caught in a collision, whether it is a
craton, a docked island arc or an arc still at sea, and where the volcanoes
are; erosion knows how deep it cut into that rock and how much sediment it
laid; climate knows how warm and wet the place is.  A geologic map is those
facts read the way a field geologist reads them:

``basement`` -- the crystalline rock under any cover:

* oceanic crust is **basalt**; where a trench thickened it into an island
  arc, **andesite**;
* continental crust is **granite** (the granodiorite of the upper crust),
  except a craton, **gneiss** (the shield); a collision belt, **schist**
  (its metamorphic core); a docked island arc, **greenstone**; and under a
  volcano, **andesite**.

``rock`` -- what is at the surface, the unit a map colours:

* a sedimentary basin (``BASIN_M`` of sediment or more) is **sandstone**, or
  **shale** on wet lowland and lake floors, **evaporite** where it is arid
  and deep or under a desert lake, **glacial till** where it is frozen;
* thinner sediment is **alluvium** (till in the cold);
* bare rock that erosion cut ``EXHUMED_M`` or more into is the basement,
  and so is any volcano;
* bare rock it barely touched still carries the cover the continent
  collected while it sat low -- the platform: **limestone** where it is warm
  and low, **shale** on other lowland, **sandstone** above;
* a continental shelf is limestone where warm and shallow, sandstone where
  shallow, shale down the slope; the deep sea floor is basalt while it is
  young and **pelagic sediment** once it has had ``PELAGIC_MY`` to collect
  it, shale under a continent's fans.

The thicknesses a cross-section draws are the model's own: ``sediment``
(erosion's) and ``crust_thickness`` (tectonics').  The platform cover has no
thickness in the model, so a section shows it as the map unit along its top
and nothing more.
"""
from __future__ import annotations

import numpy as np

NAMES = ["none", "basalt", "andesite", "granite", "gneiss", "schist", "greenstone",
         "sandstone", "shale", "limestone", "alluvium", "evaporite", "glacial till", "pelagic sediment"]
NONE, BASALT, ANDESITE, GRANITE, GNEISS, SCHIST, GREENSTONE, SANDSTONE, SHALE, LIMESTONE, ALLUVIUM, EVAPORITE, TILL, PELAGIC = range(len(NAMES))
N_ROCKS = len(NAMES)
#: the family of each class, for the key: igneous (volcanic or plutonic), metamorphic, sedimentary
FAMILY = ["", "volcanic", "volcanic", "plutonic", "metamorphic", "metamorphic", "metamorphic",
          "sedimentary", "sedimentary", "sedimentary", "sedimentary", "sedimentary", "sedimentary", "sedimentary"]
#: map colours (sRGB), after the conventions of printed geologic maps: pinks and
#: reds for igneous rock, browns and greens for metamorphic, yellows, greys and
#: blue for sedimentary
PALETTE = [(40, 40, 40), (96, 78, 122), (190, 84, 70), (236, 140, 160), (176, 132, 88), (120, 160, 132), (62, 118, 86),
           (236, 206, 98), (146, 156, 120), (104, 164, 222), (246, 238, 184), (244, 196, 224), (198, 214, 228), (164, 138, 112)]
#: how hard each rock is to erode, 0..1 (the kernel's bedrock erodibility is 1 - hardness)
HARDNESS = [0.5, 0.75, 0.7, 0.9, 0.95, 0.8, 0.8, 0.55, 0.3, 0.6, 0.1, 0.2, 0.15, 0.2]

#: codes of tectonics' ``crust_province`` diagnostic (tectonics.run)
PROVINCE_CRATON, PROVINCE_TERRANE, PROVINCE_ARC = 1, 2, 3

#: sediment (m) that makes a sedimentary basin; under it and over ALLUVIUM_M it is a skin of alluvium
BASIN_M = 50.0
ALLUVIUM_M = 5.0
#: a basin this deep in a desert has an evaporite centre
EVAPORITE_M = 500.0
#: bare rock erosion cut this far into (m) shows its basement
EXHUMED_M = 300.0
#: an edifice this thick (m) is mapped as its volcano
CONE_M = 100.0
#: continental crust last assembled this long ago (My) is shield where tectonics gave no province
SHIELD_MY = 2000.0
#: the sea floor carries a pelagic blanket once it is this old (My): ~2-5 m of clay and ooze per My
PELAGIC_MY = 20.0
#: a fan of this much sediment (m) on the deep sea floor is mapped as its shale
FAN_M = 100.0
#: climate lines: frozen ground (mean annual C), carbonate seas and platforms (C), desert (cm of rain a year)
COLD_C = -3.0
WARM_C = 18.0
ARID_CM = 35.0
#: lowland (m) where a basin's fill is mud rather than sand, and a platform's cover limestone or shale
LOWLAND_M = 200.0
PLATFORM_LOW_M = 500.0
#: a shelf sea shallower than this (m) is sand or reef; deeper, the slope's mud
SHELF_M = 200.0


def classify(surface: np.ndarray, sediment: np.ndarray, exhumed: np.ndarray, continental: np.ndarray, age_my: np.ndarray, belt: np.ndarray,
             ocean: np.ndarray, lake: np.ndarray, temperature: np.ndarray, precip_cm: np.ndarray, cone: np.ndarray | None = None,
             province: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(rock, basement)``, uint8 codes into :data:`NAMES`, over arrays of
    one shape:

    ``surface`` (m, sea level 0) and ``sediment`` (m) as erosion left them;
    ``exhumed`` (m) how far erosion cut into the bedrock tectonics handed it;
    ``continental`` (bool), ``age_my`` (the crust's age: the sea floor's
    since its ridge, a continent's since it was last assembled), ``belt``
    (bool, a collision zone), ``cone`` (m of volcanic edifice) and
    ``province`` (tectonics' craton / terrane / arc codes) from tectonics;
    ``ocean`` and ``lake`` (bool) from hydro; ``temperature`` (C at the
    surface) and ``precip_cm`` (cm a year) from climate."""
    cont = np.asarray(continental, bool)
    ocean = np.asarray(ocean, bool)
    lake = np.asarray(lake, bool) & ~ocean
    belt = np.asarray(belt, bool)
    cone = np.zeros(cont.shape) if cone is None else np.asarray(cone)
    volcano = cone >= CONE_M
    if province is None:
        craton = cont & (age_my >= SHIELD_MY)
        terrane = arc = np.zeros(cont.shape, bool)
    else:
        craton, terrane, arc = (cont & (province == PROVINCE_CRATON), cont & (province == PROVINCE_TERRANE), ~cont & (province == PROVINCE_ARC))

    # ---- the basement ---------------------------------------------------
    base = np.where(cont, GRANITE, BASALT).astype(np.uint8)
    base[craton] = GNEISS
    base[cont & belt] = SCHIST
    base[terrane] = GREENSTONE
    base[arc] = ANDESITE
    base[volcano & (cont | arc)] = ANDESITE          # a volcano on a continent or an arc; a hotspot's on the sea floor stays basalt

    # ---- what covers it -------------------------------------------------
    cold, warm, arid = temperature < COLD_C, temperature >= WARM_C, precip_cm < ARID_CM
    land = ~ocean
    rock = base.copy()
    # bare ground erosion barely cut still carries its platform cover (a belt's is folded into it)
    cover = land & cont & ~volcano & (exhumed < EXHUMED_M)
    rock[cover] = np.where(warm & (surface < PLATFORM_LOW_M), LIMESTONE, np.where(surface < LOWLAND_M, SHALE, SANDSTONE))[cover]
    # sediment the erosion laid
    skin = land & ~volcano & (sediment >= ALLUVIUM_M)
    rock[skin] = np.where(cold, TILL, ALLUVIUM)[skin]
    basin = land & ~volcano & (sediment >= BASIN_M)
    rock[basin] = np.where(cold, TILL, np.where(arid & (sediment >= EVAPORITE_M), EVAPORITE, np.where(surface < LOWLAND_M, SHALE, SANDSTONE)))[basin]
    rock[lake & ~volcano] = np.where(arid & ~cold, EVAPORITE, SHALE)[lake & ~volcano]
    # the sea floor
    shelf = ocean & cont & ~volcano
    rock[shelf] = np.where(surface > -SHELF_M, np.where(warm, LIMESTONE, SANDSTONE), SHALE)[shelf]
    deep = ocean & ~cont & ~arc & ~volcano
    rock[deep] = np.where(sediment >= FAN_M, SHALE, np.where(age_my >= PELAGIC_MY, PELAGIC, BASALT))[deep]
    return rock, base


def shares(rock: np.ndarray, area: np.ndarray, where: np.ndarray) -> dict:
    """Share of ``where``'s area under each rock (names of :data:`NAMES`, those present)."""
    w = np.where(where, area, 0.0).astype(np.float64)
    tot = np.bincount(np.asarray(rock).ravel(), weights=w.ravel(), minlength=N_ROCKS)
    return {NAMES[k]: round(float(tot[k] / max(tot.sum(), 1e-30)), 4) for k in range(N_ROCKS) if tot[k] > 0}


def run(store, params, grid, surface: np.ndarray, sediment: np.ndarray, ocean: np.ndarray, lake: np.ndarray, temperature: np.ndarray,
        precip_cm: np.ndarray, log=print) -> dict | None:
    """Classify the world at ``store`` and save the coarse ``rock`` and
    ``basement`` fields; the map's shares, or None where the world has no
    tectonic diagnostics to read (a stub tectonics).  The other arguments are
    the derive stage's own coarse interiors."""
    from ..field import FaceField

    diag = store.root / "diagnostics"

    def diagnostic(name):
        return FaceField.load(diag, name, grid).interior if FaceField.exists(diag, name) else None

    kind, age = diagnostic("crust_kind"), diagnostic("crust_age")
    if kind is None or age is None:
        log("[derive] geology: no crust diagnostics in this world, no rock map")
        return None
    belt, cone, province = diagnostic("collision_zone"), diagnostic("volcano_cone"), diagnostic("crust_province")
    height = store.load_field("height", grid).interior.astype(np.float64)
    exhumed = store.load_field("bedrock", grid).interior.astype(np.float64) - height
    rock, base = classify(surface, sediment, exhumed, kind > 0, age.astype(np.float64) * float(params.tectonics.myr_per_step),
                          np.zeros(kind.shape, bool) if belt is None else belt > 0, ocean, lake, temperature, precip_cm, cone, province)
    store.save_field(FaceField.from_interior(grid, rock, name="rock", exchange=False))
    store.save_field(FaceField.from_interior(grid, base, name="basement", exchange=False))
    area = grid.interior_cell_area
    info = {"land": shares(rock, area, ~ocean), "sea": shares(rock, area, ocean), "provinces": province is not None}
    top = sorted(info["land"].items(), key=lambda kv: -kv[1])[:6]
    log("[derive] geology: land is " + ", ".join(f"{100 * v:.0f} % {k}" for k, v in top) + ("" if province is not None else " (no crust_province: shields by age)"))
    return info


__all__ = ["NAMES", "FAMILY", "PALETTE", "HARDNESS", "N_ROCKS", "classify", "shares", "run"]
