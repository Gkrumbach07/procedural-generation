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

import math

import numpy as np
from numba import njit, prange

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


# --------------------------------------------------------------------------
# below the coarse grid: contacts that are not cell edges
# --------------------------------------------------------------------------
#: A contact between two kinds of crust is known to one coarse cell.  Below
#: the coarse grid a point reads the tectonic classes -- kind, belt, province
#: -- of the coarse cell at a place *near* it: its own position moved by a
#: smooth vector field, WARP_CELLS coarse cells at the longest wavelength
#: (WARP_WAVELENGTH cells) and half as far with each halving of it, down to
#: two cells of whatever grid is asking.  The field is noise on the sphere
#: itself, so a contact crosses a cube edge as it crosses anywhere else, and
#: it is the same line at every refinement (a finer level adds only the
#: wiggle below its parent's cell).
WARP_CELLS = 0.6
WARP_WAVELENGTH = 4.0
WARP_KEY = 5_150_123


@njit(cache=True, inline="always")
def _lattice(ix, iy, iz, seed):
    """A number in [-1, 1) for a lattice point: a 64-bit mix of its integers."""
    h = np.uint64(ix & 0xFFFFFFFF) * np.uint64(0x9E3779B97F4A7C15)
    h ^= np.uint64(iy & 0xFFFFFFFF) * np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= np.uint64(iz & 0xFFFFFFFF) * np.uint64(0x165667B19E3779F9)
    h ^= np.uint64(seed & 0xFFFFFFFF) * np.uint64(0xD6E8FEB86659FD93)
    h ^= h >> np.uint64(32)
    h *= np.uint64(0xD6E8FEB86659FD93)
    h ^= h >> np.uint64(29)
    h *= np.uint64(0x9E3779B97F4A7C15)
    h ^= h >> np.uint64(32)
    return float(h >> np.uint64(11)) * (2.0 / 9007199254740992.0) - 1.0


@njit(cache=True, inline="always")
def _noise3(x, y, z, seed):
    """Value noise in [-1, 1] at a point of space, smooth across lattice planes."""
    x0, y0, z0 = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = x - x0, y - y0, z - z0
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fz = fz * fz * (3.0 - 2.0 * fz)
    ix, iy, iz = int(x0), int(y0), int(z0)
    v = 0.0
    for dx in range(2):
        wx = fx if dx else 1.0 - fx
        for dy in range(2):
            wy = fy if dy else 1.0 - fy
            for dz in range(2):
                wz = fz if dz else 1.0 - fz
                v += wx * wy * wz * _lattice(ix + dx, iy + dy, iz + dz, seed)
    return v


@njit(cache=True, parallel=True)
def _warp_kernel(p, cell_rad, amp, wavelength, octaves, seed):
    out = np.empty_like(p)
    for k in prange(p.shape[0]):
        x, y, z = p[k, 0], p[k, 1], p[k, 2]
        dx = dy = dz = 0.0
        a = amp * cell_rad
        lam = wavelength * cell_rad
        for o in range(octaves):
            f = 1.0 / lam
            dx += a * _noise3(x * f, y * f, z * f, seed + 3 * o)
            dy += a * _noise3(x * f, y * f, z * f, seed + 3 * o + 1)
            dz += a * _noise3(x * f, y * f, z * f, seed + 3 * o + 2)
            a *= 0.5
            lam *= 0.5
        r = dx * x + dy * y + dz * z                 # keep the tangent part: the point moves on the sphere
        qx, qy, qz = x + dx - r * x, y + dy - r * y, z + dz - r * z
        n = math.sqrt(qx * qx + qy * qy + qz * qz)
        out[k, 0], out[k, 1], out[k, 2] = qx / n, qy / n, qz / n
    return out


def warp_octaves(R: int) -> int:
    """Octaves of the warp a grid ``R`` times the coarse one resolves: from
    WARP_WAVELENGTH coarse cells down to two of its own."""
    return max(1, int(math.ceil(math.log2(max(WARP_WAVELENGTH * R / 2.0, 1.0)))) + 1)


def warped_cells(p: np.ndarray, N: int, seed: int, R: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The coarse cell ``(face, i, j)`` whose tectonic classes each point of
    ``p`` (..., 3 unit vectors) reads: the cell at its warped position.  ``N``
    is the coarse grid's cells per face, ``R`` the refinement of the grid
    asking (it sets how fine the warp's last octave is), ``seed`` the
    world's."""
    from ..cubesphere import from_sphere_v

    shape = p.shape[:-1]
    flat = np.ascontiguousarray(np.asarray(p, np.float64).reshape(-1, 3))
    q = _warp_kernel(flat, 0.5 * math.pi / N, WARP_CELLS, WARP_WAVELENGTH, warp_octaves(R), int(seed) + WARP_KEY)
    f, u, v = from_sphere_v(q)
    i = np.minimum((u * N).astype(np.int64), N - 1)
    j = np.minimum((v * N).astype(np.int64), N - 1)
    return f.reshape(shape), i.reshape(shape), j.reshape(shape)


#: the tectonic diagnostics the map reads, per world root (None where the world has none)
_FACTS: dict = {}


def facts(root, grid) -> dict | None:
    """The coarse fields the map reads from a baked world, interiors
    ``(6, N, N)``: ``continental`` (bool), ``age`` (steps), ``belt`` (bool),
    ``province`` and ``cone`` (None where tectonics saved none), ``exhumed``
    (m, a FaceField: it is read bilinearly).  None where the world has no
    crust diagnostics."""
    from pathlib import Path

    from ..field import FaceField
    from ..io.world_store import WorldStore

    key = str(Path(root).resolve())
    if key not in _FACTS:
        _FACTS.clear()
        store = WorldStore(root)
        diag = store.root / "diagnostics"
        get = lambda n: FaceField.load(diag, n, grid) if FaceField.exists(diag, n) else None
        kind, age = get("crust_kind"), get("crust_age")
        if kind is None or age is None or not store.has_field("height"):
            _FACTS[key] = None
        else:
            belt, prov, cone = get("collision_zone"), get("crust_province"), get("volcano_cone")
            ex = FaceField.from_interior(grid, (store.load_field("bedrock", grid).interior.astype(np.float32)
                                                - store.load_field("height", grid).interior.astype(np.float32)), name="exhumed")
            _FACTS[key] = {"continental": kind.interior > 0, "age": age, "belt": None if belt is None else belt.interior > 0,
                           "province": None if prov is None else prov.interior, "cone": cone, "exhumed": ex}
    return _FACTS[key]


def classify_at(fx: dict, p: np.ndarray, grid, seed: int, R: int, myr_per_step: float, surface: np.ndarray, sediment: np.ndarray, ocean: np.ndarray,
                lake: np.ndarray, temperature: np.ndarray, precip_cm: np.ndarray, exhumed: np.ndarray, age: np.ndarray,
                cone: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """:func:`classify` on a grid ``R`` times the coarse one.  ``fx`` is
    :func:`facts`; ``p`` (..., 3) the cells' places on the sphere.  The
    tectonic classes come from the warped coarse cell (:func:`warped_cells`);
    ``exhumed`` (m), ``age`` (steps) and ``cone`` (m) are the coarse fields
    the caller sampled on its own cells, smooth fields whose contours are
    contacts already; the rest is the caller's own surface, water and
    climate."""
    f, i, j = warped_cells(p, grid.N, seed, R)
    cont = fx["continental"][f, i, j]
    belt = fx["belt"][f, i, j] if fx["belt"] is not None else np.zeros(cont.shape, bool)
    prov = fx["province"][f, i, j] if fx["province"] is not None else None
    return classify(surface, sediment, exhumed, cont, np.asarray(age, np.float64) * float(myr_per_step), belt, ocean, lake,
                    temperature, precip_cm, cone, prov)


#: hardness of the sedimentary cover a continent carries where erosion has not cut through it
COVER_HARDNESS = 0.5
#: the cover gives way to the basement over this much more exhumation (m), centred on EXHUMED_M
COVER_FADE_M = 200.0


def bed_hardness(basement: np.ndarray, continental: np.ndarray, exhumed: np.ndarray, cone: np.ndarray | None = None, top: float = 1.0) -> np.ndarray:
    """Resistance to erosion (0..1, float32) of the bedrock under a cell,
    for the kernel (its bedrock erodibility is 1 - hardness): the basement's
    (:data:`HARDNESS`) where erosion has cut through the cover -- ``exhumed``
    (m) past :data:`EXHUMED_M` -- or a volcano stands, or the crust is
    oceanic and never had one; :data:`COVER_HARDNESS` where the cover is
    still there; capped at ``top``.

    Only what tectonics and the erosion's own depth say goes into it.  The
    map's cover classes turn on the ground's height and the climate
    (limestone below 500 m where it is warm, shale below 200 m), which is how
    a map is drawn and not where beds lie: a hardness read off them would
    step along those contours and the erosion would cut a terrace at 200 m
    and at 500 m around the whole planet."""
    hard = np.asarray(HARDNESS, np.float32)[np.asarray(basement)]
    t = np.clip((np.asarray(exhumed, np.float32) - (EXHUMED_M - 0.5 * COVER_FADE_M)) / COVER_FADE_M, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    t = np.where(np.asarray(continental, bool), t, 1.0)
    if cone is not None:
        t = np.where(np.asarray(cone) >= CONE_M, 1.0, t)
    return np.minimum(COVER_HARDNESS + (hard - COVER_HARDNESS) * t, np.float32(top)).astype(np.float32)


def bed_hardness_at(fx: dict, p: np.ndarray, grid, seed: int, R: int, exhumed: np.ndarray, cone: np.ndarray | None, top: float = 1.0) -> np.ndarray:
    """:func:`bed_hardness` on a grid ``R`` times the coarse one: the
    basement of the warped coarse cell (:func:`warped_cells`) under each
    place ``p``, with ``exhumed`` and ``cone`` (m) as the caller sampled
    them."""
    f, i, j = warped_cells(p, grid.N, seed, R)
    cont = fx["continental"][f, i, j]
    belt = fx["belt"][f, i, j] if fx["belt"] is not None else np.zeros(cont.shape, bool)
    prov = fx["province"][f, i, j] if fx["province"] is not None else None
    z = np.zeros(cont.shape)
    age = np.zeros(cont.shape) if prov is not None else fx["age"].interior[f, i, j].astype(np.float64)
    _, base = classify(z, z, z, cont, age, belt, np.zeros(cont.shape, bool), np.zeros(cont.shape, bool), z, z, cone, prov)
    return bed_hardness(base, cont, exhumed, cone, top)


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


__all__ = ["NAMES", "FAMILY", "PALETTE", "HARDNESS", "N_ROCKS", "classify", "classify_at", "warped_cells", "warp_octaves", "facts", "bed_hardness", "bed_hardness_at", "shares", "run"]
