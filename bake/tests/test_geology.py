"""The geologic map (derive/geology.py): each rock where the bake's own facts put it."""
import json

import numpy as np

from globe.config import WorldParams
from globe.derive import geology as g
from globe.io.world_store import WorldStore
from globe.pipeline import bake


def one(**kw):
    """Classify a single cell: bare, level continental granite country at 600 m, cool and wet,
    cut 100 m by erosion, unless told otherwise."""
    d = dict(surface=600.0, sediment=0.0, exhumed=100.0, continental=True, age_my=900.0, belt=False, ocean=False, lake=False,
             temperature=8.0, precip_cm=80.0, cone=0.0, province=0)
    d.update(kw)
    a = {k: np.array([v]) for k, v in d.items()}
    rock, base = g.classify(a["surface"], a["sediment"], a["exhumed"], a["continental"], a["age_my"], a["belt"], a["ocean"], a["lake"],
                            a["temperature"], a["precip_cm"], a["cone"], a["province"])
    return g.NAMES[int(rock[0])], g.NAMES[int(base[0])]


def test_tables_agree():
    assert len(g.NAMES) == len(g.FAMILY) == len(g.PALETTE) == len(g.HARDNESS) == g.N_ROCKS
    assert g.N_ROCKS <= 32                               # the viewer's palette has 32 slots
    assert len(set(g.PALETTE)) == g.N_ROCKS
    assert all(0.0 <= h <= 1.0 for h in g.HARDNESS)


def test_basement_is_what_tectonics_made():
    assert one(exhumed=800.0) == ("granite", "granite")
    assert one(exhumed=800.0, province=g.PROVINCE_CRATON) == ("gneiss", "gneiss")
    assert one(exhumed=800.0, belt=True) == ("schist", "schist")
    assert one(exhumed=800.0, province=g.PROVINCE_TERRANE) == ("greenstone", "greenstone")
    assert one(cone=900.0) == ("andesite", "andesite")                      # a volcano on a continent
    sea = dict(continental=False, ocean=True, surface=-4000.0, age_my=5.0)
    assert one(**sea) == ("basalt", "basalt")
    assert one(**sea, province=g.PROVINCE_ARC) == ("andesite", "andesite")
    assert one(**sea, cone=3000.0) == ("basalt", "basalt")                  # a hotspot's seamount
    # a world baked before crust_province: shields by the age of the crust
    a = {k: np.array([v]) for k, v in dict(s=600.0, z=0.0, e=800.0, c=True, f=False, t=8.0, p=80.0).items()}
    for age, name in ((2500.0, "gneiss"), (900.0, "granite")):
        rock, _ = g.classify(a["s"], a["z"], a["e"], a["c"], np.array([age]), a["f"], a["f"], a["f"], a["t"], a["p"])
        assert g.NAMES[int(rock[0])] == name


def test_cover_is_what_erosion_and_climate_left():
    assert one(sediment=200.0) == ("sandstone", "granite")                   # a basin
    assert one(sediment=200.0, surface=80.0) == ("shale", "granite")          # its muddy lowland
    assert one(sediment=20.0) == ("alluvium", "granite")
    assert one(sediment=20.0, temperature=-10.0) == ("glacial till", "granite")
    assert one(sediment=800.0, precip_cm=10.0, temperature=22.0) == ("evaporite", "granite")
    assert one(lake=True, sediment=3.0) == ("shale", "granite")               # a lake floor
    assert one(lake=True, precip_cm=10.0, temperature=22.0)[0] == "evaporite"   # a desert lake
    # bare rock erosion barely cut keeps its platform cover; cut deep, it shows the basement
    assert one(exhumed=50.0, temperature=24.0, surface=300.0) == ("limestone", "granite")
    assert one(exhumed=50.0, surface=100.0) == ("shale", "granite")
    assert one(exhumed=50.0, surface=900.0) == ("sandstone", "granite")
    assert one(exhumed=50.0, belt=True, surface=900.0) == ("sandstone", "schist")
    # the sea: a shelf by depth and warmth, the deep floor by age and by a continent's fans
    shelf = dict(ocean=True, continental=True)
    assert one(**shelf, surface=-60.0, temperature=25.0)[0] == "limestone"
    assert one(**shelf, surface=-60.0)[0] == "sandstone"
    assert one(**shelf, surface=-900.0)[0] == "shale"
    deep = dict(ocean=True, continental=False, surface=-4500.0)
    assert one(**deep, age_my=60.0) == ("pelagic sediment", "basalt")
    assert one(**deep, age_my=60.0, sediment=400.0) == ("shale", "basalt")


def test_shares_are_by_area():
    rock = np.array([g.BASALT, g.GRANITE, g.GRANITE, g.SHALE])
    out = g.shares(rock, np.array([1.0, 1.0, 2.0, 4.0]), np.array([True, True, True, False]))
    assert out == {"basalt": 0.25, "granite": 0.75}


def test_a_bake_draws_the_map_and_the_viewer_carries_it(tmp_path):
    """A bake through derive writes the coarse ``rock`` and ``basement`` -- unhashed, so the
    stage's hash is what it was -- and the viewer's final frame carries them with the crust's
    thickness in a texture of their own, named and coloured, for the Geology layer and the
    cross-sections."""
    params = WorldParams.tiny_world(seed=5)
    bake(tmp_path / "w", params, logger=lambda m: None)
    store = WorldStore(tmp_path / "w")
    grid = params.coarse_grid()
    rock, base = store.load_field("rock", grid).interior, store.load_field("basement", grid).interior
    assert rock.dtype == np.uint8 and rock.min() >= 1 and rock.max() < g.N_ROCKS
    assert set(np.unique(base)) <= {g.BASALT, g.ANDESITE, g.GRANITE, g.GNEISS, g.SCHIST, g.GREENSTONE}
    assert len(np.unique(rock)) >= 4
    info = store.manifest["stages"]["derive"]["info"]["geology"]
    assert abs(sum(info["land"].values()) - 1.0) < 1e-3 and info["provinces"] is True
    meta_js = (tmp_path / "w" / "viewer" / "data" / "meta.js").read_text()
    meta = json.loads(meta_js[meta_js.index("(") + 1: meta_js.rindex(")")])
    last = meta["frames"][-1]["layers"]
    assert last["rock"][0] == last["basement"][0] == last["crust_thickness"][0] and last["rock"][0] > 0
    assert meta["channels"]["rock"]["names"] == g.NAMES and meta["channels"]["rock"]["cmap"] == "rock"
    assert meta["rock_palette"] == [list(c) for c in g.PALETTE]
    assert meta["channels"]["crust_thickness"]["unit"] == "km"


def test_contacts_below_the_coarse_grid_are_warped_and_seamless():
    """A point below the coarse grid reads the tectonic classes of the coarse
    cell at its warped place: near its own (well under two cells off), the
    same from either side of a cube edge, the same at every refinement up to
    the wiggle a finer grid adds, and not simply the cell it stands in."""
    from globe.cubesphere import to_sphere_v
    N = 32
    e = (np.arange(4 * N) + 0.5) / (4 * N)
    U, V = np.meshgrid(e, e, indexing="ij")
    p = to_sphere_v(np.full(U.shape, 1), U, V)
    f, i, j = g.warped_cells(p, N, 7, 4)
    own_i, own_j = (U * N).astype(int), (V * N).astype(int)
    inside = f == 1
    assert inside.mean() > 0.9
    off = np.hypot(i - own_i, j - own_j)[inside]
    assert off.max() <= 2.0 and 0.2 < (off > 0).mean() < 0.8        # moved, but not far
    f2, i2, j2 = g.warped_cells(p, N, 7, 4)
    assert np.array_equal(i, i2) and np.array_equal(f, f2)          # a function of the place and the seed
    assert not np.array_equal(i, g.warped_cells(p, N, 8, 4)[1])
    # the same line at a finer refinement: a few cells along the contact differ, no more
    fa, ia, ja = g.warped_cells(p, N, 7, 64)
    assert ((fa != f) | (ia != i) | (ja != j)).mean() < 0.1
    # across a cube edge: two points a hair apart, one on each face, read the same cell
    u = np.array([1.0 - 1e-9, 1.0 + 1e-9])
    for v in (0.13, 0.5, 0.82):
        q = to_sphere_v(np.zeros(2, int), u, np.full(2, v))
        fq, iq, jq = g.warped_cells(q, N, 7, 4)
        assert fq[0] == fq[1] and iq[0] == iq[1] and jq[0] == jq[1]


def test_classify_at_draws_a_contact_as_a_line(tmp_path):
    """On a finer grid the map is :func:`classify` on that grid's own ground
    with the warped tectonic classes: a contact between a craton and younger
    crust keeps its place and its two rocks, and is no longer the edge of the
    coarse cells."""
    from globe.cubesphere import to_sphere_v
    from globe.field import FaceField
    p0 = WorldParams.tiny_world(seed=5)
    grid = p0.coarse_grid()
    N, R = grid.N, 4
    cont = np.ones((6, N, N), bool)
    prov = np.zeros((6, N, N), np.uint8)
    prov[:, : N // 2] = g.PROVINCE_CRATON                    # the contact: the line i = N / 2 on every face
    zero = FaceField.from_interior(grid, np.zeros((6, N, N), np.float32))
    fx = {"continental": cont, "age": zero, "belt": np.zeros((6, N, N), bool), "province": prov, "cone": None, "exhumed": zero}
    e = (np.arange(N * R) + 0.5) / (N * R)
    U, V = np.meshgrid(e, e, indexing="ij")
    shape = U.shape
    rock, base = g.classify_at(fx, to_sphere_v(np.full(shape, 2), U, V), grid, 5, R, 0.15, np.full(shape, 600.0), np.zeros(shape),
                               np.zeros(shape, bool), np.zeros(shape, bool), np.full(shape, 8.0), np.full(shape, 80.0),
                               np.full(shape, 800.0), np.zeros(shape), None)
    assert set(np.unique(rock)) == {g.GNEISS, g.GRANITE} and np.array_equal(rock, base)
    block = np.repeat(np.repeat(prov[2] == g.PROVINCE_CRATON, R, 0), R, 1)        # the coarse cells' own answer
    gneiss = rock == g.GNEISS
    assert 0.45 < gneiss.mean() < 0.55
    differ = (gneiss != block)[2 * R:-2 * R, 2 * R:-2 * R]     # off the face's rim, where the warp reads the next face's pattern
    assert 0.003 < differ.mean() < 0.15
    rows = np.flatnonzero(differ.any(axis=1)) + 2 * R
    assert rows.min() >= (N // 2 - 2) * R and rows.max() < (N // 2 + 2) * R       # only along the contact
    edge = np.array([np.flatnonzero(~gneiss[:, c])[0] for c in range(shape[1])])   # where the contact runs, column by column
    assert edge.std() > 0.5                                                      # a line that wanders, not a cell edge
    # the bedrock's hardness: the basement's where erosion cut through the cover, the cover's where
    # it did not, a smooth step between; oceanic crust and volcanoes have no cover; capped
    cont = np.ones(shape, bool)
    cut = g.bed_hardness(base, cont, np.full(shape, 2000.0), None, 0.85)
    assert np.array_equal(cut, np.minimum(np.asarray(g.HARDNESS, np.float32)[base], np.float32(0.85)))
    kept = g.bed_hardness(base, cont, np.zeros(shape), None, 0.85)
    assert np.allclose(kept, g.COVER_HARDNESS)
    mid = g.bed_hardness(base, cont, np.full(shape, g.EXHUMED_M), None, 0.85)
    assert (mid > kept).all() and (mid < cut).all()
    assert np.array_equal(g.bed_hardness(base, ~cont, np.zeros(shape), None, 0.85), cut)
    assert np.array_equal(g.bed_hardness(base, cont, np.zeros(shape), np.full(shape, 500.0), 0.85), cut)
    assert np.array_equal(g.bed_hardness_at(fx, to_sphere_v(np.full(shape, 2), U, V), grid, 5, R, np.full(shape, 2000.0), None, 0.85), cut)
