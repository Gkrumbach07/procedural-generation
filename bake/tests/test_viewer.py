"""The HTML viewer's data: frame capture, cube-face atlases, the export."""
import json
import re

import numpy as np
import pytest

from globe.config import WorldParams
from globe.cubesphere import from_sphere_v, to_sphere_v
from globe.pipeline import bake
from globe.viz import frames as vf
from globe.viz import viewer as vw


def _centres(res: int, pad: int = 0) -> np.ndarray:
    k = np.arange(-pad, res + pad)
    I, J = np.meshgrid(k, k, indexing="ij")
    shape = (6,) + I.shape
    return to_sphere_v(np.broadcast_to(np.arange(6)[:, None, None], shape),
                       np.broadcast_to((I + 0.5) / res, shape), np.broadcast_to((J + 0.5) / res, shape))


def test_pad_reads_the_neighbouring_face():
    """A padded border cell holds the value of the cell next to it across the
    seam: for a smooth field it is within about a cell of the true value."""
    res = 32
    a = np.array([0.3, -0.5, 0.8])
    f = _centres(res) @ a
    padded = vw.pad_faces(f, 1)
    assert np.array_equal(padded[:, 1:-1, 1:-1], f)
    truth = _centres(res, 1) @ a
    step = np.linalg.norm(a) * (np.pi / 2) / res           # |∇f| · one cell
    assert np.abs(padded - truth).max() < 2.0 * step


def test_atlas_layout():
    """Face f sits at column f % 3, row f // 3; pixel (x, y) = (pad + i, pad + j)."""
    res, pad = 8, 1
    T = res + 2 * pad
    f, i, j = np.meshgrid(np.arange(6), np.arange(T), np.arange(T), indexing="ij")
    code = (f * 100 + i * 10 + j).astype(np.int64)
    img = vw.atlas([(code % 256).astype(np.uint8), (code // 256).astype(np.uint8)])
    assert img.shape == (2 * T, 3 * T, 2)
    for face in range(6):
        r, c = divmod(face, 3)
        for (ii, jj) in [(0, 0), (3, 5), (T - 1, 2)]:
            px = img[r * T + jj, c * T + ii]
            assert int(px[0]) + 256 * int(px[1]) == face * 100 + ii * 10 + jj


def test_height_encoding_round_trips():
    rng = np.random.default_rng(0)
    h = rng.uniform(-7000, 9000, (6, 10, 10))
    hi, lo, h0, h1 = vw.encode_height(h)
    q = hi.astype(np.float64) * 256 + lo
    back = h0 + q / 65535.0 * (h1 - h0)
    assert np.abs(back - h).max() <= 0.5 * (h1 - h0) / 65535 + 1e-9


def test_height_encoding_keeps_the_sign_of_every_cell():
    """Sea level sits exactly on a code and no cell crosses it in the
    rounding: a coastal plain 0.05 m above the water used to decode below
    it, so the shader drew the cell as sea (docs/viewer: the stair-stepped
    coasts of earth-v9)."""
    h = np.array([-5700.0, -0.04, -1e-6, 0.0, 1e-6, 0.05, 7834.0]).reshape(1, 1, 7) * np.ones((6, 1, 1))
    hi, lo, h0, h1 = vw.encode_height(h)
    back = h0 + (hi.astype(np.float64) * 256 + lo) / 65535.0 * (h1 - h0)
    step = (h1 - h0) / 65535.0
    assert np.all((back < 0) == (h < 0)), back[0, 0]
    assert np.abs(back - h).max() <= step + 1e-9
    assert h0 <= h.min() and h1 >= h.max()


def test_lake_depth_is_signed_so_the_shore_falls_between_cells():
    """A lake cell carries its depth, the ground near it the lake level less
    its own height (< 0), everything else the far value -- so the
    interpolated 0 crossing is where the ground meets the water.  A pond of
    one cell is too small to smooth: it keeps its cell."""
    from globe.viz import detail as dt

    surf = np.zeros((6, 12, 12), np.float32) + 50.0
    surf[0, 5, 5] = 10.0                              # a one-cell lake ...
    surf[0, 5, 6] = 40.0                              # ... whose neighbour stands 10 m above its water
    ws = surf.copy()                                  # dry ground: water surface = ground
    ws[0, 5, 5] = 30.0                                # the lake's level
    code = vw.water_code(surf, None, ws, 0.5)
    d = vw.lake_depth(surf, ws, code)
    assert d[0, 5, 5] == 20.0                         # kept whole
    assert -10.5 < d[0, 5, 6] < -9.5 and -20.5 < d[0, 5, 4] < -19.5      # the banks: their height over the water, smoothed within half a metre of it only
    assert d[0, 5, 5 + dt.SHORE_RINGS] < -19.0 and d[0, 5, 5 + dt.SHORE_RINGS + 1] == -vw.LAKE_DEPTH_RANGE_M   # the ground, that far out
    assert d[3, 5, 5] == -vw.LAKE_DEPTH_RANGE_M


def test_a_flat_shore_is_smoothed_and_its_byte_is_finest_at_the_water():
    """Lake country is flat at the water's edge: whether a cell is in the
    lake is a metre's difference from the next, and the shore drawn from that
    is the cells' outline.  ``detail.smooth_shore`` smooths the field within
    a few metres of the level, so a one-cell tooth on a straight shore is cut
    back and a one-cell bay filled, while the deep water and the high ground
    keep their values; a finger lake a cell wide, which that would smooth
    dry, keeps its cells.  And ``detail.lake_byte`` spends its steps at the
    shore: a centimetre there, where a linear byte has 1.57 m."""
    from globe.viz import detail as dt

    R = 200.0
    n = 40
    surf = np.full((n, n), 1.0, np.float32)           # banks a metre over the water
    ws = surf.copy()
    water = np.zeros((n, n), np.uint8)
    lake = np.zeros((n, n), bool)
    lake[:, :20] = True                               # a lake with a straight shore down column 19 | 20 ...
    lake[10, 20] = True                               # ... a tooth of one cell
    lake[25, 19] = False                              # ... and a bay of one
    lake[5:35, 30] = True                             # a finger lake, one cell wide
    surf[lake] = -3.0
    surf[:, :12] = -40.0                              # deep water away from the shore
    surf[:, 36:] = 60.0                               # high ground beyond
    ws[lake] = 0.0
    water[lake] = dt.WATER_LAKE
    raw = dt.face_lake_depth(surf, ws, water, R, rings=1, sigma=0.0)
    d = dt.face_lake_depth(surf, ws, water, R)
    assert raw[10, 20] > 0 > raw[25, 19]                                  # the cells' outline has both
    assert d[10, 20] < 0.0 < d[25, 19]                                    # the smoothed shore has neither
    assert d[15, 19] > 0.0 > d[15, 20]                                    # ...and is where it was
    assert d[15, 5] == pytest.approx(40.0, abs=0.01) and d[15, 37] < -55.0   # deep water and high ground untouched
    assert (d[8:32, 30] > 0.0).all()                                      # the finger keeps its cells
    b = dt.lake_byte(np.array([-R, -10.0, -0.02, 0.0, 0.02, 1.0, 10.0, R]), R).astype(int)
    assert b[0] == 0 and b[-1] == 255 and (np.diff(b) >= 0).all()
    assert b[2] < 127.5 < b[4] and b[5] - b[4] >= 7                       # two centimetres either side are apart; a metre is 7 steps
    lin = np.round(255.0 * (np.array([0.02, 1.0]) + R) / (2 * R))
    assert lin[0] == lin[1]                                               # which a linear byte cannot tell from the shore


def test_the_step_to_the_far_value_is_not_drawn_as_a_shore():
    """The ground's height against a lake's level is known ``SHORE_RINGS``
    cells out from it and is the far value beyond, one step down.  The page
    fades the shore over a pixel of the field's own slope (viewer.html,
    ``wCov``), and a pixel on that step is within its slope of zero: every
    lake drew with a dotted ring round it, four cells out.  The fade is of the
    field between the dry ground's floor (``SHORE_DRY_M``) and as much water,
    which is flat across the step.  The shader's rule, replayed along a line
    across a shore."""
    import re

    from globe.viz import detail as dt

    R = 200.0
    n = 40
    surf = np.full((n, n), 1.0, np.float32)
    water = np.zeros((n, n), np.uint8)
    surf[:, :20] = -3.0
    water[:, :20] = dt.WATER_LAKE
    ws = np.maximum(surf, 0.0)
    lt = dt.lake_byte(dt.face_lake_depth(surf, ws, water, R), R)[15] / 127.5 - 1.0
    x = np.arange(0.0, n - 1, 0.5)                                # two pixels a cell
    line = np.interp(x, np.arange(n), lt)
    far = (x > 20.0 + dt.SHORE_RINGS - 2.0) & (x < 20.0 + dt.SHORE_RINGS + 2.0)
    assert line[far].min() == pytest.approx(-1.0) and line[far].max() > -0.1   # the step is here

    def cover(g):
        sw = np.maximum(np.abs(np.gradient(g)), 1e-5)
        t = np.clip((g + sw) / (2.0 * sw), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    sc = np.sqrt(dt.SHORE_DRY_M / R)
    assert cover(line)[far].max() > 0.05                          # unclamped: the ring
    c = cover(np.clip(line, -sc, sc))
    assert c[far].max() == 0.0 and c[x > 21.0].max() == 0.0       # clamped: none, and no water past the shore's own cells
    assert (c[x < 18.0] == 1.0).all()                             # the lake is whole
    assert 18.5 < x[np.argmin(np.abs(c - 0.5))] < 20.5            # and its shore is where it was

    js = vw.TEMPLATE.read_text()
    assert float(re.search(r"SHORE_DRY_M = ([0-9.]+)", js).group(1)) == dt.SHORE_DRY_M
    assert "sg = clamp(shoreT, -sc, sc)" in js and "smoothstep(-sw, sw, sg)" in js


def test_smooth_mask_rounds_corners_but_keeps_single_cells():
    """The coast is the 0.5 contour of the smoothed ocean mask: a corner cell
    is pulled towards 0.5, and a one-cell island or inlet stays on its side."""
    ocean = np.zeros((6, 12, 12), bool)
    ocean[0, :6, :6] = True                           # a square bay: its corner should round
    ocean[1, 4, 4] = True                             # a one-cell inlet
    s = vw.smooth_mask(ocean)
    assert s[0, 5, 5] >= 0.6 and s[0, 5, 5] < 0.75    # corner cell: still sea, but only just
    assert s[0, 6, 6] <= 0.4 and s[0, 6, 6] > 0.05    # diagonal land neighbour: land, pulled up
    assert s[1, 4, 4] >= 0.6 and s[1, 3, 3] <= 0.4
    assert np.all(s[~ocean] <= 0.4) and np.all(s[ocean] >= 0.6)


def test_refined_final_frame_uses_derives_sea_and_lake_rules(tmp_path):
    """``refined=True`` draws the final frame from ``fine/`` at full
    resolution, with derive's water rules: sea is fine ground below 0 in a
    coarse ocean cell or one touching it, so a refined cell poking above the
    water offshore is land and a hollow below 0 far inland is not sea; and
    coarse-only channels are repeated onto the fine grid."""
    N, R = 8, 2
    root = tmp_path / "w"
    (root / "coarse").mkdir(parents=True)
    (root / "fine").mkdir()
    (root / "manifest.json").write_text(json.dumps({"stages": {}, "params": {"hydro": {"lake_min_depth": 0.5}}}))
    hc = np.full((6, N, N), 100.0, np.float32)
    hc[0, :, :3] = -50.0                               # a strip of ocean on face 0
    fd = np.zeros((6, N, N), np.uint8)
    fd[hc < 0] = 255
    hf = np.repeat(np.repeat(hc, R, axis=1), R, axis=2).copy()
    hf[0, 2, 1] = 3.0                                  # a refined speck above the water, offshore
    hf[3, 8, 8] = -1.0                                 # a hollow below 0, far from any ocean
    for f in range(6):
        for name, a in (("height", hc), ("sediment", np.zeros_like(hc)), ("flow_dir", fd), ("water_surface", np.maximum(hc, 0)),
                        ("discharge", np.ones_like(hc)), ("temperature", np.full_like(hc, 10.0))):
            np.save(root / "coarse" / f"{name}.f{f}.npy", a[f])
        for name, a in (("height", hf), ("sediment", np.zeros_like(hf)), ("water_surface", np.maximum(hf, 0)),
                        ("discharge", np.ones_like(hf))):
            np.save(root / "fine" / f"{name}.f{f}.npy", a[f])
    coarse, _ = vw.collect_frames(root, None, log=lambda m: None)
    fine, _ = vw.collect_frames(root, None, log=lambda m: None, refined=True)
    assert coarse[-1].res == N and fine[-1].res == N * R
    w = fine[-1].ch["water"]
    assert w[0, 0, 0] == vw.WATER_OCEAN and w[0, 2, 1] == vw.WATER_LAND   # the speck is land
    assert w[3, 8, 8] == vw.WATER_LAND                                     # the inland hollow is not sea
    assert fine[-1].ch["temperature"].shape == (6, N * R, N * R)


def test_equirect_has_its_pole_on_z():
    """Latitude follows +Z, as in Grid.latitude and the climate stage: a field
    that depends only on z is constant along every equirect row."""
    res = 32
    img = vw.equirect(vw.pad_faces(_centres(res)[..., 2], 1), 256)
    assert img.shape == (128, 256)
    assert np.ptp(img, axis=1).max() < 0.05
    assert img[0].mean() > 0.95 and img[-1].mean() < -0.95


def test_the_crust_byte_carries_both_its_kind_and_its_age():
    """One byte holds both: the kind in the top bit, the age in the seven
    below it.  A texture of its own for the age would be another 25 MB of an
    Earth export, and the kind was spending a byte on one bit -- so both
    layers read the same slot, and the page is told which bits are theirs."""
    rng = np.random.default_rng(3)
    kind = (rng.random((6, 16, 16)) < 0.5).astype(np.uint8)
    # the sea floor is recycled and the continents are as old as the run: each kind needs the
    # whole ramp over its own range, or every ocean sits in the ramp's first colours
    age = np.where(kind > 0, rng.uniform(3000, 4000, kind.shape), rng.uniform(0, 1200, kind.shape)).astype(np.float32)
    fr = vw._Frame("final", 0, "final", np.zeros((6, 16, 16), np.float32), crust=kind, crust_age=age)
    specs = vw.channel_specs(fr)
    sa = specs["crust_age"]
    assert specs["crust"]["bits"] == [7, 1] and sa["bits"] == [0, 127] and sa["unit"] == "steps"
    assert sa["kind"] == "log" and sa["hi"] < 1300 and sa["cont_lo"] > 2000 and sa["cont_hi"] > 3900   # a scale each
    b = vw._byte("crust", kind, specs, fr.ch)
    assert b.dtype == np.uint8
    assert ((b >> 7) == kind).all()                                  # the kind, as the layer reads it
    lo = np.where(kind > 0, sa["cont_lo"], sa["lo"])                 # and the age, as the page decodes it
    hi = np.where(kind > 0, sa["cont_hi"], sa["hi"])
    got = lo * np.power(hi / lo, (b & 127) / 127.0)
    inside = (age >= lo) & (age <= hi)
    assert (np.abs(got - age) / age)[inside].max() < 0.05
    # both kinds use the whole ramp, so a ridge is not lost in the first few colours
    for sel in (kind > 0, kind == 0):
        span = (b & 127)[sel]
        assert span.min() < 16 and span.max() > 111, (span.min(), span.max())
    images, meta = vw.encode_frame(fr, specs)
    assert meta["layers"]["crust_age"] == meta["layers"]["crust"], meta["layers"]
    assert len(images) == 1 + sum(1 for names in vw.FINAL_TEXTURES if any(n in fr.ch for n in names))


def test_bake_captures_frames_and_exports_a_viewer(tmp_path):
    """Frames are captured for both stages, the viewer is written, and frame
    capture changes no stage hash."""
    on = WorldParams.tiny_world(seed=5)
    off = WorldParams.tiny_world(seed=5)
    off.render.viewer = False
    bake(tmp_path / "on", on, to_stage="erosion", logger=lambda m: None)
    bake(tmp_path / "off", off, to_stage="erosion", logger=lambda m: None)
    m_on = json.loads((tmp_path / "on" / "manifest.json").read_text())
    m_off = json.loads((tmp_path / "off" / "manifest.json").read_text())
    for s in ("tectonics", "climate", "erosion"):
        assert m_on["stages"][s]["hash"] == m_off["stages"][s]["hash"], s
    assert not (tmp_path / "off" / "frames").exists()
    assert not (tmp_path / "off" / "viewer").exists()

    tect = vf.list_frames(tmp_path / "on", "tectonics")
    ero = vf.list_frames(tmp_path / "on", "erosion")
    assert len(tect) == on.render.tectonics_frames + 1
    assert [k for k, _, _ in ero][0] == 0 and [k for k, _, _ in ero][-1] == on.erosion.iterations
    index = tmp_path / "on" / "viewer" / "index.html"
    assert index.exists() and "data/meta.js" in index.read_text()
    meta_js = (tmp_path / "on" / "viewer" / "data" / "meta.js").read_text()
    meta = json.loads(meta_js[meta_js.index("(") + 1: meta_js.rindex(")")])
    assert len(meta["frames"]) == len(tect) + len(ero) + 1
    assert [f["stage"] for f in meta["frames"]][-1] == "final"
    for f in meta["frames"]:
        assert (tmp_path / "on" / "viewer" / f["file"]).exists()
    assert "plate" in meta["frames"][0]["layers"]
    # the crust's age runs through the tectonic animation too, in a texture of its own: the
    # sea floor being made at the ridges and eaten at the trenches is the whole point of it
    with np.load(tect[len(tect) // 2][1]) as z:
        assert "crust_age" in z.files and "crust_kind" in z.files
        assert z["crust_age"].shape == z["height"].shape and float(z["crust_age"].max()) > 0
    for f in meta["frames"]:
        if f["stage"] != "tectonics":
            continue
        assert f["layers"]["crust_age"] == f["layers"]["crust"] and f["layers"]["crust"][0] > 0
    # an erosion frame holds its water and its ice beside the ground, at the frame's own
    # resolution, and the page gets them as a shoreline texture of the frame's own
    N = on.world.N_c
    assert on.render.frame_res >= N                    # a toy body's frame is its whole grid
    for _, path, _ in ero:
        with np.load(path) as z:
            assert z["lake"].shape == z["ice"].shape == z["height"].shape == (6, N, N)
            assert z["lake"].dtype == np.float16 and z["ice"].dtype == np.uint8 and float(z["lake"].min()) >= 0.0
    for f in meta["frames"]:
        if f["stage"] == "erosion":
            assert f["layers"]["lake_depth"] == [1, 1] and f["layers"]["ocean"] == [1, 2] and len(f["tex"]) == 2
    assert "discharge" in meta["frames"][-1]["layers"]
    assert "water" in meta["frames"][-1]["layers"]  # what the elevation view colours as water
    # rivers on the final frame are the particles' discharge in texture 0's B,
    # eased in over a byte span, as on the erosion frames
    fin = meta["frames"][-1]
    assert fin["layers"]["discharge"] == [0, 2] and fin["river_span_byte"] >= 1
    assert meta["channels"]["ocean"]["hidden"] and fin["layers"]["ocean"][1] == 2


def test_water_code_tells_lakes_from_sea_and_from_closed_basins():
    """Sea, a closed basin below sea level, dry land, and a lake standing
    above sea level -- the last two of which `height < 0` cannot see."""
    surf = np.array([[-3000.0, -200.0, 50.0, 800.0]], np.float32)
    fd = np.array([[255, 0, 0, 0]], np.uint8)          # hydro: only the first is sea
    ws = np.array([[0.0, 20.0, 0.0, 900.0]], np.float32)
    assert list(vw.water_code(surf, fd, ws, 0.5)[0]) == [vw.WATER_OCEAN, vw.WATER_LAKE, vw.WATER_LAND, vw.WATER_LAKE]
    # without hydro the height sign is all there is: both lakes disappear and
    # the closed basin reads as ocean
    assert list(vw.water_code(surf, None, None)[0]) == [vw.WATER_OCEAN, vw.WATER_OCEAN, vw.WATER_LAND, vw.WATER_LAND]


def test_erosion_frame_records_the_lakes_and_the_ice(scratch, tmp_path):
    """A frame holds the standing water over its ground and the share of each
    cell under the glacial pass's ice (``frames.erosion_frame``).  Both kinds
    of lake are read off the state: an overflowing one is flagged and stands
    at the level the refresh solved (the routing surface once that is stale),
    a closed one at the base level of its cells.  Water no deeper than the
    marsh line is ground, and so is ground under a base level of 0 -- that is
    the sea, to the kernel and here.  The ice is zero until the pass starts,
    then the cold land plus what the last pass carved under."""
    from test_erosion import _dry_site, _land_world

    p, st = _land_world(scratch, "frame_water")
    p = p.with_overrides(erosion={"iterations": 60, "glacial_every": 10, "glacial_from": 0.5, "glacial_sticky": True, "ice_evap": 0.0},
                         hydro={"lake_min_depth": 0.5, "marsh_depth": 3.0})
    u, H, N = st.height_unit_m, st.H, st.N
    f, i0, j0 = _dry_site(st, 8)
    ia, ja = i0 + (i0 & 1), j0 + (j0 & 1)                 # on a 2 x 2 block corner, for the block mean below
    box = lambda i, j, n=2, m=2: (f, slice(H + ia + i, H + ia + i + n), slice(H + ja + j, H + ja + j + m))   # noqa: E731
    assert (st.surface()[box(0, 0, 6, 6)] > 0.0).all()
    st.route = st.surface().copy()
    st.lake_flag = np.zeros(st.height.shape, np.uint8)
    st.lake_flag[box(0, 0)] = 1
    st.route[box(0, 0)] += 12.0 / u                       # an overflowing lake, 12 m deep
    st.lake_flag[box(0, 2)] = 2
    st.route[box(0, 2)] += 2.0 / u                        # ...and 2 m of water beside it: marsh
    st.base[box(2, 0)] = st.surface()[box(2, 0)] + 7.0 / u    # a closed lake, 7 m deep
    coast = (f, H + ia + 4, H + ja)
    st.height[coast] = -5.0 / u - st.sediment[coast]      # ground gone under since the sea mask was made
    st.evap[...] = 1.0                                    # a warm planet...
    st.evap[box(2, 2)] = 0.0                              # ...with four cells of frozen land
    st.ice_prev = np.zeros(st.height.shape, bool)
    st.ice_prev[coast] = True                             # and one the last pass carved below the sea

    def frame(res, iteration):
        rec = vf.FrameRecorder(tmp_path / f"r{res}", "erosion", res)
        vf.erosion_frame(st, rec, iteration, 60, p)
        key, path, meta = vf.list_frames(tmp_path / f"r{res}", "erosion")[-1]
        with np.load(path) as z:
            return {k: z[k] for k in z.files}, meta

    cell = lambda a, i, j, n=2, m=2: a[f, ia + i:ia + i + n, ja + j:ja + j + m]   # noqa: E731
    d, meta = frame(N, 20)
    assert set(d) == {"height", "discharge", "lake", "ice"}
    assert d["lake"].shape == d["ice"].shape == (6, N, N) and d["lake"].dtype == np.float16 and d["ice"].dtype == np.uint8
    assert np.allclose(cell(d["lake"], 0, 0), 12.0, atol=0.01) and np.allclose(cell(d["lake"], 2, 0), 7.0, atol=0.01)
    assert (cell(d["lake"], 0, 2) == 0.0).all()                                   # the marsh is ground
    assert int((d["lake"] > 0).sum()) == 8 and not (d["lake"][d["height"] < 0] > 0).any()   # ...and so is the sea, the new coast with it
    assert not d["ice"].any() and "glacial" not in meta                           # the pass starts half way through
    d, meta = frame(N, 40)
    assert meta["glacial"] is True and (cell(d["ice"], 2, 2) == 255).all() and d["ice"][f, ia + 4, ja] == 255
    assert int((d["ice"] > 0).sum()) == 5
    # block means at half the grid: a lake's depth over the block, the ice as its share of it
    d, _ = frame(N // 2, 40)
    assert d["lake"].shape == d["ice"].shape == (6, N // 2, N // 2)
    b = lambda a, i, j: a[f, (ia + i) // 2, (ja + j) // 2]                        # noqa: E731
    assert b(d["lake"], 0, 0) == pytest.approx(12.0, abs=0.01) and b(d["lake"], 2, 0) == pytest.approx(7.0, abs=0.01)
    assert b(d["ice"], 2, 2) == 255 and b(d["ice"], 4, 0) == 64 and b(d["lake"], 4, 0) == 0.0
    # the level the lake refresh solved, while the surface is the one it solved it on
    st.lake_level = (st.surface()[st.interior] + 20.0 / u).astype(np.float32)
    st.lake_level_at = st.iteration
    assert np.allclose(cell(frame(N, 40)[0]["lake"], 0, 0), 20.0, atol=0.05)
    st.lake_level_at = st.iteration - 1
    assert np.allclose(cell(frame(N, 40)[0]["lake"], 0, 0), 12.0, atol=0.01)


def _frames_world(root, water: bool):
    """A hand-made world of one coarse state and two erosion frames: a lake
    7 m deep whose bed is 4 m below sea level, a strip of sea, and -- on the
    second frame, taken in the glacial phase -- ice over half of face 1."""
    N = 8
    (root / "coarse").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"stages": {}, "N_c": N, "cell_size_m": 1000.0, "seed": 0,
                                                    "params": {"hydro": {"lake_min_depth": 0.5}}}))
    h = np.full((6, N, N), 100.0, np.float32)
    h[0, :, :3] = -50.0                                # the sea
    h[2, 3:5, 3:5] = -4.0                              # the lake's bed
    for k in range(6):
        np.save(root / "coarse" / f"height.f{k}.npy", h[k])
        np.save(root / "coarse" / f"sediment.f{k}.npy", np.zeros((N, N), np.float32))
        np.save(root / "coarse" / f"discharge.f{k}.npy", np.ones((N, N), np.float32))
    lake = np.zeros((6, N, N), np.float16)
    lake[2, 3:5, 3:5] = 7.0
    ice = np.zeros((6, N, N), np.uint8)
    rec = vf.FrameRecorder(root, "erosion", N)
    for key, glacial in ((0, False), (10, True)):
        arrays = {"height": h.astype(np.float16), "discharge": np.ones((6, N, N), np.float16)}
        if water:
            if glacial:
                ice[1, :, :4] = 255
                ice[1, :, 4] = 128
            arrays.update(lake=lake, ice=ice.copy())
        rec.write(key, arrays, iteration=key, of=10, units="m", **({"glacial": True} if glacial and water else {}))
    return N


def _viewer_frames(index):
    """The frames of an exported viewer: ``[(meta, [texture payloads])]``."""
    meta_js = (index.parent / "data" / "meta.js").read_text()
    meta = json.loads(meta_js[meta_js.index("(") + 1: meta_js.rindex(")")])
    out = []
    for f in meta["frames"]:
        js = (index.parent / f["file"]).read_text()
        out.append((f, json.loads(js[js.index("["): js.rindex("]") + 1])))
    return meta, out


def test_erosion_frames_with_water_get_a_shoreline_texture_and_an_ice_layer(tmp_path):
    """A frame that recorded its lakes is drawn through the final frame's
    shore path: a second texture with the signed lake depth in G and the
    smoothed ocean mask in B, where the page looks for them, and the ice
    share in R as a layer of its own.  A lake whose bed is below sea level
    is a lake, not a hole of ocean in itself.  Frames written before any of
    this export as they did: one texture, the same bytes, no new layer."""
    N = _frames_world(tmp_path / "wet", water=True)
    _frames_world(tmp_path / "dry", water=False)
    frames, _ = vw.collect_frames(tmp_path / "wet", None, log=lambda m: None)
    first, second = [fr for fr in frames if fr.stage == "erosion"]
    assert second.label.endswith("ice age") and "ice age" not in first.label
    for fr in (first, second):
        assert fr.ch["lake_depth"][2, 3, 3] == 7.0 and fr.ch["lake_depth"][2, 2, 3] == pytest.approx(3.0 - 100.0, abs=3.0)   # the shore next to it: its level less the ground (smoothed within half a metre of the level)
        assert fr.ch["ocean"][2, 3, 3] <= 0.4 and fr.ch["ocean"][0, 4, 1] >= 0.6 and fr.ch["ocean"][3, 4, 4] == 0.0
    assert not first.ch["ice"].any() and second.ch["ice"][1, 0, 0] == 1.0 and second.ch["ice"][1, 0, 4] == pytest.approx(128 / 255)
    specs = vw.channel_specs(frames[-1], None, frames[:-1])
    assert specs["ice"]["kind"] == "linear" and not specs["ice"].get("hidden") and specs["lake_depth"]["hidden"]
    images, meta = vw.encode_frame(second, specs)
    assert len(images) == 2 and meta["layers"] == {"discharge": [0, 2], "ice": [1, 0], "lake_depth": [1, 1], "ocean": [1, 2]}
    T = N + 2 * vw.PAD
    px = lambda face, i, j: images[1][(face // 3) * T + vw.PAD + j, (face % 3) * T + vw.PAD + i]   # noqa: E731
    assert px(2, 3, 3)[1] > 127 and px(2, 0, 0)[1] < 127 and px(4, 4, 4)[1] == 0 and px(2, 3, 3)[2] == 0   # a lake: depth above the shoreline byte, its banks below, nothing far from it; no ocean
    assert px(0, 4, 1)[2] > 127 and px(0, 4, 1)[1] == 0                                # the sea
    assert px(1, 0, 0)[0] == 255 and px(1, 0, 4)[0] == 128 and px(1, 0, 7)[0] == 0     # the ice, as a share
    # the whole way out, beside the same frames without their water
    meta_w, wet = _viewer_frames(vw.export_viewer(tmp_path / "wet", log=lambda m: None))
    meta_d, dry = _viewer_frames(vw.export_viewer(tmp_path / "dry", log=lambda m: None))
    assert "ice" in meta_w["channels"] and "ice" not in meta_d["channels"]
    for (fw, tw), (fd, td) in zip(wet, dry):
        assert tw[0] == td[0]                                                           # the ground and the rivers: byte for byte
        if fw["stage"] != "erosion":
            assert fw == fd and tw == td
            continue
        assert len(tw) == 2 and len(fw["tex"]) == 2 and fw["layers"]["ocean"] == [1, 2]
        assert len(td) == 1 and "tex" not in fd and fd["layers"] == {"discharge": [0, 2]} and "ice age" not in fd["label"]
    # a run with no ice in it has no ice layer
    for _, path, meta in vf.list_frames(tmp_path / "wet", "erosion"):
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["ice"][...] = 0
        vf.FrameRecorder(tmp_path / "wet", "erosion", N).write(meta["key"], arrays, iteration=meta["key"], of=10, units="m")
    frames, _ = vw.collect_frames(tmp_path / "wet", None, log=lambda m: None)
    assert all("ice" not in fr.ch for fr in frames) and "lake_depth" in frames[0].ch
    assert "ice" not in vw.channel_specs(frames[-1], None, frames[:-1])


def test_resumed_erosion_drops_frames_past_the_resume_point(tmp_path):
    p = WorldParams.tiny_world(seed=6)
    bake(tmp_path / "w", p, to_stage="erosion", logger=lambda m: None)
    rec = vf.FrameRecorder(tmp_path / "w", "erosion", 8)
    rec.write(9999, {"height": np.zeros((6, 8, 8), np.float16)})
    kept = [k for k, _, _ in vf.list_frames(tmp_path / "w", "erosion")]
    rec.clear(after=p.erosion.iterations)
    after = [k for k, _, _ in vf.list_frames(tmp_path / "w", "erosion")]
    assert 9999 in kept and 9999 not in after
    assert after == [k for k in kept if k <= p.erosion.iterations]


# --------------------------------------------------------------------------
# the page has to parse (viewer.html)
# --------------------------------------------------------------------------
def _template_literals(js: str):
    """Every backtick-delimited run in ``js``, as (start, end) offsets, by a
    plain left-to-right scan that honours backslash escapes.  Crude, but it
    is exactly the scan a JS parser does for a template literal with no
    ``${}`` in it, which is what the shader sources are."""
    out, i, n = [], 0, len(js)
    while i < n:
        if js[i] == "\\":
            i += 2
            continue
        if js[i] == "`":
            j = i + 1
            while j < n and js[j] != "`":
                j += 2 if js[j] == "\\" else 1
            out.append((i, j))
            i = j + 1
            continue
        i += 1
    return out


def test_shader_sources_are_not_cut_short_by_a_stray_backtick():
    """The GLSL lives in JS template literals, so **a backtick anywhere in
    it ends the string** -- including one inside a GLSL comment, where it
    looks like ordinary prose markup and reads as perfectly innocent.

    That is not hypothetical: a comment written as ``// `w` is the water
    code`` truncated the fragment shader mid-file, so the whole script block
    failed to parse with "Unexpected identifier 'w'" and every viewer
    written for two commits was a blank page.  Nothing else checked that the
    emitted page is even syntactically valid JavaScript.
    """
    js = vw.TEMPLATE.read_text()
    for name, tail in (("VS", "gl_Position"), ("FS", "void main"), ("LVS", "gl_Position"), ("LFS", "void main")):
        k = js.index(f"const {name} = `")
        start = js.index("`", k)
        lit = _template_literals(js[start:])[0]
        body = js[start + lit[0] + 1 : start + lit[1]]
        assert body.startswith("#version 300 es"), name
        assert tail in body, f"{name} shader is cut short: {body[-120:]!r}"
        assert body.rstrip().endswith("}"), f"{name} shader does not end at a closing brace"
        # GLSL ES 3.00 reserves these names; one used as a variable (``float patch``)
        # fails the compile, and the page never gets past loading
        code = re.sub(r"//[^\n]*", "", body)
        for word in ("patch", "sample", "input", "output", "filter", "common", "partition", "active",
                     "noise", "cast", "namespace", "using", "sizeof", "union", "enum", "extern", "external"):
            assert not re.search(rf"\b{word}\b", code), f"{name} shader uses the reserved word {word!r}"



def test_the_river_lines_slope_and_curvature_are_the_splines_own():
    """Closer than a texel a pixel a river is drawn as the line on its
    discharge's ridge (viewer.html ``riverRidge``), found from the slope and
    the curvature of the same B-spline read the band is drawn from.  Those
    come through two one-line weight functions, and a slip in either bends
    every river on every world without failing anything -- so they are held
    to the spline here, with the numbers the shader's rule is written on: a
    channel one texel wide curves twice its height over its banks, one of
    two texels once, the inside of a wider one not at all, and one Newton
    step from beside the line lands on it."""
    js = vw.TEMPLATE.read_text()

    def glsl(name):
        m = re.search(rf"vec4 {name}\(float t\)\{{ return vec4\((.*?)\); \}}", js)
        assert m, f"{name} is not the one-liner this test reads"
        return lambda t: np.array(eval("(" + m.group(1) + ")", {"t": float(t)}))

    D, DD = glsl("bsplineD"), glsl("bsplineDD")
    B = lambda t: np.array([(1 - t) ** 3, 3 * t**3 - 6 * t**2 + 4, -3 * t**3 + 3 * t**2 + 3 * t + 1, t**3]) / 6.0  # noqa: E731
    for t in (0.0, 0.13, 0.5, 0.87, 1.0):
        e = 1e-4
        assert np.allclose(D(t), (B(t + e) - B(t - e)) / (2 * e), atol=1e-6)
        assert np.allclose(DD(t), (B(t + e) - 2 * B(t) + B(t - e)) / e**2, atol=1e-5)

    def across(profile, x):
        """Value, slope and curvature across a channel at x texels from texel 0's centre."""
        i0 = int(np.floor(x)); f = x - i0
        c = np.array([profile.get(i0 + k - 1, 0.0) for k in range(4)])
        return float(B(f) @ c), float(D(f) @ c), float(DD(f) @ c)

    one, two, wide = {0: 100.0}, {0: 100.0, 1: 100.0}, {k: 100.0 for k in range(-3, 4)}
    assert across(one, 0.0)[2] == pytest.approx(-200.0)            # twice its height
    assert across(two, 0.5)[2] == pytest.approx(-100.0)            # once
    assert across(wide, 0.0)[2] == pytest.approx(0.0, abs=1e-9)    # none: a flat top has no ridge
    for x in (-0.2, -0.05, 0.1, 0.2):                              # beside the line: one step is the distance
        v, g, h = across(one, x)
        assert x - g / h == pytest.approx(0.0, abs=0.05)
        assert v - 0.5 * g * g / h == pytest.approx(across(one, 0.0)[0], rel=0.02)   # ...and the discharge on it


def test_a_routed_river_is_a_smooth_ridge_as_high_as_it_is_strong():
    """``detail.river_strength``: a routed flow's channel is a path one cell
    wide, and the final frame draws it as the line on a ridge -- so the path
    is laid down as one, as high on its line as the channel is strong (the
    top of the river bytes at full strength, ``RIVER_BASE`` of it at the
    threshold), falling away on both sides and gone within ``RIVER_REACH``
    cells, and flow below the threshold is no river at all.  The ridge is
    narrow: two channels three cells apart are two ridges with ground between
    them, which a blur wide enough to round a path's steps made one.  And its
    steps are rounded along it: a path stepping aside and back is a line that
    barely does."""
    from globe.viz import detail as dt

    n = 64
    down = np.arange(n, dtype=np.float32)[:, None]            # a river carries more the farther down it is
    q = np.full((n, n), 0.5, np.float32)
    q[:, 16:17] = 1000.0 + down                               # trunk: full strength
    q[:, 40:41] = 10.5 + 0.001 * down                         # creek: just past the threshold
    q[:, 56:57] = 5.0                                         # below it
    q[:, 30:31] = 1000.0 + down                               # two more trunks, three cells apart
    q[:, 33:34] = 1000.0 + down
    sc = dt.strength_scale(1.0, 4000.0, 10.0, 1000.0)
    byte = lambda x: dt.log_byte(np.asarray(x), sc["lo"], sc["hi"]).astype(int)  # noqa: E731
    top = dt.RIVER_MIN_BYTE + dt.RIVER_SPAN_BYTE
    assert byte([sc["river_min"]])[0] == dt.RIVER_MIN_BYTE and byte([sc["river_full"]])[0] == top and sc["lines"]
    row = byte(dt.river_strength(q, sc)[n // 2])
    assert abs(row[16] - top) <= 2
    assert abs(row[40] - dt.RIVER_BASE * top) <= 4
    assert row[16] > row[15] > row[14] > 0 and row[16] > row[17] > row[18] > 0               # a ridge, smooth on both sides
    assert row[16 + dt.RIVER_REACH + 1] == 0 and row[16 - dt.RIVER_REACH - 1] == 0
    assert (row[52:61] == 0).all() and row[23] == 0
    assert row[30] >= top - 12 and row[33] >= top - 12 and max(row[31], row[32]) < 0.7 * top    # two ridges, not one
    # a path that steps a cell aside and back: its line moves a third of a cell, not a whole one
    q2 = np.full((n, n), 0.5, np.float32)
    col = np.full(n, 16)
    col[30] = 17
    q2[np.arange(n), col] = 1000.0 + down[:, 0]
    f2 = byte(dt.river_strength(q2, sc)).astype(np.float64) ** 4
    crest = lambda r: float((f2[r, 12:22] * np.arange(12, 22)).sum() / f2[r, 12:22].sum())  # noqa: E731
    assert abs(crest(20) - 16.0) < 0.05 and 0.1 < crest(30) - 16.0 < 0.5
    # on a finer frame the same path, at the frame's cells
    f3 = byte(dt.river_strength(q, sc, 2))
    assert f3.shape == (2 * n, 2 * n) and f3[n, 32] == f3[n, 33] >= 0.75 * top and f3[n, 30] < f3[n, 31] < f3[n, 32]


def test_the_final_frames_rivers_are_the_water_that_fills_the_lakes(tmp_path):
    """``viewer.hydro_rivers``: the final frame draws its rivers from hydro's
    routing -- the rain down the finished surface, through each lake and out
    at its spill -- not from the erosion's discharge, so a river meets a lake
    at its shore.  A routed path comes out as one smooth channel as wide as
    the frame is finer than the routing, as strong as its flow, from a share
    of hydro's river threshold up; the sea and the ground between are none;
    and a world hydro has not run on has no such rivers (the erosion's
    discharge is drawn as it was)."""
    from globe.viz import detail as dt

    N, n = 16, 32
    (tmp_path / "coarse").mkdir()
    fa = np.full((6, N, N), 1.0, np.float32)
    fd = np.zeros((6, N, N), np.uint8)
    down = np.arange(N, dtype=np.float32)
    fa[0, :, 8] = 4000.0 + down               # a trunk down face 0...
    fa[0, 4, :8] = 60.0 + down[:8]            # ...a tributary of it, over the threshold's share but under the threshold
    fa[0, 12, 9:] = 20.0                      # ...and a trickle under both
    fd[1] = 255                               # face 1 is sea
    fa[1, :, 5] = 4000.0 + down
    for f in range(6):
        np.save(tmp_path / "coarse" / f"flow_acc.f{f}.npy", fa[f])
        np.save(tmp_path / "coarse" / f"flow_dir.f{f}.npy", fd[f])
    manifest = {"stages": {"hydro": {"info": {"river_threshold_volume": 400.0}}}}
    assert vw.HYDRO_RIVER_SHARE * 400.0 < 60.0 < 400.0
    river, scale = vw.hydro_rivers(tmp_path, manifest, n)
    assert river.shape == (6, n, n) and scale["lines"] and scale["q_min"] == pytest.approx(vw.HYDRO_RIVER_SHARE * 400.0)
    b = dt.log_byte(river, scale["lo"], scale["hi"]).astype(int)
    top = dt.RIVER_MIN_BYTE + dt.RIVER_SPAN_BYTE
    row = b[0, 21]                            # across the trunk, away from the tributary: its line runs between fine cells 16 and 17
    assert row[16] == row[17] >= 0.75 * top and row[16] > row[15] > row[14] and row[17] > row[18] > row[19]
    assert row[16 - 2 * dt.RIVER_REACH - 1] == 0 and row[17 + 2 * dt.RIVER_REACH + 1] == 0
    trib = b[0, 8:10, 6]                      # the tributary: a river, weaker than the trunk
    assert dt.RIVER_BASE * top * 0.6 < trib.max() < row[16]
    assert b[0, 24:26, 26].max() == 0         # the trickle is none
    assert b[1].max() == 0                    # nor is anything in the sea
    assert vw.hydro_rivers(tmp_path, {"stages": {}}, n) == (None, None)


# --------------------------------------------------------------------------
# river lines (viewer.river_lines, globe/viz/river_lines.py)
# --------------------------------------------------------------------------
def _valley(res: int = 32):
    """Face 0: sea along i < 2, land rising along i; a valley down j = 16 whose
    discharge grows towards the sea but dips for a cell mid-way (the
    time-averaged stream map does that), a band a cell wide either side
    carrying a little less, and a tributary joining from j = 24."""
    surf = np.full((6, res, res), 500.0, np.float32)
    D = np.zeros((6, res, res), np.float32)
    water = np.zeros((6, res, res), np.uint8)
    i = np.arange(res, dtype=np.float32)[:, None]
    j = np.arange(res, dtype=np.float32)[None, :]
    surf[0] = 1.0 * i + 3.0 * np.abs(j - 16)                 # a V valley falling 1 m a cell towards the sea
    water[0, :2, :] = 2
    surf[0, :2, :] = -50.0
    D[0, 2:, 16] = 1000.0 - 20.0 * np.arange(2, res)
    D[0, 15, 16] = 500.0                                   # the dip
    D[0, 2:, 15] = D[0, 2:, 17] = 0.8 * D[0, 2:, 16]
    D[0, 10, 17:26] = 60.0 + 5.0 * np.arange(9)[::-1]      # tributary, growing towards the valley
    surf[0, 10, 17:26] -= 5.0                              # its channel, cut into the valley side
    return D, water, surf


def test_river_lines_run_down_a_valley_to_the_sea():
    from globe.viz import river_lines as rl

    D, water, surf = _valley()
    out = rl.trace(D, water, surf, q_min=50.0, min_length=3, smooth_passes=3, tolerance=0.25)
    xyz, n = out["xyz"].astype(np.float64), out["lengths"]
    assert n.sum() == xyz.shape[0] and np.allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-5)
    # the dip did not end the river: nothing ends but in the sea or at the face's edge
    assert out["mouths"] >= 1 and out["ends_elsewhere"] == 0, out
    # the band beside the channel is spurs, not rivers: every vertex lies
    # near the valley's axis or on the tributary
    f, u, v = from_sphere_v(xyz)
    ci, cj = u * 32 - 0.5, v * 32 - 0.5
    assert np.all(f == 0)
    assert np.all((np.abs(cj - 16) <= 1.01) | (np.abs(ci - 10) <= 1.01)), np.c_[ci, cj]
    # the main stem reaches the shore: its last vertex is the sea cell's centre
    assert (ci < 2).any()
    assert out["discharge"].max() == pytest.approx(D.max())


def test_river_lines_payload_round_trips():
    import base64

    lines = {"xyz": np.array([[32767, 0, 0], [0, 32767, 0], [0, 0, -32767]], np.int16), "width": np.array([30, 300, 3000], np.uint16),
             "byte": np.array([10, 200, 255], np.uint8), "lengths": np.array([2, 1], np.uint32), "channel": "flow", "source": "graph"}
    specs = {"flow": {"lo": 1.0, "hi": 1000.0, "river_min": 10.0}}
    js, info = vw.river_lines_script(lines, specs)
    payload = json.loads(js[js.index("(") + 1: js.rindex(")")])
    back = {k: np.frombuffer(base64.b64decode(payload[k]), dtype=lines[k].dtype) for k in ("xyz", "width", "byte", "lengths")}
    assert np.array_equal(back["xyz"].reshape(-1, 3), lines["xyz"]) and np.array_equal(back["width"], lines["width"])
    assert np.array_equal(back["byte"], lines["byte"]) and np.array_equal(back["lengths"], lines["lengths"])
    assert info["lines"] == 2 and info["vertices"] == 3 and info["min_byte"] == int(vw.log_byte(np.array([10.0]), 1.0, 1000.0)[0])


# --------------------------------------------------------------------------
# detail tiles (globe/viz/detail.py)
# --------------------------------------------------------------------------
def test_detail_tiles_round_trip_heights_shores_and_the_sea(tmp_path):
    """A tile decodes back to its cells: 16-bit height on a grid with sea level
    on a code (a cell's sign survives), the signed lake depth byte, and the
    ocean mask in alpha with land opaque; tiles of nothing but sea are not
    written, and the bitset says which are."""
    import base64
    import io

    from PIL import Image

    from globe.viz import detail as dt

    res = 2 * dt.TILE
    rng = np.random.default_rng(3)
    surf = rng.normal(200.0, 400.0, (6, res, res)).astype(np.float32)
    surf[:, :, :8] = 0.04                                  # a plain a few cm above the sea
    surf[:, :, 8:12] = -0.04                               # and a few cm below it
    water = np.zeros((6, res, res), np.uint8)
    water[:, dt.TILE - 2:, :] = dt.WATER_OCEAN             # every face's second tile row is sea, its pad row too
    surf[water == dt.WATER_OCEAN] = -3000.0
    water[:, 40:50, 100:120] = dt.WATER_LAKE
    ws = np.where(water == dt.WATER_LAKE, surf + 15.0, surf).astype(np.float32)
    q = np.ones_like(surf)                                 # rain everywhere, two rivers
    q[:, 10:14, :] = 500.0
    q[:, 60:62, :] = 50.0
    src = dt.RefinedSource(surf, ws, water, q)
    assert dt.levels_for(dt.TILE, res) == [(1, res)] and dt.levels_for(res, res) == []
    info = dt.export_tiles(tmp_path, src, dt.TILE, 200.0, log=lambda m: None)
    lv = info["levels"][0]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(lv["exists"]), np.uint8), bitorder="little")[: 6 * 4].reshape(6, 2, 2)
    assert bits[:, 0, :].all() and not bits[:, 1, :].any() and lv["tiles"] == 12
    js = (tmp_path / "tiles" / "L1" / "2_0_0.js").read_text()
    img = np.array(Image.open(io.BytesIO(base64.b64decode(js.split('"')[3]))).convert("RGBA"))
    assert img.shape == (2 * (dt.TILE + 2), dt.TILE + 2, 4)        # ground on top, water below
    both = img.transpose(1, 0, 2)                                   # (i, j) of the face's tile cells
    ground, wat = both[:, :dt.TILE + 2], both[:, dt.TILE + 2:]
    cell = ground[1:-1, 1:-1]
    q = cell[..., 0].astype(np.int64) * 256 + cell[..., 1]
    h = info["h0"] + q / 65535.0 * (info["h1"] - info["h0"])
    # a lake's height is its water, not its bed (the bed is the lake depth below it)
    s = np.where(water[2, :dt.TILE, :dt.TILE] == dt.WATER_LAKE, ws[2, :dt.TILE, :dt.TILE], surf[2, :dt.TILE, :dt.TILE])
    assert np.abs(h - s).max() <= (info["h1"] - info["h0"]) / 65535.0
    assert np.all((h >= 0.0) == (s >= 0.0))
    land = water[2, :dt.TILE, :dt.TILE] == dt.WATER_LAND
    assert (cell[..., 3][land & (np.arange(dt.TILE)[:, None] < dt.TILE - 8)] == 255).all()        # land away from the coast is opaque
    assert (cell[..., 3][dt.TILE - 1, :] < 128).all()                                                # the sea is not
    assert info["lake_curve"] == "sqrt"                             # the lake byte is finest at the shore (detail.lake_byte)
    t = cell[..., 2] / 127.5 - 1.0
    ld = t * np.abs(t) * 200.0
    assert (ld[41:49, 101:119] > 14.0).all() and (ld[:30, :60] < -100.0).all()
    # the water half: the log discharge byte of the same cells
    qb = wat[1:-1, 1:-1, 0]
    assert qb[10:14, :].min() > qb[60:62, :].max() and qb[60:62, :].min() > 0
    assert (qb[20:30, :] == 0).all()                                # only rain: under the line, no river
    lo, hi = info["q_lo"], info["q_hi"]
    back = lo * (hi / lo) ** (qb[10:14, :].astype(float) / 255.0)
    assert np.allclose(back, 500.0, rtol=0.02)
    # the pad is the neighbouring cells, clamped at the face edge
    assert np.array_equal(ground[0, 1:-1], ground[1, 1:-1])
