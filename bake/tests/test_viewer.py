"""The HTML viewer's data: frame capture, cube-face atlases, the export."""
import json

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
    """A lake cell carries its depth, a dry neighbour the lake level less its
    ground (< 0), everything else the far value -- so the bilinear 0
    crossing is where the ground meets the water."""
    surf = np.zeros((6, 5, 5), np.float32) + 50.0
    surf[0, 2, 2] = 10.0                              # a one-cell lake ...
    surf[0, 2, 3] = 40.0                              # ... whose neighbour stands 10 m above its water
    ws = surf.copy()                                  # dry ground: water surface = ground
    ws[0, 2, 2] = 30.0                                # the lake's level
    code = vw.water_code(surf, None, ws, 0.5)
    d = vw.lake_depth(surf, ws, code)
    assert d[0, 2, 2] == 20.0 and d[0, 2, 3] == -10.0 and d[0, 2, 1] == -20.0
    assert d[0, 0, 0] == -vw.LAKE_DEPTH_RANGE_M and d[3, 2, 2] == -vw.LAKE_DEPTH_RANGE_M
    # linear between the lake centre and the neighbour: the shore 2/3 of the way
    t = 20.0 / (20.0 + 10.0)
    assert abs(t - 2.0 / 3.0) < 1e-9


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
