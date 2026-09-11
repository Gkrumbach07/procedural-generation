"""The HTML viewer's data: frame capture, cube-face atlases, the export."""
import json

import numpy as np

from globe.config import WorldParams
from globe.cubesphere import to_sphere_v
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
