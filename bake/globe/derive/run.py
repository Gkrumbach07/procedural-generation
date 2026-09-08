"""Derive stage driver (PLAN.md section 11): biomes, vegetation, rivers,
lakes, from the fine refine output plus the coarse climate / hydro fields.

Inputs
    coarse: ``temperature``, ``bedrock``, ``precip``, ``height``, ``sediment``,
    ``water_surface``, ``flow_acc`` (optional), ``graph/drainage.json``
    (optional; empty graph tolerated), ``graph/lakes_coarse.json``
    (optional).
    fine:   ``height`` (required), ``sediment``, ``water_surface``,
    ``discharge`` (optional, missing = zeros).
Outputs
    coarse ``biome`` (u8, 0 = ocean; code table in ``derive/biomes.py``);
    ``fine/biome``, ``fine/vegetation``, ``fine/river_mask`` (u8, one
    memmapped face at a time); ``graph/rivers.json`` (points ``[f, u, v,
    width_m, height_m]``); ``graph/lakes.json``.

``temperature`` is the climate field, which was computed on the
pre-erosion *bedrock* surface: derive removes its lapse term and re-applies
it at the final surface (coarse) and at every fine cell, so peaks are
classified at the temperature of the terrain that is actually rendered.

Rivers are derived from the *fine* discharge (``derive/rivers.py``); the
coarse drainage graph only supplies Strahler orders / reach ids where a
coarse channel lies under a fine polyline.  Fine biomes are recomputed
from bilinearly upsampled temperature / precipitation with the overrides
(cliff, alpine, riparian, wetland, lake) evaluated on the fine data.
Faces are processed one at a time in three passes (discharge threshold +
blob connectivity across cube edges; rivers + lakes; biomes + vegetation
reading the neighbouring faces' cells through thin pads), so no output
stops at a cube edge except the river-mask discs.  No random numbers are
drawn.
"""
from __future__ import annotations

import time

import numpy as np

from ..climate.temperature import retarget
from ..field import FaceField
from . import biomes, soil
from . import lakes as lakes_mod
from . import rivers as rivers_mod
from .fine import coarse_face_array, face_pads, has_fine, load_face, slope_magnitude, upsample_face, upsample_nearest, write_face

FINE_OUTPUTS = ("biome", "vegetation", "river_mask")
OUTPUTS = ["biome", "graph/rivers.json", "graph/lakes.json"] + [f"fine/{n}.f{k}.npy" for n in FINE_OUTPUTS for k in range(6)]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def channel_threshold(store, params, precip_interior: np.ndarray, land: np.ndarray, drainage: dict | None) -> float:
    """Coarse channel threshold on ``flow_acc``: the hydro stage's
    ``river_threshold_volume`` when recorded in ``drainage.json``, else
    ``hydro.river_threshold`` x the mean precip volume over land."""
    if drainage and drainage.get("river_threshold_volume") is not None:
        return float(drainage["river_threshold_volume"])
    p = np.asarray(precip_interior)[np.asarray(land, dtype=bool)]
    mean_p = float(p.mean()) if p.size else 1.0
    return float(params.hydro.river_threshold) * mean_p


def _fine_surface(store, face: int, Nf: int) -> np.ndarray:
    h = np.asarray(load_face(store, "height", face), dtype=np.float32)
    if has_fine(store, "sediment"):
        h = h + np.asarray(load_face(store, "sediment", face), dtype=np.float32)
    if h.shape != (Nf, Nf):
        raise ValueError(f"fine/height.f{face}.npy has shape {h.shape}, expected {(Nf, Nf)}")
    return h


def _fine_optional(store, name: str, face: int, Nf: int, fill: float) -> np.ndarray:
    if has_fine(store, name):
        return np.asarray(load_face(store, name, face), dtype=np.float32)
    return np.full((Nf, Nf), fill, dtype=np.float32)


class _FineReader:
    """Scattered reads of a fine field of any face through the memmaps —
    used for the pads beyond the face edges (a few rows of the
    neighbouring faces, never a whole face).  ``fill`` is returned when
    the field does not exist."""

    def __init__(self, store, name: str, fill: float = 0.0):
        self.store = store
        self.name = name
        self.fill = fill
        self.exists = has_fine(store, name)
        self._mm = {}

    def __call__(self, f2, i2, j2):
        out = np.full(np.shape(i2), self.fill, dtype=np.float32)
        if not self.exists:
            return out
        for f in np.unique(np.asarray(f2)):
            f = int(f)
            if f not in self._mm:
                self._mm[f] = load_face(self.store, self.name, f)
            sel = np.asarray(f2) == f
            out[sel] = np.asarray(self._mm[f][i2[sel], j2[sel]], dtype=np.float32)
        return out


class _SurfaceReader:
    """``_FineReader`` of the fine surface (height + sediment)."""

    def __init__(self, store):
        self.h = _FineReader(store, "height")
        self.s = _FineReader(store, "sediment")

    def __call__(self, f2, i2, j2):
        return self.h(f2, i2, j2) + self.s(f2, i2, j2)


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------
def run(store, params, log=print) -> dict:
    t0 = time.time()
    grid = params.coarse_grid()
    N, H, R = grid.N, grid.H, params.world.R
    Nf = params.N_fine
    dp = params.derive
    cs_f = params.fine_cell_size_m
    depth = float(params.hydro.lake_min_depth)
    info: dict = {}

    # ---- coarse inputs -------------------------------------------------
    T = store.load_field("temperature", grid)
    P = store.load_field("precip", grid)
    h = store.load_field("height", grid)
    sed = store.load_field("sediment", grid)
    ws = store.load_field("water_surface", grid)
    acc = store.load_field("flow_acc", grid) if store.has_field("flow_acc") else None
    drainage = store.read_json("graph/drainage.json") if store.has("graph/drainage.json") else None
    coarse_lakes = store.read_json("graph/lakes_coarse.json").get("lakes") if store.has("graph/lakes_coarse.json") else None

    surface_c = (h.interior + sed.interior).astype(np.float32)
    land_c = surface_c >= 0.0
    # the stored climate temperature was computed before erosion, on the
    # *bedrock* surface (climate/run.py); erosion then moved that surface by
    # hundreds of metres.  Undo the old lapse term and re-apply it at the
    # surface the biomes are classified on (sea-level temperature in between).
    T0_c = retarget(T.interior, store.load_field("bedrock", grid).interior, 0.0, params.climate)  # sea-level field
    T_c = retarget(T0_c, 0.0, surface_c, params.climate)
    area_factor = (grid.interior_cell_area / grid.cell_size_m**2).astype(np.float32)
    wet_c = biomes.wetness(P.interior, area_factor, params.climate.precip_mean, land=land_c)
    Pcm_c = biomes.precip_cm(wet_c, dp.precip_scale_cm, dp.precip_gamma, getattr(dp, "precip_max_cm", 0.0))
    thr = channel_threshold(store, params, P.interior, land_c, drainage)
    chan_c = ((acc.interior > thr) if acc is not None else np.zeros_like(land_c)) & land_c
    chan_frac = float(chan_c.sum() / max(int(land_c.sum()), 1))
    lake_c = lakes_mod.lake_mask(surface_c, ws.interior, depth)
    coarse_lab, n_coarse_lakes = lakes_mod.coarse_lake_labels(lake_c, ws.interior, grid)
    surf_field = FaceField.from_interior(grid, surface_c, name="surface")
    slope_c = surf_field.gradient().vec_norm().interior
    cliff_slope = biomes.effective_cliff_slope(slope_c[land_c], dp)
    alpine_min = biomes.effective_alpine_min(surface_c[land_c], dp)
    river_near_c = biomes.near_faces(chan_c, dp.riparian_cells, grid)
    lake_near_c = biomes.near_faces(lake_c, dp.wetland_cells, grid)
    biome_c = biomes.classify(T_c, Pcm_c, surface_c, slope_c, lake_c, river_near_c, lake_near_c, dp, cliff_slope, alpine_min)
    store.save_field(FaceField.from_interior(grid, biome_c, name="biome", exchange=False))
    hist = np.bincount(biome_c.ravel(), minlength=biomes.N_BIOMES)
    info["coarse_biome_hist"] = {biomes.NAMES[k]: int(hist[k]) for k in range(biomes.N_BIOMES) if hist[k]}
    info["coarse_channel_fraction"] = chan_frac
    info["coarse_lakes"] = int(n_coarse_lakes)
    info["cliff_slope"] = cliff_slope
    # what the *configured* threshold would have selected: effective_cliff_slope
    # raises it to keep cliffs under cliff_max_fraction, which would otherwise
    # hide a bedrock that stands at the talus angle everywhere
    info["cliff_fraction_at_configured_slope"] = float((slope_c[land_c] > dp.cliff_slope).mean()) if land_c.any() else None
    info["alpine_min_m"] = alpine_min
    info["T_biome_land_mean"] = float(T_c[land_c].mean()) if land_c.any() else None
    info["land_slope_median"] = float(np.median(slope_c[land_c])) if land_c.any() else None
    info["t_coarse_s"] = time.time() - t0
    info["P_cm_land_quantiles"] = [float(v) for v in np.percentile(Pcm_c[land_c], [5, 25, 50, 75, 95])] if land_c.any() else []
    log(
        f"[derive] coarse biomes in {info['t_coarse_s']:.1f}s: channel fraction {chan_frac:.4f}, {n_coarse_lakes} coarse lakes, "
        f"P_cm land quantiles [5,25,50,75,95] {np.round(info['P_cm_land_quantiles']).tolist()}, cliff slope {cliff_slope:.2f} "
        f"(configured {dp.cliff_slope:.2f} would take {info['cliff_fraction_at_configured_slope'] if info['cliff_fraction_at_configured_slope'] is None else round(info['cliff_fraction_at_configured_slope'], 3)} of the land; "
        f"median land slope {info['land_slope_median'] if info['land_slope_median'] is None else round(info['land_slope_median'], 3)}), alpine above {alpine_min:.0f} m"
    )

    # ---- fine pass 1: global discharge threshold + blob connectivity -----
    if not has_fine(store, "height"):
        raise FileNotFoundError(f"derive needs fine/height.f*.npy (refine output) in {store.root}")
    t = time.time()
    counts = np.zeros(rivers_mod.LOG_HIST_BINS, dtype=np.int64)
    n_land = 0
    for f in range(6):
        q_s = rivers_mod.smooth_discharge(_fine_optional(store, "discharge", f, Nf, 0.0), dp.discharge_smooth_cells)
        land_f = _fine_surface(store, f, Nf) >= 0.0
        c, n = rivers_mod.log_histogram(q_s, land_f)
        counts += c
        n_land += n
    frac = rivers_mod.river_fraction(dp, chan_frac)
    q_thr = rivers_mod.threshold_from_histogram(counts, n_land, frac)
    q_low = rivers_mod.threshold_from_histogram(counts, n_land, frac * max(float(dp.river_hysteresis), 1.0))
    q_low = min(q_low, q_thr)
    graph_index = rivers_mod.CoarseGraphIndex(drainage, N, dp.graph_match_cells)
    # the keep decision (blob has a high cell, blob size) is taken on the
    # blobs joined across cube edges, so rivers are not cut at face edges
    conn = rivers_mod.RiverConnectivity(Nf, int(round(dp.min_river_cells * R * R)))
    for f in range(6):
        q_s = rivers_mod.smooth_discharge(_fine_optional(store, "discharge", f, Nf, 0.0), dp.discharge_smooth_cells)
        land_f = _fine_surface(store, f, Nf) >= 0.0
        conn.add_face(f, (q_s > q_low) & land_f, (q_s > q_thr) & land_f)
    conn.finalize()
    info["river_fraction"] = frac
    info["discharge_threshold"] = q_thr
    info["discharge_threshold_low"] = q_low
    info["fine_land_cells"] = int(n_land)
    info["river_blobs"] = int(conn.n_labels)
    info["river_blobs_kept"] = int(conn.keep.sum()) if conn.keep is not None else 0
    info["t_threshold_s"] = time.time() - t
    log(f"[derive] discharge threshold {q_thr:.3f} (connectivity {q_low:.3f}) for {frac:.4f} of {n_land:,} fine land cells; {info['river_blobs_kept']}/{info['river_blobs']} blobs kept ({info['t_threshold_s']:.1f}s); coarse graph edges {graph_index.n_edges}")

    # ---- fine pass 2: per face ---------------------------------------------
    Pcm_field = FaceField.from_interior(grid, Pcm_c, name="P_cm")
    T0_field = FaceField.from_interior(grid, T0_c, name="T0")  # sea-level temperature: the lapse is re-applied per fine cell
    stencil = int(dp.slope_stencil) if dp.slope_stencil > 0 else R
    lake_min_cells = max(1, int(round(dp.lake_min_cells * R * R)))
    all_rivers: list[dict] = []
    pieces: list[dict] = []
    frames: list[np.ndarray] = []
    face_info = []
    chan_covered = 0
    for f in range(6):
        t = time.time()
        surface_f = _fine_surface(store, f, Nf)
        ws_f = _fine_optional(store, "water_surface", f, Nf, 0.0)
        q_s = rivers_mod.smooth_discharge(_fine_optional(store, "discharge", f, Nf, 0.0), dp.discharge_smooth_cells)
        land_f = surface_f >= 0.0
        # rivers
        keep_f = conn.mask(f, (q_s > q_low) & land_f)
        river_mask, rivers, rinfo = rivers_mod.extract_face_rivers(f, q_s, surface_f, land_f, q_thr, dp, R, cs_f, graph_index, q_low=q_low, mask=keep_f)
        write_face(store, "river_mask", f, river_mask)
        for r in rivers:
            r["id"] = len(all_rivers)
            all_rivers.append(r)
        del q_s, keep_f
        # how much of the coarse D8 channel network the fine rivers cover (diagnostic)
        if chan_c[f].any():
            on = biomes.near((river_mask > 0).reshape(N, R, N, R).any(axis=(1, 3)), 1)
            rinfo["coarse_channel_coverage"] = float(on[chan_c[f]].mean())
            chan_covered += int(on[chan_c[f]].sum())
        # lakes
        lake_f = lakes_mod.lake_mask(surface_f, ws_f, depth)
        area_f = (upsample_nearest(coarse_face_array(grid, grid.cell_area, f), R) / float(R * R)).astype(np.float32)
        pcs, frame = lakes_mod.face_lake_pieces(f, lake_f, ws_f, surface_f, area_f, coarse_lab[f], R, lake_min_cells, piece_base=len(pieces))
        pieces += pcs
        frames.append(frame)
        del area_f, ws_f, lake_f, river_mask, surface_f
        dt = time.time() - t
        fi = {"face": f, "seconds_rivers_lakes": dt, "lake_pieces": len(pcs), **rinfo}
        face_info.append(fi)
        log(f"[derive] face {f}: mask {rinfo['mask_cells']:,} cells -> skeleton {rinfo['skeleton_cells']:,} -> {rinfo['rivers']} rivers ({rinfo['graph_matched']} matched to coarse reaches), {len(pcs)} lake pieces, {dt:.1f}s")

    # ---- fine pass 3: biomes + vegetation per face -------------------------
    # (after every face's river mask exists: the riparian / wetland bands,
    # the slope stencil and the lake mask read the neighbouring faces'
    # cells through pads, so nothing stops at a cube edge)
    surface_reader = _SurfaceReader(store)
    ws_reader = _FineReader(store, "water_surface", 0.0)
    mask_reader = _FineReader(store, "river_mask", 0.0)
    d_rip = int(dp.riparian_cells * R)
    d_wet = int(dp.wetland_cells * R)
    for f in range(6):
        t = time.time()
        surface_f = _fine_surface(store, f, Nf)
        ws_f = _fine_optional(store, "water_surface", f, Nf, 0.0)
        lake_f = lakes_mod.lake_mask(surface_f, ws_f, depth)
        river_mask = np.asarray(load_face(store, "river_mask", f))
        T_f = retarget(upsample_face(T0_field, f, R, order=1), 0.0, surface_f, params.climate)
        Pcm_f = np.maximum(upsample_face(Pcm_field, f, R, order=1), 0.0)
        pads = face_pads(surface_reader, Nf, f, stencil)
        slope_f = slope_magnitude(surface_f, coarse_face_array(grid, grid.metric_inv, f), R, cs_f, stencil, pads=pads)
        river_near_f = biomes.near_padded(river_mask > 0, [p > 0 for p in face_pads(mask_reader, Nf, f, d_rip)] if d_rip > 0 else None, d_rip)
        if d_wet > 0:
            s_pads = face_pads(surface_reader, Nf, f, d_wet)
            w_pads = face_pads(ws_reader, Nf, f, d_wet)
            lake_pads = [lakes_mod.lake_mask(a, b, depth) for a, b in zip(s_pads, w_pads)]
            del s_pads, w_pads
        else:
            lake_pads = None
        lake_near_f = biomes.near_padded(lake_f, lake_pads, d_wet)
        biome_f = biomes.classify(T_f, Pcm_f, surface_f, slope_f, lake_f, river_near_f, lake_near_f, dp, cliff_slope, alpine_min)
        land_mask_f = surface_f >= 0.0
        face_info[f]["T_land_mean"] = float(T_f[land_mask_f].mean()) if land_mask_f.any() else None
        del T_f, land_mask_f, river_near_f, lake_near_f, river_mask, pads, lake_pads, ws_f, lake_f
        sf = soil.soil_factor(_fine_optional(store, "sediment", f, Nf, 0.0), dp.soil_full_depth_m)
        veg_f = biomes.vegetation(biome_f, Pcm_f, slope_f, sf, dp, cliff_slope)
        write_face(store, "biome", f, biome_f)
        write_face(store, "vegetation", f, veg_f)
        face_info[f]["seconds_biomes"] = time.time() - t
        face_info[f]["seconds"] = face_info[f]["seconds_rivers_lakes"] + face_info[f]["seconds_biomes"]
        del surface_f, slope_f, biome_f, veg_f, sf, Pcm_f
    log(f"[derive] fine biomes + vegetation in {sum(fi['seconds_biomes'] for fi in face_info):.1f}s")

    # ---- graphs ----------------------------------------------------------------
    groups = lakes_mod.link_pieces(pieces, frames, Nf)
    lakes = lakes_mod.assemble_lakes(pieces, Nf, coarse_lakes, coarse_lab, groups=groups)
    info["coarse_channel_coverage"] = float(chan_covered / max(int(chan_c.sum()), 1)) if chan_c.any() else None
    store.write_json("graph/lakes.json", {"lakes": lakes, "lake_min_depth": depth, "N_fine": Nf, "coordinates": "corner lattice u = i / N_fine"})
    store.write_json(
        "graph/rivers.json",
        {
            "rivers": all_rivers,
            "N_fine": Nf,
            "discharge_threshold": q_thr,
            "discharge_threshold_low": q_low,
            "river_fraction": frac,
            "width": {"a": dp.river_width_a, "b": dp.river_width_b, "unit": "fine cells of (Q / threshold)^b, width_m = cells * fine_cell_size_m"},
            "coordinates": "cell centre u = (i + 0.5) / N_fine",
            "point": ["face", "u", "v", "width_m", "height_m (water surface, non-increasing downstream)"],
        },
    )
    info["faces"] = face_info
    info["n_rivers"] = len(all_rivers)
    info["n_river_points"] = int(sum(len(r["points"]) for r in all_rivers))
    info["n_lakes"] = len(lakes)
    info["max_order"] = int(max((r["order"] for r in all_rivers), default=0))
    info["t_total_s"] = time.time() - t0
    cov = info["coarse_channel_coverage"]
    log(f"[derive] {len(all_rivers)} rivers ({info['n_river_points']:,} points, max order {info['max_order']}; coarse channel coverage {cov if cov is None else round(cov, 3)}), {len(lakes)} lakes ({sum(1 for L in lakes if L['outlet'] is not None)} with outlet) in {info['t_total_s']:.1f}s")
    return info


# --------------------------------------------------------------------------
# quicklook
# --------------------------------------------------------------------------
def draw_polyline(img: np.ndarray, pts: np.ndarray, color, radius: int = 0) -> None:
    """Rasterise a polyline (pixel coordinates ``(m, 2)`` float, ``[i, j]``)
    into ``img`` ``(n, n, 3)`` with a square brush of half-width ``radius``."""
    p = np.asarray(pts, dtype=np.float64)
    n = img.shape[0]
    if p.shape[0] == 0:
        return
    if p.shape[0] == 1:
        samples = p
    else:
        d = np.diff(p, axis=0)
        cnt = np.maximum(np.ceil(np.abs(d).max(axis=1)).astype(np.int64), 1)
        idx = np.repeat(np.arange(d.shape[0]), cnt)
        off = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        tt = off / cnt[idx]
        samples = np.vstack([p[idx] + d[idx] * tt[:, None], p[-1:]])
    ii = np.rint(samples[:, 0]).astype(np.int64)
    jj = np.rint(samples[:, 1]).astype(np.int64)
    r = int(radius)
    if r > 0:
        o = np.arange(-r, r + 1)
        oi, oj = np.meshgrid(o, o, indexing="ij")
        ii = (ii[:, None] + oi.ravel()[None, :]).ravel()
        jj = (jj[:, None] + oj.ravel()[None, :]).ravel()
    ok = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
    img[ii[ok], jj[ok]] = color


def _shade(surface6: np.ndarray, cell: float) -> np.ndarray:
    from ..viz import quicklook as ql

    rng = float(np.nanpercentile(surface6, 99) - np.nanpercentile(surface6, 1)) or 1.0
    z = 0.5 * surface6.shape[-1] * cell / rng
    return ql.hillshade(surface6, cell, z_factor=z)


def render_derive(store, params, stride: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(biome + rivers image, vegetation image), each ``(6, n, n, 3)`` uint8
    ``[f, i, j]`` sampled every ``stride`` fine cells (default: faces of at
    most 1024 pixels)."""
    Nf = params.N_fine
    s = stride or max(1, Nf // 1024)
    n = Nf // s
    cs = params.fine_cell_size_m
    biome6 = np.zeros((6, n, n), dtype=np.uint8)
    veg6 = np.zeros((6, n, n), dtype=np.uint8)
    surf6 = np.zeros((6, n, n), dtype=np.float32)
    for f in range(6):
        biome6[f] = np.asarray(load_face(store, "biome", f))[: n * s : s, : n * s : s]
        veg6[f] = np.asarray(load_face(store, "vegetation", f))[: n * s : s, : n * s : s]
        hh = np.asarray(load_face(store, "height", f))[: n * s : s, : n * s : s].astype(np.float32)
        if has_fine(store, "sediment"):
            hh = hh + np.asarray(load_face(store, "sediment", f))[: n * s : s, : n * s : s]
        surf6[f] = hh
    hs = _shade(surf6, cs * s)
    shade = (0.55 + 0.45 * hs)[..., None]
    img = np.clip(biomes.colorize(biome6).astype(np.float32) * shade * 1.15, 0, 255).astype(np.uint8)
    ocean = surf6 < 0
    img[ocean] = biomes.PALETTE[biomes.OCEAN]
    # rivers
    if store.has("graph/rivers.json"):
        rj = store.read_json("graph/rivers.json")
        rivers = sorted(rj.get("rivers", []), key=lambda r: r.get("order", 1))
        omax = max((r.get("order", 1) for r in rivers), default=1)
        for r in rivers:
            pts = np.asarray(r["points"], dtype=np.float64)
            if pts.size == 0:
                continue
            f = int(pts[0, 0])
            pix = np.stack([pts[:, 1] * Nf / s - 0.5, pts[:, 2] * Nf / s - 0.5], axis=1)
            wpx = float(np.median(pts[:, 3])) / cs / s
            rad = int(max(0, round((wpx - 1) / 2)))
            t = (r.get("order", 1) - 1) / max(omax - 1, 1)
            col = (int(70 - 50 * t), int(140 - 80 * t), 255)
            draw_polyline(img[f], pix, col, rad)
    # vegetation: dry sand -> deep green, hillshaded
    t = veg6.astype(np.float32)[..., None] / 255.0
    dry = np.array([205, 190, 150], dtype=np.float32)
    green = np.array([15, 85, 25], dtype=np.float32)
    vimg = np.clip((dry * (1 - t) + green * t) * shade * 1.1, 0, 255).astype(np.uint8)
    vimg[ocean] = biomes.PALETTE[biomes.OCEAN]
    lake = biome6 == biomes.LAKE
    vimg[lake] = biomes.PALETTE[biomes.LAKE]
    return img, vimg


def quicklook(store, params, path):
    """``derive.png``: fine biome map (hillshaded) with the rivers of
    ``graph/rivers.json`` drawn at their width; ``derive_vegetation.png``:
    vegetation density."""
    from ..viz import quicklook as ql

    img, vimg = render_derive(store, params)
    ql.save_image(store.quicklook_path("derive", "vegetation"), vimg)
    return ql.save_image(path, img)


__all__ = ["OUTPUTS", "FINE_OUTPUTS", "run", "quicklook", "render_derive", "draw_polyline", "channel_threshold"]
