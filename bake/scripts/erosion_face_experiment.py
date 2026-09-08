"""Single-face erosion experiments (PLAN milestone 2 / section 16 item 2).

    cd bake && python3 scripts/erosion_face_experiment.py --case a --iters 300 \
        --tag baseline [--set erosion.k_mom=2 ...] [--out DIR]

The sweep that chose the shipped ErosionParams defaults, and every number it
produced, is in docs/erosion-tuning.md.

Cases (N x N window, cell 50 m, mask 1 inside, 1-cell frozen rim, mask-0 halo):
  a  tilted plane + small noise, uniform uplift (McDonald's classic test); ocean strip at i < 6
  b  central massif: gaussian dome above a shallow sea, uplift ∝ dome
  c  low-gradient floodplain (slope 0.002) fed by a fixed inflow block on the high edge
Every `--every` iterations: quicklook (hillshade + log-discharge overlay | log discharge)
and a metrics line; metrics.json + final state npz in <out>/runs/<tag>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from globe.config import WorldParams  # noqa: E402
from globe.erosion import particle as pk  # noqa: E402
from globe.erosion.maps import ErosionState, step  # noqa: E402
from globe.erosion.route import priority_flood_eps  # noqa: E402
from globe.viz import quicklook as ql  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "erosion_runs"
D8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


# --------------------------------------------------------------------------
# terrain synthesis
# --------------------------------------------------------------------------
def noise2d(NE: int, rng: np.random.Generator, octaves: int = 6, base: float = 3.0) -> np.ndarray:
    out = np.zeros((NE, NE))
    amp, tot = 1.0, 0.0
    x = np.linspace(0, 1, NE, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for o in range(octaves):
        f = base * 2**o
        n = int(np.ceil(f)) + 2
        lat = rng.random((n + 1, n + 1))
        qx, qy = X * f, Y * f
        i0 = np.floor(qx).astype(int)
        j0 = np.floor(qy).astype(int)
        tx, ty = qx - i0, qy - j0
        tx = tx * tx * (3 - 2 * tx)
        ty = ty * ty * (3 - 2 * ty)
        v = lat[i0, j0] * (1 - tx) * (1 - ty) + lat[i0 + 1, j0] * tx * (1 - ty) + lat[i0, j0 + 1] * (1 - tx) * ty + lat[i0 + 1, j0 + 1] * tx * ty
        out += amp * (v * 2 - 1)
        tot += amp
        amp *= 0.5
    return out / tot


def make_state(case: str, N: int, params: WorldParams, iters: int, seed: int = 0, deposit_on_exit: bool = False, tilt: float = 8.0, noise: float = 3.0, uplift: float = 5.0, plain_slope: float = 0.002) -> ErosionState:
    """Heights below are in *cell units* (cell = params.cell_size_m)."""
    H = params.world.halo
    NE = N + 2 * H
    rng = np.random.default_rng(seed)
    cs = params.cell_size_m
    x = (np.arange(NE) - H + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    ii = np.arange(NE) - H
    I, J = np.meshgrid(ii, ii, indexing="ij")
    mask = np.zeros((NE, NE), np.uint8)
    mask[H : H + N, H : H + N] = 1
    rim = (mask == 1) & ((I == 0) | (I == N - 1) | (J == 0) | (J == N - 1))
    precip = np.ones((NE, NE), np.float32)
    upl = np.zeros((NE, NE))
    nz = noise2d(NE, rng)
    if case == "a":
        # plane rising 0 -> 25 cells (1250 m) along +i, noise +-3 cells, ocean strip i < 6
        h = tilt * (X - 6.0 / N) / (1 - 6.0 / N) + noise * nz + 0.5
        h = np.where(I < 6, -20.0, h)
        upl[...] = uplift / iters  # uniform on land; the ocean strip is the fixed base level
        upl[I < 6] = 0.0
    elif case == "b":
        r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2
        dome = np.exp(-r2 / (2 * 0.2**2))
        h = -3.0 + 33.0 * dome + 2.0 * nz
        upl = 10.0 / iters * dome
        upl[h < 0] = 0.0
    elif case == "c":
        # floodplain: slope 0.002 per cell towards i = 0 (ocean at i < 4), noise +-0.05 cells
        h = 0.002 * (I - 16) + 0.05 * nz + 0.3
        h = np.where(I < 16, -20.0, h)  # deep, wide sea: a 3-column shelf fills up within 300 iterations
        precip[...] = 0.02
        inlet = (I >= N - 5) & (I < N - 1) & (np.abs(J - N // 2) <= 1)
        precip[inlet] = 150.0
        # a shallow starter channel so the inflow has a bed to follow
    elif case == "d":
        # floodplain fed by a noisy hinterland: plain (slope 0.002) for i in [4, 0.55N),
        # hills (slope 0.06 + noise) above; ocean at i < 4; uplift on the hills only
        i_hill = int(0.55 * N)
        plain = plain_slope * (I - 16) + 0.3
        hills = plain[i_hill, 0] + 0.06 * (I - i_hill) + 1.5 * noise2d(NE, rng, base=6.0)
        h = np.where(I < i_hill, plain + 0.03 * nz, hills)
        h = np.where(I < 16, -20.0, h)  # deep, wide sea: a 3-column shelf fills up within 300 iterations
        precip[...] = np.where(I < i_hill, 0.25, 1.0)
        upl[...] = np.where(I >= i_hill, uplift / iters, 0.0)
    else:
        raise ValueError(case)
    mask[rim] = 2
    precip[rim] = 0
    metric = np.zeros((NE, NE, 3), np.float32)
    metric[..., 0] = 1.0
    metric[..., 2] = 1.0
    z = np.zeros((NE, NE))
    return ErosionState.window(
        h * cs, z, z, np.zeros((NE, NE, 2)), np.full((NE, NE), 0.5, np.float32), precip, np.ones((NE, NE), np.float32), upl * cs, mask, metric, metric, H, cs, deposit_on_exit=deposit_on_exit
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def d8_downstream(filled: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Steepest-descent D8 code per interior cell on the filled surface (-1 = none)."""
    N = filled.shape[0]
    best = np.full((N, N), -1, dtype=np.int64)
    bestdrop = np.zeros((N, N))
    pad = np.pad(filled, 1, mode="edge")
    for k, (di, dj) in enumerate(D8):
        nb = pad[1 + di : 1 + di + N, 1 + dj : 1 + dj + N]
        drop = (filled - nb) / (np.hypot(di, dj))
        # neighbour must be inside the array
        ok = np.ones((N, N), bool)
        if di > 0:
            ok[-1, :] = False
        if di < 0:
            ok[0, :] = False
        if dj > 0:
            ok[:, -1] = False
        if dj < 0:
            ok[:, 0] = False
        better = ok & (drop > bestdrop)
        best[better] = k
        bestdrop[better] = drop[better]
    return best


def trace_paths(chan: np.ndarray, down: np.ndarray, land: np.ndarray, n_paths: int = 10, max_len: int = 100000):
    """Trace downstream from every channel head; return the n longest paths (lists of (i, j))."""
    N = chan.shape[0]
    has_up = np.zeros((N, N), bool)
    ci, cj = np.nonzero(chan)
    for i, j in zip(ci, cj):
        k = down[i, j]
        if k >= 0:
            ni, nj = i + D8[k][0], j + D8[k][1]
            if chan[ni, nj]:
                has_up[ni, nj] = True
    heads = [(i, j) for i, j in zip(ci, cj) if not has_up[i, j]]
    paths = []
    for i, j in heads:
        path = [(i, j)]
        seen = {(i, j)}
        while True:
            k = down[i, j]
            if k < 0:
                break
            i, j = i + D8[k][0], j + D8[k][1]
            if not (0 <= i < N and 0 <= j < N) or not chan[i, j] or not land[i, j] or (i, j) in seen or len(path) > max_len:
                break
            seen.add((i, j))
            path.append((i, j))
        paths.append(path)
    paths.sort(key=len, reverse=True)
    return paths[:n_paths]


def sinuosity(path, smooth: int = 7) -> float:
    if len(path) < 2 * smooth:
        return 1.0
    p = np.asarray(path, dtype=np.float64)
    k = np.ones(smooth) / smooth
    ps = np.stack([np.convolve(p[:, 0], k, mode="valid"), np.convolve(p[:, 1], k, mode="valid")], 1)
    L = float(np.sum(np.hypot(np.diff(ps[:, 0]), np.diff(ps[:, 1]))))
    chord = float(np.hypot(*(ps[-1] - ps[0])))
    return L / max(chord, 1e-9)


def metrics(state: ErosionState, prev_surface: np.ndarray | None, pct: float = 97.0) -> dict:
    inter = state.interior
    surf = state.surface()[inter][0]
    q = state.discharge[inter][0]
    act = state.mask[inter][0] == pk.MASK_ACTIVE
    land = (surf >= 0) & act
    ql_ = q[land]
    srt = np.sort(ql_)[::-1]
    top1 = float(srt[: max(1, srt.size // 100)].sum() / max(srt.sum(), 1e-12))
    thr = float(np.percentile(ql_, pct))
    chan = (q > thr) & land
    lab, n = ndimage.label(chan, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
    big = sizes[sizes > 100]
    # paths on the filled surface (D8), restricted to channel cells
    filled = priority_flood_eps(state.surface(), state.mask, state.owner_table(), state.H, state.N, 1e-3)[inter][0]
    down = d8_downstream(filled, land)
    paths = trace_paths(chan, down, land)
    sins = [sinuosity(p) for p in paths]
    hi = float((surf[land].mean() - surf[land].min()) / max(surf[land].max() - surf[land].min(), 1e-9)) if land.any() else 0.0
    churn = float(np.mean(np.abs(surf - prev_surface)[act] > 1.0)) if prev_surface is not None else 0.0
    out = {
        "top1_share": top1,
        "n_components": int(n),
        "n_big": int(big.size),
        "big_sizes": [int(s) for s in np.sort(big)[::-1][:15]],
        "frac_in_big": float(big.sum() / max(sizes.sum(), 1)),
        "largest": int(sizes.max()) if n else 0,
        "sinuosity_mean": float(np.mean(sins)) if sins else 1.0,
        "sinuosity_max": float(np.max(sins)) if sins else 1.0,
        "path_lengths": [len(p) for p in paths],
        "hypsometric": hi,
        "churn": churn,
        "land_fraction": float(np.mean(surf >= 0)),
        "nan": bool(~np.isfinite(state.height).all() or ~np.isfinite(state.sediment).all() or ~np.isfinite(state.discharge).all()),
        "sediment_mean": float(state.sediment[inter][0][land].mean()) if land.any() else 0.0,
        "pending": float(state.pending[inter].sum()),
        "relief": float(surf[land].max() - surf[land].min()) if land.any() else 0.0,
        "q_max": float(q.max()),
    }
    return out


def fmt(m: dict) -> str:
    return (
        f"top1 {m['top1_share']:.3f} comps {m['n_components']:3d} big {m['n_big']:2d} {m['big_sizes'][:6]} fracbig {m['frac_in_big']:.2f} "
        f"sinu {m['sinuosity_mean']:.3f}/{m['sinuosity_max']:.3f} len {m['path_lengths'][:3]} HI {m['hypsometric']:.3f} churn {m['churn']:.4f} "
        f"land {m['land_fraction']:.3f} sed {m['sediment_mean']:.3f} pend {m['pending']:.1f} relief {m['relief']:.1f} qmax {m['q_max']:.0f}{' NAN' if m['nan'] else ''}"
    )


# --------------------------------------------------------------------------
# quicklook
# --------------------------------------------------------------------------
def quicklook_face(state: ErosionState, path: Path, scale: int = 2) -> Path:
    inter = state.interior
    surf = (state.surface()[inter] * state.height_unit_m).astype(np.float32)  # (1, N, N)
    q = state.discharge[inter].astype(np.float64)
    img = ql.render_height(surf, 0.0, state.height_unit_m)
    lq = np.log1p(np.maximum(q, 0))
    thr = float(np.percentile(lq, 97)) if lq.max() > 0 else 1.0
    wgt = np.clip((lq - thr) / max(lq.max() - thr, 1e-9), 0, 1)
    img = ql.overlay(img, wgt > 0, (30, 80, 255), 0.95, weight=0.35 + 0.65 * wgt)
    qi = ql.render_scalar(q, vmin=0.0, vmax=float(lq.max()) if lq.max() > 0 else 1.0, log=True, cmap="gray")
    left = np.transpose(img[0], (1, 0, 2))
    right = np.transpose(qi[0], (1, 0, 2))
    sep = np.full((left.shape[0], 4, 3), 255, np.uint8)
    both = np.concatenate([left, sep, right], axis=1)
    im = Image.fromarray(np.ascontiguousarray(both))
    if scale != 1:
        im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(path))
    return path


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def parse_sets(sets):
    over = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        g, name = k.split(".", 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        over.setdefault(g, {})[name] = val
    return over


def run(case: str, iters: int, tag: str, N: int = 256, sets=None, every: int = 50, seed: int = 0, out: Path = OUT, quiet: bool = False, deposit_on_exit: bool = False, tilt: float = 8.0, noise: float = 3.0, uplift: float = 5.0, plain_slope: float = 0.002):
    params = WorldParams().with_overrides(**parse_sets(sets)) if sets else WorldParams()
    st = make_state(case, N, params, iters, seed=seed, deposit_on_exit=deposit_on_exit, tilt=tilt, noise=noise, uplift=uplift, plain_slope=plain_slope)
    d = out / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(json.dumps({"case": case, "iters": iters, "N": N, "sets": sets or [], "tilt": tilt, "noise": noise, "uplift": uplift, "plain_slope": plain_slope, "erosion": params.to_dict()["erosion"]}, indent=1))
    quicklook_face(st, d / "iter0000.png")
    log = []
    t0 = time.time()
    tstep = 0.0
    prev = None
    for it in range(iters):
        if (it + 1) % every == 0 or it + 1 == iters:
            prev = st.surface()[st.interior][0].copy()
        t1 = time.time()
        s = step(st, params, it)
        tstep += time.time() - t1
        if (it + 1) % every == 0 or it + 1 == iters:
            m = metrics(st, prev)
            m["iteration"] = it + 1
            m["seconds_per_iter"] = tstep / (it + 1)
            m["deaths"] = s["deaths"]
            m["steps_mean"] = s["steps_mean"]
            m["clamped"] = s["clamped"]
            log.append(m)
            quicklook_face(st, d / f"iter{it + 1:04d}.png")
            if not quiet:
                print(f"[{tag}] it {it + 1:4d} ({tstep / (it + 1):.2f}s/it) {fmt(m)} deaths {s['deaths']} steps {s['steps_mean']:.0f} clamped {s['clamped']}", flush=True)
            (d / "metrics.json").write_text(json.dumps(log, indent=1))
    np.savez_compressed(d / "final.npz", height=st.height, sediment=st.sediment, discharge=st.discharge, momentum=st.momentum, mask=st.mask, pending=st.pending)
    if not quiet:
        print(f"[{tag}] done in {time.time() - t0:.1f}s -> {d}")
    return st, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="a")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--deposit-on-exit", action="store_true")
    ap.add_argument("--tilt", type=float, default=8.0)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--uplift", type=float, default=5.0)
    ap.add_argument("--plain-slope", type=float, default=0.002)
    ap.add_argument("--out", type=Path, default=OUT, help="directory for runs/<tag>/ (default bake/erosion_runs)")
    a = ap.parse_args()
    tag = a.tag or (f"{a.case}_" + ("_".join(s.replace("erosion.", "").replace("=", "") for s in a.set) or "base"))
    run(a.case, a.iters, tag, a.N, a.set, a.every, a.seed, out=a.out, deposit_on_exit=a.deposit_on_exit, tilt=a.tilt, noise=a.noise, uplift=a.uplift, plain_slope=a.plain_slope)


if __name__ == "__main__":
    main()
