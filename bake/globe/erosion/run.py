"""Erosion stage driver (PLAN.md section 8.3): the global pass on the coarse
grid with checkpoints, resume and periodic quicklooks.

Inputs (coarse fields): ``bedrock``, ``uplift``, ``hardness`` (tectonics),
``precip``, ``evap`` (climate).  Outputs: ``height``, ``sediment``,
``discharge``, ``momentum`` (docs/DEVELOPING.md).  Sediment still parked
at the end of the run (``ErosionState.pending``: land pits and seafloor
stockpiles) is not part of the outputs; its total is reported as
``pending_total_m`` in the stage info, the submerged part of it as
``pending_sea_m``, next to ``lost_offshore_m`` (the
``erosion.offshore_writeoff`` share of the load submarine fans could not
place: it left the modelled surface for the deep ocean).

``land_fraction`` (reported next to ``land_fraction_bedrock``) is held at
:func:`globe.erosion.maps.datum_land_fraction` throughout the run --
``world.land_fraction``, or in shelf mode the bedrock's own land fraction: uplift is applied mean-free and
every iteration ends with :func:`globe.erosion.maps.hold_datum`, a rigid
shift of ``height`` onto the land-fraction order statistic of the surface.
Without it the datum drifts (uplift is a forcing, and mass leaves the
surface for the deep ocean), and hydro's one-shot re-quantile then drops
sea level onto terrain that was sculpted against a different base level.
The drift it removed is reported as ``datum_drift_m``; the residual
``land_fraction`` differs from that target only by the
tie/rounding of a cell or two.

Checkpoints: ``checkpoints/erosion_iterNNNN.npz`` (extended state arrays in
cell units) + ``checkpoints/erosion_iterNNNN.json`` (iteration, parameter
hash); ``run`` resumes from the newest checkpoint whose hash (parameters,
upstream output hashes, kernel version) matches unless ``erosion.resume``
is False.  Only the two newest checkpoints per parameter hash are retained
(the older ones can never be resumed from); other parameter families are
left alone.  ``bake.py --force`` clears the outputs but not
``checkpoints/``: with an unchanged hash the final checkpoint is reused
(that is what it is for); a kernel change bumps ``KERNEL_VERSION`` and
invalidates it.  ``erosion.iterations`` is not part of the checkpoint hash,
so lowering it resumes from the newest checkpoint at or below the new end
instead of recomputing from bedrock -- in ``erosion.uplift_mode = 'stack'``.
In ``'replay'`` the start state is ``bedrock`` less ``iterations`` times the
applied uplift (:func:`globe.erosion.maps.start_replay`), so a checkpoint
belongs to one ``iterations`` and the count stays in its hash.

``erosion.uplift_mode`` (docs/uplift-replay.md): ``'stack'`` starts from
``bedrock`` and applies ``uplift`` on top; ``'replay'`` starts from the crust
at the uplift reference step and replays the window, ending at ``bedrock``
when nothing erodes.  The start is built once, before the loop; a resumed
run's checkpoint already carries the replayed height and overwrites it.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from ..config import WorldParams
from ..field import FaceField
from ..io.world_store import WorldStore
from .maps import ErosionState, datum_land_fraction, fill_basins, held_land_fraction, hold_datum, settle_lakes, start_replay, step, uplift_cap
from .particle import KERNEL_VERSION, MASK_ACTIVE

OUTPUTS = ["height", "sediment", "discharge", "momentum"]

_CKPT_RE = re.compile(r"erosion_iter(\d+)\.npz$")


def _ckpt_hash(params: WorldParams, store: WorldStore | None = None) -> str:
    """Parameters *and* upstream outputs a checkpoint depends on: the world
    and erosion groups, the kernel version (``particle.KERNEL_VERSION``: a
    code change never resumes stale results) and the manifest hashes of the
    tectonics and climate outputs (so a re-baked upstream stage — even with
    the same parameters, e.g. a stub replaced by the real stage —
    invalidates it).

    ``erosion.iterations`` and ``erosion.resume`` are normalised out: they
    are run-control knobs, not state knobs (the per-iteration uplift that
    ``iterations`` scales is already covered by the tectonics output hash),
    so a shortened, extended or ``resume=False`` run still matches its own
    checkpoints and :func:`find_checkpoint`'s ``max_iteration`` guard — not
    the hash — is what rejects a checkpoint past the requested end.  This
    normalisation is a one-time invalidation of checkpoints written by the
    old scheme.

    Except in ``uplift_mode = 'replay'``: there the start state is lowered by
    ``iterations`` times the applied uplift (``maps.start_replay``), so a
    state at iteration k of an 800-iteration run is not a state of a
    600-iteration one (resuming it would end ``200 x uplift`` below
    ``bedrock``), and ``iterations`` stays in the hash."""
    up = f":k{KERNEL_VERSION}"
    if store is not None:
        for s in ("tectonics", "climate"):
            info = store.stage_info(s) or {}
            up += ":" + str(info.get("hash", ""))
    replay = str(params.erosion.uplift_mode) == "replay"
    norm = params.with_overrides(erosion={"iterations": int(params.erosion.iterations) if replay else 0, "resume": True})
    return norm.group_hash("world", "erosion") + ":" + params.group_hash("tectonics", "climate") + up


def save_checkpoint(store: WorldStore, state: ErosionState, params: WorldParams) -> Path:
    d = store.checkpoint_dir
    d.mkdir(parents=True, exist_ok=True)
    it = state.iteration
    p = d / f"erosion_iter{it:04d}.npz"
    tmp = p.with_suffix(".npz.tmp")
    arrays = dict(height=state.height, sediment=state.sediment, discharge=state.discharge, momentum=state.momentum, pending=state.pending,
                  lake_load=state.lake_load)
    if state.route is not None:  # the routing surface is refreshed every flood_every iterations: part of the state
        arrays["route"] = state.route
    if getattr(state, "base_at", None) is not None:  # the base level is refreshed every sea_mask_every iterations: part of the state
        arrays["base"] = state.base
    if getattr(state, "iso_acc", None) is not None:  # rebound not yet applied: part of the state
        arrays["iso_acc"] = state.iso_acc
    if getattr(state, "ice_prev", None) is not None:  # sticky ice: part of the state
        arrays["ice_prev"] = state.ice_prev.astype(np.uint8)
    if getattr(state, "strata", None) is not None:  # how deep in its beds each cell is (erosion.strata_amp)
        arrays["eroded"] = state.eroded
    if getattr(state, "sheet", None) is not None:  # the ice sheet as it stands (erosion.ice_sheet): a resume goes on from it
        arrays["sheet"] = state.sheet.astype(np.uint8)
        arrays["sheet_h"] = state.sheet_h
        arrays["sheet_fed"] = state.sheet_fed.astype(np.uint8)
        if getattr(state, "sheet_flux", None) is not None:
            arrays["sheet_flux"] = state.sheet_flux
        arrays["sea_fall"] = np.array([float(state.sea_drop), np.nan if state.sea_ref is None else float(state.sea_ref)])
    if state.lake_flag is not None:  # lakes are refreshed every flood_every iterations: part of the state
        arrays["lake_flag"] = state.lake_flag
        arrays["lake_id"] = state.lake_id        # ...and so is the room each has left for its rivers' load
        arrays["lake_room"] = state.lake_room
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(p)
    meta = {"iteration": it, "params_hash": _ckpt_hash(params, store), "height_unit_m": state.height_unit_m, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(p.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    # after the .json write, so a crash between tmp.replace(p) and it leaves
    # the previous checkpoint in place for find_checkpoint to fall back to
    _prune_checkpoints(d, meta["params_hash"], keep=2)
    return p


def _prune_checkpoints(d: Path, params_hash: str, keep: int = 2) -> None:
    """Keep only the ``keep`` newest checkpoints of this parameter family.

    :func:`find_checkpoint` only ever resumes from the newest matching
    checkpoint, so the rest are dead weight (~358 MB each at the default
    N_c=1024, ~5.7 GB over a default 800-iteration run, kept for the life of
    the world directory because ``bake --force`` preserves ``checkpoints/``).
    One spare is kept so a truncated newest checkpoint still has a fallback;
    other parameter families are left alone."""
    mine = []
    for q in d.glob("erosion_iter*.npz"):
        m = _CKPT_RE.search(q.name)
        jq = q.with_suffix(".json")
        if not m or not jq.exists():
            continue
        try:
            if json.loads(jq.read_text()).get("params_hash") != params_hash:
                continue
        except json.JSONDecodeError:
            continue
        mine.append((int(m.group(1)), q))
    for _, q in sorted(mine)[: max(0, len(mine) - keep)]:
        q.unlink(missing_ok=True)
        q.with_suffix(".json").unlink(missing_ok=True)


def find_checkpoint(store: WorldStore, params: WorldParams, max_iteration: int | None = None) -> tuple[Path, dict] | None:
    """Newest valid checkpoint (matching parameter hash, iteration <= max)."""
    d = store.checkpoint_dir
    if not d.exists():
        return None
    want = _ckpt_hash(params, store)
    best = None
    for p in d.glob("erosion_iter*.npz"):
        m = _CKPT_RE.search(p.name)
        if not m:
            continue
        jp = p.with_suffix(".json")
        if not jp.exists():
            continue
        try:
            meta = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("params_hash") != want:
            continue
        it = int(meta.get("iteration", -1))
        if max_iteration is not None and it > max_iteration:
            continue
        if best is None or it > best[1]["iteration"]:
            best = (p, meta)
    return best


def load_checkpoint(state: ErosionState, path: Path, meta: dict) -> None:
    with np.load(path) as z:
        state.height[...] = z["height"]
        state.sediment[...] = z["sediment"]
        state.discharge[...] = z["discharge"]
        state.momentum[...] = z["momentum"]
        state.pending[...] = z["pending"] if "pending" in z.files else 0.0
        state.lake_load[...] = z["lake_load"] if "lake_load" in z.files else 0.0
        state.route = np.ascontiguousarray(z["route"]) if "route" in z.files else None
        if "base" in z.files:
            state.base = np.ascontiguousarray(z["base"])
            state.base_at = int(meta["iteration"])  # a resume must see the mask the continuous run had
        else:
            state.base_at = None
        state.iso_acc = np.array(z["iso_acc"]) if "iso_acc" in z.files else None
        state.ice_prev = np.array(z["ice_prev"]).astype(bool) if "ice_prev" in z.files else None
        if getattr(state, "strata", None) is not None and "eroded" in z.files:
            state.eroded[...] = z["eroded"]
        if getattr(state, "ice_clim", None) is not None and "sheet" in z.files:
            state.sheet = np.ascontiguousarray(z["sheet"]).astype(bool)
            state.sheet_h = np.ascontiguousarray(z["sheet_h"])
            state.sheet_fed = np.ascontiguousarray(z["sheet_fed"]).astype(bool)
            state.sheet_flux = np.ascontiguousarray(z["sheet_flux"]) if "sheet_flux" in z.files else None
            if "sea_fall" in z.files:
                state.sea_drop = float(z["sea_fall"][0])
                state.sea_ref = None if np.isnan(z["sea_fall"][1]) else float(z["sea_fall"][1])
        state.lake_flag = np.ascontiguousarray(z["lake_flag"]).astype(np.uint8) if "lake_flag" in z.files else None
        if "lake_id" in z.files:
            state.lake_id = np.ascontiguousarray(z["lake_id"]).astype(np.int32)
            state.lake_room = np.ascontiguousarray(z["lake_room"]).astype(np.float64)
    state.iteration = int(meta["iteration"])


#: the active volcanic edifices tectonics hands over (globe.tectonics.run.VOLCANO_FIELD)
VOLCANO_FIELD = "volcano_active"


def active_volcanoes(store: WorldStore, grid) -> FaceField | None:
    """The cones of the edifices active at the end of tectonics (m), or None."""
    return store.load_field(VOLCANO_FIELD, grid) if store.has_field(VOLCANO_FIELD) else None


def build_state(store: WorldStore, params: WorldParams, replay: bool = True) -> ErosionState:
    """The iteration-0 state: ``bedrock`` as tectonics left it, lowered to
    the uplift reference step when ``erosion.uplift_mode`` is ``'replay'``
    (:func:`globe.erosion.maps.start_replay`) unless ``replay`` is False --
    :func:`run` takes that step itself, to measure the bedrock first.

    The active volcanic edifices (``volcano_active``, tectonics.volcanoes)
    are taken off the bedrock first: the stage erodes the ground under them
    and :func:`run` puts them back on top at its end.  An active volcano is
    being built faster than it erodes, and the coarse erosion is not a model
    of a cone 10-30 km across on 10 km cells: run on them, the shipped
    parameters cut 2-5 km cones to 0.03-0.6 km islets even at hardness 1
    (window test, proto/arcs critique).  Extinct edifices are in the
    uplift and go through the stage like any other relief."""
    grid = params.coarse_grid()
    bed = store.load_field("bedrock", grid)
    cone = active_volcanoes(store, grid)
    if cone is not None:
        bed.data = bed.data - cone.data
    hard = store.load_field("hardness", grid)
    upl = store.load_field("uplift", grid)
    pr = store.load_field("precip", grid)
    ev = store.load_field("evap", grid)
    state = ErosionState.from_grid(grid, bed, hard, pr, ev, upl, params.erosion)
    state.land_target = datum_land_fraction(params, bed.interior)
    # (a world with no climate temperature -- a stub -- has the kernel's `evap` for all three)
    has_t = store.has_field("temperature")
    t0 = float(getattr(params.hydro, "pet_t0", 0.0)) if has_t else 0.0
    follow = bool(getattr(params.erosion, "climate_at_surface", False)) and has_t
    dry = float(getattr(params.erosion, "ice_aridity", 0.0)) if has_t else 0.0
    age = float(getattr(params.erosion, "ice_age_c", 0.0)) if has_t else 0.0
    if dry > 0.0:
        # ice needs snow: the rain as a depth (the land-mean rain's), for the aridity of cold ground
        af = grid.interior_cell_area / float(grid.cell_size_m) ** 2
        state.ice_snow = (dry, float(getattr(params.hydro, "land_evap", 0.0)), np.ascontiguousarray(FaceField.from_interior(grid, (pr.interior / af).astype(np.float32)).data))
    sheet = bool(getattr(params.erosion, "ice_sheet", False)) and has_t
    if t0 > 0.0 or follow or dry > 0.0 or age > 0.0 or sheet:
        state.ice_age = age
        # the climate's temperature itself: for the lakes' evaporation, which does not stop at freezing
        # (hydro.pet_t0), and for a cold that follows the ground (erosion.climate_at_surface)
        from ..hydro.balance import insolation

        state.temp0 = np.ascontiguousarray(store.load_field("temperature", grid).data, np.float32)
        state.temp_z = np.ascontiguousarray(store.load_field("bedrock", grid).data, np.float32)      # with its cones on: what the climate saw
        state.temp_lapse = float(params.climate.lapse)
        state.temp_k = float(params.climate.k_evap)
        state.temp_follow = follow
        if t0 > 0.0:
            state.pet_law = (float(params.climate.T_eq), t0, insolation(grid.latitude()).astype(np.float32))
    if float(getattr(params.erosion, "strata_amp", 0.0)) > 0.0:
        from . import strata

        strata.setup(state, grid, params)        # bedded rock: the hardness follows the depth of the cut
    if bool(getattr(params.erosion, "ice_sheet", False)):
        from . import icesheet

        icesheet.setup(state, grid, store, params)   # the ice as a body: its climate (the sheet grows at the first iteration)
    if replay:
        start_replay(state, params)
    return state


def restore_volcanoes(state: ErosionState, cone: FaceField | None) -> dict:
    """Put the active edifices back on top of the eroded surface (in place)."""
    if cone is None:
        return {}
    add = cone.data.astype(np.float64) / state.height_unit_m
    state.height += add
    inter = state.interior
    c = cone.interior
    surf = (state.height + state.sediment)[inter]
    on = c > 0.0
    return {"volcano_cells": int(on.sum()), "volcano_max_m": float(c.max()) if on.any() else 0.0,
            "volcano_land_cells": int((on & (surf >= 0.0)).sum())}


#: the stage's ice (erosion.ice_sheet): its thickness (m) at the last maximum the stage ends on, and today's
ICE_FIELDS = ("ice_max", "ice_now")


def thaw(store: WorldStore, state: ErosionState, params: WorldParams, frame_res: int, log=print) -> dict:
    """The end of the ice age (``erosion.ice_sheet``).  The stage ends on
    its last glacial maximum; today is ``erosion.ice_age_c`` warmer, and the
    sheet that stood at the maximum is thawed back to it in
    ``render.thaw_frames`` steps on the ground as the stage left it -- half
    an iteration of time, in which the ground does not change and a quarter
    of the land comes out from under the ice.  What is left is today's ice:
    a sheet's height keeps it where none would grow (``icesheet.settle``).

    Writes the fields ``ice_max`` and ``ice_now`` (thickness, metres) and,
    with ``frame_res``, a frame of each step for the viewer (stage
    ``thaw``).  The state's sheet -- the last maximum's, settled, as
    :func:`run` leaves it for its last frame -- is that again afterwards."""
    from ..viz import frames as vf
    from . import icesheet

    grid = state.grid
    age = float(params.erosion.ice_age_c)
    n = max(int(getattr(params.render, "thaw_frames", 0)), 1)
    top = (state.sheet.copy(), state.sheet_h.copy(), state.sheet_fed.copy(), float(state.ice_age))
    low = float(getattr(state, "sea_drop", 0.0)) * float(state.height_unit_m)
    bed, sea = icesheet.ground(state)
    s_max = icesheet.stats(grid, state.sheet[state.interior], state.sheet_h[state.interior], sea)
    store.save_field(FaceField(grid, np.where(state.sheet, state.sheet_h, 0.0).astype(np.float32), name="ice_max"))
    rec = vf.FrameRecorder(store.root, "thaw", frame_res) if frame_res > 0 else None
    if rec is not None:
        rec.clear()
    s_now = s_max
    for k in range(1, n + 1):
        state.ice_age = age * (1.0 - k / n)
        res = icesheet.update(state, params, rounds=icesheet.FIRST_ROUNDS)
        s_now = res["stats"]
        if float(getattr(state, "sea_drop", 0.0)) != 0.0 or float(getattr(params.erosion, "ice_sea_share", 0.0)) > 0.0:
            # the sea comes back with the melt (erosion.ice_sea_share), to today's level at the last step:
            # over the shelf the rivers crossed, and up the valleys they cut to the lower coast
            icesheet.sea_fall(state, params, s_now["sea_level_m"])
            if k == n:
                state.sea_drop = 0.0
            hold_datum(state, held_land_fraction(state, params), float(state.sea_drop))
        if rec is not None:
            vf.erosion_frame(state, rec, k, n, params)
    store.save_field(FaceField(grid, np.where(state.sheet, state.sheet_h, 0.0).astype(np.float32), name="ice_now"))
    state.sheet, state.sheet_h, state.sheet_fed, state.ice_age = top
    drop = s_max["sea_level_m"] - s_now["sea_level_m"]
    log(f"[erosion] ice: on {100.0 * s_max['land_share']:.1f} % of the land at the last maximum ({s_max['thickness_mean_m']:.0f} m thick on average, "
        f"the sea {drop:.0f} m lower than today) and {100.0 * s_now['land_share']:.1f} % today ({s_now['thickness_mean_m']:.0f} m, {s_now['sea_level_m']:.0f} m of sea level in it)")
    return {"ice_max_land_share": s_max["land_share"], "ice_now_land_share": s_now["land_share"], "ice_max_thickness_m": s_max["thickness_mean_m"],
            "ice_now_thickness_m": s_now["thickness_mean_m"], "ice_now_sea_level_m": s_now["sea_level_m"], "ice_max_sea_drop_m": drop,
            # how far under today's the stage's sea stood at its end (erosion.ice_sea_share; 0 = it did not move)
            "ice_sea_lowstand_m": low}


def write_outputs(store: WorldStore, state: ErosionState) -> None:
    for f in state.fields().values():
        store.save_field(f)


def run(store: WorldStore, params: WorldParams, log=print) -> dict:
    ep = params.erosion
    grid = params.coarse_grid()
    state = build_state(store, params, replay=False)
    n_iter = int(ep.iterations)
    land0 = float(np.mean((state.height + state.sediment)[state.interior] >= 0))
    # uplift_mode 'replay': lower the start to the reference-step crust (None
    # in 'stack').  A resume overwrites the height with the checkpoint's.
    rep = start_replay(state, params)
    # the basins the bedrock arrives with become sedimentary plains (maps.fill_basins); a
    # resume overwrites this with the checkpoint's sediment, which already carries it
    fill = fill_basins(state, params)
    ck = find_checkpoint(store, params, max_iteration=n_iter) if ep.resume else None
    if ck is not None:
        load_checkpoint(state, *ck)
        log(f"[erosion] resumed from {ck[0].name} (iteration {state.iteration})")
    log(f"[erosion] {grid.describe()}; heights in units of {state.height_unit_m:.1f} m; {n_iter} iterations, {ep.particles_per_cell} particles/cell")
    # the uplift ceiling (maps.apply_uplift): how many cells it touches is the
    # number the next Earth bake's after-number is read from
    cap = uplift_cap(state, ep)
    iact = state.mask[state.interior] == MASK_ACTIVE
    n_capped = int(np.sum(state.uplift[state.interior][iact] > cap)) if cap is not None else 0
    log(f"[erosion] uplift cap {0.0 if cap is None else ep.uplift_max_m:g} m/iteration"
        f" ({n_capped} of {int(iact.sum())} active cells above it; field max {state.uplift[state.interior][iact].max() * state.height_unit_m:.2f} m/iteration)")
    if fill is not None and ck is None:
        log(f"[erosion] basin fill: {fill['basin_area_km2'] / 1e6:.2f} Mkm2 of closed basins laid with {fill['fill_km3'] / 1e6:.2f} Mkm3 of sediment"
            f" (mean {fill['fill_mean_m']:.0f} m, max {fill['fill_max_m']:.0f} m)")
    if rep is not None:
        log(f"[erosion] uplift replay: start lowered by up to {rep['replay_max'] * state.height_unit_m:.1f} m"
            f" and raised by up to {-rep['replay_min'] * state.height_unit_m:.1f} m to the reference-step crust"
            f" (land fraction {rep['land_fraction']:.3f} before the datum hold moved the surface {-rep['datum_shift'] * state.height_unit_m:+.1f} m)")
    # viewer timeline frames (hash-exempt, read-only).  Frames past the
    # resume point belong to whichever run wrote them, not to this one.
    from ..viz import frames as vf

    rp = params.render
    frame_every = int(rp.erosion_frame_every) if rp.viewer else 0
    rec = vf.FrameRecorder(store.root, "erosion", min(int(rp.frame_res), grid.N))
    rec.clear(after=state.iteration)

    def frame(st: ErosionState) -> None:
        # Frame k is taken inside iteration k's step, after it has refreshed the sea, the routing
        # surface and the lakes and before its particles run (maps.step) -- the surface iteration
        # k - 1 left, with the water that stands on it.  Taken when that step returned, as the
        # frames were while they held only the ground, the water is a refresh old: on the small
        # preset the last frame showed 404 cells of lake where hydro finds 4, the levels of
        # before the last glacial pass over the bed that pass had lowered.  (Only where
        # `erosion_frame_every` is a multiple of `flood_every`, as the defaults are; a frame
        # between two refreshes has the water of the one before it.)
        vf.erosion_frame(st, rec, st.iteration, n_iter, params)

    times = []
    clamped = 0
    lost_offshore = 0.0
    datum_shift = 0.0
    sea = None  # last `maps.refresh_base` census: what erosion called sea
    lakes = None  # last `maps.refresh_lakes` census
    lake_laid = lake_passed = 0.0  # sediment the lakes kept and laid, and what full lakes passed on (cell units)
    while state.iteration < n_iter:
        it = state.iteration
        t0 = time.time()
        st = step(state, params, it, after_refresh=frame if frame_every > 0 and it % frame_every == 0 else None)
        dt = time.time() - t0
        times.append(dt)
        clamped += int(st.get("clamped", 0))
        lost_offshore += float(st.get("lost_offshore", 0.0))
        datum_shift += float(st.get("datum_shift", 0.0))
        sea = st.get("sea", sea)
        lakes = st.get("lakes", lakes)
        if "lakes" in st:
            lk = st["lakes"]
            lake_laid += float(lk.get("load_laid", 0.0))
            lake_passed += float(lk.get("load_stray", 0.0))
            if "load_parked" in lk and (it % 50 == 0 or it + 10 >= n_iter):
                u = state.height_unit_m
                log(f"[erosion] lakes at {it}: {lk['lake_cells']} cells ({lk['overflowing']} overflowing, {lk['closed']} closed); their rivers left "
                    f"{lk['load_parked'] * u:,.0f} m-cells, {lk['load_laid'] * u:,.0f} laid, {lk['load_stray'] * u:,.0f} where the lake had gone; "
                    f"{lk['load_lakes']} lakes have room for {lk['load_room'] * u:,.0f}")
        if "ice" in st and (it % 50 == 0 or it + 10 >= n_iter):
            si = st["ice"]
            log(f"[erosion] ice at {it}: {state.ice_age:+.1f} C, on {100.0 * si['land_share']:.1f} % of the land, {si['thickness_mean_m']:.0f} m thick on average "
                f"({si['thickness_max_m']:.0f} at most), {si['sea_level_m']:.0f} m of sea level; {si['rounds']} rounds{'' if si['settled'] else ' (not settled)'}"
                + (f"; the sea {float(state.sea_drop) * state.height_unit_m:.0f} m under today's" if float(getattr(state, "sea_drop", 0.0)) > 0.0 else ""))
        d = st.get("deaths", {})
        log(
            f"[erosion] iter {it + 1}/{n_iter}: {st['particles']} particles, mean {st['steps_mean']:.0f} steps, "
            f"deaths ocean {d.get('ocean', 0)} pit {d.get('pit', 0)} age {d.get('age', 0)}, clamped {st.get('clamped', 0)}, "
            f"pending {st.get('pending_total', 0.0):.1f}, {st['seconds_particles']:.2f}s particles, {dt:.2f}s total"
        )
        done = state.iteration
        if ep.checkpoint_every > 0 and (done % ep.checkpoint_every == 0 or done == n_iter):
            p = save_checkpoint(store, state, params)
            log(f"[erosion] checkpoint -> {p.name}")
        if ep.quicklook_every > 0 and done % ep.quicklook_every == 0 and done != n_iter:
            try:
                qp = quicklook_state(state, store.quicklook_path("erosion", f"iter{done:04d}"))
                log(f"[erosion] quicklook -> {qp}")
            except Exception as e:  # never break a bake on a picture
                log(f"[erosion] quicklook failed: {e!r}")
    if bool(getattr(ep, "ice_history", False)) and state.temp0 is not None:
        from .glacial import ice_cooling

        state.ice_age = ice_cooling(ep, n_iter)                # the stage ends on its last maximum (erosion.ice_history)
    # the lakes lay what the last iterations brought them (a run ends between two lake refreshes)
    lk = settle_lakes(state, params)
    if lk is not None and "load_parked" in lk:
        lake_laid += float(lk.get("load_laid", 0.0))
        lake_passed += float(lk.get("load_stray", 0.0))
        lakes = lk
        log(f"[erosion] lakes at the end: {lk['lake_cells']} cells, {lk['load_laid'] * state.height_unit_m:,.0f} m-cells laid of "
            f"{lk['load_parked'] * state.height_unit_m:,.0f} their rivers left")
    # the last frame: the surface the run ends on, with the lakes as that refresh has just found
    # them (without `erosion.lake_fill` there was none, and its water is the last refresh's)
    sheeted = getattr(state, "sheet", None) is not None
    if sheeted:
        from . import icesheet

        icesheet.update(state, params, rounds=icesheet.FIRST_ROUNDS)      # the last maximum's sheet, settled on the ground the stage ends with
    if frame_every > 0:
        frame(state)
    ice_info = thaw(store, state, params, min(int(rp.frame_res), grid.N) if frame_every > 0 else 0, log) if sheeted else {}
    # the active volcanic edifices go back on top of what the stage made of the ground under them
    volc = restore_volcanoes(state, active_volcanoes(store, grid))
    if volc:
        log(f"[erosion] active volcanoes back on top: {volc['volcano_cells']} cells, max {volc['volcano_max_m']:.0f} m, "
            f"{volc['volcano_land_cells']} of them land")
    write_outputs(store, state)
    surf = (state.height + state.sediment)[state.interior]
    info = {
        "iterations": n_iter,
        "height_unit_m": state.height_unit_m,
        "seconds_per_iteration": float(np.mean(times)) if times else 0.0,
        "land_fraction": float(np.mean(surf >= 0)),
        "land_fraction_bedrock": land0,
        "max_discharge": float(state.discharge[state.interior].max()),
        "sediment_mean_m": float(state.sediment[state.interior].mean() * state.height_unit_m),
        "sediment_p99_m": float(np.percentile(state.sediment[state.interior], 99) * state.height_unit_m),
        "pending_total_m": float(state.pending[state.interior].sum() * state.height_unit_m),
        # what the lakes kept of their rivers' sediment (erosion.lake_fill): laid over their floors,
        # and what was parked on a lake that had drained by its next refresh (sent on as pending)
        "lake_fill_laid_m": lake_laid * state.height_unit_m,
        "lake_fill_stray_m": lake_passed * state.height_unit_m,
        # the part of it still walking on the seafloor (a sea death's surplus
        # less the write-off); like the land part it is absent from the outputs
        "pending_sea_m": float(state.pending[state.interior][surf < state.base[state.interior]].sum() * state.height_unit_m),
        "clamped_entries": clamped,
        # metres of surface drift the in-loop datum hold removed over the run
        # (positive = the planet would have inflated by this much); the same
        # sign convention as hydro's ``height_shift_m``
        "datum_drift_m": datum_shift * state.height_unit_m,
        "lost_offshore_m": lost_offshore * state.height_unit_m,
        # the ceiling on the per-iteration uplift (0 = off) and the number of
        # active interior cells whose field exceeds it, i.e. how much of the
        # tectonics field the stage refused (docs/uplift-ceiling.md)
        "uplift_cap_m_per_iter": 0.0 if cap is None else float(ep.uplift_max_m),
        "uplift_capped_cells": n_capped,
        # 'stack' or 'replay' (docs/uplift-replay.md)
        "uplift_mode": str(ep.uplift_mode),
        **volc,
        # the ice sheet at the last maximum and today (erosion.ice_sheet; `thaw`)
        **ice_info,
    }
    if rep is not None:
        # the start was the reference-step crust: the most a cell was lowered
        # (raised) to build it, its land fraction before the datum was held
        # on it, and that hold (metres, datum_drift_m's sign: positive = the
        # surface was shifted down; not part of datum_drift_m, the loop's)
        info["replay_lowered_max_m"] = rep["replay_max"] * state.height_unit_m
        info["replay_raised_max_m"] = -rep["replay_min"] * state.height_unit_m
        info["land_fraction_reference"] = rep["land_fraction"]
        info["replay_datum_start_m"] = rep["datum_shift"] * state.height_unit_m
    if sea is not None:
        # what the kernel called sea at the last refresh (maps.refresh_base):
        # the classification the whole stage ran on, worth having on record
        info["sea"] = sea
    if lakes is not None:
        info["lakes"] = lakes
    return info


def quicklook_state(state: ErosionState, path) -> Path:
    from ..viz import quicklook as ql

    grid = state.grid
    h = FaceField(grid, (state.height + state.sediment).astype(np.float32) * state.height_unit_m, name="surface")
    q = FaceField(grid, state.discharge.astype(np.float32), name="discharge")
    return ql.quicklook_height(path, h, 0.0, grid.cell_size_m, discharge=q)


def quicklook(store: WorldStore, params: WorldParams, path) -> Path:
    """Hillshade of ``height + sediment`` with the discharge overlay."""
    from ..viz import quicklook as ql

    grid = params.coarse_grid()
    h = store.load_field("height", grid)
    h.data += store.load_field("sediment", grid).data
    q = store.load_field("discharge", grid)
    return ql.quicklook_height(path, h, 0.0, grid.cell_size_m, discharge=q)


__all__ = ["OUTPUTS", "run", "quicklook", "build_state", "save_checkpoint", "find_checkpoint", "load_checkpoint", "quicklook_state"]
