"""A scorecard for the tectonics stage: what the plates, the crust and the
rendered map are doing, every few hundred steps, in numbers that can be put
next to Earth's (``scripts/tect_scorecard.py`` runs it over seeds and prints
the table).

It is a *pure observer*.  It steps a :class:`~globe.tectonics.run.TectonicSim`
the way :meth:`~globe.tectonics.run.TectonicSim.run` does and samples it
between steps; where a number only exists inside a step it wraps the
module-level function that makes it -- ``run.collide`` (which pairs collided,
by crust type and plate, read off copies taken *before* the call, because the
call relabels a shortened loser onto the survivor's plate and turns an arc's
survivor continental) and ``intraplate.rift`` (the plate's centre before the
cut) -- with a wrapper that calls the original and copies what it needs.  The
segment cloud's ``compress`` / ``append`` are wrapped the same way, per
instance, to keep the observer's own per-segment arrays aligned with it.
Nothing it computes is written back, so a run with the observer attached is
bit-identical to one without (``tests/test_tectonics.py`` pins it on `tiny`).

Which continental crust was born during the run
-----------------------------------------------
The fields cannot say.  An island arc is the overriding *oceanic* segment of
an ocean-ocean collision relabelled continental (``collision._apply_collisions``),
so it keeps its ``age``; and the ocean floor that was there at step 0 carries
exactly the age of the continents that were (both start at 0 and age one a
step).  ``rework`` is set to 0 at an arc's birth, but it also falls towards 0
wherever a collision stacks crust into a belt, and ``craton`` is 0 on every
belt as well as every arc.  So the observer keeps two arrays of its own: the
step each segment was born (-1 for the initial cloud) and the crust kind it was
born with.  **An arc is continental crust that was born oceanic** -- whatever
turned it, so the rule holds for any arc mechanism that relabels ocean floor --
and crust *born* continental mid-run is the interior-void fill of
``collision.spawn_segments``, reported apart.  The field-only proxy
(``kind == CONTINENTAL and age < steps run``) is reported as a cross-check; it
misses the arcs built on the initial ocean floor.

The map numbers (rendered continental share, land share, land components,
hypsometry) are read off :func:`~globe.tectonics.run.frame_bed` -- the bed the
viewer's timeline draws, on the tect grid -- and put in metres with
:func:`~globe.tectonics.run.metres_per_unit`, the scale ``finalise`` uses.  They
are *before* the coarse-grid finishing ``finalise`` adds (``inject_detail``,
``inject_ranges``), so the maximum and the high-ground shares read lower than a
baked world's; what they measure is the crust the simulation built.

Units: speeds in km per step and cm/yr, ages in steps and My, through
``myr_per_step`` (``tectonics.myr_per_step``, a reporting unit the simulation
never reads).  Areas are shares of the planet's *extent* (``Segments.ext``,
the ground each segment covers), which with ``variable_extent`` off is the
same for every segment.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import intraplate
from . import run as tect_run
from .collision import DENSITY_EPS, build_tree, interior_centers_flat, plate_pair_polarity, weighted_quantile
from .segments import CONTINENTAL, OCEANIC

#: thickness unit -> km, for volumes only (``continental_thickness`` 1.0 is ~35 km, config.py)
THICKNESS_KM = 35.0
#: a plate that counts: share of the planet's extent (Earth has ~12 plates over 1 %)
MAJOR_SHARE = 0.01
#: plates entering the speed correlations and speed classes: >= this share (100 segments at 20k)
CORR_SHARE = 0.005
#: link radius (spacings) of the landmass graph -- scripts/supercontinent.py uses 1.6 and 1.1-2.5
LANDMASS_LINK = 1.5
#: two segments of different plates this close (spacings) are a boundary pair
BOUNDARY_LINK = 1.5
#: a continental segment with an oceanic one this close (spacings) is coastal
COAST_LINK = 1.2
#: an arc with no other plate this close (spacings) is stranded inside its plate
STRANDED = 3.0
#: |normal| / |relative velocity| above which a boundary pair is convergent / divergent (60 deg)
OBLIQUITY = 0.5
#: a rift's halves are watched for this many steps after the cut
RIFT_WINDOW = 100
#: a plate that lives fewer steps than this is short-lived
SHORT_LIFE = 10
#: land component area bins, km^2
AREA_BINS = (0.0, 1e3, 1e4, 1e5, 1e6, np.inf)
AREA_BIN_NAMES = ("lt1e3", "1e3_1e4", "1e4_1e5", "1e5_1e6", "gt1e6")


def _chord(theta: float) -> float:
    """Chord length of an angle (radians): the KD-trees hold unit vectors."""
    return 2.0 * math.sin(0.5 * min(float(theta), math.pi))


def _f(x):
    """A JSON-able float (None for NaN / inf / None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    from scipy.stats import spearmanr

    return _f(spearmanr(a, b).statistic)


def _wq(v, w, q):
    v, w = np.asarray(v, np.float64), np.asarray(w, np.float64)
    if v.size == 0 or w.sum() <= 0:
        return None
    return _f(weighted_quantile(v, w, q))


def _median(v):
    v = np.asarray(v, np.float64)
    return _f(np.median(v)) if v.size else None


class Observer:
    """Attach to a freshly initialised sim, call :meth:`after_step` after
    every ``sim.step()`` and :meth:`sample` whenever a row is wanted;
    :meth:`report` returns everything as plain JSON-able data.  Use it as a
    context manager (or :meth:`attach` / :meth:`detach`) so the wrapped
    module functions are restored however the run ends."""

    def __init__(self, sim, myr_per_step: float | None = None):
        self.sim = sim
        tp = sim.tp
        self.myr = float(myr_per_step if myr_per_step is not None else getattr(tp, "myr_per_step", 0.15))
        self.R_km = float(sim.params.R_planet) / 1000.0
        self.spacing = float(sim.spacing)
        self._seg = sim.seg
        # per-segment origin, aligned with the cloud by the compress / append wrappers
        self.born_step = np.full(sim.seg.M, -1, np.int32)
        self.born_kind = sim.seg.kind.astype(np.int8).copy()
        # plate genealogy: id -> (birth step, cause); id -> death step
        self.plate_birth: dict[int, tuple[int, str]] = {q: (-1, "initial") for q in range(sim.plates.P)}
        self.plate_death: dict[int, int] = {}
        self._retired: list[tuple[int, int, str]] = []     # (birth, death, cause) of plates a reorganise ended
        self._alive_prev = sim.plates.alive.copy()
        self._P_prev = int(sim.plates.P)
        self.rifts: list[dict] = []
        self._active_rifts: list[dict] = []
        self._step_rifts: list[dict] = []
        self.samples: list[dict] = []
        self._restore: list = []
        self._t0 = time.time()
        self._reset_window()

    # -- wiring ---------------------------------------------------------------
    def attach(self) -> "Observer":
        obs, sim = self, self.sim
        orig_collide = tect_run.collide

        def collide(seg, tree, radius, omega_dt, alive, *a, **kw):
            if seg is not obs.sim.seg:
                return orig_collide(seg, tree, radius, omega_dt, alive, *a, **kw)
            pid0, kind0, pos0 = seg.plate_id.copy(), seg.kind.copy(), seg.pos.copy()
            out = orig_collide(seg, tree, radius, omega_dt, alive, *a, **kw)
            obs._on_collide(seg, pid0, kind0, pos0, np.asarray(out[0]), np.asarray(out[1]))
            return out

        orig_rift = intraplate.rift

        def rift(sim_, rng, *a, **kw):
            if sim_ is not obs.sim:
                return orig_rift(sim_, rng, *a, **kw)
            com0 = sim_.plates.com.copy()
            seg = sim_.seg
            ext0 = np.bincount(seg.plate_id, weights=seg.ext, minlength=sim_.plates.P)
            ev = orig_rift(sim_, rng, *a, **kw)
            obs._on_rift(sim_, ev, com0, ext0)
            return ev

        seg = sim.seg
        orig_compress, orig_append = seg.compress, seg.append

        def compress(keep):
            k = np.array(keep, dtype=bool)          # a copy, read before the cloud changes
            orig_compress(keep)
            if not k.all():
                obs.born_step = obs.born_step[k]
                obs.born_kind = obs.born_kind[k]

        def append(other):
            orig_append(other)
            if other.M:
                obs.born_step = np.concatenate([obs.born_step, np.full(other.M, obs.sim.step_index, np.int32)])
                obs.born_kind = np.concatenate([obs.born_kind, np.asarray(other.kind, np.int8)])

        tect_run.collide = collide
        intraplate.rift = rift
        seg.compress = compress
        seg.append = append
        self._restore = [(tect_run, "collide", orig_collide), (intraplate, "rift", orig_rift)]
        return self

    def detach(self) -> None:
        for mod, name, orig in reversed(self._restore):
            setattr(mod, name, orig)
        self._restore = []
        for name in ("compress", "append"):
            self._seg.__dict__.pop(name, None)

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    # -- per-step bookkeeping -------------------------------------------------
    def _reset_window(self) -> None:
        self.win = {"steps": 0, "coll_cc": 0, "coll_oc": 0, "coll_oo": 0, "arc_births": 0, "births_rift": 0,
                    "births_split": 0, "births_other": 0, "deaths": 0, "rifts": 0, "spawned": 0, "gap_cells": 0}

    def _on_collide(self, seg, pid0, kind0, pos0, losers, survivors) -> None:
        lk, sk = kind0[losers], kind0[survivors]
        w = self.win
        w["coll_cc"] += int(((lk == CONTINENTAL) & (sk == CONTINENTAL)).sum())
        w["coll_oc"] += int(((lk != sk)).sum())
        w["coll_oo"] += int(((lk == OCEANIC) & (sk == OCEANIC)).sum())
        # arcs: crust that went into the call oceanic and came out continental
        w["arc_births"] += int(((kind0 == OCEANIC) & (seg.kind == CONTINENTAL)).sum())
        if not self._active_rifts or losers.size == 0:
            return
        k = self.sim.step_index
        pl, ps = pid0[losers], pid0[survivors]
        for r in list(self._active_rifts):
            if k >= r["step"] + RIFT_WINDOW:
                self._active_rifts.remove(r)
                continue
            a, b = r["target"], r["new"]
            pair = ((pl == a) & (ps == b)) | ((pl == b) & (ps == a))
            n = int(pair.sum())
            if not n:
                continue
            far = pos0[losers[pair]] @ np.asarray(r["centre"]) < 0.0      # > 90 deg from the rift centre
            r["coll"] += n
            r["coll_far"] += int(far.sum())
            r["coll_cc"] += int(((lk[pair] == CONTINENTAL) & (sk[pair] == CONTINENTAL)).sum())

    def _on_rift(self, sim, ev, com0, ext0) -> None:
        seg = sim.seg
        tot = float(seg.ext.sum())
        ext_now = np.bincount(seg.plate_id, weights=seg.ext, minlength=sim.plates.P)
        cont_now = np.bincount(seg.plate_id[seg.kind == CONTINENTAL], weights=seg.ext[seg.kind == CONTINENTAL],
                               minlength=sim.plates.P)
        for a, b in ev.get("pairs", ()):
            a, b = int(a), int(b)
            c = com0[a] if a < com0.shape[0] else np.zeros(3)
            if np.linalg.norm(c) < 1e-9:                     # as _rift_one: the halves' mean position
                m = (seg.plate_id == a) | (seg.plate_id == b)
                c = seg.pos[m].mean(axis=0)
            c = c / max(float(np.linalg.norm(c)), 1e-12)
            rec = {"step": int(sim.step_index), "target": a, "new": b, "centre": [float(x) for x in c],
                   "area_before": _f(ext0[a] / tot) if a < ext0.shape[0] else None,
                   "area_a": _f(ext_now[a] / tot), "area_b": _f(ext_now[b] / tot),
                   "cont_a": _f(cont_now[a] / max(ext_now[a], 1e-30)), "cont_b": _f(cont_now[b] / max(ext_now[b], 1e-30)),
                   "coll": 0, "coll_far": 0, "coll_cc": 0}
            self.rifts.append(rec)
            self._active_rifts.append(rec)
            self._step_rifts.append(rec)

    def after_step(self) -> None:
        """Book what the step just did: plate births by cause and deaths."""
        sim = self.sim
        if sim.seg is not self._seg or self.born_step.size != sim.seg.M:
            raise RuntimeError("the segment cloud was replaced or reindexed outside compress/append; "
                               "the scorecard's per-segment arrays no longer line up with it")
        k = sim.step_index - 1                   # the step that just ran
        pl = sim.plates
        P = int(pl.P)
        w = self.win
        w["steps"] += 1
        info = sim.stats[-1] if sim.stats else {}
        w["spawned"] += int(info.get("spawned", 0))
        w["gap_cells"] += int(info.get("gap_cells", 0))
        rift_new = [r["new"] for r in self._step_rifts]
        w["rifts"] += len(rift_new)
        self._step_rifts = []
        if any(ev.get("event") == "reorganise" for ev in sim.events):
            # every plate is new: the old ones are retired dead and the new ones born
            for q, (b, c) in self.plate_birth.items():
                if b >= 0:
                    self._retired.append((b, self.plate_death.get(q, k), c))
            w["deaths"] += int(self._alive_prev.sum())
            self.plate_birth = {q: (k, "other") for q in range(P)}
            self.plate_death = {}
            w["births_other"] += int(pl.alive.sum())
        else:
            split_n = sum(int(ev.get("split", 0)) for ev in sim.events
                          if ev.get("event") == "split" and isinstance(ev.get("split"), (int, np.integer)))
            new_ids = list(range(self._P_prev, P))
            rift_set = set(rift_new)
            rest = [q for q in new_ids if q not in rift_set]
            for q in new_ids:
                if q in rift_set:
                    cause = "rift"
                elif rest.index(q) < split_n:
                    cause = "split"
                else:
                    cause = "other"
                self.plate_birth[q] = (k, cause)
                w["births_" + cause] += 1
            prev = np.zeros(P, bool)
            prev[:min(P, self._alive_prev.size)] = self._alive_prev[:P]
            prev[self._P_prev:] = True               # born this step: a death if they are already gone
            for q in np.flatnonzero(prev & ~pl.alive):
                self.plate_death[int(q)] = k
                w["deaths"] += 1
            for q in np.flatnonzero(pl.alive[:min(P, self._alive_prev.size)] & ~self._alive_prev[:P]):
                self.plate_death.pop(int(q), None)   # a dead id given crust again (should not happen)
        self._alive_prev = pl.alive.copy()
        self._P_prev = P

    # -- the sample -----------------------------------------------------------
    def sample(self) -> dict:
        """One row: every metric at the current step, and the window since
        the previous row."""
        t0 = time.time()
        sim, seg, pl, tp = self.sim, self.sim.seg, self.sim.plates, self.sim.tp
        k = int(sim.step_index)
        sp, R_km, myr = self.spacing, self.R_km, self.myr
        cms = R_km * 0.1 / myr                    # rad/step -> cm/yr
        tree = build_tree(seg)
        pid = seg.plate_id.astype(np.int64)
        P = int(pl.P)
        cont = seg.kind == CONTINENTAL
        ocean = ~cont
        ext = seg.ext.astype(np.float64)
        ext_tot = float(ext.sum())
        row: dict = {"step": k, "myr": _f(k * myr), "M": int(seg.M)}

        # ---- crust books --------------------------------------------------
        bed, c = tect_run.frame_bed(sim, tree, with_c=True)
        area = sim.grid.interior_cell_area.astype(np.float64)
        A = float(area.sum())
        land = bed > 0.0
        rc = c > 0.5
        cvol = float((ext[cont] * seg.thickness[cont]).sum())
        row["books"] = {
            "cont_extent_share": _f(ext[cont].sum() / ext_tot),
            "cont_volume": _f(cvol),
            "cont_volume_km3": _f(cvol * R_km ** 2 * THICKNESS_KM),
            "crust_mass": _f(seg.crust_mass()),
            "total_mass": _f(seg.total_mass()),
            "n_continental": int(cont.sum()),
            "rendered_cont_share": _f(area[rc].sum() / A),
            "land_share": _f(area[land].sum() / A),
            "land_of_rendered_cont": _f(area[land & rc].sum() / max(area[rc].sum(), 1e-30)),
            "ledger": {kk: _f(v) for kk, v in sim.ledger.items()},
        }

        # ---- continents (landmasses on the cloud) -------------------------
        row["continents"] = self._landmasses(seg, cont, ext)

        # ---- plates -------------------------------------------------------
        A_p = np.bincount(pid, weights=ext, minlength=P)[:P]
        C_p = np.bincount(pid[cont], weights=ext[cont], minlength=P)[:P]
        share = A_p / ext_tot
        cshare = C_p / np.maximum(A_p, 1e-30)
        alive = np.flatnonzero(pl.alive[:P] & (A_p > 0))
        order = alive[np.argsort(-share[alive], kind="stable")]
        v_seg = np.linalg.norm(np.cross(pl.omega[pid], seg.pos), axis=1)          # rad/step
        v2_p = np.bincount(pid, weights=ext * v_seg ** 2, minlength=P)[:P]
        v_rms_p = np.sqrt(v2_p / np.maximum(A_p, 1e-30))
        speed = pl.speeds()[:P]
        capped = speed >= 0.999 * float(sim.max_omega) if sim.max_omega > 0 else np.zeros(P, bool)
        shown = order[share[order] >= 1e-3]
        row["plates"] = {
            "alive": int(pl.n_alive()),
            "n_ge_1pct": int((share[alive] >= MAJOR_SHARE).sum()),
            "largest_share": _f(share[order[0]]) if order.size else None,
            "top7_share": _f(share[order[:7]].sum()) if order.size else None,
            "capped_share": _f(capped[alive].mean()) if alive.size else None,
            "shares": [round(float(x), 5) for x in share[shown]],
            "cont_share": [round(float(x), 4) for x in cshare[shown]],
            "speed_cmyr": [round(float(x * cms), 3) for x in v_rms_p[shown]],
            "speed_kmstep": [round(float(x * R_km), 3) for x in v_rms_p[shown]],
            "n_below_0p1pct": int(order.size - shown.size),
        }

        # ---- boundaries ---------------------------------------------------
        B = self._boundary_pairs(seg, pl, tree)
        trench_share = np.zeros(P)
        bnd_km = {}
        if B is not None:
            i, j, vn, obl, down = B["i"], B["j"], B["vn"], B["obl"], B["down"]
            ki, kj = seg.kind[i], seg.kind[j]
            cc = (ki == CONTINENTAL) & (kj == CONTINENTAL)
            conv, div = obl > OBLIQUITY, obl < -OBLIQUITY
            trans = ~(conv | div)
            seg_len = sp * R_km / 2.0                 # each boundary has a segment on either side

            def length(mask):
                return _f(np.unique(np.concatenate([i[mask], j[mask]])).size * seg_len)

            bnd_km = {"total_km": length(np.ones(i.size, bool)), "convergent_km": length(conv),
                      "divergent_km": length(div), "transform_km": length(trans),
                      "subduction_km": length(conv & ~cc), "collision_cc_km": length(conv & cc),
                      "convergent_any_km": length(vn > 0.0), "pairs": int(i.size)}
            # trench share: the plate's boundary segments that are the *subducting* side of
            # an approaching pair (threshold 0 -- only the sign of the approach, not its
            # speed, so a fast plate does not get more trench by being fast)
            bnd_seg = np.unique(np.concatenate([i, j]))
            app = vn > 0.0
            sub_seg = np.unique(np.concatenate([i[app & (down == 0)], j[app & (down == 1)]]))
            nb = np.bincount(pid[bnd_seg], minlength=P)[:P]
            ns = np.bincount(pid[sub_seg], minlength=P)[:P]
            trench_share = ns / np.maximum(nb, 1)
            conv0_seg = np.zeros(seg.M, bool)
            conv0_seg[i[app]] = True
            conv0_seg[j[app]] = True
        else:
            conv0_seg = np.zeros(seg.M, bool)
        row["bnd"] = bnd_km

        # ---- kinematics ---------------------------------------------------
        wgt = ext * (4.0 * math.pi / ext_tot)
        v_mean = float((wgt * v_seg).sum() / (4.0 * math.pi))
        v_rms = math.sqrt(float((wgt * v_seg ** 2).sum() / (4.0 * math.pi)))
        om = pl.omega[pid]
        nr = (3.0 / (8.0 * math.pi)) * np.sum(wgt[:, None] * (om - np.sum(om * seg.pos, axis=1, keepdims=True) * seg.pos), axis=0)
        v_nr = float(np.linalg.norm(nr)) * math.sqrt(2.0 / 3.0)
        big = alive[share[alive] >= CORR_SHARE]
        oce = big[cshare[big] < 0.2]
        con = big[cshare[big] > 0.5]
        row["kin"] = {
            "v_mean_kmstep": _f(v_mean * R_km), "v_mean_cmyr": _f(v_mean * cms),
            "v_median_cmyr": _wq(v_seg * cms, ext, 0.5),
            "v_rms_cmyr": _f(v_rms * cms),
            "v_max_cmyr": _f(v_rms_p[alive].max() * cms) if alive.size else None,
            "v_ocean_median_cmyr": _median(v_rms_p[oce] * cms), "n_ocean_plates": int(oce.size),
            "v_cont_median_cmyr": _median(v_rms_p[con] * cms), "n_cont_plates": int(con.size),
            "net_rotation_ratio": _f(v_nr / max(v_rms, 1e-30)),
            "spearman_speed_trench": _spearman(v_rms_p[big], trench_share[big]),
            "spearman_speed_trench_ocean": _spearman(v_rms_p[oce], trench_share[oce]),
            "spearman_speed_cont": _spearman(v_rms_p[big], cshare[big]),
            "n_corr_plates": int(big.size),
        }

        # ---- coastal continental segments ---------------------------------
        row["coast"] = self._coast(seg, pl, cont)

        # ---- collisions, births and deaths in the window --------------------
        w = dict(self.win)
        n_c = w["coll_cc"] + w["coll_oc"] + w["coll_oo"]
        st = max(w["steps"], 1)
        row["window"] = {
            **w,
            "coll_total": n_c,
            "coll_per_step": _f(n_c / st),
            "cc_share": _f(w["coll_cc"] / n_c) if n_c else None,
            "oc_share": _f(w["coll_oc"] / n_c) if n_c else None,
            "oo_share": _f(w["coll_oo"] / n_c) if n_c else None,
            "births": w["births_rift"] + w["births_split"] + w["births_other"],
        }
        self._reset_window()

        # ---- ocean floor ----------------------------------------------------
        if ocean.any():
            age, wo = seg.age[ocean], ext[ocean]
            mean = float((age * wo).sum() / wo.sum())
            row["ocean"] = {
                "mean_age_steps": _f(mean), "mean_age_myr": _f(mean * myr),
                "median_over_mean": _f(weighted_quantile(age, wo, 0.5) / max(mean, 1e-30)),
                "p90_over_mean": _f(weighted_quantile(age, wo, 0.9) / max(mean, 1e-30)),
                "max_over_mean": _f(age.max() / max(mean, 1e-30)),
            }
        else:
            row["ocean"] = {}

        # ---- arcs -------------------------------------------------------------
        row["arcs"] = self._arcs(seg, tree, cont, ext, ext_tot, conv0_seg, bed, k)

        # ---- hypsometry on the tect grid, metres ----------------------------
        scale = tect_run.metres_per_unit(bed, area, tp, sim.spacing, sim.params.R_planet)
        bm = bed * scale
        lw, ow = area[land], area[~land]
        row["hyps"] = {
            "scale_m_per_unit": _f(scale),
            "land_median_m": _wq(bm[land], lw, 0.5), "ocean_median_m": _wq(bm[~land], ow, 0.5),
            "land_mean_m": _f((bm[land] * lw).sum() / lw.sum()) if land.any() else None,
            "land_gt1km_pct": _f(100.0 * lw[bm[land] > 1000.0].sum() / lw.sum()) if land.any() else None,
            "land_gt2km_pct": _f(100.0 * lw[bm[land] > 2000.0].sum() / lw.sum()) if land.any() else None,
            "max_m": _f(bm.max()), "min_m": _f(bm.min()),
        }
        row["sample_seconds"] = _f(time.time() - t0)
        row["wall_seconds"] = _f(time.time() - self._t0)
        self.samples.append(row)
        return row

    # -- metric blocks --------------------------------------------------------
    def _landmasses(self, seg, cont, ext) -> dict:
        """Largest connected landmass as a share of the continental extent and the
        number of landmasses holding at least 1 % of it (continental segments
        within LANDMASS_LINK spacings are one landmass, as scripts/supercontinent.py)."""
        idx = np.flatnonzero(cont)
        if idx.size == 0:
            return {"largest_share": None, "n_ge_1pct": 0, "components": 0}
        pairs = cKDTree(seg.pos[idx]).query_pairs(_chord(LANDMASS_LINK * self.spacing), output_type="ndarray")
        n = idx.size
        g = coo_matrix((np.ones(pairs.shape[0], np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if pairs.size \
            else coo_matrix((n, n))
        ncomp, lab = connected_components(g, directed=False)
        per = np.bincount(lab, weights=ext[idx], minlength=ncomp)
        tot = float(per.sum())
        return {"largest_share": _f(per.max() / tot), "n_ge_1pct": int((per / tot >= 0.01).sum()),
                "components": int(ncomp)}

    def _boundary_pairs(self, seg, pl, tree):
        """Every pair of segments of different plates within BOUNDARY_LINK
        spacings, with the relative velocity at their midpoint: ``vn`` its
        component along the line of centres (> 0 approaching), ``obl`` that
        over its magnitude, and ``down`` which side would subduct (0 = i,
        1 = j, -1 = continent-continent), by collide's own rule: the oceanic
        member of a mixed pair, else the denser, else the plate whose crust
        is older along that boundary (``plate_pair_polarity``)."""
        pairs = tree.query_pairs(_chord(BOUNDARY_LINK * self.spacing), output_type="ndarray")
        if pairs.size == 0:
            return None
        pid = seg.plate_id.astype(np.int64)
        pairs = pairs[pid[pairs[:, 0]] != pid[pairs[:, 1]]]
        if pairs.shape[0] == 0:
            return None
        i, j = pairs[:, 0], pairs[:, 1]
        xi, xj = seg.pos[i], seg.pos[j]
        mid = xi + xj
        mid /= np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-12)
        u = xj - xi
        u -= np.sum(u * mid, axis=1, keepdims=True) * mid
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        vrel = np.cross(pl.omega[pid[i]], mid) - np.cross(pl.omega[pid[j]], mid)
        vn = np.sum(vrel * u, axis=1)
        vm = np.linalg.norm(vrel, axis=1)
        obl = np.where(vm > 1e-30, vn / np.maximum(vm, 1e-30), 0.0)
        ki, kj = seg.kind[i], seg.kind[j]
        down = np.full(i.size, -1, np.int8)
        mixed = ki != kj
        down[mixed & (ki == OCEANIC)] = 0
        down[mixed & (kj == OCEANIC)] = 1
        oo = (ki == OCEANIC) & (kj == OCEANIC)
        di, dj = seg.density[i], seg.density[j]
        heavy_i = oo & (di > dj + DENSITY_EPS)
        heavy_j = oo & (dj > di + DENSITY_EPS)
        down[heavy_i] = 0
        down[heavy_j] = 1
        tie = oo & ~heavy_i & ~heavy_j
        if tie.any():
            ids, cid = np.unique(pid, return_inverse=True)       # compact ids: P is in the thousands late on
            pol = plate_pair_polarity(cid.astype(np.int64), seg.age, seg.kind, np.ascontiguousarray(pairs), ids.size)
            ci, cj = cid[i], cid[j]
            down[tie & (pol[ci, cj] == 1)] = 0
            down[tie & (pol[ci, cj] != 1)] = 1
        return {"i": i, "j": j, "vn": vn, "obl": obl, "down": down}

    def _coast(self, seg, pl, cont) -> dict:
        """Continental segments with an oceanic one within COAST_LINK spacings,
        by what the nearest such ocean floor is doing: on the same plate
        (a passive margin), on another plate and approaching (an active,
        converging margin), or on another plate and not (active, other)."""
        ci, oi = np.flatnonzero(cont), np.flatnonzero(~cont)
        if ci.size == 0 or oi.size == 0:
            return {"n": 0}
        d, jj = cKDTree(seg.pos[oi]).query(seg.pos[ci], k=1, distance_upper_bound=_chord(COAST_LINK * self.spacing))
        m = np.isfinite(d)
        a, b = ci[m], oi[jj[m]]
        n = int(a.size)
        if n == 0:
            return {"n": 0}
        pid = seg.plate_id
        same = pid[a] == pid[b]
        va = np.cross(pl.omega[pid[a]], seg.pos[a])
        vb = np.cross(pl.omega[pid[b]], seg.pos[b])
        app = np.sum((va - vb) * (seg.pos[b] - seg.pos[a]), axis=1) > 0.0
        return {"n": n, "passive_share": _f(same.mean()), "active_conv_share": _f((~same & app).mean()),
                "active_other_share": _f((~same & ~app).mean())}

    def _arcs(self, seg, tree, cont, ext, ext_tot, conv0_seg, bed, k) -> dict:
        arc = cont & (self.born_kind == OCEANIC)
        void = cont & (self.born_kind == CONTINENTAL) & (self.born_step >= 0)
        proxy = cont & (seg.age < float(k) - 0.5)
        out = {"n": int(arc.sum()), "planet_share": _f(ext[arc].sum() / ext_tot),
               "cont_share": _f(ext[arc].sum() / max(ext[cont].sum(), 1e-30)),
               "n_void_fill": int(void.sum()), "void_fill_share": _f(ext[void].sum() / ext_tot),
               "n_field_proxy": int(proxy.sum())}
        ai = np.flatnonzero(arc)
        if ai.size:
            out["on_converging_share"] = _f(conv0_seg[ai].mean())
            pid = seg.plate_id
            balls = tree.query_ball_point(seg.pos[ai], _chord(STRANDED * self.spacing))
            stranded = np.fromiter((not np.any(pid[np.asarray(nb, np.int64)] != pid[q]) for q, nb in zip(ai, balls)),
                                   dtype=bool, count=ai.size)
            out["stranded_share"] = _f(stranded.mean())
        else:
            out["on_converging_share"] = None
            out["stranded_share"] = None
        out.update(self._land_components(seg, tree, cont, arc, bed))
        return out

    def _land_components(self, seg, tree, cont, arc, bed) -> dict:
        """Land (bed > 0) on the tect grid in connected pieces across the cube
        faces -- cells whose centres are within 1.6 nominal cells are joined,
        which takes in the 8-neighbours on a face and across a face edge -- by
        area bin, and how many are arc-dominated (over half their area nearest
        to arc crust) or touch no other continental crust at all."""
        grid = self.sim.grid
        land = (bed > 0.0).ravel()
        li = np.flatnonzero(land)
        out = {"land_components": 0}
        for nm in AREA_BIN_NAMES:
            out["land_" + nm] = 0
        out.update({"arc_dominated": 0, "arc_only": 0, "arc_land_km2": 0.0})
        if li.size == 0:
            return out
        cen = interior_centers_flat(grid)[li].astype(np.float64)
        area_km2 = grid.interior_cell_area.astype(np.float64).ravel()[li] / 1e6
        pairs = cKDTree(cen).query_pairs(1.6 * (math.pi / 2.0) / grid.N, output_type="ndarray")
        n = li.size
        g = coo_matrix((np.ones(pairs.shape[0], np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if pairs.size \
            else coo_matrix((n, n))
        ncomp, comp = connected_components(g, directed=False)
        ca = np.bincount(comp, weights=area_km2, minlength=ncomp)
        hist = np.histogram(ca, bins=np.asarray(AREA_BINS))[0]
        _, near = tree.query(cen, k=1)
        arc_c = arc[near]
        other_c = cont[near] & ~arc_c
        arc_a = np.bincount(comp, weights=area_km2 * arc_c, minlength=ncomp)
        touches = np.bincount(comp, weights=other_c.astype(np.float64), minlength=ncomp) > 0
        dom = arc_a > 0.5 * ca
        out["land_components"] = int(ncomp)
        for nm, h in zip(AREA_BIN_NAMES, hist):
            out["land_" + nm] = int(h)
        out["arc_dominated"] = int(dom.sum())
        out["arc_only"] = int((dom & ~touches).sum())
        out["arc_land_km2"] = _f(arc_a.sum())
        return out

    # -- the end ----------------------------------------------------------------
    def final(self) -> dict:
        """Run totals: plate lifetimes and births by cause, and the rifts."""
        k = int(self.sim.step_index)
        born = {q: v for q, v in self.plate_birth.items() if v[0] >= 0}
        old = [d - b for b, d, _ in self._retired]
        life_dead = np.array([self.plate_death[q] - born[q][0] for q in born if q in self.plate_death] + old, np.float64)
        life_all = np.array([self.plate_death.get(q, k) - born[q][0] for q in born] + old, np.float64)
        causes = {}
        for c in [v[1] for v in born.values()] + [r[2] for r in self._retired]:
            causes[c] = causes.get(c, 0) + 1
        rifts = [dict(r) for r in self.rifts]
        for r in rifts:
            r["far_share"] = _f(r["coll_far"] / r["coll"]) if r["coll"] else None
            r["cc_share"] = _f(r["coll_cc"] / r["coll"]) if r["coll"] else None
        tot = sum(r["coll"] for r in rifts)
        return {
            "steps": k,
            "plates_born": int(len(born) + len(self._retired)),
            "plates_born_rift": int(causes.get("rift", 0)),
            "plates_born_split": int(causes.get("split", 0)),
            "plates_born_other": int(causes.get("other", 0)),
            "plates_died": int(life_dead.size),
            "lifetime_median_dead": _median(life_dead),
            "lifetime_median_dead_myr": _f(np.median(life_dead) * self.myr) if life_dead.size else None,
            "short_lived_share_dead": _f((life_dead < SHORT_LIFE).mean()) if life_dead.size else None,
            "lifetime_median_all": _median(life_all),
            "short_lived_share_all": _f((life_all < SHORT_LIFE).mean()) if life_all.size else None,
            "rifts": int(len(rifts)),
            "rift_far_share": _f(sum(r["coll_far"] for r in rifts) / tot) if tot else None,
            "rift_far_share_mean": _f(np.mean([r["far_share"] for r in rifts if r["far_share"] is not None]))
            if any(r["far_share"] is not None for r in rifts) else None,
            "rift_cc_share": _f(sum(r["coll_cc"] for r in rifts) / tot) if tot else None,
            "rift_halves_min_area_median": _median([min(r["area_a"], r["area_b"]) for r in rifts]),
            **hemisphere_rift(rifts),
            "rift_list": rifts,
        }

    def report(self) -> dict:
        return {"myr_per_step": self.myr, "R_km": self.R_km, "spacing_rad": self.spacing,
                "spacing_km": self.spacing * self.R_km, "samples": self.samples, "final": self.final()}


def hemisphere_rift(rifts: list[dict]) -> dict:
    """The first rift through a plate covering more than half the planet --
    on the shipped start, the supercontinent plate.  A cut longer than 180 deg
    *must* partly converge (the normal component of a rigid relative rotation
    along a great circle goes as cos), so the share of its halves' collisions
    > 90 deg from its centre is the number to watch; the later, smaller rifts
    average it away in ``rift_far_share``.  ``rift_hemi_n`` counts such rifts."""
    big = [x for x in rifts if (x.get("area_before") or 0.0) > 0.5]
    r = next((x for x in big if x.get("coll")), None)
    out = {"rift_hemi_n": len(big), "rift_hemi_step": None, "rift_hemi_far_share": None,
           "rift_hemi_cc_share": None, "rift_hemi_area_before": None}
    if r is not None:
        out.update({"rift_hemi_step": r["step"], "rift_hemi_far_share": _f(r["coll_far"] / r["coll"]),
                    "rift_hemi_cc_share": _f(r["coll_cc"] / r["coll"]), "rift_hemi_area_before": r.get("area_before")})
    return out


def observe(sim, steps: int, every: int, myr_per_step: float | None = None, log=None) -> dict:
    """Step ``sim`` ``steps`` times under an :class:`Observer`, sampling at
    step 0, every ``every`` steps and at the end; returns the report."""
    obs = Observer(sim, myr_per_step)
    with obs:
        obs.sample()
        for i in range(int(steps)):
            sim.step()
            obs.after_step()
            if sim.step_index % max(int(every), 1) == 0 or i == int(steps) - 1:
                row = obs.sample()
                if log is not None:
                    log(headline(row))
    return obs.report()


def headline(row: dict) -> str:
    """One progress line per sample."""
    b, p, k, w = row["books"], row["plates"], row["kin"], row["window"]

    def g(x, f="{:.3f}"):
        return "-" if x is None else f.format(x)

    return (f"[scorecard] step {row['step']:5d} ({g(row['myr'], '{:.0f}')} My) M={row['M']} "
            f"cont_ext {g(b['cont_extent_share'])} rendered {g(b['rendered_cont_share'])} land {g(b['land_share'])} | "
            f"plates {p['alive']} (>=1% {p['n_ge_1pct']}, top {g(p['largest_share'], '{:.2f}')}) | "
            f"v {g(k['v_mean_cmyr'], '{:.1f}')} cm/yr ocean {g(k['v_ocean_median_cmyr'], '{:.1f}')} "
            f"cont {g(k['v_cont_median_cmyr'], '{:.1f}')} | coll/step {g(w['coll_per_step'], '{:.0f}')} "
            f"cc {g(w['cc_share'], '{:.2f}')} births {w['births']} | largest mass {g(row['continents']['largest_share'], '{:.2f}')} "
            f"| {row['sample_seconds']:.1f}s sample, {row['wall_seconds']:.0f}s")


# ------------------------------------------------------------------------------
# summaries over seeds
# ------------------------------------------------------------------------------
def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> {'a.b': scalar}, lists and non-numbers dropped."""
    out = {}
    for key, v in d.items():
        name = f"{prefix}{key}"
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        elif isinstance(v, bool):
            out[name] = float(v)
        elif isinstance(v, (int, float)) or v is None:
            out[name] = v
    return out


def _stats(vals) -> dict:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)}


def checkpoints_for(steps: int) -> list[int]:
    cps = [s for s in (1000, 2000, 4000, 8000) if s <= steps]
    if steps not in cps:
        cps.append(int(steps))
    return cps


def summarise(runs: list[dict], checkpoints: list[int] | None = None) -> dict:
    """Mean / min / max over seeds of every scalar metric at each checkpoint
    step (the sample taken at exactly that step), and of the run totals."""
    if checkpoints is None:
        checkpoints = checkpoints_for(min(int(r["final"]["steps"]) for r in runs))
    out = {"seeds": [r.get("seed") for r in runs], "checkpoints": {}, "final": {}}
    for cp in checkpoints:
        rows = []
        for r in runs:
            m = [s for s in r["samples"] if s["step"] == cp]
            if m:
                rows.append(flatten(m[0]))
        keys = sorted({k for row in rows for k in row})
        out["checkpoints"][str(cp)] = {k: _stats([row.get(k) for row in rows]) for k in keys}
    # (hemisphere_rift again: a report written before it existed still has the rift list)
    fins = [flatten({**hemisphere_rift(r["final"].get("rift_list", [])),
                     **{kk: vv for kk, vv in r["final"].items() if kk != "rift_list"}}) for r in runs]
    keys = sorted({k for f in fins for k in f})
    out["final"] = {k: _stats([f.get(k) for f in fins]) for k in keys}
    return out


#: Earth's numbers, from the diagnosis brief's "Earth targets" (cited and verified there) and,
#: for the hypsometry, the classic hypsographic curve scripts/hypsometry.py carries.  The
#: scorecard prints them beside the model; they are not tuned against inside this module.
EARTH_TARGETS = {
    "books.cont_extent_share": "~0.40 (continental crust incl. shelves)",
    "books.rendered_cont_share": "~0.40",
    "books.land_share": "0.29",
    "continents.largest_share": ">=0.75 assembled (Pangaea, ~335-175 Ma)",
    "plates.alive": "52 today (Bird 2003); >=10 at ~200 Ma",
    "plates.n_ge_1pct": "~12 today",
    "plates.largest_share": "~0.20 (Pacific)",
    "plates.top7_share": "~0.90",
    "kin.v_median_cmyr": "~4.2 (overall median)",
    "kin.v_ocean_median_cmyr": "7.9-8.1 (slab-attached)",
    "kin.v_cont_median_cmyr": "2.8 (>50 % continental)",
    "kin.v_max_cmyr": "limits ~20 ocean / ~10 continent",
    "kin.spearman_speed_cont": "r = -0.77",
    "kin.spearman_speed_trench": "> 0 (slab-attached plates fastest)",
    "kin.net_rotation_ratio": "small",
    "bnd.subduction_km": "42,400 today; 65,500 at 235 Ma",
    "bnd.divergent_km": "~65,000 (ridges)",
    "ocean.mean_age_myr": "64",
    "ocean.median_over_mean": "0.88 (triangular)",
    "ocean.max_over_mean": "~3",
    "arcs.planet_share": "~0.01 (intra-oceanic arc crust)",
    "final.lifetime_median_dead_myr": "small plates live < 10-20 My",
    "hyps.land_median_m": "~350",
    "hyps.ocean_median_m": "~-4070",
    "hyps.land_gt1km_pct": "28.8",
    "hyps.land_gt2km_pct": "13.4",
    "hyps.max_m": "8849",
}


__all__ = ["Observer", "observe", "hemisphere_rift", "summarise", "flatten", "checkpoints_for", "headline", "EARTH_TARGETS"]
