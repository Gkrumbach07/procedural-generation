"""Island arcs: arc crust at the volcanic front, and volcanic edifices as points
riding the plates (te/arcs, from the proto/arcs prototype and its critique).

Why two things.  An intra-oceanic island arc is two things at two scales.  The *ridge*
-- 20-35 km of arc crust, 100-200 km wide, its crest 1-3 km under water -- is crust, and
the segment cloud carries it as ocean floor thickened at the volcanic front
(:func:`arc_front`: the overriding plate's share of every ocean-ocean slab goes to the
oceanic segment nearest the point ``arc_front_km`` behind the trench).  The *islands* are
the volcanoes standing on that ridge: cones 10-30 km across, 50-100 km apart, most of them
seamounts and a few breaking the surface (the Marianas: ~1,000 km2 of islands along 2,800
km of arc).  At 160 km between segments no reconstruction of the cloud can show a 20 km
cone -- the splat is one spacing wide by design -- and more segments do not help (the
diagnosis: the splat is resolution-invariant, and 80k segments cost 6-8x the wall time).
So the cones are not crust in the cloud at all: they are a sparse set of edifices, each a
point fixed to the plate it grew on, that finalise stamps into the bedrock at its own size
(:func:`stamp`).

Where volcanoes grow.  The prototype founded a vent at every ocean-ocean subduction,
including the contacts between short-lived sliver plates, while arc crust builds only
where a trench persists, so 69-88 % of its active vents stood more than 100 km from any
arc crust, on -3 to -4 km of abyssal floor.  Here a slab event founds or feeds a vent
only where the trench is *persistent* -- the slab hanging under it (``TectonicSim.slab``,
the mantle-frame field slab pull reads; ``g = min(S / slab_sat_km, 1)``, the ``g`` of
``TectonicSim.slab_info``) is at least ``volc_slab_g`` -- and the overriding segment at
the front carries arc crust.  A vent is fed by the events within ``volc_sep_km`` of it and
a new one is founded only farther than that from every standing vent, so the vents fill
the front at 1-2x the separation (the prototype's 120 km feed radius set its spacing at
175-220 km).  A vent not fed for ``volc_active_my`` is extinct and loses height with
``volc_decay_my``; a vent whose crust is subducted goes with it.

Hotspot volcanoes are the same object fed by a mantle-fixed plume instead of a slab: the
plate drifts over the plume and the chain ages away from it (Hawaii-Emperor), each new
edifice founded once the last has drifted a jittered distance off the plume.

Nothing here moves crust except :func:`arc_front` (volume for volume, so the books do not
see it); the edifices are a reconstruction detail, like ``ranges``, and only finalise and
the diagnostics read them.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ..cubesphere import from_sphere_v
from .plates import _rotate_kernel
from .segments import CONTINENTAL, OCEANIC, greedy_accept

ARC_OCEAN, ARC_CONT, HOTSPOT = 0, 1, 2
#: arc crust thinner than this (thickness units, ~3.5 km above the 7 km ocean floor) is
#: accretion noise, not an arc: the ridge is drawn from it and a vent needs it under it
ARC_MIN_TH = 0.1
#: an edifice below this many metres is gone
MIN_HEIGHT_M = 100.0


def _chord(theta):
    return 2.0 * np.sin(0.5 * np.minimum(theta, math.pi))


def cell_index(grid, pos: np.ndarray) -> np.ndarray:
    """Flat interior index (face, i, j) -> (face * N + i) * N + j of the cell holding each
    unit vector."""
    f, u, v = from_sphere_v(np.asarray(pos, dtype=np.float64).reshape(-1, 3))
    N = grid.N
    i = np.minimum((u * N).astype(np.int64), N - 1)
    j = np.minimum((v * N).astype(np.int64), N - 1)
    return (f * N + i) * N + j


def _flat_centers(grid) -> np.ndarray:
    from .collision import interior_centers_flat

    return interior_centers_flat(grid)


# ------------------------------------------------------------------------------
# the volcanic front
# ------------------------------------------------------------------------------
def front_points(p_lo: np.ndarray, p_su: np.ndarray, theta: float):
    """The volcanic-front point ``theta`` radians behind each trench contact, into the
    overriding plate: from the midpoint of the pair along the tangent from the slab to the
    survivor.  Returns ``(points (n, 3), ok (n,), mid (n, 3), dirn (n, 3))``."""
    c = p_lo + p_su
    c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    d = p_su - p_lo
    d = d - c * np.sum(d * c, axis=1, keepdims=True)
    nd = np.linalg.norm(d, axis=1)
    ok = nd > 1e-12
    d = d / np.maximum(nd, 1e-12)[:, None]
    f = c * math.cos(theta) + d * math.sin(theta)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f, ok, c, d


def front_targets(seg, tree, losers, survivors, alive, theta: float) -> dict:
    """This step's ocean-under-ocean subductions and, for each, the overriding plate's
    oceanic segment nearest its volcanic-front point (among the front point's 8 nearest
    on the step's tree).  ``sel`` indexes ``losers`` / ``survivors``."""
    dead = ~alive[losers]
    sel = np.flatnonzero(dead & (seg.kind[losers] == OCEANIC) & (seg.kind[survivors] == OCEANIC) & alive[survivors])
    out = {"sel": sel, "f": np.zeros((0, 3)), "tgt": np.zeros(0, np.int64), "has": np.zeros(0, bool),
           "mid": np.zeros((0, 3)), "dirn": np.zeros((0, 3))}
    if sel.size == 0:
        return out
    lo, su = losers[sel], survivors[sel]
    f, ok, c, d = front_points(seg.pos[lo], seg.pos[su], theta)
    sel, lo, su, f, c, d = sel[ok], lo[ok], su[ok], f[ok], c[ok], d[ok]
    if sel.size == 0:
        return out
    k = min(8, tree.n)
    _, nb = tree.query(f, k=k)
    nb = np.atleast_2d(nb).reshape(sel.size, k)
    good = alive[nb] & (seg.plate_id[nb] == seg.plate_id[su][:, None]) & (seg.kind[nb] == OCEANIC)
    has = good.any(axis=1)
    tgt = nb[np.arange(sel.size), np.argmax(good, axis=1)]
    out.update(sel=sel, f=f, tgt=tgt, has=has, mid=c, dirn=d)
    return out


def arc_front(seg, ft: dict, losers, survivors, received, share: float, accretion: float) -> float:
    """Arc magmatism where Earth puts it.  Every ocean-floor segment that went down under
    ocean floor this step handed the overriding segment it met a fraction of itself
    (``collide``); ``share`` of that moves on to the front target (:func:`front_targets`)
    as arc crust -- volume for volume (thickness scaled by the two extents), so
    ``sum(ext * mass)`` does not move -- and only the rest is left for the belt profile to
    spread.  ``received`` (what each survivor was handed) is reduced in place to match.
    Returns the volume moved."""
    sel = ft["sel"]
    if sel.size == 0:
        return 0.0
    lo, su, tgt, has = losers[sel], survivors[sel], ft["tgt"], ft["has"]
    if received is not None:
        rth, rm = received
        th_in = np.where(np.isfinite(rth[sel]), rth[sel], accretion * seg.thickness[lo])
        m_in = np.where(np.isfinite(rm[sel]), rm[sel], accretion * seg.mass[lo])
    else:
        th_in, m_in = accretion * seg.thickness[lo], accretion * seg.mass[lo]
    dth = share * np.maximum(th_in, 0.0) * has
    dm = share * np.maximum(m_in, 0.0) * has
    # never more than the survivor holds
    room = np.maximum(seg.thickness[su] - 1e-3, 0.0)
    g = np.minimum(1.0, room / np.maximum(dth, 1e-30))
    dth, dm = dth * g, dm * g
    if not (dth > 0).any():
        return 0.0
    # where the front is the survivor itself (a cloud coarser than the front's distance) the
    # arc crust stays on it; elsewhere it moves on to the front segment
    move = tgt != su
    if move.any():
        s_, t_, a_, b_ = su[move], tgt[move], dth[move], dm[move]
        np.subtract.at(seg.thickness, s_, a_)
        np.subtract.at(seg.mass, s_, b_)
        r = seg.ext[s_] / np.maximum(seg.ext[t_], 1e-30)
        np.add.at(seg.thickness, t_, a_ * r)
        np.add.at(seg.mass, t_, b_ * r)
        for a in (s_, t_):
            seg.density[a] = seg.mass[a] / np.maximum(seg.thickness[a], 1e-12)
    if received is not None:
        # ...and the belt profile spreads only the rest
        rth[sel] = np.maximum(th_in - dth, 0.0)
        rm[sel] = np.maximum(m_in - dm, 0.0)
    return float((dth * seg.ext[su]).sum())


def arc_crust(seg, ocean_base: float) -> np.ndarray:
    """Arc crust per segment, thickness units: the oceanic column above the birth
    thickness (ocean floor never thickens otherwise -- crystallise skips it -- so the
    excess is exactly convergent-margin crust); 0 on continental crust."""
    return np.where(seg.kind == OCEANIC, np.maximum(seg.thickness - float(ocean_base), 0.0), 0.0)


def arc_excess(seg, tp) -> np.ndarray:
    """The arc's relief per segment, bedrock units: its arc crust times (1 - density),
    where that crust is at least ARC_MIN_TH."""
    a = arc_crust(seg, float(tp.oceanic_thickness))
    return np.where(a >= ARC_MIN_TH, a * (1.0 - seg.density), 0.0)


# ------------------------------------------------------------------------------
# the edifices
# ------------------------------------------------------------------------------
class Volcanoes:
    """Edifice markers: ``pos`` (n, 3) unit vectors, ``plate`` (n,) the plate they ride,
    ``kind`` (n,) ARC_OCEAN / ARC_CONT / HOTSPOT, ``born`` / ``fed`` (n,) steps, ``flux``
    (n,) recent magma supply (steradians of slab, decayed over ``volc_active_my``), ``jit``
    (n,) a fixed lognormal size factor, ``peak`` (n,) the flux ratio when last fed, ``src``
    (n,) the plume index (-1 for arcs).  Lengths convert through the tectonic reference
    radius (``TectonicSim.R_km``), times through ``myr_per_step``."""

    FIELDS = ("pos", "plate", "kind", "born", "fed", "flux", "jit", "peak", "src")

    def __init__(self, sim, n_plumes: int = 0):
        tp = sim.tp
        self.tp = tp
        self.R_km = float(sim.R_km)
        self.spacing = float(sim.spacing)
        self.dt_my = max(float(tp.myr_per_step), 1e-9)
        self.rng = sim.params.rng("tectonics", 21)
        self.pos = np.zeros((0, 3))
        self.plate = np.zeros(0, np.int32)
        self.kind = np.zeros(0, np.int8)
        self.born = np.zeros(0)
        self.fed = np.zeros(0)
        self.flux = np.zeros(0)
        self.jit = np.zeros(0)
        self.peak = np.zeros(0)
        self.src = np.zeros(0, np.int32)
        self.sep = float(tp.volc_sep_km) / self.R_km
        self.active_steps = float(tp.volc_active_my) / self.dt_my
        self.hot_active_steps = float(tp.volc_hot_active_my) / self.dt_my
        # the supply an average vent gets at a 5 cm/yr trench: a slab strip one vent
        # separation long converging at 5 cm/yr (50 km/My), held over the activity window
        v_ref = 50.0 * self.dt_my / self.R_km                         # rad/step
        self.flux_ref = v_ref * self.sep * self.active_steps
        self.hot_jit = np.exp(self.rng.normal(0.0, 0.35, size=n_plumes)) if n_plumes else np.zeros(0)
        self.hot_next = self._hot_gap(n_plumes)
        self.events = {"born": 0, "fed": 0, "gated_slab": 0, "gated_arc": 0, "killed_subducted": 0, "killed_faded": 0}

    def _hot_gap(self, n: int) -> np.ndarray:
        j = float(self.tp.volc_hot_jitter)
        return float(self.tp.volc_hot_sep_km) / self.R_km * (1.0 + j * (2.0 * self.rng.random(n) - 1.0))

    # -- state -------------------------------------------------------------
    def __len__(self):
        return self.pos.shape[0]

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS}

    def _keep(self, keep):
        for k in self.FIELDS:
            setattr(self, k, getattr(self, k)[keep])

    def _append(self, pos, plate, kind, step, flux, src, sigma: float = 0.3):
        n = pos.shape[0]
        if n == 0:
            return
        jit = np.exp(self.rng.normal(0.0, sigma, size=n))
        self.pos = np.concatenate([self.pos, pos])
        self.plate = np.concatenate([self.plate, plate.astype(np.int32)])
        self.kind = np.concatenate([self.kind, kind.astype(np.int8)])
        self.born = np.concatenate([self.born, np.full(n, float(step))])
        self.fed = np.concatenate([self.fed, np.full(n, float(step))])
        self.flux = np.concatenate([self.flux, flux])
        self.jit = np.concatenate([self.jit, jit])
        self.peak = np.concatenate([self.peak, flux / max(self.flux_ref, 1e-30)])
        self.src = np.concatenate([self.src, src.astype(np.int32)])
        self.events["born"] += n

    # -- per step ----------------------------------------------------------
    def move(self, plates) -> None:
        """Ride the plates (right after the segments are rotated)."""
        if len(self) == 0:
            return
        ok = self.plate < plates.omega.shape[0]
        if not ok.all():
            self._keep(ok)
        _rotate_kernel(self.pos, self.plate, plates.omega, 1.0)

    def refresh(self, seg, tree, step: int) -> None:
        """Each vent rides the plate of the crust under it (plates split, merge and
        renumber), the slab supply decays, and edifices that have faded below
        MIN_HEIGHT_M, or have no crust under them, are gone.  Call with the step's tree."""
        if len(self) == 0:
            return
        self.flux *= math.exp(-1.0 / max(self.active_steps, 1e-9))
        d, j = tree.query(self.pos, k=1)
        self.plate = seg.plate_id[j].astype(np.int32)
        gone = (self.height(step) < MIN_HEIGHT_M) & ~self.active(step)
        gone |= d > 2.0 * self.spacing
        if gone.any():
            self.events["killed_faded"] += int(gone.sum())
            self._keep(~gone)

    def on_subduction(self, seg, losers, survivors, alive, step: int, ft: dict, g_lo: np.ndarray,
                      tree=None, theta: float = 0.0) -> None:
        """Found and feed arc vents from this step's slab, and drop the vents whose own crust
        went down.  Call before ``seg.compress`` (``losers`` index the pre-collision cloud).
        ``ft`` is :func:`front_targets` of this step (after :func:`arc_front` put the arc
        crust there); ``g_lo`` the slab field's ``g`` at every loser."""
        tp = self.tp
        dead = ~alive[losers]
        sub = dead & (seg.kind[losers] == OCEANIC)
        if not sub.any():
            return
        lo_all = losers[sub]
        if len(self):
            t = cKDTree(seg.pos[lo_all])
            d, j = t.query(self.pos, k=1, distance_upper_bound=0.75 * self.spacing)
            hit = np.isfinite(d)
            hit[hit] = seg.plate_id[lo_all[j[hit]]] == self.plate[hit]
            if hit.any():
                self.events["killed_subducted"] += int(hit.sum())
                self._keep(~hit)
        slab_ok = g_lo >= float(tp.volc_slab_g)
        # ocean arcs: the front target must carry arc crust, at a persistent trench
        sel = ft["sel"]
        pts, plates, kinds, supply = [], [], [], []
        if sel.size:
            ob = float(tp.oceanic_thickness)
            tgt, has = ft["tgt"], ft["has"]
            arc_ok = has & (seg.thickness[tgt] - ob >= ARC_MIN_TH)
            self.events["gated_slab"] += int((~slab_ok[sel]).sum())
            self.events["gated_arc"] += int((slab_ok[sel] & ~arc_ok).sum())
            use = slab_ok[sel] & arc_ok
            if use.any():
                # onto the arc line: the front target (where arc_front put the arc crust),
                # shifted along strike by the event's own offset -- the raw point's
                # across-strike scatter (+-80 km, from the pair geometry) sowed the prototype's
                # first vents over a 300 km band
                f, pt = ft["f"][use], seg.pos[tgt[use]]
                t_ = np.cross(ft["mid"][use], ft["dirn"][use])
                t_ /= np.maximum(np.linalg.norm(t_, axis=1, keepdims=True), 1e-12)
                off = np.sum((f - pt) * t_, axis=1, keepdims=True)
                g = pt + off * t_
                g /= np.linalg.norm(g, axis=1, keepdims=True)
                lo_u, su_u = losers[sel[use]], survivors[sel[use]]
                # ...and along the slab segment's own length of trench: a segment is ~160 km of
                # strike, so one point per slab event put the vents at the cloud's spacing
                # (130-140 km apart, nearest neighbour, on the first Earth runs) instead of the
                # magma's.  Each event feeds or founds one candidate per volc_sep_km of its
                # length, jittered by a third of a separation, and splits its supply over them
                L = np.sqrt(np.maximum(seg.ext[lo_u], 0.0))
                n = np.maximum(1, np.rint(L / self.sep)).astype(np.int64)
                ev = np.repeat(np.arange(g.shape[0]), n)
                kk = np.arange(ev.size) - np.repeat(np.cumsum(n) - n, n)
                o = ((kk + 0.5) / n[ev] - 0.5) * L[ev] + (self.rng.random(ev.size) - 0.5) * (2.0 / 3.0) * (L[ev] / n[ev])
                gg = g[ev] + o[:, None] * t_[ev]
                gg /= np.linalg.norm(gg, axis=1, keepdims=True)
                pts.append(gg)
                plates.append(seg.plate_id[su_u][ev])
                kinds.append(np.full(ev.size, ARC_OCEAN, np.int8))
                supply.append(seg.ext[lo_u][ev] / n[ev])
        if bool(tp.volc_continental):
            oc = np.flatnonzero(sub & alive[survivors] & (seg.kind[survivors] == CONTINENTAL) & slab_ok)
            if oc.size:
                f, ok, _, _ = front_points(seg.pos[losers[oc]], seg.pos[survivors[oc]], theta)
                oc = oc[ok]
                pts.append(f[ok])
                plates.append(seg.plate_id[survivors[oc]])
                kinds.append(np.full(oc.size, ARC_CONT, np.int8))
                supply.append(seg.ext[losers[oc]])
        if not pts:
            return
        f = np.concatenate(pts)
        plate = np.concatenate(plates)
        kind = np.concatenate(kinds)
        sup = np.concatenate(supply)
        unfed = np.ones(f.shape[0], dtype=bool)
        stand = (self.kind != HOTSPOT) & ((self.height(step) >= MIN_HEIGHT_M) | self.active(step))
        if stand.any():
            # the feed radius is the separation: an event feeds the nearest standing vent of its
            # plate within volc_sep_km, else founds one (a wider feed radius sets the spacing)
            ia = np.flatnonzero(stand)
            dd, jj = cKDTree(self.pos[ia]).query(f, k=1, distance_upper_bound=_chord(self.sep))
            near = np.isfinite(dd)
            near[near] = self.plate[ia[jj[near]]] == plate[near]
            if near.any():
                tg = ia[jj[near]]
                np.add.at(self.flux, tg, sup[near])
                self.fed[tg] = float(step)
                self.events["fed"] += int(near.sum())
                unfed[near] = False
        if unfed.any():
            cand = f[unfed]
            existing = self.pos[stand] if stand.any() else np.zeros((0, 3))
            acc = greedy_accept(cand, existing, float(_chord(self.sep)))
            if acc.any():
                idx = np.flatnonzero(unfed)[acc]
                self._append(f[idx], plate[idx], kind[idx], step, sup[idx], np.full(idx.size, -1))
        self.note_peak(step)

    def hotspots(self, seg, tree, spots: np.ndarray, step: int) -> None:
        """Feed the edifice standing over every plume (mantle-fixed ``spots``), or found a
        new one once the plate has carried the last a jittered ``volc_hot_sep_km`` off it."""
        n = spots.shape[0]
        if n == 0:
            return
        if self.hot_next.size < n:
            self.hot_next = np.concatenate([self.hot_next, self._hot_gap(n - self.hot_next.size)])
            self.hot_jit = np.concatenate([self.hot_jit, np.exp(self.rng.normal(0.0, 0.35, size=n - self.hot_jit.size))])
        _, j = tree.query(spots, k=1)
        plate = seg.plate_id[j]
        hot = np.flatnonzero(self.kind == HOTSPOT)
        new = []
        for h in range(n):
            mine = hot[(self.src[hot] == h) & (self.plate[hot] == plate[h])]
            if mine.size:
                dd = np.linalg.norm(self.pos[mine] - spots[h], axis=1)
                m = int(np.argmin(dd))
                if dd[m] < _chord(self.hot_next[h]):
                    self.fed[mine[m]] = float(step)
                    continue
            new.append(h)
            self.hot_next[h] = self._hot_gap(1)[0]
        if new:
            new = np.asarray(new)
            self._append(spots[new], plate[new], np.full(new.size, HOTSPOT, np.int8), step, np.zeros(new.size), new,
                         sigma=float(self.tp.volc_hot_size_jitter))

    def note_peak(self, step: int) -> None:
        """Remember each fed arc vent's flux ratio, so an extinct one keeps its size."""
        if len(self) == 0:
            return
        fresh = (step - self.fed) <= 0.5
        self.peak[fresh] = np.maximum(self.flux[fresh], 0.0) / max(self.flux_ref, 1e-30)

    # -- what finalise stamps ----------------------------------------------
    def active(self, step: float) -> np.ndarray:
        """Fed within ``volc_active_my`` (arcs) or ``volc_hot_active_my`` (hotspots)."""
        win = np.where(self.kind == HOTSPOT, self.hot_active_steps, self.active_steps)
        return (float(step) - self.fed) <= win

    def height(self, step: float) -> np.ndarray:
        """Edifice height (m, at Earth's vertical scale) above the ground it stands on."""
        tp = self.tp
        n = len(self)
        if n == 0:
            return np.zeros(0)
        arc = self.kind != HOTSPOT
        q_now = np.sqrt(np.maximum(self.flux, 0.0) / max(self.flux_ref, 1e-30))
        q_pk = np.sqrt(np.maximum(self.peak, 0.0))
        h = np.empty(n)
        # an arc vent is full height (its supply now) while it is fed; once extinct its height
        # is frozen at its last-fed supply and decays.  A plume feeds one edifice at a time and
        # the plate carries it straight off, so a hotspot edifice decays from its last feed
        h_arc = float(tp.volc_height_m) * self.jit * np.clip(np.where(self.active(step), q_now, q_pk), 0.35, 1.6)
        hj = self.hot_jit[np.clip(self.src, 0, max(self.hot_jit.size - 1, 0))] if self.hot_jit.size else np.ones(n)
        h_hot = float(tp.volc_hot_height_m) * hj * self.jit
        since = np.where(arc, np.maximum(float(step) - self.fed - self.active_steps, 0.0),
                         np.maximum(float(step) - self.fed, 0.0)) * self.dt_my
        tau = np.where(arc, float(tp.volc_decay_my), float(tp.volc_hot_decay_my))
        h = np.where(arc, h_arc, h_hot) * np.exp(-since / np.maximum(tau, 1e-9))
        if float(tp.volc_height_max_m) > 0.0:
            h = np.where(arc, np.minimum(h, float(tp.volc_height_max_m)), h)
        if float(tp.volc_hot_height_max_m) > 0.0:
            h = np.where(arc, h, np.minimum(h, float(tp.volc_hot_height_max_m)))
        return h


# ------------------------------------------------------------------------------
# finalise: the ridge at its own width, the cones at their own size
# ------------------------------------------------------------------------------
def ridge_field(seg, tp, grid, e: np.ndarray, scale: float, spacing: float, R_km: float, R_planet_m: float,
                ctree=None) -> tuple[np.ndarray, dict]:
    """The arc ridge at its own width on ``grid`` (metres, (6, N, N)).

    The splat reconstructs the cloud with a kernel one spacing (160 km) wide, so arc crust
    carried by a line of single segments -- which is what an arc is at this resolution --
    comes out as a swell 400 km across at half its column's relief.  Here each segment's
    arc relief ``e`` (bedrock units, :func:`arc_excess`) is laid out as an anisotropic
    Gaussian -- ``arc_ridge_km`` across the arc's strike, 0.8 of the spacing to its arc
    neighbours along it (a line of them sums to a flat crest) -- normalised on the grid to
    hold exactly the segment's volume ``e * ext``.  The across-strike sigma widens where a
    segment's ground is wider than such a ridge could hold at its column's relief, and an
    isolated segment gets a round kernel no narrower than its own ground, so the ridge
    never stands above its column.  The prototype normalised overlapping unit-peak kernels
    as ``sum e g / max(sum g, 1)``: a slope break at ``sum g = 1`` (creases in the
    hillshade) and an isolated segment at 1.9x its volume.  Strike is the principal axis
    of the same-plate arc segments within 1.6 spacings.  Returns ``(field, info)``."""
    N = grid.N
    centers = _flat_centers(grid)
    out = np.zeros(centers.shape[0])
    arc = np.flatnonzero(e > 0.0)
    info = {"arc_segments": int(arc.size), "volume_m_sr": 0.0}
    if arc.size == 0:
        return out.reshape(6, N, N), info
    ctree = cKDTree(centers) if ctree is None else ctree
    area_sr = grid.interior_cell_area.reshape(-1).astype(np.float64) / float(R_planet_m) ** 2
    # never narrower than a cell: a sub-cell sigma piles the volume into one row of cells
    cell = 0.5 * math.pi / N
    sb0 = max(float(tp.arc_ridge_km) / float(R_km), cell)
    pos = seg.pos[arc]
    at = cKDTree(pos)
    nbl = at.query_ball_point(pos, _chord(1.6 * spacing))
    pid = seg.plate_id[arc]
    vol = 0.0
    for r, i in enumerate(arc):
        p = seg.pos[i]
        ext = float(seg.ext[i])
        nb = [k for k in nbl[r] if k != r and pid[k] == pid[r]]
        if nb:
            q = pos[nb] - p
            q = q - np.outer(q @ p, p)
            dist = np.linalg.norm(q, axis=1)
            _, v = np.linalg.eigh(q.T @ q)                    # principal axis = strike
            t = v[:, -1] - p * (v[:, -1] @ p)
            t /= max(np.linalg.norm(t), 1e-12)
            b = np.cross(p, t)
            gap = max(float(np.mean(np.sort(dist)[:2])), 0.5 * math.sqrt(ext))
            sa = max(0.8 * gap, cell)
            sb = max(sb0, ext / (math.sqrt(2.0 * math.pi) * gap))
        else:
            t = b = None
            sa = sb = max(sb0, math.sqrt(ext / (2.0 * math.pi)))
        reach = 3.0 * max(sa, sb)
        cells = np.asarray(ctree.query_ball_point(p, float(_chord(reach))), dtype=np.int64)
        if cells.size == 0:
            continue
        rel = centers[cells] - p
        if t is not None:
            g = np.exp(-0.5 * ((rel @ t) / sa) ** 2 - 0.5 * ((rel @ b) / sb) ** 2)
        else:
            g = np.exp(-0.5 * np.sum(rel * rel, axis=1) / sa ** 2)
        norm = float((g * area_sr[cells]).sum())
        if norm <= 0.0:
            continue
        v_i = float(e[i]) * scale * ext
        np.add.at(out, cells, g * (v_i / norm))
        vol += v_i
    info["volume_m_sr"] = vol
    info["crest_max_m"] = float(out.max())
    return out.reshape(6, N, N), info


def stamp(vol: Volcanoes, grid, cont: np.ndarray, step: float, vfac: float = 1.0, continental: bool = False,
          R_planet_m: float | None = None, ctree=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Cones on ``grid``: each edifice's profile ``H (1 - r / R)^1.5`` over its base
    radius ``R = H / tan(flank)`` (on the tectonic sphere, so a toy body's cones keep their
    footprint relative to the plates), heights times ``vfac`` (the planet's vertical scale
    over Earth's).  Overlapping edifices take the higher of the two (they merge into one
    massif, they do not stack), and **each edifice's own cell is raised to its full H**:
    with 9-11 km cones on 9.77 km cells the cell-centre sample kept a median 0.42-0.58 of the
    peak and 0 at the 10th percentile, so whether a vent became an island was decided by
    where it fell between cell centres.  Only cells off the continental mask ``cont`` unless
    ``continental``.  Returns ``(cone, active cone, kind map, info)`` -- metres (float32),
    the cones of the vents active now, and per cell the kind of the edifice that made it
    (-1 none, else ARC_OCEAN / ARC_CONT / HOTSPOT)."""
    tp = vol.tp
    N = grid.N
    n_cells = 6 * N * N
    cone = np.zeros(n_cells, np.float32)
    cone_act = np.zeros(n_cells, np.float32)
    who = np.full(n_cells, -1, np.int8)
    info = {"volcanoes": len(vol), "stamped": 0}
    if len(vol) == 0:
        return cone.reshape(6, N, N), cone_act.reshape(6, N, N), who.reshape(6, N, N), info
    H_e = vol.height(step)
    act = vol.active(step)
    centers = _flat_centers(grid)
    ctree = cKDTree(centers) if ctree is None else ctree
    cell = cell_index(grid, vol.pos)
    cflat = cont.reshape(-1).astype(bool)
    use = H_e > MIN_HEIGHT_M
    if not continental:
        use &= ~cflat[cell] & (vol.kind != ARC_CONT)
    flank = np.where(vol.kind == HOTSPOT, float(tp.volc_hot_flank_deg), float(tp.volc_flank_deg))
    R_rad = H_e / np.tan(np.radians(flank)) / (vol.R_km * 1000.0)       # on the tectonic sphere
    H = H_e * float(vfac)
    order = np.flatnonzero(use)
    for i in order:
        cells = np.asarray(ctree.query_ball_point(vol.pos[i], float(_chord(R_rad[i]))), dtype=np.int64)
        cells = np.union1d(cells, [cell[i]])
        d = np.arccos(np.clip(centers[cells] @ vol.pos[i], -1.0, 1.0))
        h = (H[i] * np.clip(1.0 - d / max(R_rad[i], 1e-12), 0.0, 1.0) ** 1.5).astype(np.float32)
        h[cells == cell[i]] = np.float32(H[i])
        up = h > cone[cells]
        who[cells[up]] = vol.kind[i]
        np.maximum.at(cone, cells, h)
        if act[i]:
            np.maximum.at(cone_act, cells, h)
    if not continental:
        cone[cflat] = 0.0
        cone_act[cflat] = 0.0
        who[cflat] = -1
    who[cone <= 0.0] = -1
    info["stamped"] = int(order.size)
    info["active"] = int((act & use).sum())
    info["stamped_arc"] = int((use & (vol.kind != HOTSPOT)).sum())
    info["stamped_hotspot"] = int((use & (vol.kind == HOTSPOT)).sum())
    info["H_median_m"] = float(np.median(H[order])) if order.size else 0.0
    return cone.reshape(6, N, N), cone_act.reshape(6, N, N), who.reshape(6, N, N), info


__all__ = ["Volcanoes", "cell_index", "front_points", "front_targets", "arc_front", "arc_crust", "arc_excess", "ridge_field",
           "stamp", "ARC_OCEAN", "ARC_CONT", "HOTSPOT", "ARC_MIN_TH"]
