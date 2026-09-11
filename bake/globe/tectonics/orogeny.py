"""Mountain belts with a cross-section, instead of a symmetric bump.

A collision used to hand the survivor the subducted segment's mass, which
:func:`~globe.tectonics.collision.spread_collisions` then shared over its
neighbours with a Gaussian. That builds a belt of the right *width* and the
wrong *shape*: symmetric, single-peaked, and identical whatever collided
with what. Real orogens are none of those.

They are asymmetric, because subduction has a direction. Across a live
margin, going from the trench inland, you cross an outer-arc ridge, a
forearc basin, the range front, a plateau, then a back slope down into a
back-arc. Across a continent-continent collision you cross a foreland basin
loaded down by the thrust sheets, the range, the plateau, and the far side.
The foreland basin is *below* the surrounding land -- an orogen digs a moat
in front of itself, which a Gaussian cannot represent at all.

And they come in kinds. The four here are the ones the geological literature
names, and they differ in ways that show from orbit:

``andean``
    Ocean under continent. An arc on thick crust: outer-arc ridge, forearc
    basin, a narrow high plateau, back-arc behind. The Andes.
``himalayan``
    Continent into continent. No arc, a deep foreland basin (the Ganges
    plain), and the widest, highest plateau of the four. Tibet.
``laramide``
    Flat-slab subduction: the slab shallows, so deformation jumps far
    inland and the plateau is enormously wide but lower. The Rockies and
    the Colorado Plateau.
``island_arc``
    Ocean under ocean. No continent is involved, so there is no thick crust
    to raise and no plateau: a trench, a narrow volcanic ridge that mostly
    stays under water, and a back-arc basin opening behind it. The Marianas,
    the Aleutians, the Lesser Antilles.
``ural``
    A former orogen. Continent-continent, long dead, worn down to a low
    welt with its root still under it. The Urals, the Appalachians, the
    Caledonides -- the belts that make continental interiors interesting.
    This one is **not** a collision outcome -- see :func:`classify`.

Zone widths and heights come from the reference profiles; each zone is a
(width_km, height_m) pair and the profile is piecewise-linear across them,
so the shape survives at any resolution that can hold the belt at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A zone of an orogen cross-section: how far it runs, and the height it
#: reaches at its far edge. Heights are relative to the local crust, so a
#: negative one is a basin (the foreland moat) rather than a low hill.
Zone = tuple[str, float, float]      # (name, width_km, height_m)


@dataclass(frozen=True)
class Orogen:
    """One belt type, as a cross-section from the down-going side inland."""

    name: str
    zones: tuple[Zone, ...]

    @property
    def width_km(self) -> float:
        return sum(z[1] for z in self.zones)

    @property
    def crest_m(self) -> float:
        return max(z[2] for z in self.zones)

    @property
    def lead_km(self) -> float:
        """How far the profile reaches onto the *down-going* plate."""
        return self.zones[0][1]

    @property
    def reach_km(self) -> float:
        """How far it reaches onto the *overriding* plate."""
        return self.width_km - self.zones[0][1]

    def profile(self, x_km: np.ndarray) -> np.ndarray:
        """Height (m) at signed cross-belt distance `x_km` from the suture.

        `x` runs from the down-going side (negative: the trench, or the
        foreland basin flexed down under the thrust load) through the range
        and out the far side. Outside the belt the profile is 0, so a
        segment beyond it is untouched.
        """
        edges = np.cumsum([0.0] + [z[1] for z in self.zones]) - self.zones[0][1]
        heights = np.array([0.0] + [z[2] for z in self.zones], dtype=np.float64)
        return np.interp(np.asarray(x_km, dtype=np.float64), edges, heights, left=0.0, right=0.0)


#: The reference profiles. Widths in km, heights in m relative to the
#: surrounding crust.
TYPES: dict[str, Orogen] = {
    "andean": Orogen("andean", (
        ("outer_arc_ridge", 100.0, 1500.0),
        ("forearc_basin", 150.0, -500.0),
        ("fore_slope", 100.0, 3000.0),
        ("plateau", 200.0, 6000.0),
        ("back_slope", 200.0, 800.0),
        ("back_arc", 300.0, 0.0),
    )),
    "himalayan": Orogen("himalayan", (
        ("foreland_basin", 300.0, -1500.0),
        ("fore_slope", 150.0, 4000.0),
        ("plateau", 700.0, 5500.0),
        ("back_slope", 200.0, 800.0),
    )),
    "laramide": Orogen("laramide", (
        ("outer_arc_ridge", 100.0, 1200.0),
        ("forearc_basin", 150.0, -400.0),
        ("fore_slope", 100.0, 2500.0),
        ("plateau", 900.0, 3500.0),      # flat slab: wide and lower
        ("back_slope", 200.0, 800.0),
        ("back_arc", 300.0, 0.0),
    )),
    "island_arc": Orogen("island_arc", (
        # An intra-oceanic arc is not a small Andes: there is no continental
        # crust under it, so the whole structure is 2-3 km of relief on a
        # 4 km-deep plain and most of it never breaks the surface. Measured
        # against the Izu-Bonin-Mariana system, which is ~400 km from trench
        # to back-arc with a ridge standing ~2.5 km over the plain.
        ("trench", 100.0, -2000.0),
        ("fore_slope", 80.0, 1200.0),
        ("arc_ridge", 120.0, 2500.0),
        ("back_arc", 200.0, -500.0),
    )),
    "ural": Orogen("ural", (
        ("foreland_basin", 150.0, -300.0),
        ("fore_slope", 80.0, 800.0),
        ("plateau", 150.0, 1200.0),      # worn down; the root is still there
        ("back_slope", 120.0, 200.0),
    )),
}


#: Lower edges of the subducting-slab age histogram the census keeps, in
#: tectonic steps. The interesting boundary is ``flat_slab_age`` (60): a
#: model whose trenches mostly eat crust younger than that builds
#: ``laramide`` belts as the rule rather than the exception Earth has.
SLAB_AGE_BINS = (0, 15, 30, 60, 120, 240, 400)


def classify(kind_lo: int, kind_su: int, flat_slab: bool, continental: int) -> str:
    """Which belt a collision builds, from what met what.

    Ocean going under continent makes an arc; a flat slab makes that arc
    wide and low and pushes it inland; continent meeting continent makes a
    plateau with a foreland moat and no arc at all, because there is no
    longer a slab to melt; ocean under *ocean* makes none of those, because
    every one of them is a structure built on continental crust.

    **``ural`` is deliberately unreachable from here.** It was read as a
    missing branch -- the classifier names four types and returns three --
    but a Ural is not a kind of collision, it is the *end* of one: the
    Urals were a full continent-continent collision, and what makes them
    Urals rather than a Himalaya is 250 My of standing still afterwards.
    Building one at birth would mean a belt that was born dead. The type is
    used, and used where that age belongs: ``TYPES["ural"].crest_m`` is the
    ``orogen_floor_m`` that :func:`relax_orogens` decays every dead belt
    down to, so a former orogen is *produced* by the decay rather than
    classified into existence.

    ``island_arc`` is the branch that really was missing. Measured on an
    Earth-scale run of 93,092 classified collisions, **44 % were ocean on
    ocean** and every one of them built an ``andean`` or ``laramide``
    cross-section -- a 6000 m or 3500 m plateau, 1050 to 1750 km wide, out
    of ocean floor. Those profiles describe an arc standing on thick
    continental crust and there is none under an intra-oceanic arc.
    See docs/crust-audit.md.
    """
    if kind_lo == continental and kind_su == continental:
        return "himalayan"
    if kind_su != continental:
        # no continent on either side: a Mariana, not an Andes. Checked
        # before `flat_slab`, which routes a buoyant slab to a *continental*
        # profile it has no business building here either.
        return "island_arc"
    if flat_slab:
        return "laramide"
    return "andean"


def relax_orogens(seg, baseline_m: float, floor_m: float, height_unit_m: float,
                  rate: float, continental: int) -> float:
    """Wear active orogens down into former ones. Returns the mass shed.

    An orogen is only high while it is being built. Once convergence moves
    elsewhere the range comes down -- erosion strips it, the thickened root
    delaminates -- and what survives is a low welt with its root still under
    it: the Urals, the Appalachians, the Caledonides, the belts that make a
    continental interior interesting rather than flat.

    Without this every belt a world ever built stays at full height forever.
    Measured on the Earth preset at step 1500 with the profile belts in and
    no decay, **31.2 %** of the land stood above 2 km against Earth's ~11 %:
    not one Tibet but fifty, because nothing had ever taken one down.

    Height above ``baseline_m + floor_m`` decays at ``rate`` per step and the
    crust goes back to the mantle, so it is a real sink and shows in the
    ledger. Keying on *height* rather than thickness is what keeps a craton
    safe: a craton is thick but floats at the baseline, so its excess is zero
    and it never decays, while a belt at Tibetan thickness is 5 km above it
    and comes down. The floor is ``TYPES["ural"].crest_m`` -- a
    dead belt settles at the height of a dead belt, not at zero.

    An active belt is fed faster than this takes it away, so the two need no
    coordination and no per-segment clock: convergence keeps a range up, and
    the moment it stops the range starts down.
    """
    if rate <= 0.0:
        return 0.0
    buoy = np.maximum(1.0 - seg.density, 1e-3) * height_unit_m
    excess = seg.thickness * buoy - (baseline_m + floor_m)
    hot = (seg.kind == continental) & (excess > 0.0)
    if not hot.any():
        return 0.0
    dth = float(rate) * excess[hot] / buoy[hot]
    shed = float((dth * seg.density[hot]).sum())
    seg.thickness[hot] -= dth
    seg.mass[hot] = seg.thickness[hot] * seg.density[hot]
    return shed


def shape_belt(seg, tree, losers, survivors, alive, spacing_rad: float, R_planet_m: float,
               height_unit_m: float, strength: float, continental: int, accretion: float = 1.0,
               flat_slab_age: float = 0.0, along_strike: float = 1.5, census: dict | None = None) -> float:
    """Build each collision belt with a cross-section. Returns thickness moved.

    This *replaces* :func:`~globe.tectonics.collision.spread_collisions` for
    the events it handles: the accreted crust is shared out along the
    orogen's profile instead of a Gaussian, so the belt gets its shape from
    the same mass that gives it its height. Two transfers, each conserving
    mass on its own:

    **Accretion.** The survivor holds what the collision just handed it --
    all of a continental partner, ``accretion`` of an oceanic slab, the rest
    having gone back to the mantle. That is taken off the survivor and shared
    over the belt footprint weighted by the *positive* part of the profile,
    so it lands as a range with a plateau rather than as a bump.

    **Fold and thrust.** The down-going plate's upper crust is peeled off
    along the décollement and stacked into the range, which leaves the
    foreland lower and loads it into a flexural moat. That is a zero-sum
    transfer from the negative part of the profile into the positive part,
    relaxing toward the target depth.

    Four things this has to get right, all of which a first cut got wrong.
    Measured with one collision on a lattice of production-spaced segments,
    against the ``himalayan`` profile's −1500 m moat and +5500 m crest:

    **Where the profile is anchored.** ``x = 0`` is the *suture* -- the
    midpoint of the colliding pair -- not the survivor. Anchored on the
    survivor, the moat lands on the overriding plate and the range is pushed
    a zone too far inland; on Earth the Ganges foreland is on India, the
    down-going plate, and Tibet is on Asia, the overriding one.

    **How far it reaches.** A fixed ``knn`` disc is a fixed *radius*, and the
    belts are 500-1750 km wide, so 48 neighbours (624 km at 160 km spacing)
    truncated every profile: the measured crest sat at the disc edge, 636 km
    out, with the plateau and back slope simply missing. The footprint is now
    a ball query at the type's own reach, so ``ural`` costs what ``ural``
    needs.

    **Along strike.** One collision is one *point* on a belt, but a ball
    query is a disc, so an isolated event would paint a 3300 km circle of
    plateau. The profile fades out of the plane of its own collision over
    ``along_strike`` spacings; neighbouring events fill the belt in.

    **What sets the amplitude.** A belt cannot be built by robbing its own
    foreland. Sharing the moat's donation out over the positive part makes
    the crest a function of *footprint geometry* -- a sliver of moat against
    a disc of plateau -- rather than of the profile: measured, a 353 m moat
    and a 230 m crest against an intended 1500 and 5500. Rescaling the two
    sides to balance only moved that to 1066 and 271. The height has to come
    from the accreted crust, which is where it comes from on Earth.
    """
    if losers.size == 0:
        return 0.0
    km_per_rad = R_planet_m / 1000.0
    sigma_km = float(along_strike) * spacing_rad * km_per_rad
    moved = 0.0
    for e in range(losers.size):
        su, lo = int(survivors[e]), int(losers[e])
        if not alive[su]:
            continue  # subducted later in this step; its mass has already moved on
        # belt frame at the suture: `c` up, `d` across (convergence), `t` along strike
        c = seg.pos[su] + seg.pos[lo]
        nc = float(np.linalg.norm(c))
        if nc < 1e-12:
            continue
        c /= nc
        d = seg.pos[su] - seg.pos[lo]
        d = d - c * float(d @ c)
        nd = float(np.linalg.norm(d))
        if nd < 1e-12:
            continue
        d /= nd
        t = np.cross(c, d)
        flat = flat_slab_age > 0.0 and float(seg.age[lo]) < flat_slab_age
        name = classify(int(seg.kind[lo]), int(seg.kind[su]), flat, continental)
        typ = TYPES[name]
        if census is not None:
            # every collision that reaches here, whether or not it finds a
            # footprint to build on: the census counts what the classifier
            # *decided*, which is the thing being audited
            census[name] = census.get(name, 0) + 1
            pair = ("c" if int(seg.kind[lo]) == continental else "o") + \
                   ("c" if int(seg.kind[su]) == continental else "o")
            # `su` is read *after* `collide`, so an ocean-ocean event whose
            # survivor `arc_birth` has just converted counts as `oc` here
            census["pair_" + pair] = census.get("pair_" + pair, 0) + 1
            if pair[0] == "o":
                # the age of the down-going slab is what `flat_slab_age`
                # gates on, so it is the distribution that decides whether
                # `laramide` is the exception it is on Earth or the rule
                h = census.setdefault("slab_age_hist", [0] * len(SLAB_AGE_BINS))
                a = float(seg.age[lo])
                for b in range(len(SLAB_AGE_BINS) - 1, -1, -1):
                    if a >= SLAB_AGE_BINS[b]:
                        h[b] += 1
                        break

        # A belt can be wider than a small planet: `small` has R = 4.1 km
        # against belt reaches of 300-950 km, so theta runs to hundreds of
        # radians and `2 sin(theta/2)` is an arbitrary number in [-2, 2] --
        # a footprint radius that jumps around with the belt *type* for no
        # geometric reason, and can be negative.  Clamping at pi means the
        # query covers the whole sphere once the belt does, which is the
        # honest answer for a world smaller than one orogen.
        theta = min(max(typ.reach_km, typ.lead_km) / km_per_rad, np.pi)
        idx = np.asarray(tree.query_ball_point(c, r=2.0 * np.sin(0.5 * theta)), dtype=np.int64)
        if idx.size == 0:
            continue
        # a belt is built on the overriding plate, out of its own kind of crust
        idx = idx[alive[idx] & (seg.plate_id[idx] == seg.plate_id[su]) & (seg.kind[idx] == seg.kind[su])]
        if idx.size < 3:
            continue
        rel = seg.pos[idx] - c
        x = (rel @ d) * km_per_rad
        y = (rel @ t) * km_per_rad
        prof = typ.profile(x) * np.exp(-0.5 * (y / sigma_km) ** 2)
        up = prof > 0.0
        if not up.any():
            continue
        share = prof[up] / prof[up].sum()

        # 1. accretion: what the collision handed the survivor, laid out as a range
        f = 1.0 if int(seg.kind[lo]) == continental else float(accretion)
        th_in, m_in = f * float(seg.thickness[lo]), f * float(seg.mass[lo])
        # never hand out more than the survivor is holding: clamping the
        # thickness afterwards would conjure the shortfall out of nothing
        if th_in > float(seg.thickness[su]) - 1e-3:
            g = max(float(seg.thickness[su]) - 1e-3, 0.0) / max(th_in, 1e-12)
            th_in, m_in = th_in * g, m_in * g
        if th_in > 0.0:
            seg.thickness[su] -= th_in
            seg.mass[su] -= m_in
            np.add.at(seg.thickness, idx[up], th_in * share)
            np.add.at(seg.mass, idx[up], m_in * share)
            moved += th_in

        # 2. fold and thrust: peel the foreland into the range, zero sum
        low = prof < -1.0
        if strength > 0.0 and low.any():
            buoy = np.maximum(1.0 - seg.density[idx], 1e-3) * height_unit_m  # m per thickness unit
            h = seg.thickness[idx] * buoy
            # measure the moat against the *undeformed* crust around the belt,
            # not against the footprint mean: the range is inside that mean and
            # drags it up, so a growing range shuts its own foreland basin off
            ref = h[np.abs(prof) < 50.0]
            base = float(np.median(ref if ref.size >= 3 else h))
            take = np.zeros(idx.size, dtype=np.float64)
            take[low] = strength * (h[low] - (base + prof[low])) / buoy[low]
            take = np.clip(take, 0.0, 0.4 * seg.thickness[idx])
            pot = float((take * seg.density[idx]).sum())
            if pot > 0.0:
                seg.thickness[idx] -= take
                seg.mass[idx] -= take * seg.density[idx]
                np.add.at(seg.thickness, idx[up], take.sum() * share)
                np.add.at(seg.mass, idx[up], pot * share)
                moved += float(take.sum())
        seg.density[idx] = seg.mass[idx] / np.maximum(seg.thickness[idx], 1e-9)
        seg.density[su] = seg.mass[su] / max(seg.thickness[su], 1e-9)
    return moved


__all__ = ["Orogen", "TYPES", "classify", "relax_orogens", "shape_belt"]
