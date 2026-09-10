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
``ural``
    A former orogen. Continent-continent, long dead, worn down to a low
    welt with its root still under it. The Urals, the Appalachians, the
    Caledonides -- the belts that make continental interiors interesting.

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

    def profile(self, x_km: np.ndarray) -> np.ndarray:
        """Height (m) at signed cross-belt distance `x_km`.

        `x` runs from the down-going side (negative, the trench or the
        foreland) through the range to the far side. Outside the belt the
        profile is 0, so a segment beyond it is untouched.
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
    "ural": Orogen("ural", (
        ("foreland_basin", 150.0, -300.0),
        ("fore_slope", 80.0, 800.0),
        ("plateau", 150.0, 1200.0),      # worn down; the root is still there
        ("back_slope", 120.0, 200.0),
    )),
}


def classify(kind_lo: int, kind_su: int, craton_su: int, flat_slab: bool, continental: int) -> str:
    """Which belt a collision builds, from what met what.

    Ocean going under continent makes an arc; a flat slab makes that arc
    wide and low and pushes it inland; continent meeting continent makes a
    plateau with a foreland moat and no arc at all, because there is no
    longer a slab to melt.
    """
    if kind_lo == continental and kind_su == continental:
        return "himalayan"
    if flat_slab:
        return "laramide"
    return "andean"


def age_to_ural(h: np.ndarray, ages: np.ndarray, half_life: float) -> np.ndarray:
    """Relax an orogen's relief toward the `ural` profile as it ages.

    An orogen is only high while it is being built. Once convergence stops
    the range decays -- erosion strips it, the thickened root relaxes -- and
    what survives is a low welt with a deep crustal root under it, which is
    what the Urals and the Appalachians are. Without this, every belt a
    world ever built stays at full height forever and the continents end up
    a mess of ranges of every age at the same elevation.
    """
    return h * np.exp(-np.asarray(ages, dtype=np.float64) / max(float(half_life), 1e-6))


def shape_belt(seg, tree, losers, survivors, alive, spacing_rad: float, R_planet_m: float,
               height_unit_m: float, strength: float, continental: int, knn: int = 48) -> float:
    """Give each collision belt its cross-section. Returns the mass moved.

    Runs *after* the mass-conserving share in
    :func:`~globe.tectonics.collision.spread_collisions`, and only
    redistributes what is already there: thickness is moved from the
    foreland side into the range, following the type's profile. So the belt
    keeps the mass the collision gave it, and gains a shape -- a moat in
    front, a crest, a back slope -- that a symmetric Gaussian cannot have.

    The cross-belt axis comes from the collision itself: the direction from
    the subducted segment to the survivor is the convergence direction, so
    distance measured along it is exactly the profile's `x`.
    """
    if losers.size == 0 or strength <= 0.0:
        return 0.0
    kk = min(int(knn), tree.n)
    _, nb = tree.query(seg.pos[survivors], k=kk, workers=-1)
    nb = np.atleast_2d(nb).reshape(survivors.size, kk)
    moved = 0.0
    km_per_rad = R_planet_m / 1000.0
    for e in range(losers.size):
        su, lo = int(survivors[e]), int(losers[e])
        if not alive[su]:
            continue
        d = seg.pos[su] - seg.pos[lo]
        d = d - seg.pos[su] * float(d @ seg.pos[su])       # tangential
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            continue
        d /= n
        typ = TYPES[classify(int(seg.kind[lo]), int(seg.kind[su]), int(seg.craton[su]), False, continental)]
        idx = nb[e]
        idx = idx[alive[idx]]
        if idx.size < 4:
            continue
        # signed cross-belt distance in km, measured from the survivor
        x = ((seg.pos[idx] - seg.pos[su]) @ d) * km_per_rad
        w = typ.profile(x) / max(typ.crest_m, 1.0)         # -1 .. 1 shape
        if not np.any(w > 0):
            continue
        # zero-sum: take from the moat, give to the range, scaled by what the
        # belt already carries so a young belt is shaped gently
        pos_w = np.clip(w, 0.0, None)
        neg_w = np.clip(-w, 0.0, None)
        take = strength * seg.thickness[idx] * neg_w
        take = np.minimum(take, 0.4 * seg.thickness[idx])
        pot = float(take.sum())
        if pot <= 0.0 or pos_w.sum() <= 0.0:
            continue
        give = pot * pos_w / pos_w.sum()
        dth = give - take
        seg.thickness[idx] = np.maximum(seg.thickness[idx] + dth, 1e-3)
        seg.mass[idx] = seg.thickness[idx] * seg.density[idx]
        moved += pot
    return moved


__all__ = ["Orogen", "TYPES", "classify", "age_to_ural", "shape_belt"]
