"""Belt types: the cross-sections, what `classify` returns, and the census.

`orogeny.py` had no tests, which is how `classify` came to build a 6000 m
Andean plateau out of ocean floor for 40 % of a world's collisions without
anything noticing (docs/crust-audit.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from globe.config import WorldParams
from globe.tectonics import orogeny
from globe.tectonics.collision import build_tree
from globe.tectonics.segments import CONTINENTAL, OCEANIC, Segments


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(orogeny.TYPES))
def test_profile_is_zero_outside_the_belt_and_hits_every_zone_height(name):
    t = orogeny.TYPES[name]
    assert t.width_km == pytest.approx(t.lead_km + t.reach_km)
    # `x` is signed from the suture: -lead_km on the down-going side,
    # +reach_km inland.  Outside that the profile must not touch anything.
    outside = np.array([-t.lead_km - 1.0, -1e4, t.reach_km + 1.0, 1e4])
    assert np.all(t.profile(outside) == 0.0)
    # sample the zone boundaries exactly -- a linspace misses them and the
    # crest then reads a few metres low on a steep fore-slope
    edges = np.cumsum([0.0] + [z[1] for z in t.zones]) - t.lead_km
    heights = np.array([0.0] + [z[2] for z in t.zones])
    assert t.profile(edges) == pytest.approx(heights)
    assert heights.max() == pytest.approx(t.crest_m)
    # every type has a basin somewhere: a foreland moat, a forearc basin or
    # (the intra-oceanic arc) a trench and a back-arc
    assert heights.min() < 0.0


def test_ural_is_the_floor_a_dead_belt_settles_at():
    # `orogen_floor_m` is documented as the `ural` crest; if one moves and
    # the other does not, `relax_orogens` stops meaning what it says
    assert orogeny.TYPES["ural"].crest_m == pytest.approx(WorldParams().tectonics.orogen_floor_m)


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------
def test_classify_covers_every_pairing_and_never_builds_a_dead_belt():
    c, o = CONTINENTAL, OCEANIC
    assert orogeny.classify(c, c, False, c) == "himalayan"
    assert orogeny.classify(c, c, True, c) == "himalayan"       # a continent is never a flat slab
    assert orogeny.classify(o, c, False, c) == "andean"
    assert orogeny.classify(o, c, True, c) == "laramide"
    # ocean under ocean has no continent to stand an arc on, and the
    # flat-slab branch must not divert it to a continental profile either
    assert orogeny.classify(o, o, False, c) == "island_arc"
    assert orogeny.classify(o, o, True, c) == "island_arc"
    # `ural` is an age, not a collision: see classify's docstring
    every = {orogeny.classify(a, b, f, c) for a in (c, o) for b in (c, o) for f in (False, True)}
    assert "ural" not in every
    assert every <= set(orogeny.TYPES)


# --------------------------------------------------------------------------
# the census
# --------------------------------------------------------------------------
def _lattice(n: int = 400, seed: int = 3):
    """A ring of segments, alternating crust type, with a plate boundary."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3))
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    kind = np.where(np.arange(n) % 2 == 0, CONTINENTAL, OCEANIC).astype(np.int8)
    return Segments(pos, 1.0, 0.5, np.arange(n, dtype=float), 0, 4 * np.pi / n, kind=kind)


def _pairs(seg, n: int, seed: int):
    """`n` (loser, survivor) pairs that `collide` could actually produce.

    A continental segment never subducts under an oceanic one at any density
    -- that irreversibility is the whole of the crust-type model -- so a
    continental loser is always given a continental survivor.  Pairing at
    random would manufacture `pair_co` events and the census identities
    below would be testing a configuration the simulation cannot reach.
    """
    rng = np.random.default_rng(seed)
    cont = np.flatnonzero(seg.kind == CONTINENTAL)
    ocean = np.flatnonzero(seg.kind == OCEANIC)
    losers = rng.choice(seg.M, n, replace=False)
    survivors = np.empty(n, dtype=np.int64)
    for i, lo in enumerate(losers):
        pool = cont if seg.kind[lo] == CONTINENTAL else np.concatenate([cont, ocean])
        pool = pool[pool != lo]
        survivors[i] = rng.choice(pool)
    return losers, survivors


def test_census_counts_one_entry_per_collision_and_the_pairings_agree():
    seg = _lattice()
    tree = build_tree(seg)
    losers, survivors = _pairs(seg, 40, seed=11)
    alive = np.ones(seg.M, dtype=bool)
    census: dict = {}
    orogeny.shape_belt(seg, tree, losers, survivors, alive, 0.1, 6.371e6, 26400.0, 0.25,
                       CONTINENTAL, accretion=0.15, flat_slab_age=60.0, census=census)

    pairs = {k: v for k, v in census.items() if k.startswith("pair_")}
    belts = {k: v for k, v in census.items() if k in orogeny.TYPES}
    assert sum(pairs.values()) == sum(belts.values()) == losers.size
    assert pairs.get("pair_co", 0) == 0  # by construction, and by the model
    assert pairs.get("pair_oo", 0) > 0 and pairs.get("pair_cc", 0) > 0
    # the identities that say the classifier and the crust types agree, and
    # that docs/crust-audit.md reads off a real run
    assert belts.get("himalayan", 0) == pairs.get("pair_cc", 0)
    assert belts.get("island_arc", 0) == pairs.get("pair_oo", 0)
    assert belts.get("andean", 0) + belts.get("laramide", 0) == pairs.get("pair_oc", 0)
    # a slab-age histogram entry for every oceanic slab, and no more
    hist = census.get("slab_age_hist", [])
    assert sum(hist) == pairs.get("pair_oo", 0) + pairs.get("pair_oc", 0)
    assert len(hist) == len(orogeny.SLAB_AGE_BINS)


def test_census_is_optional_and_shape_belt_is_unchanged_without_it():
    a, b = _lattice(), _lattice()
    tree_a, tree_b = build_tree(a), build_tree(b)
    losers, survivors = _pairs(a, 30, seed=5)
    args = (losers, survivors, np.ones(a.M, dtype=bool), 0.1, 6.371e6, 26400.0, 0.25, CONTINENTAL)
    m_a = orogeny.shape_belt(a, tree_a, *args, accretion=0.15, flat_slab_age=60.0)
    m_b = orogeny.shape_belt(b, tree_b, *args, accretion=0.15, flat_slab_age=60.0, census={})
    assert m_a == pytest.approx(m_b)
    assert np.array_equal(a.thickness, b.thickness)
    assert np.array_equal(a.mass, b.mass)
