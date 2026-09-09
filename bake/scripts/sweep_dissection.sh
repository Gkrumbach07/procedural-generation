#!/usr/bin/env bash
# Is the valley network reachable by tuning erosion?  (Measured answer: no.)
#
#   cd bake && bash scripts/sweep_dissection.sh [OUTDIR] [ITERS] [N]
#
# docs/terrain-realism.md measures the shipped world at ~1.0 km/km2 of
# drainage density and 475 m median hillslopes, against 4-8 km/km2 and
# 50-200 m for real landscapes -- roughly 5x under-dissected, and that is
# what "bland" actually is.
#
# This script re-runs the parameter sweeps that were tried against it.
# Every one is a null result: drainage density does not move.  It exists so
# the nulls can be re-checked rather than taken on trust, and so nobody
# spends another day tuning these knobs.
#
# Read the drain.dens column.  p99 slope must stay under the talus angle
# (1.2); a row that improves beta by exceeding it is just noise, not
# valleys, and terrain_stats.py flags it.
set -u

OUT="${1:-$(pwd)/dissect_runs}"
ITERS="${2:-120}"
N="${3:-256}"
CASE="${CASE:-a}"

run () {
  local tag="$1"; shift
  [ -f "$OUT/runs/$tag/final.npz" ] && { echo "skip $tag"; return; }
  echo "=== $tag"
  python3 scripts/erosion_face_experiment.py \
      --case "$CASE" --iters "$ITERS" --N "$N" --tag "$tag" \
      --out "$OUT" --every 100000 "$@" > "$OUT/$tag.log" 2>&1 \
    || echo "  FAILED (see $OUT/$tag.log)"
}
mkdir -p "$OUT"

run base

# 1. fluvial work: 4x the particles.  null (0.42 -> 0.42 km/km2)
run ppc1.0     --set erosion.particles_per_cell=1.0

# 2. hillslope diffusion.  moves beta, but the WRONG WAY on the network:
#    less diffusion gives FEWER, larger channels and longer hillslopes,
#    while p99 slope runs past the talus angle.  Low beta here is noise.
run creep0     --set erosion.creep_rate=0.0
run nodiff     --set erosion.creep_rate=0.0 --set erosion.thermal_rate=0.0
run talus2x    --set erosion.creep_rate=0.0 --set erosion.talus_slope_soft=1.2 --set erosion.talus_slope_hard=2.0

# 3. channel initiation: where discharge starts to enhance incision.
#    null across a 16x change (0.42 km/km2 at every setting)
run ds8        --set erosion.disc_saturation=8
run ds2        --set erosion.disc_saturation=2

# 4. erodibility
run k0.4       --set erosion.erodibility=0.4

# 5. initial terrain.  the ONLY knob that moves density -- and it moves it
#    the wrong way: more noise gives fewer, larger channels.
run noise0.5   --noise 0.5
run noise12    --noise 12.0

echo
echo "==================== results ===================="
python3 scripts/terrain_stats.py "$OUT"/runs/*/final.npz
echo
echo "target: drainage density 4-8 km/km2, hillslope 50-200 m,"
echo "        with p99 slope under the talus angle (1.2)."
echo "None of the above gets close.  The deficit is structural, not a tuning"
echo "problem -- see docs/terrain-realism.md."
