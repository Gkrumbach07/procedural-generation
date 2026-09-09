# Stochastic geomorphological transport (McDonald & Cordonnier 2026) — prototype

A standalone prototype of the SIGGRAPH 2026 method, on a plain 2-D grid, built
to answer one question: **can we run the 2026 model per-basin and get the
detail our particle kernel misses?**

Not wired into the bake. Nothing here is imported by `bake/` or `game/`.

    python3 cone_test.py                  # transport validation (paper Fig. 2)
    python3 run_flat.py 192 1200 20       # flat start + uniform uplift
    python3 cutbasin.py && python3 run_basin.py 200   # a real basin from a bake

## What works

`transport.py` implements Algorithm 1 — the quasi-static conservation law
`div(phi v) = S - R phi` solved by superposing attenuated upstream
contributions, sampled by Monte Carlo. It reproduces the paper's convergence
cone, with the plateau past 1 sample/cell that Fig. 2 describes:

| K   | samples/cell | MSRE   |
|-----|--------------|--------|
| 128 | 0.25         | 0.1458 |
| 128 | 1.00         | 0.0689 |
| 128 | 4.00         | 0.0480 |
| 256 | 0.25         | 0.0976 |
| 256 | 1.00         | 0.0588 |
| 256 | 4.00         | 0.0487 |

Per-thread accumulators rather than the paper's atomic adds, so the result is
deterministic and this project's byte-reproducibility survives.

`geoerode.py` is the erosion driver: Table 3 parameters, the paper's time step,
and Eq. 1 over a geological timestep. It runs stable and NaN-free on both a
flat start and a real basin, with `E_f` and `D_f` in balance to two figures and
under 0.2 % of cells touching any limiter.

## What does not work, and why

**1. The model stays in a sheet-flow attractor — no channel network.**

This is the one that matters, because detail was the whole point. Water depth
has two self-consistent states, and the drainage length `|v|/k_e` decides which:

| state       | h_f    | \|v\| | drainage length |
|-------------|--------|-------|-----------------|
| sheet flow  | 63 µm  | 5 mm/s| ~10 m           |
| channelised | 3.4 mm | 0.5 m/s | ~940 m        |

With `k_e = 5e-4 s^-1` (Table 3), water evaporates within a few cells unless
flow has already concentrated. A flat start never escapes: measured water p50
3.7e-5 against p99 1.3e-4, a ratio of 3.4 where a real network is orders of
magnitude. A real baked basin does better — `|v|` rises to 0.088 m/s and the
drainage length to 176 m — but that is still a fraction of a 4.8 km basin, so
the network does not close. The channelised branch is self-consistent and
reachable in principle (a 1 km² catchment gives 3.4 mm at 0.47 m/s); getting
there needs flow already concentrated into channels of roughly the right width.

**2. Throughput is three to four orders of magnitude short of globe scale.**

Measured: 200 steps in 51 s on 384x304 cells (18 km² at 12.5 m) covering
0.2 ky, because the stability-limited step pins to its 1 y floor.

| simulated | steps   | per basin | x 2300 basins |
|-----------|---------|-----------|---------------|
| 10 ky     | 10,000  | 0.7 h     | 1,600 h       |
| 100 ky    | 100,000 | 7.1 h     | 16,300 h      |
| 625 ky    | 625,000 | 44.3 h    | 101,800 h     |

The paper reports interactive rates on CUDA; this is single-machine numpy +
numba, and the gap is far more than an implementation constant.

## Defects found along the way

Recorded because each was measured, and several were mine rather than the
paper's.

- **`tau_f` was missing Eq. 3's factor of 1/8, and the gravity-friction root
  its matching 8.** They cancel exactly in the balance limit
  (`tau = rho g h S`, the depth-slope product), which is why the cone
  validation passed while they were both wrong — but they leave `|v|` low by
  sqrt(8) = 2.83x everywhere inertia matters, and `|v|` sets the drainage
  length. Fixed.

- **Debris decay degeneracy — the cause of the 2500-step stall.** Using
  `R_d = D_d/h_d` with the paper's "set D_d to 0 if h_d = 0" gives `R_d = 0` at
  bootstrap. A transport with no decay integrates its source along the whole
  streamline, so `h_d` overflowed, `tau_d` and `drive` both reached inf,
  `drive - tau_d` became `inf - inf = NaN`, and the NaN spread through `v_d`
  to every cell. `nan_to_num` then rewrote the erosion to 0 — a terrain frozen
  mid-run that still reported `finite True`, relief pinned at 19.7 m with
  `dz rms 0.0000` for 2250 of 2500 steps. Fixed by evaluating `R_d` on a
  floored thickness and taking `D_d = R_d h_d`, so deposition vanishes with the
  debris instead of diverging.

- **My own mass limiters were a mass *source*.** `D <= h/dt` looks
  conservative, but `h_s` and `h_d` are quasi-static fields re-solved every
  iteration rather than carried as stock, so spending them never depletes them:
  `dz = D dt = h` is added every step at any `dt`. That is why the divergence
  survived a 250x cut in the timestep, growing ~1 m per *step* whether the step
  was 250 y or 1 y. Removed; deposition now comes from the same decay the
  transport is given, which bounds it by construction.

- **Eq. 5 is a stiff penalty, not a rate.** `k_l rho_d g` = 61.3 m/s per unit
  of slope past repose, so one degree over asks for 4.8e11 m in a 250 y step;
  measured at 2.6e9 m/y, dwarfing `E_f` at 0.05 and `U` at 0.001. One-sided
  removal on a gradient magnitude also oscillates — a limit cycle of ~3400 m
  per cell at every resolution tried. Replaced by a mass-conserving repose
  relaxation swept to convergence, which is what a landslide actually does.

- **The debris channel is degenerate at Table 3's `tau_y = 2 MPa`.**
  Mobilising debris needs `rho_d g h_d |grad z| > 2 MPa` — 82 m of debris on a
  45 degree slope — so `tau_d` sits at ~2 MPa, the momentum decay reaches
  8e14 s^-1 and annihilates `v_d`, and the deposition decay reaches 4e11 s^-1.
  Multiplying that by a 1e-12 m thickness produced 1.9e7 m/y on single cells.
  Stiff, not unstable: no timestep fixes it. Carried for reporting, not
  applied.

- **An early rate clamp was set below the uplift rate.** `cfl * dx / dt` came
  out at 0.4 x `U`, throttling uplift itself by 60 %, and later it was also
  smaller than what the repose limiter needed to do its job.

## Verdict

The transport solver is sound and validated. The erosion driver around it is
now numerically clean. But on the two things that would justify adopting it —
detail, and cost at globe scale — it does not currently clear the bar, so it
stays an experiment rather than a stage.
