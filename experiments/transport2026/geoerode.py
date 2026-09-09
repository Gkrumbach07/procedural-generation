"""Erosion driver for the 2026 stochastic-transport model, on one basin.

Implements the paper's time step (Sec. 6, "Time stepping"):

  1. erosion / deposition rates from the state at t   (Eqs. 2, 4, 5, 6, 7)
  2. velocities at t+dt from momentum conservation    (Eq. 12, via Alg. 1)
  3. water / sediment / debris at t+dt                (Eqs. 10, 11, via Alg. 1)
  4. elevation update                                 (Eq. 1)
  5. temporal exponential filter with beta            (Eq. 23)

Defaults are Table 3 of the paper.

Bootstrap: the paper starts from a flat terrain with zeroed fields, where
v = 0 is consistent because grad z = 0 too.  We start from a real basin,
so v = 0 would leave R = tau/(rho h |v|) singular and nothing could advect.
We therefore seed v from the gravity-friction balance that the Stream Power
Law assumes (|v| = sqrt(g h |grad z| / f_D), directed downslope) and let
momentum conservation take over from the first step -- which is exactly the
assumption the paper relaxes, used only to get the iteration started.
"""
from __future__ import annotations

import numpy as np

import transport

# Thickness floor used to linearise the debris decay (see `erode`).
HD_FLOOR = 1e-4


class P:
    # Table 3
    rho_f, rho_d = 1000.0, 2500.0
    f_D = 0.06
    a = 2.0
    k_f = 4.5e-8
    k_df = 0.05
    k_l = 0.0025
    g = 9.81
    theta = np.radians(37.0)
    k_d = 0.01
    k_dd = 0.05
    tau_y = 2e6
    mu_f, mu_d = 1e-3, 4e-3
    p_f = 1.0 / (365.25 * 24 * 3600)      # 1 m/y  -> m/s
    k_e = 5e-4
    dt = 250.0 * 365.25 * 24 * 3600       # 250 y  -> s
    beta = 0.1
    U = 0.001 / (365.25 * 24 * 3600)      # 0.001 m/y -> m/s


def _grad(z, dx):
    gy, gx = np.gradient(z, dx)
    return gx, gy


def _lap(f, dx):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f) / (dx * dx)


def _solve_vec(Sx, Sy, R, vx, vy, mask, dx, npc, seed, fbx=None, fby=None):
    """Momentum: phi is a vector, so two scalar transports advected by the
    same v and sharing the same decay."""
    ax = transport.solve(Sx, R, vx, vy, mask, dx, n_per_cell=npc, seed=seed, fallback=fbx)
    ay = transport.solve(Sy, R, vx, vy, mask, dx, n_per_cell=npc, seed=seed + 1, fallback=fby)
    return ax, ay


def _balance(gx, gy, slope, h, f_D, g_):
    """Gravity-friction root, the stagnant-flow limit of Eq. 12 and the
    closure the Stream Power Law assumes.  Dropping the inertial term of
    Eq. 12 leaves g |grad z| = tau/(rho h), and with Eq. 3's tau =
    (1/8) f_D rho |v|^2 that is

        |v| = sqrt(8 g h |grad z| / f_D),

    which returns the textbook depth-slope product tau = rho g h S.

    The 8 here and the 1/8 in Eq. 3 cancel *in this limit*, so omitting
    both (as an earlier version did) still gave the right tau and passed
    the cone validation -- but left |v| low by sqrt(8) = 2.83x everywhere
    else, and |v| is what sets the drainage length |v|/k_e.  At these
    depths that was 29 m instead of 82 m, so no catchment could grow.
    """
    sp = np.sqrt(8.0 * g_ * np.maximum(h, 1e-9) * slope / f_D)
    inv = np.where(slope > 1e-12, 1.0 / np.maximum(slope, 1e-12), 0.0)
    return -gx * inv * sp, -gy * inv * sp


def repose_relax(z, dx, tan_theta, inside, frac=0.5, max_sweeps=30, tol=1e-3):
    """Move material downhill until no cell stands more than the angle of
    repose above a neighbour.  Mass conserving: what one cell loses, its
    lower neighbours gain.

    This stands in for the debris channel (Eqs. 5-8), which is degenerate at
    this parameterisation.  With Table 3's tau_y = 2 MPa, mobilising debris
    needs rho_d g h_d |grad z| > 2 MPa -- 82 m of debris on a 45 deg slope --
    so tau_d sits at ~2 MPa always.  That makes the debris momentum decay
    tau_d/(rho_d h_d |v_d|) about 8e14 per second, which annihilates v_d;
    with |v_d| then at its floor the deposition decay R_d reaches ~4e11 per
    second, and multiplying it by a 1e-12 m thickness turned a numerically
    empty field into 1.9e7 m/y of deposition on single cells.  No timestep
    fixes that -- it is stiff, not unstable.

    E_l's own form has the same trouble: k_l rho_d g = 61.3 m/s per unit of
    slope past repose is a penalty, not a rate, and one-sided removal on a
    gradient magnitude oscillates (measured: a limit cycle of ~3400 m per
    cell at every resolution tried).  Moving the excess to the neighbours
    that are actually lower is both stable and what a landslide does.
    """
    # Sweep to convergence, not once.  A single sweep moves only ~12 % of a
    # cell's excess, so a 90 m step between neighbours survives for many
    # iterations -- long enough for E_f (quartic in |v|) to run away on it.
    # Measured with one sweep: cells at slope 5-7 feeding h_s 1.7e5 x its
    # local equilibrium downstream, and D_f reaching 1.1e8 m/y.
    for _ in range(int(max_sweeps)):
        worst = 0.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            zn = np.roll(np.roll(z, -di, axis=0), -dj, axis=1)
            over = np.maximum(z - zn - tan_theta * dx, 0.0)
            w = float(over.max())
            if w > worst:
                worst = w
            excess = over * (0.25 * frac)
            z = z + np.where(inside, np.roll(np.roll(excess, di, axis=0), dj, axis=1) - excess, 0.0)
        if worst < tol:
            break
    return z


def erode(z, mask, dx, steps, npc=1.0, seed=0, uplift=None, log=None, p=P, snapshot_every=0,
          zcap=1e4, vcap=3.0, dz_target=0.25, dt_min=1.0 * 365.25 * 24 * 3600):
    """Run `steps` geological timesteps on one basin.  `mask` is 1 inside the
    basin (Dirichlet z outside, i.e. the coarse solution holds the rim)."""
    z = z.astype(np.float64).copy()
    m = mask.astype(np.uint8)
    inside = m == 1
    # the model's own local balance for water: div(v h) = p_f - k_e h with
    # no divergence gives h = p_f/k_e, ~0.06 mm.  Seeding a millimetre film
    # instead puts |v| ~4x too high, tau ~16x, and E_f ~250x, which diverges
    # within three steps.
    h_f = np.full_like(z, p.p_f / p.k_e)
    h_s = np.zeros_like(z)
    h_d = np.zeros_like(z)
    U = np.zeros_like(z) if uplift is None else uplift.astype(np.float64)

    gx, gy = _grad(z, dx)
    slope = np.hypot(gx, gy)
    # bootstrap (see module docstring)
    sp = np.sqrt(p.g * np.maximum(h_f, 1e-6) * slope / p.f_D)
    vfx = np.where(slope > 1e-12, -gx / np.maximum(slope, 1e-12) * sp, 0.0)
    vfy = np.where(slope > 1e-12, -gy / np.maximum(slope, 1e-12) * sp, 0.0)
    vdx, vdy = vfx * 0.1, vfy * 0.1
    snaps = []
    t_sim = 0.0
    # A NaN in the erosion terms used to be swallowed by nan_to_num, which
    # froze the terrain while leaving z finite.  Record it instead.
    self_report = {"warned": False, "first_bad_it": -1}

    for it in range(steps):
        gx, gy = _grad(z, dx)
        slope = np.hypot(gx, gy)
        sf = np.maximum(np.hypot(vfx, vfy), 1e-6)
        sd = np.maximum(np.hypot(vdx, vdy), 1e-6)
        hf = np.maximum(h_f, 1e-6)
        hd = np.maximum(h_d, 1e-6)

        # 1. rates at time t
        tau_f = 0.125 * p.f_D * p.rho_f * sf ** 2                           # Eq. 3
        E_f = p.k_f * tau_f ** p.a                                          # Eq. 2
        # Eq. 4, through the same decay the sediment transport is given.
        # D_f = k_df h_s/h_f divides by the water depth, so a cell with
        # almost no water asks for almost unbounded deposition -- measured at
        # 1e6 m/y once h_f reached its floor, while every median stayed
        # healthy.  Sharing one floored h_f between the rate and the decay
        # makes the pair self-consistent: h_s solves div(v h_s) = E_f - R_s
        # h_s, so away from strong divergence D_f = R_s h_s settles at E_f
        # rather than running off on its own.
        R_s = p.k_df / hf
        D_f = R_s * h_s                                                     # Eq. 4
        # Eq. 5.  As written this is a *stiff penalty*, not a rate to
        # integrate: k_l rho_d g = 61.3 m/s per unit of slope past repose, so
        # a slope one degree over the angle asks for 0.6 m/s -- 4.8e11 m in a
        # 250 y step.  Measured, it reached 2.6e9 m/y and drove the whole
        # divergence (E_f was 0.05 m/y and U 0.001 m/y beside it).  Its
        # physical content is that a slope past repose fails and planes back
        # to repose, so cap it at exactly the excess height a cell carries
        # above the angle: (slope - tan theta) * dx, delivered over one step.
        excess = np.maximum(slope - np.tan(p.theta), 0.0)
        E_l = np.minimum(p.k_l * p.rho_d * p.g * excess, excess * dx / p.dt)
        tau_d = p.tau_y + p.rho_d * p.g * hd * np.tan(p.theta) + p.mu_d * sd / hd  # Eq. 8
        drive = p.rho_d * p.g * hd * slope
        E_d = p.k_d * np.maximum(drive - tau_d, 0.0) / (p.rho_d * sd)       # Eq. 6
        # Eq. 7, written through the transport's own decay so the two agree.
        #
        # Taken literally, D_d = k_dd (tau_d - drive)_+ / (rho_d |v_d|) does
        # not vanish with h_d: as the debris thins, tau_d -> tau_y = 2 MPa and
        # D_d tends to a large constant, which measured 1.3e15 m/y.  The
        # paper's "we set D_d to 0 if h_d = 0" patches the h_d == 0 case but
        # not the thin-but-nonzero one.
        #
        # It also has to be the same relation the transport assumes.  h_d
        # solves div(v h_d) = (E_d + E_l) - R_d h_d, so the deposition implied
        # by that solution is exactly R_d h_d and nothing else; using a
        # different D_d in Eq. 1 puts mass in the terrain that the transport
        # never took out of the flow.  Evaluating R_d on a floored thickness
        # keeps it finite, then multiplying back by the true h_d makes D_d
        # vanish with the debris, as it must.
        #
        # Using D_d/h_d as the decay instead -- with D_d forced to 0 at
        # h_d = 0 -- is what killed the 2500-step run: R_d = 0 at bootstrap
        # left the transport with no sink, so it integrated its source along
        # the whole streamline, h_d overflowed, tau_d and drive both reached
        # inf, `drive - tau_d` became inf - inf = NaN, and the NaN spread
        # through v_d to every cell.  `raw` was then NaN everywhere and the
        # nan_to_num guard rewrote it to 0: a terrain frozen mid-run that
        # still reported `finite True`.
        hd_r = np.maximum(h_d, HD_FLOOR)
        tau_dr = p.tau_y + p.rho_d * p.g * hd_r * np.tan(p.theta) + p.mu_d * sd / hd_r
        R_d = p.k_dd * np.maximum(tau_dr - p.rho_d * p.g * hd_r * slope, 0.0) / (p.rho_d * sd * hd_r)
        D_d = R_d * h_d
        # NO mass limiter here.  Capping deposition at "the material on hand
        # over one step" (D <= h/dt) looks conservative and is in fact a mass
        # *source*: h_s and h_d are quasi-static fields, re-solved from the
        # transport every iteration rather than carried as a stock, so
        # spending them never depletes them.  dz = D dt = h is then added
        # every step at any dt -- which is why the divergence survived a 250x
        # cut in the timestep, growing ~1 m per step whether the step was
        # 250 y or 1 y.
        #
        # The consistent closure is the paper's own: h_s solves
        # div(v h_s) = E_f - D_f with the decay R = k_df/h_f, so D_f =
        # R h_s = k_df h_s/h_f is bounded by construction, and locally
        # E_f - D_f nets to the transport divergence rather than to a
        # free-floating stock.
        # Stability-limited timestep.  The paper notes that dt is "limited
        # numerically by instabilities in the erosion stepping" for Eq. 1,
        # and that is exactly what is seen here: with every median healthy
        # (slope 0.001, E_f 4e-8 m/y) a sub-1% tail of cells reaching ~0.3
        # m/s incises ~240 m in one 250 y step, punches a pit and cascades.
        # So size the step by the fastest cell rather than clamping it after
        # the fact.  Only the erosion terms set the limit -- deposition is
        # already bounded by the mass on hand.
        # An earlier attempt at this collapsed to ~2 y per step, but that was
        # while E_l was running at 2.6e9 m/y; with Eq. 5 limited to the
        # repose excess the step stays usable.
        # only the terms Eq. 1 actually applies may set the step
        drop = np.abs(U - E_f + D_f)
        peak = float(drop[inside].max()) if inside.any() else 0.0
        dt = p.dt if peak <= 0.0 else min(p.dt, max(dt_min, dz_target * dx / peak))

        # 2. momentum at t+dt  (Eq. 12); R from the friction term
        Sfx = -p.g * gx + (p.mu_f / p.rho_f) * _lap(vfx, dx)
        Sfy = -p.g * gy + (p.mu_f / p.rho_f) * _lap(vfy, dx)
        Rf = tau_f / (p.rho_f * hf * sf)
        bfx, bfy = _balance(gx, gy, slope, hf, p.f_D, p.g)
        nfx, nfy = _solve_vec(Sfx, Sfy, Rf, vfx, vfy, m, dx, npc, seed + 7 * it, bfx, bfy)
        Sdx = -p.g * gx + (p.mu_d / p.rho_d) * _lap(vdx, dx)
        Sdy = -p.g * gy + (p.mu_d / p.rho_d) * _lap(vdy, dx)
        Rd = tau_d / (p.rho_d * hd * sd)
        bdx, bdy = _balance(gx, gy, slope, hd, p.f_D, p.g)
        ndx, ndy = _solve_vec(Sdx, Sdy, Rd, vdx, vdy, m, dx, npc, seed + 7 * it + 3, bdx, bdy)
        vfx = p.beta * nfx + (1 - p.beta) * vfx                             # Eq. 23
        vfy = p.beta * nfy + (1 - p.beta) * vfy
        vdx = p.beta * ndx + (1 - p.beta) * vdx
        vdy = p.beta * ndy + (1 - p.beta) * vdy
        # Terminal-velocity clamp.  Momentum conservation adds inertia on top
        # of the gravity-friction balance -- that is the point, it is what
        # makes meanders -- but it should not exceed it by a large factor.
        # E_f is quartic in |v| (tau ~ |v|^2, a = 2), so an overshoot feeds
        # back through erosion into slope into more velocity and diverges in
        # two steps.  Cap at `vcap` x the friction-limited root.
        def _clip(ax, ay, bx, by):
            sp = np.hypot(ax, ay); lim = vcap * np.hypot(bx, by) + 1e-9
            sc = np.where(sp > lim, lim / np.maximum(sp, 1e-12), 1.0)
            return ax * sc, ay * sc
        vfx, vfy = _clip(vfx, vfy, bfx, bfy)
        vdx, vdy = _clip(vdx, vdy, bdx, bdy)

        # 3. transported quantities at t+dt along the new velocities
        nh_f = transport.solve(np.full_like(z, p.p_f), np.full_like(z, p.k_e),
                               vfx, vfy, m, dx, n_per_cell=npc, seed=seed + 7 * it + 5)
        nh_s = transport.solve(E_f, R_s,
                               vfx, vfy, m, dx, n_per_cell=npc, seed=seed + 7 * it + 6)
        # Same R_d the deposition above was built from.  With Table 3's
        # tau_y = 2 MPa this is large -- mobilising debris needs
        # rho_d g h_d |grad z| > 2 MPa, i.e. 82 m of debris on a 45 deg
        # slope -- so debris is deposited essentially where it is made.
        # That is the correct reading of these parameters at this scale:
        # the debris channel is inert here, not explosive.
        nh_d = transport.solve(E_d + E_l, R_d,
                               vdx, vdy, m, dx, n_per_cell=npc, seed=seed + 7 * it + 4)
        h_f = p.beta * nh_f + (1 - p.beta) * h_f
        h_s = p.beta * nh_s + (1 - p.beta) * h_s
        h_d = p.beta * nh_d + (1 - p.beta) * h_d

        # 4. elevation (Eq. 1); Dirichlet outside the basin.
        #
        # Adaptive timestep.  The paper notes the transport scheme is
        # unconditionally stable but "local instabilities can occur in the
        # erosion model, as we use explicit timestepping for Eq. 1", so dt is
        # "limited numerically by instabilities in the erosion stepping".
        # E_f is quartic in |v| (tau ~ |v|^2, then a = 2), so a cell that cuts
        # a few metres on a 5 m grid steepens, accelerates and runs away
        # within two steps.  Cap the step so no cell moves more than
        # `cfl` x dx of elevation, which is the erosion analogue of a CFL
        # condition and self-tunes as the terrain matures.
        # Clamp the per-cell RATE rather than throttling the global dt: the
        # limit is set by a handful of runaway cells, and shrinking dt for
        # everyone stalls geological time (measured: dt collapsed to ~2 y, so
        # 200 steps covered 2.2 ky where the paper needs 200-1000 ky).
        # Clamping the outliers keeps the step at dt and bounds any single
        # cell to `cfl` x dx of elevation change.
        # Eq. 1 without the debris channel: E_d, E_l and D_d are carried for
        # reporting but not applied, because at this parameterisation they
        # cancel to zero net (everything mobilised is deposited where it was
        # made) while being far too stiff to integrate.  The repose
        # relaxation below carries their geomorphic content instead.
        raw = U - E_f + D_f
        # The cap is a multiple of the uplift a step delivers, not a fraction
        # of dx.  `cfl * dx / dt` with the earlier defaults came out at 0.4 x
        # U, so the cap throttled *uplift itself* by 60 % and no terrain could
        # reach the relief its uplift rate implies.
        lim = zcap * p.U
        nbad = int((~np.isfinite(raw))[inside].sum())
        if nbad and not self_report["warned"]:
            self_report["warned"] = True
            self_report["first_bad_it"] = it
        rate = np.clip(np.nan_to_num(raw, nan=0.0, posinf=lim, neginf=-lim), -lim, lim)
        nclamp = int((np.abs(raw) > lim)[inside].sum())
        dz = rate * dt
        z = np.where(inside, z + dz, z)
        z = repose_relax(z, dx, np.tan(p.theta), inside)
        t_sim += dt

        if log and (it % max(1, steps // 12) == 0 or it == steps - 1):
            n = int(inside.sum())
            yr = 365.25 * 24 * 3600
            def _m(a):
                return float(np.median(a[inside])) * yr if np.ndim(a) else float(a) * yr
            log("  it %4d t %6.1f ky dt %7.1f y |vf| %7.4f h_f %8.2e slope %6.3f relief %7.1f m "
                "dzrms %7.3f clamp %5.1f%% nan %4.1f%% | m/y: U %.1e E_f %.1e D_f %.1e E_l %.1e E_d %.1e D_d %.1e | raw p99 %.1e"
                % (it, t_sim / yr / 1000, dt / yr,
                   np.median(sf[inside]), np.median(h_f[inside]), float(np.median(slope[inside])),
                   float(np.ptp(z[inside])), float(np.sqrt(np.mean(dz[inside] ** 2))),
                   100.0 * nclamp / n, 100.0 * nbad / n,
                   _m(U), _m(E_f), _m(D_f), _m(E_l), _m(E_d), _m(D_d),
                   float(np.percentile(np.abs(raw[inside]), 99)) * yr))
            # the worst single cell, term by term -- percentiles hide a tail
            # of a few cells, which is exactly where these runaways live
            fl = np.where(inside, np.abs(raw), 0.0)
            k = np.unravel_index(np.argmax(fl), fl.shape)
            log("        worst cell %s: raw %+.3e | E_f %.3e D_f %.3e E_l %.3e E_d %.3e D_d %.3e "
                "| h_f %.3e h_s %.3e h_d %.3e |v_f| %.4f |v_d| %.4f slope %.3f"
                % (k, raw[k] * yr, E_f[k] * yr, D_f[k] * yr, E_l[k] * yr, E_d[k] * yr, D_d[k] * yr,
                   h_f[k], h_s[k], h_d[k], sf[k], sd[k], slope[k]))
        if snapshot_every and it % snapshot_every == 0:
            snaps.append(z.copy())
    return z, dict(t_sim=t_sim, h_f=h_f, h_s=h_s, h_d=h_d, vfx=vfx, vfy=vfy, snaps=snaps,
                   nan_free=not self_report["warned"], first_bad_it=self_report["first_bad_it"])
