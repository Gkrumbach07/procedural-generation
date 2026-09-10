"""Plates: clustering, rigid rotation about Euler poles, convection forces
(PLAN.md section 6.1 / 6.2 steps 2-3).

A plate is a rigid body on the unit sphere described by its angular
velocity ``omega`` (3-vector, **radians per tectonic step**; the axis is
the Euler pole).  Per step the plate rotates by the angle ``|omega|``.

Units and force model
---------------------
* ``heat`` is dimensionless in [0, 1] on the tect grid.
* :func:`heat_gradient_3d` returns ∇heat at segment positions as a tangent
  3-vector in **heat units per radian** of great-circle arc (resolution and
  planet-radius independent).  It converts the contravariant cell
  components of ``FaceField.gradient()`` (whose basis is ``E_i = R_planet
  * J_u / N`` metres per cell) with the EAC Jacobian:
  ``∇f [per radian] = (a * J_u + b * J_v) * R_planet**2 / N``.
* Force per segment ``f_i = area_i * ∇heat_i`` (``area_i`` in steradians,
  so the total force scales with plate area); plates are pushed *towards*
  hot regions (PLAN 6.2.2, ``f = convection * ∇heat``).  Subduction warms
  and new crust cools the heat field, so convergent boundaries attract and
  rifts repel — the feedback that keeps plates moving coherently.
* Torque about the planet centre ``τ = Σ pos_i × f_i``; scalar inertia
  ``I = Σ area_i * mass_i``; :func:`update_omega` does
  ``omega = (1 - damping) * omega + gain * τ / I`` with
  ``gain = convection * force_scale * spacing`` (run.py), so the angular
  acceleration in *spacings per step²* is ``convection * force_scale *
  |∇heat| / (mass per area)`` and the terminal speed under a unit gradient
  is ``convection * force_scale / (mass * damping)`` spacings per step.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..cubesphere import from_sphere_v, jacobian_v
from ..field import FaceField
from ..stubs import fbm_at
from .segments import CONTINENTAL, Segments, random_unit_vectors


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
class Plates:
    """Per-plate arrays (P plates).  ``alive`` is False once a plate has no
    segments left; its ``omega`` is frozen at zero."""

    def __init__(self, P: int):
        self.P = int(P)
        self.omega = np.zeros((P, 3), dtype=np.float64)
        self.com = np.zeros((P, 3), dtype=np.float64)
        self.inertia = np.zeros(P, dtype=np.float64)
        self.mass = np.zeros(P, dtype=np.float64)
        self.area = np.zeros(P, dtype=np.float64)
        self.count = np.zeros(P, dtype=np.int64)
        self.alive = np.ones(P, dtype=bool)

    def update_stats(self, seg: Segments) -> None:
        """Recompute centre of mass, inertia, mass, area and segment counts."""
        P = self.P
        pid = seg.plate_id
        self.count = np.bincount(pid, minlength=P)[:P]
        self.mass = np.bincount(pid, weights=seg.mass, minlength=P)[:P]
        self.area = np.bincount(pid, weights=seg.area, minlength=P)[:P]
        self.inertia = np.bincount(pid, weights=seg.area * seg.mass, minlength=P)[:P]
        com = np.stack([np.bincount(pid, weights=seg.pos[:, k] * seg.area, minlength=P)[:P] for k in range(3)], axis=1)
        n = np.linalg.norm(com, axis=1, keepdims=True)
        self.com = np.where(n > 1e-12, com / np.maximum(n, 1e-12), 0.0)
        self.alive = self.count > 0
        self.omega[~self.alive] = 0.0

    def n_alive(self) -> int:
        return int(self.alive.sum())

    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.omega, axis=1)


# --------------------------------------------------------------------------
# initial clustering
# --------------------------------------------------------------------------
def _value_noise_points(p: np.ndarray, rng: np.random.Generator, octaves: int = 3, base_freq: float = 3.0) -> np.ndarray:
    """Smooth value noise evaluated at unit vectors ``p`` (M, 3) — the same
    lattice construction as ``globe.stubs.fbm_noise`` but at arbitrary
    points.  Roughly in [-1, 1]."""
    out = np.zeros(p.shape[0], dtype=np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        f = base_freq * (2**o)
        n = int(np.ceil(f)) + 2
        lattice = rng.random((n + 1, n + 1, n + 1))
        q = (p + 1.0) * 0.5 * f
        i0 = np.floor(q).astype(np.int64)
        t = q - i0
        t = t * t * (3 - 2 * t)
        i0 = np.clip(i0, 0, n - 1)
        acc = np.zeros_like(out)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2])
                    acc += w * lattice[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        out += amp * (acc * 2 - 1)
        total += amp
        amp *= 0.5
    return out / total


def cluster_plates(pos: np.ndarray, n_plates: int, rng: np.random.Generator, iterations: int = 12, size_jitter: float = 0.35, boundary_noise: float = 0.25) -> np.ndarray:
    """Initial plate assignment (PLAN 6.1): spherical k-means from
    farthest-point seeds, then a final assignment with per-plate distance
    weights (unequal plate sizes) and low-frequency noise on the distances
    (irregular, non-polygonal boundaries).  Returns plate ids (M,) int32
    in ``0..n_plates-1``; every plate gets at least one segment when
    ``M >= n_plates``."""
    M = pos.shape[0]
    P = int(max(1, min(n_plates, M)))
    # farthest-point seeds from a random start
    seeds = np.empty((P, 3), dtype=np.float64)
    seeds[0] = pos[int(rng.integers(M))]
    dmin = np.full(M, np.inf)
    for k in range(1, P):
        dmin = np.minimum(dmin, np.linalg.norm(pos - seeds[k - 1], axis=1))
        seeds[k] = pos[int(np.argmax(dmin))]
    centres = seeds
    for _ in range(int(iterations)):
        d = 2.0 - 2.0 * pos @ centres.T  # squared chord distance
        lab = np.argmin(d, axis=1)
        for k in range(P):
            sel = lab == k
            if sel.any():
                c = pos[sel].mean(axis=0)
                n = np.linalg.norm(c)
                if n > 1e-12:
                    centres[k] = c / n
    w = 1.0 + size_jitter * (2.0 * rng.random(P) - 1.0)
    d = np.sqrt(np.maximum(2.0 - 2.0 * pos @ centres.T, 0.0)) * w[None, :]
    if boundary_noise > 0:
        for k in range(P):
            d[:, k] *= 1.0 + boundary_noise * _value_noise_points(pos, rng)
    lab = np.argmin(d, axis=1).astype(np.int32)
    # guarantee every plate owns at least one segment (its k-means centre's nearest point)
    for k in range(P):
        if not (lab == k).any():
            lab[int(np.argmin(np.linalg.norm(pos - centres[k], axis=1)))] = k
    return lab


def random_initial_omega(plates: Plates, rng: np.random.Generator, speed: float) -> None:
    """Give every plate a random Euler pole with angular speed ``speed``
    (radians per step)."""
    plates.omega[:] = random_unit_vectors(rng, (plates.P,)) * float(speed)
    plates.omega[~plates.alive] = 0.0


# --------------------------------------------------------------------------
# forces
# --------------------------------------------------------------------------
def heat_gradient_3d(heat: FaceField, pos: np.ndarray) -> np.ndarray:
    """∇heat at unit vectors ``pos`` (M, 3) as tangent 3-vectors in heat
    units per radian (see module docstring).  ``heat`` must have exchanged
    halos."""
    grad = heat.gradient()
    face, u, v = from_sphere_v(pos)
    ab = grad.sample_bilinear(face, u, v).astype(np.float64)  # (M, 2) owner-face cell components
    ju, jv = jacobian_v(face, u, v)
    g = heat.grid
    scale = g.R_planet**2 / g.N
    return (ab[:, :1] * ju + ab[:, 1:] * jv) * scale


def plate_torques(seg: Segments, grad3: np.ndarray, P: int) -> np.ndarray:
    """Torque (P, 3) about the planet centre from the per-segment forces
    ``f_i = area_i * grad3_i`` (steradian × heat per radian)."""
    f = grad3 * seg.area[:, None]
    tau = np.cross(seg.pos, f)
    return np.stack([np.bincount(seg.plate_id, weights=tau[:, k], minlength=P)[:P] for k in range(3)], axis=1)


def update_omega(plates: Plates, torque: np.ndarray, gain: float, damping: float, max_speed: float = 0.0) -> None:
    """``omega = (1 - damping) * omega + gain * τ / I`` for live plates
    (radians per step), optionally capped at ``max_speed`` (radians per
    step, i.e. the speed of a segment on the rotation equator)."""
    inertia = np.maximum(plates.inertia, 1e-12)
    om = (1.0 - damping) * plates.omega + gain * torque / inertia[:, None]
    if max_speed > 0:
        s = np.linalg.norm(om, axis=1, keepdims=True)
        om = np.where(s > max_speed, om * (max_speed / np.maximum(s, 1e-30)), om)
    om[~plates.alive] = 0.0
    plates.omega = om


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------
@njit(cache=True)
def _rotate_kernel(pos, plate_id, omega, dt):
    P = omega.shape[0]
    axis = np.zeros((P, 3), dtype=np.float64)
    cs = np.ones(P, dtype=np.float64)
    sn = np.zeros(P, dtype=np.float64)
    for p in range(P):
        w = np.sqrt(omega[p, 0] ** 2 + omega[p, 1] ** 2 + omega[p, 2] ** 2)
        if w > 0.0:
            axis[p, 0] = omega[p, 0] / w
            axis[p, 1] = omega[p, 1] / w
            axis[p, 2] = omega[p, 2] / w
            cs[p] = np.cos(w * dt)
            sn[p] = np.sin(w * dt)
    for i in range(pos.shape[0]):
        p = plate_id[i]
        if sn[p] == 0.0 and cs[p] == 1.0:
            continue
        kx, ky, kz = axis[p, 0], axis[p, 1], axis[p, 2]
        x, y, z = pos[i, 0], pos[i, 1], pos[i, 2]
        c, s = cs[p], sn[p]
        kd = kx * x + ky * y + kz * z
        # Rodrigues: p cosθ + (k × p) sinθ + k (k·p)(1 − cosθ)
        rx = x * c + (ky * z - kz * y) * s + kx * kd * (1.0 - c)
        ry = y * c + (kz * x - kx * z) * s + ky * kd * (1.0 - c)
        rz = z * c + (kx * y - ky * x) * s + kz * kd * (1.0 - c)
        inv = 1.0 / np.sqrt(rx * rx + ry * ry + rz * rz)
        pos[i, 0] = rx * inv
        pos[i, 1] = ry * inv
        pos[i, 2] = rz * inv


def rotate_segments(seg: Segments, plates: Plates, dt: float = 1.0) -> None:
    """Rigidly rotate every segment about its plate's Euler pole by
    ``|omega| * dt`` (Rodrigues' formula, renormalised; ``dt`` = 1 step by
    default).  In place."""
    _rotate_kernel(seg.pos, seg.plate_id, plates.omega, float(dt))


def segment_velocities(seg: Segments, plates: Plates, dt: float = 1.0) -> np.ndarray:
    """Displacement per step ``(omega_plate * dt) × pos`` as tangent
    3-vectors (M, 3), radians per step."""
    return np.cross(plates.omega[seg.plate_id] * dt, seg.pos)


def tangent_to_cell_components(grid, face, u, v, v3: np.ndarray) -> np.ndarray:
    """Express tangent 3-vectors ``v3`` (..., 3) (unit-sphere units per
    step) at ``(face, u, v)`` as contravariant *cell* components (cells per
    step) of that face — the least-squares 2x2 solve used by
    ``cubesphere.transfer_vector`` and ``field.rotation_field``."""
    ju, jv = jacobian_v(face, u, v)
    guu = np.sum(ju * ju, -1)
    guv = np.sum(ju * jv, -1)
    gvv = np.sum(jv * jv, -1)
    wu = np.sum(v3 * ju, -1)
    wv = np.sum(v3 * jv, -1)
    det = guu * gvv - guv * guv
    out = np.empty(v3.shape[:-1] + (2,), dtype=np.float64)
    out[..., 0] = (gvv * wu - guv * wv) / det * grid.N
    out[..., 1] = (guu * wv - guv * wu) / det * grid.N
    return out


def seed_supercontinent(pos: np.ndarray, continental_fraction: float, craton_fraction: float,
                        n_cratons: int, rng, roughness: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """One assembled supercontinent with cratons inside it.

    Returns ``(kind, craton)``: an (M,) int8 of :data:`OCEANIC` /
    :data:`CONTINENTAL`, and an (M,) int8 naming the Archean core each
    segment belongs to (0 = none, 1..n = which nucleus). The *index* matters
    rather than a bare flag: a rift has to keep each craton whole, which
    means knowing which craton a segment is part of, not merely that it is
    in one.

    Scattering continental crust as independent blobs -- what this used to
    do -- starts the world mid-dispersal and, since the model has no force
    that gathers continents deliberately, it tends to stay there. Real
    continental crust spends most of its life assembled: Nuna, Rodinia,
    Pangaea. So the initial condition is a *supercontinent*, and breakup is
    something the simulation does rather than something it starts from.

    The margin is a spherical-noise perturbation of one cap, which gives an
    irregular coastline with embayments and promontories instead of a
    circle; the threshold is taken as a quantile of the perturbed distance,
    so the target area comes out exact however rough the margin is.

    The cratons are grown inside that continent as weighted-Voronoi blobs,
    the way the Gondwana reconstruction looks: Amazonia, West Africa, Congo,
    Kalahari and the rest as discrete old nuclei, separated and surrounded
    by the younger mobile-belt crust that welded them together.
    """
    M = pos.shape[0]
    kind = np.zeros(M, dtype=np.int8)
    craton = np.zeros(M, dtype=np.int8)
    target = int(round(float(continental_fraction) * M))
    if target <= 0:
        return kind, craton

    c = random_unit_vectors(rng, (1,))[0]
    d = np.arccos(np.clip(pos @ c, -1.0, 1.0))                 # angular distance
    if roughness > 0.0:
        d = d + roughness * fbm_at(pos, rng, octaves=4, base_freq=2.0)
    cont = np.argsort(d, kind="stable")[:min(target, M)]
    kind[cont] = CONTINENTAL

    n = int(n_cratons)
    k_target = int(round(float(craton_fraction) * cont.size))
    if n <= 0 or k_target <= 0:
        return kind, craton
    # seeds drawn from the continent itself, so no craton lands in the ocean
    seeds = pos[rng.choice(cont, size=min(n, cont.size), replace=False)]
    w = 1.0 + 0.5 * (rng.random(seeds.shape[0]) - 0.5) * 2.0   # 0.5 .. 1.5 sizes
    dc = np.sqrt(np.maximum(2.0 - 2.0 * (pos[cont] @ seeds.T), 0.0)) / w[None, :]
    take = np.argsort(dc.min(axis=1), kind="stable")[:k_target]
    craton[cont[take]] = (np.argmin(dc[take], axis=1) + 1).astype(np.int8)
    return kind, craton


def snap_cratons(seg) -> int:
    """Give every craton wholly to the plate that holds most of it.

    Plate boundaries follow weak lithosphere. A craton is the opposite of
    that -- thick, cold, depleted, and strong enough that it has survived
    every cycle since the Archean -- so a boundary generally goes around one
    rather than through it. `cluster_plates` knows nothing about cratons and
    was leaving 11 of 24 straddling a boundary at step 0, which then breaks
    them up as the plates diverge.

    Returns the number of cratons that had to be re-assigned. Mutates
    `seg.plate_id` in place.
    """
    cr, pid = seg.craton, seg.plate_id
    moved = 0
    for c in np.unique(cr[cr > 0]):
        m = cr == c
        ids, counts = np.unique(pid[m], return_counts=True)
        if ids.size > 1:
            pid[m] = ids[np.argmax(counts)]
            moved += 1
    return moved


__all__ = ["seed_supercontinent", "snap_cratons", "Plates", "cluster_plates", "random_initial_omega", "heat_gradient_3d", "plate_torques", "update_omega", "rotate_segments", "segment_velocities", "tangent_to_cell_components"]
