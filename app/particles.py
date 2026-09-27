"""Glowing dust that makes the vortices and the pen visibly move.

A fixed-size pool of particles stored as NumPy arrays (structure of arrays).
Dead slots are reused when new particles are spawned, so nothing is
allocated per frame. Every particle is advected by the velocity field of all
vortices, so dust from the pen gets swallowed by a nearby whirlpool too.
"""

from __future__ import annotations

import math

import numpy as np

from .config import CANVAS_H, CANVAS_W, MAX_PARTICLES
from .vortex import Vortex

_rng = np.random.default_rng()


class ParticleSystem:
    def __init__(self, capacity: int = MAX_PARTICLES):
        self.n = capacity
        self.pos = np.zeros((capacity, 2), dtype=np.float32)
        self.vel = np.zeros((capacity, 2), dtype=np.float32)
        self.life = np.zeros(capacity, dtype=np.float32)      # seconds left
        self.max_life = np.ones(capacity, dtype=np.float32)
        self.size = np.ones(capacity, dtype=np.float32)
        self.hue = np.zeros(capacity, dtype=np.float32)       # 0 cyan .. 1 magenta
        self.drag = np.zeros(capacity, dtype=np.float32)
        self._spawn_debt: dict[int, float] = {}

    def clear(self) -> None:
        self.life.fill(0.0)

    # ---------------------------------------------------------------- spawn
    def _free_slots(self, count: int) -> np.ndarray:
        dead = np.flatnonzero(self.life <= 0.0)
        return dead[:count]

    def spawn(self, pos: np.ndarray, vel: np.ndarray, life: np.ndarray,
              size: np.ndarray, hue: np.ndarray, drag: float) -> None:
        idx = self._free_slots(len(pos))
        k = len(idx)
        if k == 0:
            return
        self.pos[idx] = pos[:k]
        self.vel[idx] = vel[:k]
        self.life[idx] = life[:k]
        self.max_life[idx] = life[:k]
        self.size[idx] = size[:k]
        self.hue[idx] = hue[:k]
        self.drag[idx] = drag

    def emit_pen(self, pos: np.ndarray, pen_vel: np.ndarray, dt: float) -> None:
        """Sparks shed by the pen tip while it presses into the fabric."""
        speed = float(np.linalg.norm(pen_vel))
        count = _rng.poisson((40.0 + speed * 0.12) * dt)
        if count == 0:
            return
        ang = _rng.uniform(0, 2 * math.pi, count)
        mag = _rng.uniform(10, 45, count)
        vel = np.stack([np.cos(ang) * mag, np.sin(ang) * mag], 1) + pen_vel * 0.12
        jitter = _rng.normal(0, 4.0, (count, 2))
        self.spawn(
            pos=np.repeat(pos[None, :], count, 0) + jitter,
            vel=vel,
            life=_rng.uniform(0.45, 1.0, count),
            size=_rng.uniform(1.5, 3.2, count),
            hue=_rng.uniform(0.55, 1.0, count),
            drag=2.2,
        )

    def emit_vortices(self, vortices: list[Vortex], dt: float) -> None:
        """Dust spawned on the rim of each vortex, which then spirals in."""
        for v in vortices:
            rate = 160.0 * abs(v.strength)
            key = id(v)
            debt = self._spawn_debt.get(key, 0.0) + rate * dt
            count = int(debt)
            self._spawn_debt[key] = debt - count
            if count == 0:
                continue
            ang = _rng.uniform(0, 2 * math.pi, count)
            rad = v.radius * _rng.uniform(0.6, 1.25, count)
            pos = v.pos[None, :] + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None]
            self.spawn(
                pos=pos.astype(np.float32),
                vel=np.zeros((count, 2), np.float32),
                life=_rng.uniform(1.6, 3.2, count),
                size=_rng.uniform(1.2, 3.0, count),
                hue=np.clip(rad / (v.radius * 1.25), 0, 1) * 0.6,
                drag=0.0,
            )
        live = {id(v) for v in vortices}
        self._spawn_debt = {k: d for k, d in self._spawn_debt.items() if k in live}

    # --------------------------------------------------------------- update
    def update(self, vortices: list[Vortex], dt: float) -> None:
        alive = self.life > 0.0
        if not alive.any():
            return
        idx = np.flatnonzero(alive)
        p = self.pos[idx]
        v = self.vel[idx]

        # Free motion with per-particle drag (pen sparks slow down quickly).
        v *= np.exp(-self.drag[idx] * dt)[:, None]
        p += v * dt

        swallowed = np.zeros(len(idx), dtype=bool)
        for vx in vortices:
            c = vx.pos.astype(np.float32)
            rel = p - c
            r2 = (rel * rel).sum(1) + 1e-3
            r = np.sqrt(r2)
            R = vx.radius
            # Angular speed rises towards the core (like a real whirlpool),
            # capped so the core does not become a blur, and fades far away.
            omega = 2 * math.pi * vx.spin * np.minimum(R * R / (r2 + (0.35 * R) ** 2), 2.0)
            omega *= np.exp(-(r / (1.7 * R)) ** 2)
            # Rotate exactly instead of stepping along the tangent: a straight
            # tangent step would slowly fling particles outwards.
            ang = omega * dt
            cos, sin = np.cos(ang), np.sin(ang)
            rel = np.stack([cos * rel[:, 0] - sin * rel[:, 1], sin * rel[:, 0] + cos * rel[:, 1]], 1)
            # Gentle pull towards the centre turns circles into spirals.
            pull = 60.0 * abs(vx.strength) * np.exp(-(r / (1.5 * R)) ** 2) * dt
            rel -= rel / r[:, None] * np.minimum(pull, r * 0.5)[:, None]
            p = c + rel
            swallowed |= r < 0.1 * R

        self.pos[idx] = p
        self.vel[idx] = v
        life = self.life[idx] - dt
        life[swallowed] = np.minimum(life[swallowed], 0.12)
        out = (p[:, 0] < -20) | (p[:, 0] > CANVAS_W + 20) | (p[:, 1] < -20) | (p[:, 1] > CANVAS_H + 20)
        life[out] = 0.0
        self.life[idx] = life

    # --------------------------------------------------------------- render
    def vertex_data(self) -> np.ndarray:
        """Pack live particles as (x, y, size, alpha, hue) for the GPU."""
        idx = np.flatnonzero(self.life > 0.0)
        if len(idx) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        t = self.life[idx] / self.max_life[idx]          # 1 -> 0 over the lifetime
        alpha = np.minimum(1.0, (1.0 - t) * 6.0) * np.minimum(1.0, t * 2.5)
        out = np.empty((len(idx), 5), dtype=np.float32)
        out[:, 0:2] = self.pos[idx]
        out[:, 2] = self.size[idx]
        out[:, 3] = alpha
        out[:, 4] = self.hue[idx]
        return out
