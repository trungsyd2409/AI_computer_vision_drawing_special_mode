"""Whirlpools created by drawing circles with the fingertip.

A vortex is a small record (centre, radius, signed spin). While the finger
keeps circling, the vortex spins up towards the finger's angular speed;
once the finger stops, the spin decays exponentially and the vortex is
removed when it becomes too weak to see. The shader and the particle system
both read this list every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import MAX_VORTICES
from .gestures import SpinEvent

MAX_SPIN = 2.5         # turns per second at full strength
_SPIN_UP_TIME = 0.35   # seconds to follow the finger's speed
_FEED_GRACE = 0.15     # seconds without new input before decay begins


@dataclass
class Vortex:
    pos: np.ndarray
    radius: float
    spin: float = 0.0     # signed turns / s, positive = clockwise on screen
    phase: float = 0.0    # accumulated rotation, animates the spiral arms
    idle: float = 0.0     # seconds since the finger last fed this vortex

    @property
    def strength(self) -> float:
        """Signed strength in -1..1 used by the shader."""
        return float(np.clip(self.spin / MAX_SPIN, -1.0, 1.0))


class VortexField:
    def __init__(self):
        self.vortices: list[Vortex] = []
        self.decay_seconds = 2.6

    def clear(self) -> None:
        self.vortices.clear()

    def feed(self, event: SpinEvent, dt: float) -> None:
        """Spin up the vortex under the finger, creating one if needed."""
        radius = float(np.clip(event.radius * 2.0, 60.0, 190.0))
        target = float(np.clip(event.turns_per_sec * 1.4, -MAX_SPIN, MAX_SPIN))

        vortex = self._nearest(event.center, max_dist=radius * 0.8)
        if vortex is None:
            if len(self.vortices) >= MAX_VORTICES:
                # Recycle the weakest one instead of refusing the gesture.
                self.vortices.sort(key=lambda v: abs(v.spin))
                self.vortices.pop(0)
            vortex = Vortex(pos=event.center.astype(np.float64).copy(), radius=radius)
            self.vortices.append(vortex)

        follow = 1.0 - math.exp(-dt / _SPIN_UP_TIME)
        # Reversing direction first has to cancel the existing spin, which
        # feels like pushing against real water.
        vortex.spin += (target - vortex.spin) * follow
        vortex.pos += (event.center - vortex.pos) * min(1.0, dt * 3.0)
        vortex.radius += (radius - vortex.radius) * min(1.0, dt * 2.0)
        vortex.idle = 0.0

    def update(self, dt: float) -> None:
        tau = max(self.decay_seconds, 0.2)
        alive = []
        for v in self.vortices:
            v.idle += dt
            if v.idle > _FEED_GRACE:
                v.spin *= math.exp(-dt / tau)
            v.phase += v.spin * dt * 2 * math.pi
            if abs(v.spin) > 0.04:
                alive.append(v)
        self.vortices = alive

    def _nearest(self, pos: np.ndarray, max_dist: float) -> Vortex | None:
        best, best_d = None, max_dist
        for v in self.vortices:
            d = float(np.linalg.norm(v.pos - pos))
            if d < best_d:
                best, best_d = v, d
        return best
