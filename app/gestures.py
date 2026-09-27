"""Turning raw landmarks into intent: smoothing, pinch and circular motion.

Nothing in this module knows about the camera or the GPU, which keeps the
logic easy to test with synthetic data.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al., CHI 2012).

    Slow movements get heavy smoothing to kill jitter, fast movements get
    little smoothing so the pen does not lag behind the finger.
    """

    def __init__(self, min_cutoff: float = 1.4, beta: float = 0.012, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx = np.zeros(2)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = np.zeros(2)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or dt <= 0:
            self._x = x.copy()
            return x
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class PinchDetector:
    """Thumb-to-pinky contact with hysteresis.

    The distance between the two fingertips is divided by the palm length so
    the gesture works the same close to the camera and far away from it.
    Two thresholds (on/off) stop the state from flickering at the boundary.
    """

    def __init__(self, on_ratio: float = 0.42, off_ratio: float = 0.60):
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.active = False
        self.closeness = 0.0  # 0 = fingers apart, 1 = touching (for the UI)

    def update(self, thumb: np.ndarray, pinky: np.ndarray, wrist: np.ndarray, middle_mcp: np.ndarray) -> bool:
        palm = float(np.linalg.norm(middle_mcp - wrist)) + 1e-6
        ratio = float(np.linalg.norm(thumb - pinky)) / palm
        if self.active:
            self.active = ratio < self.off_ratio
        else:
            self.active = ratio < self.on_ratio
        # Map ratio 1.4 (open hand) .. on_ratio (touching) to 0..1.
        span = max(1.4 - self.on_ratio, 1e-3)
        self.closeness = float(np.clip((1.4 - ratio) / span, 0.0, 1.0))
        return self.active

    def release(self) -> None:
        self.active = False
        self.closeness = 0.0


@dataclass
class SpinEvent:
    center: np.ndarray   # canvas px
    radius: float        # canvas px
    turns_per_sec: float # signed: positive = clockwise on screen


class SpinDetector:
    """Recognises the fingertip drawing circles.

    Keeps a short history of pen positions. The path is considered a circle
    when, around its own centroid, it sweeps a large angle while staying at
    a roughly constant distance. The swept angle and the time it took give a
    signed angular speed, which drives the vortex.
    """

    def __init__(self, window_s: float = 0.75, min_turn: float = 1.5 * math.pi,
                 min_radius: float = 10.0, max_radius: float = 150.0):
        self.window_s = window_s
        self.min_turn = min_turn
        self.min_radius = min_radius
        self.max_radius = max_radius
        self._hist: deque[tuple[float, float, float]] = deque()

    def reset(self) -> None:
        self._hist.clear()

    def update(self, t: float, pos: np.ndarray) -> SpinEvent | None:
        self._hist.append((t, float(pos[0]), float(pos[1])))
        while self._hist and t - self._hist[0][0] > self.window_s:
            self._hist.popleft()
        if len(self._hist) < 8:
            return None

        arr = np.array(self._hist)
        ts, pts = arr[:, 0], arr[:, 1:]
        duration = ts[-1] - ts[0]
        if duration < 0.2:
            return None

        center = pts.mean(axis=0)
        rel = pts - center
        radii = np.hypot(rel[:, 0], rel[:, 1])
        mean_r = float(radii.mean())
        if not (self.min_radius <= mean_r <= self.max_radius):
            return None
        # A circle keeps its distance to the centre; a zig-zag does not.
        if float(radii.std()) > 0.45 * mean_r:
            return None

        # Screen y points down, so a positive swept angle is clockwise.
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        steps = np.diff(ang)
        steps = (steps + math.pi) % (2 * math.pi) - math.pi
        swept = float(steps.sum())
        # Reject paths that go back and forth: most steps must share a sign.
        if abs(swept) < self.min_turn or abs(swept) < 0.7 * float(np.abs(steps).sum()):
            return None

        return SpinEvent(center=center, radius=mean_r,
                         turns_per_sec=swept / (2 * math.pi) / duration)
