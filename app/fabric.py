"""The elastic "spacetime fabric" that the pen presses into.

The fabric is a coarse grid of heights (0 = flat, negative = pushed in).
Each cell behaves like a damped spring pulled back to zero and coupled to
its neighbours, which gives a rubber-sheet feel: grooves smooth out, spread
slightly and settle back over a few seconds.

The grid is tiny (200 x 150), so NumPy handles it in well under a
millisecond per frame. The GPU only reads it as a texture.
"""

from __future__ import annotations

import math

import numpy as np

from .config import CANVAS_H, CANVAS_W, FABRIC_H, FABRIC_W

_SUBSTEP = 1.0 / 120.0
_DAMPING_RATIO = 1.15  # slightly over-damped: settles without wobbling forever
_COUPLING = 30.0       # neighbour coupling (cells^2 / s^2); higher = stiffer sheet


class Fabric:
    def __init__(self, width: int = FABRIC_W, height: int = FABRIC_H):
        self.w = width
        self.h = height
        self.height = np.zeros((height, width), dtype=np.float32)
        self.velocity = np.zeros_like(self.height)
        self._sx = width / CANVAS_W   # canvas px -> cells
        self._sy = height / CANVAS_H
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self._cx = xs + 0.5            # cell centres in cell units
        self._cy = ys + 0.5
        self._acc = 0.0
        self.set_recovery(3.0)

    def set_recovery(self, seconds: float) -> None:
        """Choose spring constants so a groove is ~95% flat after `seconds`.

        For a damped spring with ratio z the slow decay rate is
        w0 * (z - sqrt(z^2 - 1)); three time constants give ~95%.
        """
        seconds = max(seconds, 0.3)
        slow = _DAMPING_RATIO - math.sqrt(_DAMPING_RATIO ** 2 - 1.0)
        w0 = 3.0 / (slow * seconds)
        self._k = w0 * w0
        self._c = 2.0 * _DAMPING_RATIO * w0

    def clear(self) -> None:
        self.height.fill(0.0)
        self.velocity.fill(0.0)

    def press(self, p0: np.ndarray, p1: np.ndarray, radius_px: float, depth: float) -> None:
        """Carve a groove along the segment p0 -> p1 (canvas px).

        The groove profile is a smooth bowl. Cells only ever get pushed down
        (``minimum``) so drawing over an existing groove does not lift it.
        """
        a = np.array([p0[0] * self._sx, p0[1] * self._sy], dtype=np.float32)
        b = np.array([p1[0] * self._sx, p1[1] * self._sy], dtype=np.float32)
        r = radius_px * self._sx

        x0 = int(max(min(a[0], b[0]) - r - 1, 0))
        x1 = int(min(max(a[0], b[0]) + r + 2, self.w))
        y0 = int(max(min(a[1], b[1]) - r - 1, 0))
        y1 = int(min(max(a[1], b[1]) + r + 2, self.h))
        if x0 >= x1 or y0 >= y1:
            return

        cx = self._cx[y0:y1, x0:x1]
        cy = self._cy[y0:y1, x0:x1]
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-6:
            t = np.zeros_like(cx)
        else:
            t = np.clip(((cx - a[0]) * ab[0] + (cy - a[1]) * ab[1]) / denom, 0.0, 1.0)
        dx = cx - (a[0] + t * ab[0])
        dy = cy - (a[1] + t * ab[1])
        d2 = (dx * dx + dy * dy) / (r * r)

        target = -depth * np.exp(-2.6 * d2)
        region = self.height[y0:y1, x0:x1]
        pushed = target < region
        region[pushed] = target[pushed]
        # Held under the pen, the sheet should not keep bouncing.
        self.velocity[y0:y1, x0:x1][pushed] = 0.0

    def step(self, dt: float) -> None:
        """Advance the spring simulation with fixed sub-steps for stability."""
        self._acc = min(self._acc + dt, 0.1)
        h, v = self.height, self.velocity
        k, c, cpl = self._k, self._c, _COUPLING
        damp = math.exp(-c * _SUBSTEP)
        while self._acc >= _SUBSTEP:
            self._acc -= _SUBSTEP
            # Laplacian with the border pinned at zero, like a sheet clamped
            # to a frame.
            p = np.pad(h, 1)
            lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * h
            v += (cpl * lap - k * h) * _SUBSTEP
            v *= damp
            h += v * _SUBSTEP
        # Tiny values are invisible; snapping them to zero avoids denormals.
        np.putmask(h, np.abs(h) < 1e-4, 0.0)
