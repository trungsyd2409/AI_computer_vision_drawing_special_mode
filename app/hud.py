"""On-screen text (FPS, status, hints) drawn with Pillow.

Text changes a few times per second at most, so it is cheaper and simpler to
rasterise it on the CPU into a transparent image and let the GPU blend that
image on top, than to build a font atlas in OpenGL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import CANVAS_H

# Tried in order; the first font found on the machine wins.
_SANS = ["seguisb.ttf", "segoeui.ttf", "Inter-SemiBold.ttf", "DejaVuSans.ttf", "Arial.ttf"]
_MONO = ["CascadiaMono.ttf", "consola.ttf", "JetBrainsMono-Regular.ttf", "DejaVuSansMono.ttf"]
_FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype"),
]


def _font(candidates: list[str], size: int) -> ImageFont.ImageFont:
    for name in candidates:
        for folder in _FONT_DIRS:
            path = folder / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
        try:
            return ImageFont.truetype(name, size)  # lets Pillow search too
        except OSError:
            continue
    return ImageFont.load_default(size)


@dataclass(frozen=True)
class HudInfo:
    fps: int
    cam_fps: int
    track_fps: int
    status: str
    status_live: bool
    warning: str = ""


WHITE = (236, 242, 255)
DIM = (160, 172, 200)
CYAN = (90, 232, 255)
MAGENTA = (255, 96, 200)
AMBER = (255, 190, 90)

HINT = "LEFT THUMB + PINKY  press      CIRCLE INDEX  vortex      X  settings      C  clear"


class Hud:
    def __init__(self):
        self._size = (0, 0)
        self._last: HudInfo | None = None
        self._fonts: dict[str, ImageFont.ImageFont] = {}

    def _build_fonts(self, scale: float) -> None:
        self._fonts = {
            "fps": _font(_MONO, round(22 * scale)),
            "small": _font(_SANS, round(10.5 * scale)),
            "mono_small": _font(_MONO, round(10.5 * scale)),
            "status": _font(_SANS, round(11.5 * scale)),
        }

    def render(self, info: HudInfo, size: tuple[int, int]) -> np.ndarray | None:
        """Return a new RGBA image, or None when nothing changed."""
        if info == self._last and size == self._size:
            return None
        if size != self._size:
            self._build_fonts(size[1] / CANVAS_H)
            self._size = size
        self._last = info

        w, h = size
        s = h / CANVAS_H
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        f = self._fonts
        margin = round(16 * s)

        # --- FPS, top right ---------------------------------------------------
        fps_txt = f"{info.fps}"
        label = "FPS"
        lw = d.textlength(label, font=f["small"])
        fw = d.textlength(fps_txt, font=f["fps"])
        x_label = w - margin - lw
        y = margin
        d.text((x_label - fw - 6 * s, y), fps_txt, font=f["fps"], fill=WHITE + (235,))
        d.text((x_label, y + 9 * s), label, font=f["small"], fill=DIM + (220,))
        sub = f"CAM {info.cam_fps}  ·  HAND {info.track_fps}"
        sw = d.textlength(sub, font=f["mono_small"])
        d.text((w - margin - sw, y + 30 * s), sub, font=f["mono_small"], fill=DIM + (170,))

        # --- Status chip, top left -----------------------------------------------
        colour = MAGENTA if info.status_live else CYAN
        tw = d.textlength(info.status, font=f["status"])
        chip_h = round(24 * s)
        chip = (margin, margin, margin + tw + round(34 * s), margin + chip_h)
        d.rounded_rectangle(chip, radius=chip_h // 2, fill=(10, 14, 28, 150), outline=colour + (120,), width=max(1, round(s)))
        cy = margin + chip_h / 2
        r = 3.6 * s
        d.ellipse((margin + 12 * s - r, cy - r, margin + 12 * s + r, cy + r), fill=colour + (255,))
        d.text((margin + 22 * s, cy), info.status, font=f["status"], fill=WHITE + (235,), anchor="lm")

        # --- Warning + hint, bottom left -----------------------------------------
        y_hint = h - margin - round(12 * s)
        d.text((margin, y_hint), HINT, font=f["small"], fill=DIM + (150,))
        if info.warning:
            d.text((margin, y_hint - round(20 * s)), info.warning, font=f["status"], fill=AMBER + (235,))

        return np.asarray(img, dtype=np.uint8)
