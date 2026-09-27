"""All OpenGL work lives here (via ModernGL).

The renderer is deliberately "dumb": it receives plain data (camera frame,
height grid, vortex list, particle buffer...) and draws it. It never decides
anything about gestures or physics, so it can also run off-screen for
screenshots and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import moderngl
import numpy as np

from .config import CANVAS_ASPECT, CANVAS_H, CANVAS_W, MAX_VORTICES, SHADER_DIR, Settings
from .vortex import Vortex


def camera_crop_scale(cam_w: int, cam_h: int) -> tuple[float, float]:
    """Portion of the camera image visible on the 4:3 canvas ("cover" fit)."""
    cam_aspect = cam_w / max(cam_h, 1)
    if cam_aspect > CANVAS_ASPECT:
        return CANVAS_ASPECT / cam_aspect, 1.0
    return 1.0, cam_aspect / CANVAS_ASPECT


@dataclass
class FrameState:
    """Everything the renderer needs for one frame, in canvas pixels."""
    time: float = 0.0
    vortices: list[Vortex] = field(default_factory=list)
    pen: tuple[float, float] | None = None
    pen_down: bool = False
    clutch: tuple[float, float] | None = None
    clutch_closeness: float = 0.0
    clutch_on: bool = False
    particles: np.ndarray | None = None   # (N, 5) x, y, size, alpha, hue
    skeleton: np.ndarray | None = None    # (M, 6) x, y, r, g, b, a  (line list)
    joints: np.ndarray | None = None      # (K, 5) same layout as particles


def _load(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        vert = _load("fullscreen.vert")
        self.fabric_prog = ctx.program(vertex_shader=vert, fragment_shader=_load("fabric.frag"))
        self.hud_prog = ctx.program(vertex_shader=vert, fragment_shader=_load("hud.frag"))
        self.points_prog = ctx.program(vertex_shader=_load("points.vert"), fragment_shader=_load("points.frag"))
        self.lines_prog = ctx.program(vertex_shader=_load("lines.vert"), fragment_shader=_load("lines.frag"))

        tri = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self._tri_vbo = ctx.buffer(tri.tobytes())
        self.fabric_vao = ctx.vertex_array(self.fabric_prog, [(self._tri_vbo, "2f", "in_pos")])
        self.hud_vao = ctx.vertex_array(self.hud_prog, [(self._tri_vbo, "2f", "in_pos")])

        # Dynamic buffers grow on demand.
        self._points_vbo = ctx.buffer(reserve=4 * 5 * 4096, dynamic=True)
        self.points_vao = ctx.vertex_array(
            self.points_prog, [(self._points_vbo, "2f 1f 1f 1f", "in_pos", "in_size", "in_alpha", "in_hue")]
        )
        self._lines_vbo = ctx.buffer(reserve=4 * 6 * 512, dynamic=True)
        self.lines_vao = ctx.vertex_array(self.lines_prog, [(self._lines_vbo, "2f 4f", "in_pos", "in_color")])

        self.cam_tex: moderngl.Texture | None = None
        self.cam_size = (0, 0)
        self.height_tex: moderngl.Texture | None = None
        self.hud_tex: moderngl.Texture | None = None

        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    # ------------------------------------------------------------- uploads
    def upload_camera(self, frame_rgb: np.ndarray) -> None:
        h, w = frame_rgb.shape[:2]
        if self.cam_tex is None or self.cam_size != (w, h):
            if self.cam_tex is not None:
                self.cam_tex.release()
            self.cam_tex = self.ctx.texture((w, h), 3)
            self.cam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.cam_tex.repeat_x = self.cam_tex.repeat_y = False
            self.cam_size = (w, h)
        self.cam_tex.write(np.ascontiguousarray(frame_rgb), alignment=1)

    def upload_height(self, height: np.ndarray) -> None:
        h, w = height.shape
        if self.height_tex is None or self.height_tex.size != (w, h):
            self.height_tex = self.ctx.texture((w, h), 1, dtype="f4")
            self.height_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.height_tex.repeat_x = self.height_tex.repeat_y = False
        self.height_tex.write(np.ascontiguousarray(height, dtype=np.float32))

    def upload_hud(self, rgba: np.ndarray) -> None:
        h, w = rgba.shape[:2]
        if self.hud_tex is None or self.hud_tex.size != (w, h):
            if self.hud_tex is not None:
                self.hud_tex.release()
            self.hud_tex = self.ctx.texture((w, h), 4)
            self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_tex.write(np.ascontiguousarray(rgba))

    # -------------------------------------------------------------- render
    def render(self, state: FrameState, settings: Settings, viewport: tuple[int, int]) -> None:
        ctx = self.ctx
        ctx.viewport = (0, 0, viewport[0], viewport[1])
        ctx.disable(moderngl.BLEND)
        self._draw_fabric(state, settings)

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE      # additive glow
        px_scale = viewport[1] / CANVAS_H
        if settings.show_skeleton and state.skeleton is not None and len(state.skeleton):
            self._draw_lines(state.skeleton)
        if settings.show_skeleton and state.joints is not None and len(state.joints):
            self._draw_points(state.joints, px_scale)
        if state.particles is not None and len(state.particles):
            self._draw_points(state.particles, px_scale)

        if self.hud_tex is not None:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.hud_tex.use(0)
            self.hud_prog["u_hud"].value = 0
            self.hud_vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.disable(moderngl.BLEND)

    def _set(self, name: str, value) -> None:
        # Uniforms that the GLSL compiler optimised away are simply skipped.
        if name in self.fabric_prog:
            self.fabric_prog[name].value = value

    def _draw_fabric(self, state: FrameState, s: Settings) -> None:
        to_p = 1.0 / CANVAS_H   # canvas px -> aspect space

        has_cam = self.cam_tex is not None
        if has_cam:
            self.cam_tex.use(0)
            self._set("u_cam_scale", camera_crop_scale(*self.cam_size))
        self._set("u_cam", 0)
        if self.height_tex is not None:
            self.height_tex.use(1)
        self._set("u_height", 1)

        self._set("u_has_cam", has_cam)
        self._set("u_aspect", CANVAS_ASPECT)
        self._set("u_canvas_h", float(CANVAS_H))
        self._set("u_time", state.time)
        self._set("u_dim", s.camera_dim)
        self._set("u_brightness", s.brightness)
        self._set("u_contrast", s.contrast)
        self._set("u_saturation", s.saturation)
        self._set("u_grid", s.grid_opacity)

        vort = np.zeros((MAX_VORTICES, 4), dtype="f4")
        phase = np.zeros(MAX_VORTICES, dtype="f4")
        for i, v in enumerate(state.vortices[:MAX_VORTICES]):
            vort[i] = (v.pos[0] * to_p, v.pos[1] * to_p, v.radius * to_p, v.strength)
            phase[i] = v.phase
        self._set("u_vort_count", min(len(state.vortices), MAX_VORTICES))
        if "u_vort" in self.fabric_prog:
            self.fabric_prog["u_vort"].write(vort.tobytes())
        if "u_vort_phase" in self.fabric_prog:
            self.fabric_prog["u_vort_phase"].write(phase.tobytes())

        if state.pen is not None:
            self._set("u_pen", (state.pen[0] * to_p, state.pen[1] * to_p, float(state.pen_down), 1.0))
        else:
            self._set("u_pen", (0.0, 0.0, 0.0, 0.0))
        if state.clutch is not None:
            self._set("u_clutch", (state.clutch[0] * to_p, state.clutch[1] * to_p, state.clutch_closeness, 1.0))
        else:
            self._set("u_clutch", (0.0, 0.0, 0.0, 0.0))
        self._set("u_clutch_on", float(state.clutch_on))

        self.fabric_vao.render(moderngl.TRIANGLES, vertices=3)

    def _draw_points(self, data: np.ndarray, px_scale: float) -> None:
        raw = np.ascontiguousarray(data, dtype="f4").tobytes()
        if len(raw) > self._points_vbo.size:
            self._points_vbo.orphan(len(raw) * 2)
        self._points_vbo.write(raw)
        self.points_prog["u_canvas"].value = (float(CANVAS_W), float(CANVAS_H))
        self.points_prog["u_px_scale"].value = px_scale
        self.points_vao.render(moderngl.POINTS, vertices=len(data))

    def _draw_lines(self, data: np.ndarray) -> None:
        raw = np.ascontiguousarray(data, dtype="f4").tobytes()
        if len(raw) > self._lines_vbo.size:
            self._lines_vbo.orphan(len(raw) * 2)
        self._lines_vbo.write(raw)
        self.lines_prog["u_canvas"].value = (float(CANVAS_W), float(CANVAS_H))
        self.lines_vao.render(moderngl.LINES, vertices=len(data))
