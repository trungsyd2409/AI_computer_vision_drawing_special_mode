"""Separate "Camera Settings" window, opened with the X key.

Built with Tkinter because it ships with Python on every platform. The window
does not run its own ``mainloop``; the main render loop calls :meth:`pump`
once per frame instead, so both windows share one thread and stay responsive.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import fields
from tkinter import ttk

from .camera import CameraStream
from .config import BACKENDS, FPS_CHOICES, RESOLUTIONS, Settings

BG = "#0d1018"
CARD = "#151a26"
LINE = "#232a3b"
TEXT = "#e6ecff"
MUTED = "#8a94ad"
ACCENT = "#5ae8ff"
GOOD = "#5cf2a8"
WARN = "#ffbe5a"
BAD = "#ff5c7a"


class SettingsWindow:
    def __init__(self, settings: Settings, camera: CameraStream, on_clear=None):
        self.settings = settings
        self.camera = camera
        self.on_clear = on_clear
        self._defaults = Settings()
        self.root = tk.Tk()
        self.root.withdraw()          # hidden root; the visible window is a Toplevel
        self.win: tk.Toplevel | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._status: tk.Label | None = None
        self._alive = True
        self._style_ready = False

    # ------------------------------------------------------------------ API
    @property
    def is_open(self) -> bool:
        return self.win is not None

    def toggle(self) -> None:
        if self.win is None:
            self._build()
        else:
            self.close()

    def close(self) -> None:
        if self.win is not None:
            self.win.destroy()
            self.win = None
            self._vars.clear()

    def pump(self) -> None:
        """Process pending Tk events without blocking."""
        if not self._alive:
            return
        try:
            self.root.update()
        except tk.TclError:
            self._alive = False

    def destroy(self) -> None:
        if self._alive:
            self._alive = False
            self.root.destroy()

    # ---------------------------------------------------------------- build
    def _setup_style(self) -> None:
        if self._style_ready:
            return
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=CARD, foreground=TEXT, fieldbackground=BG,
                        bordercolor=LINE, lightcolor=CARD, darkcolor=CARD, font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=CARD)
        style.configure("Bg.TFrame", background=BG)
        style.configure("TLabel", background=CARD, foreground=TEXT)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED)
        style.configure("Head.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 13))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED)
        style.configure("Section.TLabel", background=CARD, foreground=ACCENT, font=("Segoe UI Semibold", 9))
        style.configure("TCheckbutton", background=CARD, foreground=TEXT, indicatorbackground=BG,
                        indicatorforeground=ACCENT)
        style.map("TCheckbutton", background=[("active", CARD)], indicatorbackground=[("selected", BG)])
        style.configure("TCombobox", fieldbackground=BG, background=CARD, foreground=TEXT,
                        arrowcolor=ACCENT, selectbackground=BG, selectforeground=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", BG)], foreground=[("readonly", TEXT)])
        style.configure("Horizontal.TScale", background=ACCENT, troughcolor=LINE,
                        bordercolor=CARD, lightcolor=ACCENT, darkcolor=ACCENT, sliderthickness=12)
        style.configure("TButton", background=LINE, foreground=TEXT, borderwidth=0, padding=(12, 6))
        style.map("TButton", background=[("active", "#2e3850")])
        self.root.option_add("*TCombobox*Listbox.background", BG)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#243049")
        self._style_ready = True

    def _build(self) -> None:
        self._setup_style()
        win = tk.Toplevel(self.root)
        win.title("Camera Settings")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", self.close)
        win.bind("<Key-x>", lambda _e: self.close())
        win.bind("<Escape>", lambda _e: self.close())
        self.win = win

        outer = ttk.Frame(win, style="Bg.TFrame", padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Camera Settings", style="Head.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self._status = tk.Label(outer, text="", bg=BG, fg=MUTED, font=("Consolas", 9), anchor="w")
        self._status.grid(row=1, column=0, columnspan=2, sticky="we", pady=(2, 12))

        left = ttk.Frame(outer, style="Bg.TFrame")
        right = ttk.Frame(outer, style="Bg.TFrame")
        left.grid(row=2, column=0, sticky="n", padx=(0, 8))
        right.grid(row=2, column=1, sticky="n", padx=(8, 0))

        s = self.settings
        dev = self._card(left, "DEVICE  ·  reopens the camera")
        self._combo(dev, "Camera", "camera_index", [str(i) for i in range(5)], str(s.camera_index),
                    lambda v: int(v), reopen=True)
        self._combo(dev, "Backend", "backend", BACKENDS, s.backend, str, reopen=True)
        self._combo(dev, "Resolution", "resolution", [f"{w} x {h}" for w, h in RESOLUTIONS],
                    f"{s.resolution[0]} x {s.resolution[1]}",
                    lambda v: tuple(int(x) for x in v.split(" x ")), reopen=True)
        self._combo(dev, "Target FPS", "target_fps", [str(f) for f in FPS_CHOICES], str(s.target_fps),
                    int, reopen=True)
        self._check(dev, "MJPG stream (needed for 60 fps on most webcams)", "use_mjpg", reopen=True)

        exp = self._card(left, "EXPOSURE  ·  driver dependent")
        self._check(exp, "Auto exposure (turn off if FPS drops in dim light)", "auto_exposure", live=True)
        self._slider(exp, "Exposure", "exposure", -13.0, -1.0, "{:.0f}", live=True)

        img = self._card(left, "IMAGE")
        self._check(img, "Mirror view", "mirror")
        self._slider(img, "Brightness", "brightness", -0.5, 0.5, "{:+.2f}")
        self._slider(img, "Contrast", "contrast", 0.5, 1.8, "{:.2f}")
        self._slider(img, "Saturation", "saturation", 0.0, 1.6, "{:.2f}")
        self._slider(img, "Camera visibility", "camera_dim", 0.2, 1.0, "{:.2f}")

        fx = self._card(right, "EFFECT")
        self._slider(fx, "Brush size (px)", "brush_radius", 14.0, 70.0, "{:.0f}")
        self._slider(fx, "Dent depth", "dent_depth", 0.3, 1.6, "{:.2f}")
        self._slider(fx, "Recovery time (s)", "recovery_seconds", 0.8, 8.0, "{:.1f}")
        self._slider(fx, "Vortex fade (s)", "vortex_decay", 0.6, 8.0, "{:.1f}")
        self._slider(fx, "Grid glow", "grid_opacity", 0.0, 1.5, "{:.2f}")
        self._check(fx, "Show hand skeleton", "show_skeleton")

        ges = self._card(right, "GESTURE")
        self._slider(ges, "Pinch trigger distance", "pinch_on", 0.25, 0.65, "{:.2f}")
        ttk.Label(ges, text="Higher = easier to trigger, lower = fewer false presses.",
                  style="Muted.TLabel", wraplength=250).pack(anchor="w", pady=(2, 0))

        buttons = ttk.Frame(right, style="Bg.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Reset defaults", command=self._reset).pack(side="left")
        if self.on_clear is not None:
            ttk.Button(buttons, text="Clear canvas", command=self.on_clear).pack(side="left", padx=(8, 0))

        self._refresh_status()

    # ------------------------------------------------------------- widgets
    def _card(self, parent, title: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 10, 14, 12))
        card.pack(fill="x", pady=(0, 10))
        ttk.Label(card, text=title, style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        return card

    def _row(self, parent, label: str) -> ttk.Frame:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=20).pack(side="left")
        return row

    def _combo(self, parent, label, key, values, current, parse, reopen=False):
        row = self._row(parent, label)
        var = tk.StringVar(value=current)
        box = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=16)
        box.pack(side="right")

        def changed(_event=None):
            setattr(self.settings, key, parse(var.get()))
            if reopen:
                self.camera.request_reopen()
        box.bind("<<ComboboxSelected>>", changed)
        self._vars[key] = var

    def _check(self, parent, label, key, reopen=False, live=False):
        var = tk.BooleanVar(value=bool(getattr(self.settings, key)))

        def changed():
            setattr(self.settings, key, bool(var.get()))
            if reopen:
                self.camera.request_reopen()
            if live:
                self.camera.request_live_update()
        ttk.Checkbutton(parent, text=label, variable=var, command=changed).pack(anchor="w", pady=3)
        self._vars[key] = var

    def _slider(self, parent, label, key, lo, hi, fmt, live=False):
        row = self._row(parent, label)
        var = tk.DoubleVar(value=float(getattr(self.settings, key)))
        value_lbl = ttk.Label(row, text=fmt.format(var.get()), width=6, style="Muted.TLabel", anchor="e")
        value_lbl.pack(side="right")
        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal", length=150)
        scale.pack(side="right", padx=(6, 6))

        def changed(*_):
            v = float(var.get())
            setattr(self.settings, key, v)
            if key == "pinch_on":
                self.settings.pinch_off = v + 0.18   # keep the hysteresis gap
            value_lbl.configure(text=fmt.format(v))
            if live:
                self.camera.request_live_update()
        var.trace_add("write", changed)
        self._vars[key] = var

    # --------------------------------------------------------------- status
    def _refresh_status(self) -> None:
        if self.win is None or self._status is None:
            return
        cam = self.camera
        s = self.settings
        if not cam.is_open:
            text, colour = f"●  {cam.error or 'Opening camera...'}", BAD
        else:
            w, h = cam.actual_size
            fps = cam.fps
            colour = GOOD if fps >= min(s.target_fps, 30) * 0.9 else WARN if fps >= 15 else BAD
            text = (f"●  Live {w} x {h}  ·  {fps:4.1f} fps measured  ·  "
                    f"target {s.target_fps}  ·  driver reports {cam.reported_fps:.0f}")
        self._status.configure(text=text, fg=colour)
        self.win.after(500, self._refresh_status)

    def _reset(self) -> None:
        reopen_keys = {"camera_index", "backend", "resolution", "target_fps", "use_mjpg"}
        needs_reopen = any(getattr(self.settings, k) != getattr(self._defaults, k) for k in reopen_keys)
        for f in fields(Settings):
            setattr(self.settings, f.name, getattr(self._defaults, f.name))
        if needs_reopen:
            self.camera.request_reopen()
        self.camera.request_live_update()
        # Rebuild so every widget shows the default value again.
        self.close()
        self._build()
