"""
Vinyl Transfer Cleanup Utility
A desktop tool for cleaning up audio digitized from vinyl records.

Requires: numpy, scipy, matplotlib
Optional:  sounddevice (in-app playback)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, stft, istft, bilinear_zpk, zpk2tf
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import threading
import os
import traceback

# NOTE: scipy.signal.medfilt is intentionally NOT imported.
# It has a known heap-corruption bug when called repeatedly, causing
# "free(): invalid size" crashes. All median filtering is done via
# numpy or scipy.ndimage instead.

try:
    import sounddevice as sd
    PLAYBACK_AVAILABLE = True
except ImportError:
    PLAYBACK_AVAILABLE = False


# ---------------------------------------------------------------------------
# TOOLTIP HELPER
# ---------------------------------------------------------------------------

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tw = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _show(self, _event=None):
        self._hide()
        x = self.widget.winfo_rootx() + 28
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tw = tk.Toplevel(self.widget)
        self._tw.wm_overrideredirect(True)
        self._tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tw, text=self.text, justify=tk.LEFT,
            background="#ffffcc", relief="solid", borderwidth=1,
            font=("Arial", 9), wraplength=320, padx=6, pady=4
        ).pack()

    def _hide(self, _event=None):
        if self._tw:
            self._tw.destroy()
            self._tw = None


def tip(widget, text):
    Tooltip(widget, text)
    return widget


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

class VinylCleanupApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vinyl Transfer Cleanup Utility")
        self.root.geometry("1200x820")
        self.root.minsize(960, 660)

        self.filepath = None
        self.audio_data = None        # original, never mutated
        self.sample_rate = None
        self.processed_data = None    # full-track result
        self.preview_data = None      # region preview result
        self.noise_profile = None
        self.undo_stack = []

        self.region_start = None      # seconds
        self.region_end = None

        self.is_playing = False

        self._setup_ui()

    # -----------------------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------------------

    def _setup_ui(self):
        info_bar = ttk.Frame(self.root, padding=(10, 4))
        info_bar.pack(side=tk.TOP, fill=tk.X)
        self.lbl_file_info = ttk.Label(info_bar, text="No file loaded", font=("Courier", 10))
        self.lbl_file_info.pack(side=tk.LEFT)
        self.lbl_status = ttk.Label(info_bar, text="Ready", foreground="gray")
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        bottom_bar = ttk.Frame(self.root, padding=(10, 6))
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_bottom_bar(bottom_bar)

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=3)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

    def _build_bottom_bar(self, bar):
        self.progress = ttk.Progressbar(bar, mode="determinate", length=150)
        self.progress.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_preview = tip(
            ttk.Button(bar, text="Process Preview Region",
                       command=self.start_preview, state=tk.DISABLED),
            "Process ONLY the selected region — much faster than a full run.\n"
            "Select a region by dragging on the top waveform, or type a range.\n\n"
            "Use this to dial in settings before committing to the whole file."
        )
        self.btn_preview.pack(side=tk.LEFT, padx=3)

        self.btn_apply_full = tip(
            ttk.Button(bar, text="Apply to Full Track",
                       command=self.start_full_processing, state=tk.DISABLED),
            "Apply the current settings to the ENTIRE file.\n"
            "Always processes from the original — re-running replaces the previous result."
        )
        self.btn_apply_full.pack(side=tk.LEFT, padx=3)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.btn_save = tip(
            ttk.Button(bar, text="Save Processed WAV",
                       command=self.save_file, state=tk.DISABLED),
            "Export the full-track result to a new WAV file.\n"
            "Only available after 'Apply to Full Track'.\n"
            "The original file is never modified."
        )
        self.btn_save.pack(side=tk.LEFT, padx=3)

        self.btn_undo = tip(
            ttk.Button(bar, text="Undo", command=self.undo, state=tk.DISABLED),
            "Step back to the previous full-track result. Keeps up to 10 levels."
        )
        self.btn_undo.pack(side=tk.LEFT, padx=3)

        self.btn_reset = tip(
            ttk.Button(bar, text="Reset to Original",
                       command=self.reset_to_original, state=tk.DISABLED),
            "Discard all processing and return to the original loaded file."
        )
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        if PLAYBACK_AVAILABLE:
            ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

            self.btn_play_orig = tip(
                ttk.Button(bar, text="Play Original",
                           command=self.play_original, state=tk.DISABLED),
                "Play the original audio.\n\n"
                "If a preview region is selected, only that region plays — so you\n"
                "can A/B compare it directly against the processed preview."
            )
            self.btn_play_orig.pack(side=tk.LEFT, padx=3)

            self.btn_play_preview = tip(
                ttk.Button(bar, text="Play Preview",
                           command=self.play_preview, state=tk.DISABLED),
                "Play the processed preview region.\n"
                "Compare against 'Play Original' — both will play the same time range."
            )
            self.btn_play_preview.pack(side=tk.LEFT, padx=3)

            self.btn_play_proc = tip(
                ttk.Button(bar, text="Play Full Result",
                           command=self.play_processed, state=tk.DISABLED),
                "Play the entire full-track processed result."
            )
            self.btn_play_proc.pack(side=tk.LEFT, padx=3)

            self.btn_stop = tip(
                ttk.Button(bar, text="Stop", command=self.stop_playback, state=tk.DISABLED),
                "Stop any audio currently playing."
            )
            self.btn_stop.pack(side=tk.LEFT, padx=3)

    def _build_left_panel(self, parent):
        file_frame = ttk.LabelFrame(parent, text="File", padding=8)
        file_frame.pack(fill=tk.X, pady=(0, 6))
        tip(
            ttk.Button(file_frame, text="Load WAV File", command=self.load_file),
            "Open a WAV file. Stereo and mono are both supported.\n"
            "The original file is never modified."
        ).pack(fill=tk.X)

        region_frame = ttk.LabelFrame(parent, text="Preview Region", padding=8)
        region_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            region_frame,
            text="Drag on the top waveform to select a region, or type a "
                 "range below. 'Play Original' will also respect this region "
                 "for a direct A/B comparison.",
            foreground="gray", wraplength=210, justify=tk.LEFT, font=("Arial", 8)
        ).pack(anchor=tk.W, pady=(0, 5))

        self.lbl_region = ttk.Label(region_frame, text="No region selected", foreground="gray")
        self.lbl_region.pack(anchor=tk.W)

        range_row = ttk.Frame(region_frame)
        range_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(range_row, text="Start (s):").pack(side=tk.LEFT)
        self.manual_start = tk.DoubleVar(value=0.0)
        ttk.Spinbox(range_row, from_=0.0, to=99999.0, increment=1.0,
                    textvariable=self.manual_start, width=6, format="%.1f").pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(range_row, text="End (s):").pack(side=tk.LEFT)
        self.manual_end = tk.DoubleVar(value=60.0)
        ttk.Spinbox(range_row, from_=0.0, to=99999.0, increment=1.0,
                    textvariable=self.manual_end, width=6, format="%.1f").pack(side=tk.LEFT, padx=2)

        tip(
            ttk.Button(region_frame, text="Use Typed Range", command=self._use_typed_range),
            "Set the region to the start/end times above.\n"
            "Example: 0 to 60 to test just the first minute."
        ).pack(fill=tk.X, pady=(6, 0))

        ttk.Button(
            region_frame, text="Clear Region (use full file)",
            command=self._clear_region
        ).pack(fill=tk.X, pady=(3, 0))

        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self._build_tab_clicks(nb)
        self._build_tab_filters(nb)
        self._build_tab_levels(nb)

    def _slider_row(self, parent, variable, from_, to, fmt, label_width=7):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)
        lbl = ttk.Label(frame, text=fmt.format(variable.get()), width=label_width, anchor=tk.E)
        lbl.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL,
            command=lambda v: lbl.config(text=fmt.format(float(v)))
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_tab_clicks(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Clicks & Noise")

        ttk.Label(tab, text="De-Click / De-Pop", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.do_declick = tk.BooleanVar(value=True)
        tip(
            ttk.Checkbutton(tab, text="Enable De-Click", variable=self.do_declick),
            "Detects sharp impulse noises (clicks, pops, crackle) using a second-order\n"
            "difference detector, which responds to discontinuities rather than amplitude.\n"
            "This means it does NOT introduce artifacts on high-frequency content.\n\n"
            "Detected regions are repaired by linear interpolation from the samples\n"
            "on either side of the click — not replaced with a flat reference.\n\n"
            "Safe to leave on for virtually every vinyl transfer."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text="Detection Sensitivity\n"
                 "Higher = less aggressive (misses subtle clicks)\n"
                 "Lower = more aggressive (may soften loud transients like snare hits)",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 0))
        self.click_sens = tk.DoubleVar(value=10.0)
        self._slider_row(tab, self.click_sens, 1.0, 30.0, "{:.1f}")

        ttk.Label(
            tab,
            text="Repair Window (ms)\n"
                 "Width of the region replaced around each detected click.\n"
                 "1-3ms handles crackle. Up to 8-10ms for heavy pops.",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(5, 0))
        self.click_window = tk.DoubleVar(value=2.0)
        self._slider_row(tab, self.click_window, 0.5, 10.0, "{:.1f}")

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="Spectral Noise Reduction", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.do_noise_reduce = tk.BooleanVar(value=False)
        tip(
            ttk.Checkbutton(tab, text="Enable Noise Reduction", variable=self.do_noise_reduce),
            "Targets continuous broadband noise (hiss, hum, surface noise) in the\n"
            "frequency domain. Has no effect on clicks — that is what de-click is for.\n\n"
            "You MUST capture a noise profile first (see below).\n\n"
            "Start with Reduction Strength around 2.0. Above 4.0 causes a warbling\n"
            "metallic artifact called 'musical noise' that is worse than the hiss."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text="Capture a noise profile first:\n"
                 "1. Find a section with no music — the lead-in groove before\n"
                 "   the music starts is ideal (pure surface noise only).\n"
                 "2. Enter the time range and click Capture.",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 2))

        time_frame = ttk.Frame(tab)
        time_frame.pack(fill=tk.X)
        ttk.Label(time_frame, text="Start:").pack(side=tk.LEFT)
        self.noise_start = tk.DoubleVar(value=0.0)
        ttk.Spinbox(time_frame, from_=0.0, to=9999.0, increment=0.1,
                    textvariable=self.noise_start, width=5, format="%.2f").pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(time_frame, text="End (s):").pack(side=tk.LEFT)
        self.noise_end = tk.DoubleVar(value=1.0)
        ttk.Spinbox(time_frame, from_=0.0, to=9999.0, increment=0.1,
                    textvariable=self.noise_end, width=5, format="%.2f").pack(side=tk.LEFT, padx=2)

        tip(
            ttk.Button(tab, text="Capture Noise Profile", command=self.capture_noise_profile),
            "Records the frequency fingerprint of the selected region.\n"
            "This is subtracted from every audio frame during processing.\n"
            "Always taken from the original file."
        ).pack(fill=tk.X, pady=6)

        self.lbl_noise_profile = ttk.Label(tab, text="No profile captured",
                                            foreground="gray", font=("Arial", 8))
        self.lbl_noise_profile.pack(anchor=tk.W)

        ttk.Label(
            tab,
            text="Reduction Strength (alpha)\n"
                 "1.5-3.0 is a good starting range.\n"
                 "Above 4.0 introduces metallic warbling artifacts.",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(8, 0))
        self.noise_alpha = tk.DoubleVar(value=2.0)
        self._slider_row(tab, self.noise_alpha, 0.5, 6.0, "{:.1f}")

        ttk.Label(
            tab,
            text="Spectral Floor (beta)\n"
                 "Minimum retained signal per bin. Raise to reduce warbling at\n"
                 "the cost of slightly less noise removed. Keep 0.01-0.05.",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(5, 0))
        self.noise_beta = tk.DoubleVar(value=0.02)
        self._slider_row(tab, self.noise_beta, 0.001, 0.2, "{:.3f}", label_width=6)

    def _build_tab_filters(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Filters")

        ttk.Label(tab, text="Rumble Filter (High-Pass)", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.do_rumble = tk.BooleanVar(value=True)
        tip(
            ttk.Checkbutton(tab, text="Enable Rumble Filter", variable=self.do_rumble),
            "Cuts everything below the cutoff frequency.\n\n"
            "Turntable motors produce low-frequency rumble (5-30 Hz) that wastes\n"
            "headroom and causes woofer pumping. The default 30 Hz cutoff is safe\n"
            "for virtually all music. Only raise it for severe motor noise.\n"
            "Going above 80 Hz starts audibly cutting bass in the music."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text="Cutoff Frequency (Hz)\n"
                 "30 Hz default is safe. Raise to 50-80 Hz for severe rumble only.",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 0))
        self.rumble_freq = tk.DoubleVar(value=30.0)
        self._slider_row(tab, self.rumble_freq, 10.0, 150.0, "{:.0f}")

        ttk.Label(tab, text="Filter Steepness (order)").pack(anchor=tk.W, pady=(5, 0))
        self.rumble_order = tk.IntVar(value=4)
        rof = ttk.Frame(tab)
        rof.pack(fill=tk.X)
        for o, desc in ((2, "gentle"), (4, "standard"), (6, "steep"), (8, "very steep")):
            ttk.Radiobutton(rof, text=f"{o} ({desc})", variable=self.rumble_order, value=o).pack(anchor=tk.W)

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="Hiss Filter (Low-Pass)", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        self.do_hiss = tk.BooleanVar(value=False)
        tip(
            ttk.Checkbutton(tab, text="Enable Hiss Filter", variable=self.do_hiss),
            "Cuts everything above the cutoff frequency.\n\n"
            "WARNING: This is a blunt cut. It removes high-frequency musical content\n"
            "(cymbals, string overtones, vocal air) along with the hiss.\n\n"
            "Use Spectral Noise Reduction instead — it is far more targeted.\n"
            "Only enable this if you specifically need a hard bandwidth limit."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text="Cutoff Frequency (Hz)\n"
                 "14000+ retains most musical content.\n"
                 "Below 10000 will audibly dull the recording.",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 0))
        self.hiss_freq = tk.DoubleVar(value=14000.0)
        self._slider_row(tab, self.hiss_freq, 3000.0, 20000.0, "{:.0f}", label_width=7)

        ttk.Label(tab, text="Filter Steepness (order)").pack(anchor=tk.W, pady=(5, 0))
        self.hiss_order = tk.IntVar(value=4)
        hof = ttk.Frame(tab)
        hof.pack(fill=tk.X)
        for o, desc in ((2, "gentle"), (4, "standard"), (6, "steep"), (8, "very steep")):
            ttk.Radiobutton(hof, text=f"{o} ({desc})", variable=self.hiss_order, value=o).pack(anchor=tk.W)

    def _build_tab_levels(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Levels")

        ttk.Label(tab, text="DC Offset Removal", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_dc_remove = tk.BooleanVar(value=True)
        tip(
            ttk.Checkbutton(tab, text="Remove DC Offset", variable=self.do_dc_remove),
            "Removes a constant voltage bias from the waveform.\n\n"
            "Some phono preamps and ADCs shift the signal above or below zero,\n"
            "wasting headroom and causing asymmetric clipping.\n"
            "Always safe to leave on."
        ).pack(anchor=tk.W)

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="Normalization", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_normalize = tk.BooleanVar(value=True)
        tip(
            ttk.Checkbutton(tab, text="Enable Normalization", variable=self.do_normalize),
            "Adjusts the overall volume of the recording.\n\n"
            "PEAK mode: Scales so the loudest sample hits your target. Always safe.\n\n"
            "RMS mode: Scales to a target average loudness. Can apply very large\n"
            "gain on quiet recordings. A hard +18 dB cap is enforced, but always\n"
            "test with preview before applying to the full track."
        ).pack(anchor=tk.W)

        self.norm_mode = tk.StringVar(value="peak")
        tip(
            ttk.Radiobutton(tab, text="Peak  --  safe, recommended",
                            variable=self.norm_mode, value="peak"),
            "Scales so the loudest sample hits your target.\n"
            "Never clips. Start here."
        ).pack(anchor=tk.W, pady=(8, 0))
        tip(
            ttk.Radiobutton(tab, text="RMS  --  perceived loudness (use with caution)",
                            variable=self.norm_mode, value="rms"),
            "Scales to a target average loudness level.\n"
            "Can apply massive gain on quiet recordings.\n"
            "ALWAYS preview before applying to full track."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text="Target Level (dBFS)\n"
                 "Peak: -1.0 dBFS is standard (1 dB headroom).\n"
                 "RMS: -18 to -14 dBFS is typical for vinyl.\n"
                 "Hard +18 dB gain cap enforced in both modes.",
            foreground="gray", font=("Arial", 8), justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(8, 0))
        self.norm_target = tk.DoubleVar(value=-1.0)
        self._slider_row(tab, self.norm_target, -30.0, -0.1, "{:.1f}")

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="RIAA De-Emphasis", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_riaa = tk.BooleanVar(value=False)
        tip(
            ttk.Checkbutton(tab, text="Apply RIAA De-Emphasis", variable=self.do_riaa),
            "Applies the standard vinyl playback EQ curve (IEC 60098).\n\n"
            "LEAVE THIS OFF in almost every situation.\n\n"
            "Every standard phono preamp already applies RIAA equalization.\n"
            "You only need this if you connected your cartridge directly to a\n"
            "line-level input with NO phono preamp — the recording will sound\n"
            "extremely bass-heavy and muffled if that is the case.\n\n"
            "Enabling this on a normally-recorded file makes it sound thin and harsh."
        ).pack(anchor=tk.W)

    def _build_right_panel(self, parent):
        view_row = ttk.Frame(parent)
        view_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(view_row, text="View:").pack(side=tk.LEFT)
        self.view_mode = tk.StringVar(value="waveform")
        for label, val in (("Waveform", "waveform"), ("Spectrum", "spectrum"), ("Both", "both")):
            ttk.Radiobutton(view_row, text=label, variable=self.view_mode,
                            value=val, command=self._refresh_plots).pack(side=tk.LEFT, padx=5)

        self.lbl_selection_display = ttk.Label(
            view_row,
            text="Drag on the top waveform to select a preview region",
            foreground="gray", font=("Arial", 8)
        )
        self.lbl_selection_display.pack(side=tk.RIGHT, padx=10)

        self.fig = plt.Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._span_selector = None
        self._init_axes()

    # -----------------------------------------------------------------------
    # AXES / PLOTS
    # -----------------------------------------------------------------------

    def _init_axes(self):
        self.fig.clear()
        mode = self.view_mode.get()
        if mode == "both":
            axes = self.fig.subplots(2, 2)
            self._ax_ow, self._ax_os = axes[0]
            self._ax_pw, self._ax_ps = axes[1]
        else:
            axes = self.fig.subplots(2, 1)
            self._ax_ow, self._ax_pw = axes
            self._ax_os = self._ax_ps = None

        self._ax_ow.set_title("Original")
        self._ax_pw.set_title("Processed / Preview")
        self.fig.tight_layout(pad=2.0)
        self._attach_span_selector()
        self.canvas.draw()

    def _attach_span_selector(self):
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
            except Exception:
                pass
            self._span_selector = None

        self._span_selector = SpanSelector(
            self._ax_ow,
            onselect=self._on_span_select,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="steelblue"),
            interactive=True,
            drag_from_anywhere=True,
        )

    def _on_span_select(self, xmin, xmax):
        if self.audio_data is None:
            return
        duration = len(self.audio_data) / self.sample_rate
        xmin = max(0.0, xmin)
        xmax = min(duration, xmax)
        if xmax - xmin < 0.05:
            return
        self.region_start = xmin
        self.region_end = xmax
        self.manual_start.set(round(xmin, 2))
        self.manual_end.set(round(xmax, 2))
        self._update_region_label()

    def _use_typed_range(self):
        if self.audio_data is None:
            messagebox.showwarning("No File", "Load a WAV file first.")
            return
        s = float(self.manual_start.get())
        e = float(self.manual_end.get())
        duration = len(self.audio_data) / self.sample_rate
        if s >= e:
            messagebox.showerror("Invalid Range", "Start must be less than end.")
            return
        e = min(e, duration)
        if e - s < 0.1:
            messagebox.showerror("Invalid Range", "Range must be at least 0.1 seconds.")
            return
        self.region_start = s
        self.region_end = e
        self._update_region_label()

    def _clear_region(self):
        self.region_start = None
        self.region_end = None
        if self._span_selector is not None:
            try:
                self._span_selector.set_visible(False)
                self.canvas.draw_idle()
            except Exception:
                pass
        self.lbl_region.config(
            text="No region selected — full track will be processed",
            foreground="gray"
        )
        self.lbl_selection_display.config(text="No region selected")
        if self.audio_data is not None:
            self.btn_preview.config(state=tk.DISABLED)

    def _update_region_label(self):
        if self.region_start is None:
            return
        dur = self.region_end - self.region_start
        text = f"{self.region_start:.2f}s  to  {self.region_end:.2f}s  ({dur:.1f}s)"
        self.lbl_region.config(text=text, foreground="darkgreen")
        self.lbl_selection_display.config(text=f"Region: {text}")
        if self.audio_data is not None:
            self.btn_preview.config(state=tk.NORMAL)

    def _refresh_plots(self):
        self._init_axes()
        mode = self.view_mode.get()

        if mode in ("waveform", "both"):
            if self.audio_data is not None:
                self._draw_waveform(self.audio_data, self._ax_ow, "Original")
            lower = self.processed_data if self.processed_data is not None else self.preview_data
            label = "Processed (Full Track)" if self.processed_data is not None else "Preview Region"
            if lower is not None:
                self._draw_waveform(lower, self._ax_pw, label)

        if mode in ("spectrum", "both"):
            ax_o = self._ax_os if mode == "both" else self._ax_ow
            ax_p = self._ax_ps if mode == "both" else self._ax_pw
            if self.audio_data is not None:
                self._draw_spectrum(self.audio_data, ax_o, "Original" if mode == "spectrum" else "Spectrum (original)")
            lower = self.processed_data if self.processed_data is not None else self.preview_data
            if lower is not None:
                self._draw_spectrum(lower, ax_p, "Processed" if mode == "spectrum" else "Spectrum (processed)")

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def _draw_waveform(self, data, ax, title):
        ax.clear()
        ax.set_title(title, fontsize=9)
        if data is None:
            return
        step = max(1, len(data) // 60000)
        plot = data[::step]
        t = np.linspace(0, len(data) / self.sample_rate, len(plot))
        if plot.ndim > 1:
            ax.plot(t, plot[:, 0], color="#2196F3", alpha=0.7, linewidth=0.4, label="L")
            ax.plot(t, plot[:, 1], color="#FF5722", alpha=0.7, linewidth=0.4, label="R")
            ax.legend(loc="upper right", fontsize=7)
        else:
            ax.plot(t, plot, color="#2196F3", alpha=0.8, linewidth=0.4)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    def _draw_spectrum(self, data, ax, title):
        ax.clear()
        ax.set_title(title, fontsize=9)
        if data is None or self.sample_rate is None:
            return
        ch = data[:, 0].astype(np.float64) if data.ndim > 1 else data.astype(np.float64)
        fft_len = min(len(ch), 131072)
        seg = ch[:fft_len]
        window = np.hanning(len(seg))
        spectrum = np.abs(np.fft.rfft(seg * window))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / self.sample_rate)
        db = 20 * np.log10(np.maximum(spectrum / (spectrum.max() + 1e-10), 1e-10))
        ax.semilogx(freqs[1:], db[1:], color="#2196F3", alpha=0.85, linewidth=0.5)
        ax.set_xlim(20, self.sample_rate / 2)
        ax.set_ylim(-80, 5)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Level (dB)", fontsize=8)
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=7)

    # -----------------------------------------------------------------------
    # FILE I/O
    # -----------------------------------------------------------------------

    def load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            self.lbl_status.config(text="Loading...")
            self.root.update()
            rate, data = wavfile.read(filepath)

            self.filepath = filepath
            self.sample_rate = rate
            self.audio_data = data
            self.processed_data = None
            self.preview_data = None
            self.noise_profile = None
            self.undo_stack = []
            self.region_start = None
            self.region_end = None

            dur = len(data) / rate
            ch = "Stereo" if data.ndim > 1 else "Mono"
            self.lbl_file_info.config(
                text=(f"{os.path.basename(filepath)}  |  {rate} Hz  |  "
                      f"{data.dtype}  |  {ch}  |  {dur:.1f}s  ({dur/60:.1f} min)")
            )

            self.btn_preview.config(state=tk.DISABLED)
            self.btn_apply_full.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)
            self.btn_reset.config(state=tk.DISABLED)
            self.lbl_noise_profile.config(text="No profile captured", foreground="gray")
            self.lbl_region.config(
                text="No region selected — full track will be processed",
                foreground="gray"
            )
            self.lbl_selection_display.config(
                text="Drag on the top waveform to select a preview region"
            )

            if PLAYBACK_AVAILABLE:
                self.btn_play_orig.config(state=tk.NORMAL)
                self.btn_play_preview.config(state=tk.DISABLED)
                self.btn_play_proc.config(state=tk.DISABLED)

            self.manual_end.set(min(60.0, round(dur, 1)))
            self._refresh_plots()
            self.lbl_status.config(text=f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load file:\n{e}")
            self.lbl_status.config(text="Error loading file.")

    def save_file(self):
        if self.processed_data is None:
            return
        base = os.path.basename(self.filepath) if self.filepath else "audio.wav"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile="cleaned_" + base
        )
        if filepath:
            try:
                wavfile.write(filepath, self.sample_rate, self.processed_data)
                self.lbl_status.config(text=f"Saved: {os.path.basename(filepath)}")
                messagebox.showinfo("Saved", f"File saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n{e}")

    # -----------------------------------------------------------------------
    # PLAYBACK
    # -----------------------------------------------------------------------

    def _to_float32(self, data):
        if data.dtype == np.int16:
            return data.astype(np.float32) / 32768.0
        if data.dtype == np.int32:
            return data.astype(np.float32) / 2147483648.0
        return data.astype(np.float32)

    def play_original(self):
        """
        If a region is selected, play only that slice of the original so the
        listener is comparing the same passage as the preview. If no region
        is selected, play the whole file.
        """
        if self.audio_data is None:
            return
        if self.region_start is not None and self.region_end is not None:
            si = int(self.region_start * self.sample_rate)
            ei = int(self.region_end * self.sample_rate)
            self._play_audio(self.audio_data[si:ei])
        else:
            self._play_audio(self.audio_data)

    def play_preview(self):
        if self.preview_data is not None:
            self._play_audio(self.preview_data)

    def play_processed(self):
        if self.processed_data is not None:
            self._play_audio(self.processed_data)

    def _play_audio(self, data):
        if not PLAYBACK_AVAILABLE or data is None:
            return
        self.stop_playback()
        self.is_playing = True
        self.btn_stop.config(state=tk.NORMAL)

        # Copy to a new array so the playback thread does not share memory
        # with anything the processing thread might touch.
        play_buf = np.array(self._to_float32(data), copy=True)

        def _run():
            try:
                sd.play(play_buf, self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self.is_playing = False
                self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

        threading.Thread(target=_run, daemon=True).start()

    def stop_playback(self):
        if PLAYBACK_AVAILABLE:
            sd.stop()
        self.is_playing = False
        if PLAYBACK_AVAILABLE:
            self.btn_stop.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # NOISE PROFILE
    # -----------------------------------------------------------------------

    def capture_noise_profile(self):
        if self.audio_data is None:
            messagebox.showwarning("No File", "Load a WAV file first.")
            return
        try:
            s0 = float(self.noise_start.get())
            s1 = float(self.noise_end.get())
            dur = len(self.audio_data) / self.sample_rate

            if s0 >= s1:
                messagebox.showerror("Invalid Range", "Start must be less than end.")
                return
            if s1 > dur:
                messagebox.showerror("Invalid Range", f"End exceeds file duration ({dur:.2f}s).")
                return
            if s1 - s0 < 0.1:
                messagebox.showerror("Invalid Range", "Noise sample must be at least 0.1s long.")
                return

            data = self.audio_data.astype(np.float64)
            si, ei = int(s0 * self.sample_rate), int(s1 * self.sample_rate)
            region = data[si:ei, 0] if data.ndim > 1 else data[si:ei]

            nperseg = 2048
            _, _, Zxx = stft(region, fs=self.sample_rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
            self.noise_profile = np.mean(np.abs(Zxx), axis=1)

            self.lbl_noise_profile.config(
                text=f"Profile: {s0:.2f}s - {s1:.2f}s ({s1-s0:.2f}s)",
                foreground="green"
            )
            self.do_noise_reduce.set(True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture noise profile:\n{e}")

    # -----------------------------------------------------------------------
    # DSP
    # -----------------------------------------------------------------------

    def _butter_hp(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype="high", analog=False)
        return filtfilt(b, a, data)

    def _butter_lp(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype="low", analog=False)
        return filtfilt(b, a, data)

    def _declick(self, ch_data, sensitivity, window_ms):
        """
        Click detection via second-order difference (discrete Laplacian).

        WHY: A median-filter residual approach was used previously but has two
        problems: (1) scipy.signal.medfilt has a known heap-corruption bug that
        causes crashes after repeated calls, and (2) for a short kernel the
        median of a high-frequency sinusoid is approximately zero, so the
        residual contains genuine audio which then gets flagged as clicks and
        replaced with near-zero values, introducing high-frequency crackle.

        The second-order difference (d2[n] = x[n+1] - 2*x[n] + x[n-1]) is a
        high-pass operator that responds strongly to sharp discontinuities and
        weakly to smooth audio content regardless of frequency. It does not
        require any reference signal and introduces no frequency-dependent bias.

        Detected regions are repaired by linear interpolation from the clean
        samples on either side, not replaced with a flat reference value.
        """
        ch = ch_data.astype(np.float64)

        # Pad by one sample on each end so the diff array is the same length
        d2 = np.diff(ch, n=2, prepend=[ch[0]], append=[ch[-1]])

        # Robust threshold: median absolute deviation scaled to a sigma equivalent.
        # MAD is used instead of std because std is heavily influenced by the
        # clicks themselves, which would raise the threshold and miss them.
        mad = np.median(np.abs(d2))
        if mad < 1e-10:
            return ch_data.copy()

        # 1.4826 is the consistency factor that makes MAD equivalent to sigma
        # for a Gaussian distribution.
        threshold = sensitivity * mad * 1.4826
        mask = np.abs(d2) > threshold

        if not np.any(mask):
            return ch_data.copy()

        # Dilate the mask to cover the full extent of each impulse
        dilation = max(1, int(self.sample_rate * window_ms / 1000.0))
        mask = binary_dilation(mask, structure=np.ones(dilation, dtype=bool))

        cleaned = ch.copy()
        indices = np.arange(len(ch))
        good = ~mask

        if np.sum(good) < 2:
            # If almost everything is masked the signal is probably corrupt;
            # return it untouched rather than interpolating garbage.
            return ch_data.copy()

        # Linear interpolation across each masked (click) region
        cleaned[mask] = np.interp(indices[mask], indices[good], ch[good])
        return cleaned

    def _spectral_nr(self, ch_data, noise_profile, alpha, beta):
        nperseg = 2048
        noverlap = nperseg * 3 // 4
        _, _, Zxx = stft(ch_data, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        n_bins = mag.shape[0]
        if len(noise_profile) != n_bins:
            noise_profile = np.interp(
                np.linspace(0, 1, n_bins),
                np.linspace(0, 1, len(noise_profile)),
                noise_profile
            )
        clean_mag = np.maximum(mag - alpha * noise_profile[:, np.newaxis], beta * mag)
        _, result = istft(clean_mag * np.exp(1j * phase),
                          fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        n = len(ch_data)
        return result[:n] if len(result) >= n else np.pad(result, (0, n - len(result)))

    def _riaa_deemphasis(self, ch_data, fs):
        t1, t2, t3 = 3180e-6, 318e-6, 75e-6
        z_a = [-1.0 / t2]
        p_a = [-1.0 / t1, -1.0 / t3]
        k_a = t1 / t2
        z_d, p_d, k_d = bilinear_zpk(z_a, p_a, k_a, fs=fs)
        b, a = zpk2tf(z_d, p_d, k_d)
        return filtfilt(b, a, ch_data)

    def _normalize(self, data, mode, target_dbfs, dtype):
        """
        Normalize with a hard +18 dB gain cap.
        RMS mode on a quiet or sparse recording would otherwise request
        enormous gain and clip everything to a solid wall.
        """
        target_lin = 10.0 ** (target_dbfs / 20.0)
        MAX_GAIN = 10.0 ** (18.0 / 20.0)

        if dtype == np.int16:
            max_val = 32767.0
        elif dtype == np.int32:
            max_val = 2147483647.0
        else:
            max_val = 1.0

        if mode == "peak":
            current = np.max(np.abs(data))
            if current == 0:
                return data
            scale = (target_lin * max_val) / current
        else:
            rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))
            if rms == 0:
                return data
            scale = (target_lin * max_val) / rms

        scale = min(scale, MAX_GAIN)
        return data * scale

    # -----------------------------------------------------------------------
    # PROCESSING PIPELINE
    # -----------------------------------------------------------------------

    def _run_pipeline(self, source_data):
        data = source_data
        fs = self.sample_rate
        is_stereo = data.ndim > 1
        n_ch = data.shape[1] if is_stereo else 1
        dtype = self.audio_data.dtype
        processed_channels = []

        for ch in range(n_ch):
            ch_data = data[:, ch].copy() if is_stereo else data.flatten().copy()
            label = f"ch {ch+1}/{n_ch}"

            if self.do_dc_remove.get():
                self._update_progress(None, f"DC offset removal ({label})...")
                ch_data -= np.mean(ch_data)

            if self.do_riaa.get():
                self._update_progress(None, f"RIAA de-emphasis ({label})...")
                ch_data = self._riaa_deemphasis(ch_data, fs)

            if self.do_rumble.get():
                self._update_progress(None, f"Rumble filter ({label})...")
                ch_data = self._butter_hp(ch_data, self.rumble_freq.get(), fs, self.rumble_order.get())

            if self.do_declick.get():
                self._update_progress(None, f"De-click ({label})...")
                ch_data = self._declick(ch_data, self.click_sens.get(), self.click_window.get())

            if self.do_noise_reduce.get() and self.noise_profile is not None:
                self._update_progress(None, f"Spectral noise reduction ({label})...")
                ch_data = self._spectral_nr(ch_data, self.noise_profile,
                                             self.noise_alpha.get(), self.noise_beta.get())

            if self.do_hiss.get():
                self._update_progress(None, f"Hiss filter ({label})...")
                ch_data = self._butter_lp(ch_data, self.hiss_freq.get(), fs, self.hiss_order.get())

            processed_channels.append(ch_data)

        result = np.column_stack(processed_channels) if is_stereo else processed_channels[0]

        if self.do_normalize.get():
            self._update_progress(None, "Normalizing...")
            result = self._normalize(result, self.norm_mode.get(), self.norm_target.get(), dtype)

        if dtype == np.int16:
            np.clip(result, -32768, 32767, out=result)
        elif dtype == np.int32:
            np.clip(result, -2147483648, 2147483647, out=result)
        elif dtype in (np.float32, np.float64):
            np.clip(result, -1.0, 1.0, out=result)

        return result.astype(dtype)

    def _check_steps(self):
        if not any([self.do_dc_remove.get(), self.do_riaa.get(), self.do_rumble.get(),
                    self.do_declick.get(), self.do_noise_reduce.get(),
                    self.do_hiss.get(), self.do_normalize.get()]):
            messagebox.showinfo("Nothing to do", "Enable at least one processing step.")
            return False
        if self.do_noise_reduce.get() and self.noise_profile is None:
            return messagebox.askyesno(
                "No Noise Profile",
                "Spectral noise reduction is enabled but no noise profile has been captured.\n\n"
                "Proceed anyway? (Noise reduction will be skipped for this run.)"
            )
        return True

    def _lock_ui(self):
        self.btn_preview.config(state=tk.DISABLED)
        self.btn_apply_full.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)

    def _unlock_ui_after(self, mode):
        self.btn_apply_full.config(state=tk.NORMAL)
        if self.region_start is not None:
            self.btn_preview.config(state=tk.NORMAL)
        if mode == "preview" and PLAYBACK_AVAILABLE:
            self.btn_play_preview.config(state=tk.NORMAL)
        if mode == "full":
            self.btn_save.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)
            self.btn_undo.config(state=tk.NORMAL if self.undo_stack else tk.DISABLED)
            if PLAYBACK_AVAILABLE:
                self.btn_play_proc.config(state=tk.NORMAL)

    def start_preview(self):
        if self.audio_data is None or not self._check_steps():
            return
        self._lock_ui()
        self.progress["value"] = 0
        threading.Thread(target=self._process_preview, daemon=True).start()

    def _process_preview(self):
        try:
            s = self.region_start if self.region_start is not None else 0.0
            e = self.region_end if self.region_end is not None else (
                len(self.audio_data) / self.sample_rate
            )
            si = int(s * self.sample_rate)
            ei = int(e * self.sample_rate)

            self._update_progress(10, f"Processing preview ({s:.1f}s - {e:.1f}s)...")
            # Work on an explicit copy so the original array is never touched
            segment = np.array(self.audio_data[si:ei], dtype=np.float64, copy=True)
            result = self._run_pipeline(segment)
            self.preview_data = result
            self.root.after(0, self._finish_preview)

        except Exception as exc:
            err = f"{exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Preview Error", err))
            self.root.after(0, lambda: self._update_progress(0, "Preview failed."))
            self.root.after(0, lambda: self._unlock_ui_after("preview"))

    def _finish_preview(self):
        self._update_progress(100,
            "Preview done. Play and compare. If it sounds right, click 'Apply to Full Track'.")
        self._refresh_plots()
        self._unlock_ui_after("preview")

    def start_full_processing(self):
        if self.audio_data is None or not self._check_steps():
            return
        self._save_undo()
        self._lock_ui()
        self.progress["value"] = 0
        threading.Thread(target=self._process_full, daemon=True).start()

    def _process_full(self):
        try:
            self._update_progress(5, "Processing full track...")
            data = np.array(self.audio_data, dtype=np.float64, copy=True)
            self.processed_data = self._run_pipeline(data)
            self.preview_data = None
            self.root.after(0, self._finish_full)
        except Exception as exc:
            err = f"{exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Processing Error", err))
            self.root.after(0, lambda: self._update_progress(0, "Processing failed."))
            self.root.after(0, lambda: self._unlock_ui_after("full"))

    def _finish_full(self):
        self._update_progress(100, "Full track done. Click 'Save Processed WAV' to export.")
        self._refresh_plots()
        self._unlock_ui_after("full")

    def _update_progress(self, pct, text):
        if pct is not None:
            self.progress["value"] = pct
        self.lbl_status.config(text=text)
        self.root.update_idletasks()

    def _save_undo(self):
        if self.processed_data is not None:
            self.undo_stack.append(self.processed_data.copy())
            if len(self.undo_stack) > 10:
                self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.processed_data = self.undo_stack.pop()
            self.btn_undo.config(state=tk.NORMAL if self.undo_stack else tk.DISABLED)
            self._refresh_plots()
            self.lbl_status.config(text="Undo applied.")
        else:
            self.btn_undo.config(state=tk.DISABLED)

    def reset_to_original(self):
        if self.audio_data is None:
            return
        self._save_undo()
        self.processed_data = None
        self.preview_data = None
        self._refresh_plots()
        self.btn_save.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        if PLAYBACK_AVAILABLE:
            self.btn_play_proc.config(state=tk.DISABLED)
            self.btn_play_preview.config(state=tk.DISABLED)
        self.lbl_status.config(text="Reset to original.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = VinylCleanupApp(root)
    root.mainloop()
