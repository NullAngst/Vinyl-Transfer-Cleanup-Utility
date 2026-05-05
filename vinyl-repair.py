"""
Vinyl Transfer Cleanup Utility
A desktop tool for cleaning up audio digitized from vinyl records.

Requires: numpy, scipy, matplotlib
Optional: sounddevice (in-app playback)
"""

import os
import sys
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

from scipy.io import wavfile
from scipy.signal import butter, filtfilt, stft, istft, bilinear_zpk, zpk2tf, welch
from scipy.ndimage import binary_dilation

# NOTE: scipy.signal.medfilt is intentionally NOT imported.
# It has a known heap-corruption bug when called repeatedly, causing
# "free(): invalid size" crashes. All median filtering is done via
# numpy or scipy.ndimage instead.

try:
    import sounddevice as sd
    PLAYBACK_AVAILABLE = True
except Exception:
    # ImportError, OSError (PortAudio missing), etc.
    PLAYBACK_AVAILABLE = False


# Apply Windows DPI awareness BEFORE creating the Tk root.
# Calling this after Tk() has no effect on the existing root window.
def _apply_dpi_awareness():
    if sys.platform == "win32":
        try:
            from ctypes import windll
            # Per-monitor DPI awareness v2 if available, fall back to system DPI
            try:
                windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

UNDO_LEVELS = 10
MAX_NORMALIZATION_GAIN_DB = 18.0
SUPPORTED_EXTENSIONS = (".wav",)


# ---------------------------------------------------------------------------
# TOOLTIP HELPER
# ---------------------------------------------------------------------------

class Tooltip:
    """Simple tooltip popup. Shows after a short delay so it doesn't flash."""

    DELAY_MS = 450

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tw = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._tw is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 24
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return
        self._tw = tk.Toplevel(self.widget)
        self._tw.wm_overrideredirect(True)
        self._tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("Segoe UI", 9), wraplength=340, padx=8, pady=5
        ).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tw is not None:
            try:
                self._tw.destroy()
            except Exception:
                pass
            self._tw = None


def tip(widget, text):
    Tooltip(widget, text)
    return widget


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

class VinylCleanupApp:
    APP_TITLE = "Vinyl Transfer Cleanup Utility"

    def __init__(self, root):
        self.root = root
        self.root.title(self.APP_TITLE)
        self.root.geometry("1280x860")
        self.root.minsize(1000, 680)

        # Audio state
        self.filepath = None
        self.audio_data = None        # original, never mutated
        self.sample_rate = None
        self.processed_data = None    # full-track result (int/float matching original)
        self.preview_data = None      # preview region result
        self.noise_profile = None     # per-channel list of arrays, or single array for mono
        self.noise_profile_range = None  # tuple (start_s, end_s)
        self.undo_stack = []

        # Region selection
        self.region_start = None
        self.region_end = None

        # Playback state
        self.is_playing = False
        self._play_thread = None

        # Worker state
        self._processing = False

        # Style configuration
        self._configure_style()

        self._setup_ui()
        self._bind_shortcuts()

        # Update window title hook on close to confirm if processing
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    # STYLE
    # -----------------------------------------------------------------------

    def _configure_style(self):
        style = ttk.Style()
        # Try to use a clean theme; fall back silently if unavailable.
        for theme in ("vista", "winnative", "clam", "default"):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue

        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#555555", font=("Segoe UI", 8))
        style.configure("Status.TLabel", foreground="#444444")
        style.configure("Good.TLabel", foreground="#0a6e0a", font=("Segoe UI", 9, "bold"))
        style.configure("Warn.TLabel", foreground="#a05a00", font=("Segoe UI", 9))

    # -----------------------------------------------------------------------
    # SHORTCUTS
    # -----------------------------------------------------------------------

    def _bind_shortcuts(self):
        self.root.bind_all("<Control-o>", lambda e: self.load_file())
        self.root.bind_all("<Control-O>", lambda e: self.load_file())
        self.root.bind_all("<Control-s>", lambda e: self._shortcut_save())
        self.root.bind_all("<Control-S>", lambda e: self._shortcut_save())
        self.root.bind_all("<Control-z>", lambda e: self.undo())
        self.root.bind_all("<Control-Z>", lambda e: self.undo())
        self.root.bind_all("<F5>", lambda e: self._shortcut_full_process())
        self.root.bind_all("<F6>", lambda e: self._shortcut_preview())
        self.root.bind_all("<space>", self._shortcut_play_toggle)

    def _shortcut_save(self):
        if str(self.btn_save["state"]) == tk.NORMAL:
            self.save_file()

    def _shortcut_full_process(self):
        if str(self.btn_apply_full["state"]) == tk.NORMAL:
            self.start_full_processing()

    def _shortcut_preview(self):
        if str(self.btn_preview["state"]) == tk.NORMAL:
            self.start_preview()

    def _shortcut_play_toggle(self, event):
        # Don't hijack space inside spinboxes / entries
        widget = event.widget
        if isinstance(widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        if not PLAYBACK_AVAILABLE:
            return
        if self.is_playing:
            self.stop_playback()
        else:
            # Prefer preview if available, then full result, then original
            if self.preview_data is not None:
                self.play_preview()
            elif self.processed_data is not None:
                self.play_processed()
            elif self.audio_data is not None:
                self.play_original()

    # -----------------------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------------------

    def _setup_ui(self):
        # Menu bar
        self._build_menu()

        # Top: file info bar
        info_bar = ttk.Frame(self.root, padding=(10, 4))
        info_bar.pack(side=tk.TOP, fill=tk.X)
        self.lbl_file_info = ttk.Label(
            info_bar, text="No file loaded",
            font=("Consolas", 10) if sys.platform == "win32" else ("Courier", 10)
        )
        self.lbl_file_info.pack(side=tk.LEFT)

        # Status / playback indicator on the right
        self.lbl_playback = ttk.Label(info_bar, text="", foreground="#0066aa")
        self.lbl_playback.pack(side=tk.RIGHT, padx=(0, 10))
        self.lbl_status = ttk.Label(info_bar, text="Ready", style="Status.TLabel")
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # Bottom: action bar
        bottom_outer = ttk.Frame(self.root)
        bottom_outer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(bottom_outer, orient=tk.HORIZONTAL).pack(fill=tk.X)
        bottom_bar = ttk.Frame(bottom_outer, padding=(10, 8))
        bottom_bar.pack(fill=tk.X)
        self._build_bottom_bar(bottom_bar)

        # Main split: left controls, right plots
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=3)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open WAV...   Ctrl+O", command=self.load_file)
        file_menu.add_command(label="Save Processed...   Ctrl+S", command=self._shortcut_save)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Undo Last Process   Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Reset to Original", command=self.reset_to_original)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        process_menu = tk.Menu(menubar, tearoff=False)
        process_menu.add_command(label="Process Preview Region   F6", command=self._shortcut_preview)
        process_menu.add_command(label="Apply to Full Track   F5", command=self._shortcut_full_process)
        menubar.add_cascade(label="Process", menu=process_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Quick Start Guide", command=self._show_quickstart)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _show_quickstart(self):
        text = (
            "QUICK START\n"
            "\n"
            "1. File → Open WAV (or Ctrl+O) and select your vinyl transfer.\n"
            "\n"
            "2. (Recommended) Drag on the top waveform to select 5–10 seconds\n"
            "   that contains the worst noise. Click 'Process Preview Region'\n"
            "   to test settings on just that segment — much faster than a\n"
            "   full run.\n"
            "\n"
            "3. For Spectral Noise Reduction: in the 'Clicks & Noise' tab,\n"
            "   set the Start/End to a segment with NO music (lead-in groove\n"
            "   is ideal), then click 'Capture Noise Profile'.\n"
            "\n"
            "4. Use 'Play Original' and 'Play Preview' to A/B compare.\n"
            "\n"
            "5. When happy, click 'Apply to Full Track' (F5), then\n"
            "   'Save Processed WAV' (Ctrl+S).\n"
            "\n"
            "TIPS\n"
            "• Processing always starts from the original. You don't need to\n"
            "  reset between attempts — just change settings and re-run.\n"
            "• If the result is wrong, Ctrl+Z to undo (10 levels).\n"
            "• Hover any control for a detailed tooltip.\n"
        )
        messagebox.showinfo("Quick Start Guide", text)

    def _show_about(self):
        text = (
            "Vinyl Transfer Cleanup Utility\n"
            "\n"
            "An open-source tool for cleaning up audio digitized from vinyl\n"
            "records. De-click, spectral noise reduction, rumble filter,\n"
            "RIAA de-emphasis, normalization.\n"
            "\n"
            "Original processing is never modified. All work is non-destructive.\n"
        )
        messagebox.showinfo("About", text)

    def _build_bottom_bar(self, bar):
        self.progress = ttk.Progressbar(bar, mode="determinate", length=180)
        self.progress.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_preview = tip(
            ttk.Button(bar, text="Process Preview Region",
                       command=self.start_preview, state=tk.DISABLED),
            "Process ONLY the selected region — much faster than a full run.\n"
            "Select a region by dragging on the top waveform, or type a range.\n\n"
            "Use this to dial in settings before committing to the whole file.\n\n"
            "Shortcut: F6"
        )
        self.btn_preview.pack(side=tk.LEFT, padx=3)

        self.btn_apply_full = tip(
            ttk.Button(bar, text="Apply to Full Track",
                       command=self.start_full_processing,
                       state=tk.DISABLED, style="Accent.TButton"),
            "Apply the current settings to the ENTIRE file.\n"
            "Always processes from the original — re-running replaces the previous result.\n\n"
            "Shortcut: F5"
        )
        self.btn_apply_full.pack(side=tk.LEFT, padx=3)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_save = tip(
            ttk.Button(bar, text="Save Processed WAV",
                       command=self.save_file, state=tk.DISABLED),
            "Export the full-track result to a new WAV file.\n"
            "Only available after 'Apply to Full Track'.\n"
            "The original file is never modified.\n\n"
            "Shortcut: Ctrl+S"
        )
        self.btn_save.pack(side=tk.LEFT, padx=3)

        self.btn_save_preview = tip(
            ttk.Button(bar, text="Save Preview Region",
                       command=self.save_preview_file, state=tk.DISABLED),
            "Save just the processed preview region to a WAV file.\n"
            "Useful for sharing test snippets."
        )
        self.btn_save_preview.pack(side=tk.LEFT, padx=3)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_undo = tip(
            ttk.Button(bar, text="Undo", command=self.undo, state=tk.DISABLED),
            "Step back to the previous full-track result. Keeps up to 10 levels.\n\n"
            "Shortcut: Ctrl+Z"
        )
        self.btn_undo.pack(side=tk.LEFT, padx=3)

        self.btn_reset = tip(
            ttk.Button(bar, text="Reset to Original",
                       command=self.reset_to_original, state=tk.DISABLED),
            "Discard all processing and return to the original loaded file."
        )
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        if PLAYBACK_AVAILABLE:
            ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

            self.btn_play_orig = tip(
                ttk.Button(bar, text="▶ Original",
                           command=self.play_original, state=tk.DISABLED),
                "Play the original audio.\n\n"
                "If a preview region is selected, only that region plays — so\n"
                "you can A/B compare it directly against the processed preview.\n\n"
                "Shortcut: Space (when not editing a number)"
            )
            self.btn_play_orig.pack(side=tk.LEFT, padx=3)

            self.btn_play_preview = tip(
                ttk.Button(bar, text="▶ Preview",
                           command=self.play_preview, state=tk.DISABLED),
                "Play the processed preview region.\n"
                "Compare against 'Play Original' — both will play the same time range."
            )
            self.btn_play_preview.pack(side=tk.LEFT, padx=3)

            self.btn_play_proc = tip(
                ttk.Button(bar, text="▶ Full Result",
                           command=self.play_processed, state=tk.DISABLED),
                "Play the entire full-track processed result."
            )
            self.btn_play_proc.pack(side=tk.LEFT, padx=3)

            self.btn_stop = tip(
                ttk.Button(bar, text="■ Stop", command=self.stop_playback, state=tk.DISABLED),
                "Stop any audio currently playing."
            )
            self.btn_stop.pack(side=tk.LEFT, padx=3)
        else:
            ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
            ttk.Label(
                bar,
                text="Playback unavailable (install 'sounddevice' to enable)",
                style="Hint.TLabel"
            ).pack(side=tk.LEFT, padx=4)

    def _build_left_panel(self, parent):
        file_frame = ttk.LabelFrame(parent, text="File", padding=8)
        file_frame.pack(fill=tk.X, pady=(0, 6))
        tip(
            ttk.Button(file_frame, text="Load WAV File   (Ctrl+O)", command=self.load_file),
            "Open a WAV file. Stereo and mono are both supported.\n"
            "16-bit, 24-bit (as int32), and float32 WAV files are all accepted.\n"
            "The original file is never modified."
        ).pack(fill=tk.X)

        region_frame = ttk.LabelFrame(parent, text="Preview Region", padding=8)
        region_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            region_frame,
            text=(
                "Drag on the top waveform to select a region, or type a "
                "range below. 'Play Original' will also use this region "
                "for direct A/B comparison."
            ),
            style="Hint.TLabel", wraplength=240, justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 5))

        self.lbl_region = ttk.Label(region_frame, text="No region selected",
                                    style="Status.TLabel")
        self.lbl_region.pack(anchor=tk.W)

        range_row = ttk.Frame(region_frame)
        range_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(range_row, text="Start (s):").pack(side=tk.LEFT)
        self.manual_start = tk.DoubleVar(value=0.0)
        ttk.Spinbox(range_row, from_=0.0, to=99999.0, increment=1.0,
                    textvariable=self.manual_start, width=7,
                    format="%.2f").pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(range_row, text="End (s):").pack(side=tk.LEFT)
        self.manual_end = tk.DoubleVar(value=60.0)
        ttk.Spinbox(range_row, from_=0.0, to=99999.0, increment=1.0,
                    textvariable=self.manual_end, width=7,
                    format="%.2f").pack(side=tk.LEFT, padx=2)

        btn_row = ttk.Frame(region_frame)
        btn_row.pack(fill=tk.X, pady=(6, 0))
        tip(
            ttk.Button(btn_row, text="Use Typed Range", command=self._use_typed_range),
            "Set the region to the start/end times above.\n"
            "Example: 0 to 60 to test just the first minute."
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        ttk.Button(btn_row, text="Clear Region",
                   command=self._clear_region).pack(side=tk.LEFT, fill=tk.X,
                                                    expand=True, padx=(3, 0))

        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self._build_tab_clicks(nb)
        self._build_tab_filters(nb)
        self._build_tab_levels(nb)
        self._build_tab_summary(nb)

    def _slider_row(self, parent, variable, from_, to, fmt, label_width=8):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)
        lbl = ttk.Label(frame, text=fmt.format(variable.get()),
                        width=label_width, anchor=tk.E)
        lbl.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL,
            command=lambda v: lbl.config(text=fmt.format(float(v)))
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_tab_clicks(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Clicks & Noise")

        ttk.Label(tab, text="De-Click / De-Pop", style="Section.TLabel").pack(anchor=tk.W)

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
            text=("Detection Sensitivity\n"
                  "Higher = less aggressive (misses subtle clicks).\n"
                  "Lower = more aggressive (may soften loud transients like snare hits)."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 0))
        self.click_sens = tk.DoubleVar(value=10.0)
        self._slider_row(tab, self.click_sens, 1.0, 30.0, "{:.1f}")

        ttk.Label(
            tab,
            text=("Repair Window (ms)\n"
                  "Width of the region replaced around each detected click.\n"
                  "1–3 ms handles crackle. Up to 8–10 ms for heavy pops."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(5, 0))
        self.click_window = tk.DoubleVar(value=2.0)
        self._slider_row(tab, self.click_window, 0.5, 10.0, "{:.1f}")

        # Optional second pass for heavy damage
        self.declick_passes = tk.IntVar(value=1)
        ttk.Label(tab, text="Passes:").pack(anchor=tk.W, pady=(6, 0))
        pass_row = ttk.Frame(tab)
        pass_row.pack(fill=tk.X)
        for n, lab in ((1, "1 (standard)"), (2, "2 (heavy damage)")):
            ttk.Radiobutton(pass_row, text=lab, variable=self.declick_passes,
                            value=n).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="Spectral Noise Reduction",
                  style="Section.TLabel").pack(anchor=tk.W)

        self.do_noise_reduce = tk.BooleanVar(value=False)
        tip(
            ttk.Checkbutton(tab, text="Enable Noise Reduction",
                            variable=self.do_noise_reduce),
            "Targets continuous broadband noise (hiss, hum, surface noise) in the\n"
            "frequency domain. Has no effect on clicks — that is what de-click is for.\n\n"
            "You MUST capture a noise profile first (see below).\n\n"
            "Start with Reduction Strength around 2.0. Above 4.0 causes a warbling\n"
            "metallic artifact called 'musical noise' that is worse than the hiss."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text=("Capture a noise profile first:\n"
                  "1. Find a section with no music — the lead-in groove before\n"
                  "   the music starts is ideal (pure surface noise only).\n"
                  "2. Enter the time range and click Capture."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 2))

        time_frame = ttk.Frame(tab)
        time_frame.pack(fill=tk.X)
        ttk.Label(time_frame, text="Start:").pack(side=tk.LEFT)
        self.noise_start = tk.DoubleVar(value=0.0)
        ttk.Spinbox(time_frame, from_=0.0, to=9999.0, increment=0.1,
                    textvariable=self.noise_start, width=6,
                    format="%.2f").pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(time_frame, text="End (s):").pack(side=tk.LEFT)
        self.noise_end = tk.DoubleVar(value=1.0)
        ttk.Spinbox(time_frame, from_=0.0, to=9999.0, increment=0.1,
                    textvariable=self.noise_end, width=6,
                    format="%.2f").pack(side=tk.LEFT, padx=2)

        capture_row = ttk.Frame(tab)
        capture_row.pack(fill=tk.X, pady=6)
        tip(
            ttk.Button(capture_row, text="Capture Noise Profile",
                       command=self.capture_noise_profile),
            "Records the frequency fingerprint of the selected region.\n"
            "This is subtracted from every audio frame during processing.\n"
            "Always taken from the original file."
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tip(
            ttk.Button(capture_row, text="From Selection",
                       command=self._capture_from_region),
            "Use the currently-selected preview region (drag on the waveform)\n"
            "as the noise sample."
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self.lbl_noise_profile = ttk.Label(tab, text="No profile captured",
                                           style="Hint.TLabel")
        self.lbl_noise_profile.pack(anchor=tk.W)

        ttk.Label(
            tab,
            text=("Reduction Strength (alpha)\n"
                  "1.5–3.0 is a good starting range.\n"
                  "Above 4.0 introduces metallic warbling artifacts."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(8, 0))
        self.noise_alpha = tk.DoubleVar(value=2.0)
        self._slider_row(tab, self.noise_alpha, 0.5, 6.0, "{:.2f}")

        ttk.Label(
            tab,
            text=("Spectral Floor (beta)\n"
                  "Minimum retained signal per bin. Raise to reduce warbling at\n"
                  "the cost of slightly less noise removed. Keep 0.01–0.05."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(5, 0))
        self.noise_beta = tk.DoubleVar(value=0.02)
        self._slider_row(tab, self.noise_beta, 0.001, 0.2, "{:.3f}", label_width=7)

    def _build_tab_filters(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Filters")

        ttk.Label(tab, text="Rumble Filter (High-Pass)",
                  style="Section.TLabel").pack(anchor=tk.W)

        self.do_rumble = tk.BooleanVar(value=True)
        tip(
            ttk.Checkbutton(tab, text="Enable Rumble Filter", variable=self.do_rumble),
            "Cuts everything below the cutoff frequency.\n\n"
            "Turntable motors produce low-frequency rumble (5–30 Hz) that wastes\n"
            "headroom and causes woofer pumping. The default 30 Hz cutoff is safe\n"
            "for virtually all music. Only raise it for severe motor noise.\n"
            "Going above 80 Hz starts audibly cutting bass in the music."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text=("Cutoff Frequency (Hz)\n"
                  "30 Hz default is safe. Raise to 50–80 Hz for severe rumble only."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 0))
        self.rumble_freq = tk.DoubleVar(value=30.0)
        self._slider_row(tab, self.rumble_freq, 10.0, 150.0, "{:.0f}")

        ttk.Label(tab, text="Filter Steepness (order)").pack(anchor=tk.W, pady=(5, 0))
        self.rumble_order = tk.IntVar(value=4)
        rof = ttk.Frame(tab)
        rof.pack(fill=tk.X)
        for o, desc in ((2, "gentle"), (4, "standard"), (6, "steep"), (8, "very steep")):
            ttk.Radiobutton(rof, text=f"{o}  ({desc})",
                            variable=self.rumble_order, value=o).pack(anchor=tk.W)

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="Hiss Filter (Low-Pass)",
                  style="Section.TLabel").pack(anchor=tk.W)

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
            text=("Cutoff Frequency (Hz)\n"
                  "14000+ retains most musical content.\n"
                  "Below 10000 will audibly dull the recording."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(6, 0))
        self.hiss_freq = tk.DoubleVar(value=14000.0)
        self._slider_row(tab, self.hiss_freq, 3000.0, 20000.0, "{:.0f}")

        ttk.Label(tab, text="Filter Steepness (order)").pack(anchor=tk.W, pady=(5, 0))
        self.hiss_order = tk.IntVar(value=4)
        hof = ttk.Frame(tab)
        hof.pack(fill=tk.X)
        for o, desc in ((2, "gentle"), (4, "standard"), (6, "steep"), (8, "very steep")):
            ttk.Radiobutton(hof, text=f"{o}  ({desc})",
                            variable=self.hiss_order, value=o).pack(anchor=tk.W)

    def _build_tab_levels(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Levels")

        ttk.Label(tab, text="DC Offset Removal", style="Section.TLabel").pack(anchor=tk.W)
        self.do_dc_remove = tk.BooleanVar(value=True)
        tip(
            ttk.Checkbutton(tab, text="Remove DC Offset", variable=self.do_dc_remove),
            "Removes a constant voltage bias from the waveform.\n\n"
            "Some phono preamps and ADCs shift the signal above or below zero,\n"
            "wasting headroom and causing asymmetric clipping.\n"
            "Always safe to leave on."
        ).pack(anchor=tk.W)

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="Normalization", style="Section.TLabel").pack(anchor=tk.W)
        self.do_normalize = tk.BooleanVar(value=True)
        tip(
            ttk.Checkbutton(tab, text="Enable Normalization", variable=self.do_normalize),
            "Adjusts the overall volume of the recording.\n\n"
            "PEAK mode: scales so the loudest sample hits your target. Always safe.\n\n"
            "RMS mode: scales to a target average loudness. Can apply very large\n"
            f"gain on quiet recordings. A hard +{MAX_NORMALIZATION_GAIN_DB:.0f} dB cap is enforced, but always\n"
            "test with preview before applying to the full track."
        ).pack(anchor=tk.W)

        self.norm_mode = tk.StringVar(value="peak")
        tip(
            ttk.Radiobutton(tab, text="Peak  —  safe, recommended",
                            variable=self.norm_mode, value="peak"),
            "Scales so the loudest sample hits your target.\n"
            "Never clips. Start here."
        ).pack(anchor=tk.W, pady=(8, 0))
        tip(
            ttk.Radiobutton(tab, text="RMS  —  perceived loudness (use with caution)",
                            variable=self.norm_mode, value="rms"),
            "Scales to a target average loudness level.\n"
            "Can apply massive gain on quiet recordings.\n"
            "ALWAYS preview before applying to full track."
        ).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text=("Target Level (dBFS)\n"
                  "Peak: -1.0 dBFS is standard (1 dB headroom).\n"
                  "RMS: -18 to -14 dBFS is typical for vinyl.\n"
                  f"Hard +{MAX_NORMALIZATION_GAIN_DB:.0f} dB gain cap enforced in both modes."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(8, 0))
        self.norm_target = tk.DoubleVar(value=-1.0)
        self._slider_row(tab, self.norm_target, -30.0, -0.1, "{:.1f}")

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        ttk.Label(tab, text="RIAA De-Emphasis", style="Section.TLabel").pack(anchor=tk.W)
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

    def _build_tab_summary(self, nb):
        """A read-only summary of which steps will run and in what order."""
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Summary")

        ttk.Label(tab, text="Processing chain (in order)",
                  style="Section.TLabel").pack(anchor=tk.W)

        ttk.Label(
            tab,
            text=("Steps run in a fixed, sensible order regardless of how the\n"
                  "tabs are laid out. Refresh to see the current chain."),
            style="Hint.TLabel", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(2, 8))

        self.txt_summary = tk.Text(tab, height=14, width=34, wrap="word",
                                   relief="solid", borderwidth=1,
                                   font=("Consolas", 9), state=tk.DISABLED,
                                   background="#fafafa")
        self.txt_summary.pack(fill=tk.BOTH, expand=True)

        ttk.Button(tab, text="Refresh", command=self._update_summary).pack(
            fill=tk.X, pady=(6, 0))

    def _update_summary(self):
        steps = []
        n = 1

        def add(label, detail=""):
            nonlocal n
            line = f"{n}. {label}"
            if detail:
                line += f"\n    {detail}"
            steps.append(line)
            n += 1

        if self.do_dc_remove.get():
            add("DC offset removal")
        if self.do_riaa.get():
            add("RIAA de-emphasis", "(only for line-level captures)")
        if self.do_rumble.get():
            add("Rumble filter (high-pass)",
                f"{self.rumble_freq.get():.0f} Hz, order {self.rumble_order.get()}")
        if self.do_declick.get():
            passes = self.declick_passes.get()
            pass_str = "1 pass" if passes == 1 else f"{passes} passes"
            add("De-click / de-pop",
                f"sensitivity {self.click_sens.get():.1f}, "
                f"window {self.click_window.get():.1f} ms, {pass_str}")
        if self.do_noise_reduce.get():
            if self.noise_profile is not None:
                add("Spectral noise reduction",
                    f"alpha {self.noise_alpha.get():.2f}, "
                    f"beta {self.noise_beta.get():.3f}")
            else:
                add("Spectral noise reduction",
                    "*** SKIPPED — no noise profile captured ***")
        if self.do_hiss.get():
            add("Hiss filter (low-pass)",
                f"{self.hiss_freq.get():.0f} Hz, order {self.hiss_order.get()}")
        if self.do_normalize.get():
            add(f"Normalize ({self.norm_mode.get().upper()})",
                f"target {self.norm_target.get():.1f} dBFS")

        if not steps:
            text = "(no steps enabled — enable at least one)"
        else:
            text = "\n\n".join(steps)

        self.txt_summary.config(state=tk.NORMAL)
        self.txt_summary.delete("1.0", tk.END)
        self.txt_summary.insert("1.0", text)
        self.txt_summary.config(state=tk.DISABLED)

    def _build_right_panel(self, parent):
        view_row = ttk.Frame(parent)
        view_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(view_row, text="View:").pack(side=tk.LEFT)
        self.view_mode = tk.StringVar(value="waveform")
        for label, val in (("Waveform", "waveform"),
                           ("Spectrum", "spectrum"),
                           ("Both", "both")):
            ttk.Radiobutton(view_row, text=label, variable=self.view_mode,
                            value=val, command=self._refresh_plots).pack(
                side=tk.LEFT, padx=5)

        ttk.Separator(view_row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Stats display
        self.lbl_stats = ttk.Label(view_row, text="", style="Hint.TLabel")
        self.lbl_stats.pack(side=tk.LEFT)

        self.lbl_selection_display = ttk.Label(
            view_row,
            text="Drag on the top waveform to select a preview region",
            style="Hint.TLabel"
        )
        self.lbl_selection_display.pack(side=tk.RIGHT, padx=10)

        self.fig = plt.Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Matplotlib navigation toolbar (zoom, pan, home)
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT)

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
        try:
            s = float(self.manual_start.get())
            e = float(self.manual_end.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid Range",
                                 "Start and End must be numbers (seconds).")
            return
        duration = len(self.audio_data) / self.sample_rate
        if s < 0:
            s = 0.0
        # Clamp end first, then check ordering — fixes the bug where
        # start > duration would silently invalidate the range.
        e = min(e, duration)
        if s >= e:
            messagebox.showerror(
                "Invalid Range",
                f"Start ({s:.2f}s) must be less than End ({e:.2f}s).\n"
                f"File duration is {duration:.2f}s."
            )
            return
        if e - s < 0.1:
            messagebox.showerror("Invalid Range", "Range must be at least 0.1 seconds.")
            return
        self.region_start = s
        self.region_end = e
        self.manual_start.set(round(s, 2))
        self.manual_end.set(round(e, 2))
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
            style="Status.TLabel"
        )
        self.lbl_selection_display.config(text="No region selected")
        if self.audio_data is not None:
            self.btn_preview.config(state=tk.DISABLED)

    def _update_region_label(self):
        if self.region_start is None:
            return
        dur = self.region_end - self.region_start
        text = f"{self.region_start:.2f}s  to  {self.region_end:.2f}s  ({dur:.1f}s)"
        self.lbl_region.config(text=text, style="Good.TLabel")
        self.lbl_selection_display.config(text=f"Region: {text}")
        if self.audio_data is not None and not self._processing:
            self.btn_preview.config(state=tk.NORMAL)

    def _refresh_plots(self):
        self._init_axes()
        mode = self.view_mode.get()

        if mode in ("waveform", "both"):
            if self.audio_data is not None:
                self._draw_waveform(self.audio_data, self._ax_ow, "Original")
                self._highlight_region(self._ax_ow)
            lower = self.processed_data if self.processed_data is not None else self.preview_data
            label = ("Processed (Full Track)" if self.processed_data is not None
                     else ("Preview Region" if self.preview_data is not None else "Processed"))
            if lower is not None:
                self._draw_waveform(lower, self._ax_pw, label,
                                    is_preview=(self.processed_data is None
                                                and self.preview_data is not None))

        if mode in ("spectrum", "both"):
            ax_o = self._ax_os if mode == "both" else self._ax_ow
            ax_p = self._ax_ps if mode == "both" else self._ax_pw
            if self.audio_data is not None:
                self._draw_spectrum(
                    self.audio_data, ax_o,
                    "Original" if mode == "spectrum" else "Spectrum (original)"
                )
            lower = self.processed_data if self.processed_data is not None else self.preview_data
            if lower is not None:
                self._draw_spectrum(
                    lower, ax_p,
                    "Processed" if mode == "spectrum" else "Spectrum (processed)"
                )

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
        self._update_stats_display()

    def _highlight_region(self, ax):
        """Shade the selected region on the original waveform."""
        if self.region_start is not None and self.region_end is not None:
            ax.axvspan(self.region_start, self.region_end,
                       alpha=0.18, color="steelblue", zorder=0)

    def _draw_waveform(self, data, ax, title, is_preview=False):
        ax.clear()
        ax.set_title(title, fontsize=9)
        if data is None:
            return
        # Decimate for display performance
        step = max(1, len(data) // 60000)
        plot = data[::step]
        # X-axis: if this is a preview clip, reflect its actual time range
        if is_preview and self.region_start is not None:
            t = np.linspace(self.region_start,
                            self.region_start + len(data) / self.sample_rate,
                            len(plot))
        else:
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
        """Use Welch's method (averaged periodograms) for a stable spectrum
        of long files instead of taking only the first segment."""
        ax.clear()
        ax.set_title(title, fontsize=9)
        if data is None or self.sample_rate is None:
            return
        ch = data[:, 0].astype(np.float64) if data.ndim > 1 else data.astype(np.float64)
        # Welch averages many windows; a 16k segment gives ~3 Hz resolution
        nperseg = min(16384, len(ch))
        if nperseg < 64:
            return
        try:
            freqs, psd = welch(ch, fs=self.sample_rate, nperseg=nperseg,
                               noverlap=nperseg // 2, window="hann",
                               scaling="spectrum")
        except Exception:
            return
        # Normalize to peak so the dB scale is comparable across plots
        peak = psd.max() if psd.size else 0.0
        if peak <= 0:
            return
        db = 10.0 * np.log10(np.maximum(psd / peak, 1e-12))
        ax.semilogx(freqs[1:], db[1:], color="#2196F3", alpha=0.85, linewidth=0.6)
        ax.set_xlim(20, self.sample_rate / 2)
        ax.set_ylim(-90, 5)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Level (dB, normalized)", fontsize=8)
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=7)

    def _update_stats_display(self):
        """Show before/after peak and RMS levels in dBFS."""
        if self.audio_data is None:
            self.lbl_stats.config(text="")
            return

        def stats(data, dtype):
            arr = data.astype(np.float64)
            if dtype == np.int16:
                arr = arr / 32768.0
            elif dtype == np.int32:
                arr = arr / 2147483648.0
            peak = np.max(np.abs(arr)) if arr.size else 0.0
            rms = np.sqrt(np.mean(arr ** 2)) if arr.size else 0.0
            peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
            rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
            return peak_db, rms_db

        dtype = self.audio_data.dtype
        op, orms = stats(self.audio_data, dtype)
        result = self.processed_data if self.processed_data is not None else self.preview_data
        text = f"Original  peak {op:+.1f} dBFS  RMS {orms:+.1f} dBFS"
        if result is not None:
            pp, prms = stats(result, dtype)
            text += f"   |   Processed  peak {pp:+.1f} dBFS  RMS {prms:+.1f} dBFS"
        self.lbl_stats.config(text=text)

    # -----------------------------------------------------------------------
    # FILE I/O
    # -----------------------------------------------------------------------

    def load_file(self):
        if self._processing:
            messagebox.showinfo("Busy", "Please wait for the current job to finish.")
            return
        filepath = filedialog.askopenfilename(
            title="Open WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            self._update_progress(0, "Loading...")
            self.root.update()
            rate, data = wavfile.read(filepath)

            # Validate
            if data.size == 0:
                raise ValueError("File contains no audio samples.")
            if data.ndim > 2:
                raise ValueError("Only mono or stereo WAV files are supported.")
            if data.ndim == 2 and data.shape[1] > 2:
                raise ValueError(
                    f"This file has {data.shape[1]} channels. "
                    "Only mono and stereo are supported."
                )

            self.filepath = filepath
            self.sample_rate = rate
            self.audio_data = data
            self.processed_data = None
            self.preview_data = None
            self.noise_profile = None
            self.noise_profile_range = None
            self.undo_stack = []
            self.region_start = None
            self.region_end = None

            dur = len(data) / rate
            ch = "Stereo" if data.ndim > 1 else "Mono"
            bits = self._dtype_to_bits(data.dtype)
            self.lbl_file_info.config(
                text=(f"{os.path.basename(filepath)}  |  {rate} Hz  |  "
                      f"{bits}  |  {ch}  |  {dur:.1f}s  ({dur/60:.1f} min)")
            )

            self.root.title(f"{self.APP_TITLE} — {os.path.basename(filepath)}")

            self.btn_preview.config(state=tk.DISABLED)
            self.btn_apply_full.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            self.btn_save_preview.config(state=tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)
            self.btn_reset.config(state=tk.DISABLED)
            self.lbl_noise_profile.config(text="No profile captured", style="Hint.TLabel")
            self.lbl_region.config(
                text="No region selected — full track will be processed",
                style="Status.TLabel"
            )
            self.lbl_selection_display.config(
                text="Drag on the top waveform to select a preview region"
            )

            if PLAYBACK_AVAILABLE:
                self.btn_play_orig.config(state=tk.NORMAL)
                self.btn_play_preview.config(state=tk.DISABLED)
                self.btn_play_proc.config(state=tk.DISABLED)

            # Default suggested noise/preview range
            default_end = min(60.0, dur)
            self.manual_end.set(round(default_end, 2))
            self.noise_end.set(round(min(1.0, dur), 2))

            self._refresh_plots()
            self._update_summary()
            self._update_progress(0, f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load file:\n{e}")
            self._update_progress(0, "Error loading file.")

    def _dtype_to_bits(self, dtype):
        if dtype == np.int16:
            return "16-bit int"
        if dtype == np.int32:
            return "32-bit int"
        if dtype == np.float32:
            return "32-bit float"
        if dtype == np.float64:
            return "64-bit float"
        return str(dtype)

    def save_file(self):
        if self.processed_data is None:
            return
        initial_dir = os.path.dirname(self.filepath) if self.filepath else None
        base = os.path.basename(self.filepath) if self.filepath else "audio.wav"
        filepath = filedialog.asksaveasfilename(
            title="Save processed WAV",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialdir=initial_dir,
            initialfile="cleaned_" + base
        )
        if not filepath:
            return
        try:
            wavfile.write(filepath, self.sample_rate, self.processed_data)
            self._update_progress(self.progress["value"],
                                  f"Saved: {os.path.basename(filepath)}")
            messagebox.showinfo("Saved", f"File saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save file:\n{e}")

    def save_preview_file(self):
        if self.preview_data is None:
            return
        initial_dir = os.path.dirname(self.filepath) if self.filepath else None
        base = os.path.basename(self.filepath) if self.filepath else "audio.wav"
        s = self.region_start or 0.0
        e = self.region_end or (s + len(self.preview_data) / self.sample_rate)
        suggested = f"preview_{int(s)}-{int(e)}s_{base}"
        filepath = filedialog.asksaveasfilename(
            title="Save preview region",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialdir=initial_dir,
            initialfile=suggested
        )
        if not filepath:
            return
        try:
            wavfile.write(filepath, self.sample_rate, self.preview_data)
            messagebox.showinfo("Saved", f"Preview saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save preview:\n{e}")

    # -----------------------------------------------------------------------
    # PLAYBACK
    # -----------------------------------------------------------------------

    def _to_float32(self, data):
        if data.dtype == np.int16:
            return data.astype(np.float32) / 32768.0
        if data.dtype == np.int32:
            return data.astype(np.float32) / 2147483648.0
        # Float types — clamp to [-1, 1] for sounddevice's sake
        out = data.astype(np.float32)
        if np.max(np.abs(out)) > 1.0:
            out = np.clip(out, -1.0, 1.0)
        return out

    def play_original(self):
        """Play the original. If a region is selected, only play that range."""
        if self.audio_data is None:
            return
        if self.region_start is not None and self.region_end is not None:
            si = int(self.region_start * self.sample_rate)
            ei = int(self.region_end * self.sample_rate)
            self._play_audio(self.audio_data[si:ei], label="original (region)")
        else:
            self._play_audio(self.audio_data, label="original")

    def play_preview(self):
        if self.preview_data is not None:
            self._play_audio(self.preview_data, label="processed preview")

    def play_processed(self):
        if self.processed_data is not None:
            self._play_audio(self.processed_data, label="processed (full)")

    def _play_audio(self, data, label=""):
        if not PLAYBACK_AVAILABLE or data is None:
            return
        self.stop_playback()
        self.is_playing = True
        self.btn_stop.config(state=tk.NORMAL)
        if label:
            self.lbl_playback.config(text=f"▶ Playing: {label}")

        # Copy so the playback thread does not share memory with anything
        # the processing thread might touch.
        play_buf = np.array(self._to_float32(data), copy=True)

        def _run():
            try:
                sd.play(play_buf, self.sample_rate)
                sd.wait()
            except Exception as e:
                self.root.after(0, lambda: self.lbl_playback.config(
                    text=f"Playback error: {e}"))
            finally:
                self.is_playing = False
                self.root.after(0, self._on_playback_finished)

        self._play_thread = threading.Thread(target=_run, daemon=True)
        self._play_thread.start()

    def _on_playback_finished(self):
        self.btn_stop.config(state=tk.DISABLED)
        # Only clear the label if it's the playback indicator (not an error)
        text = self.lbl_playback.cget("text")
        if text.startswith("▶"):
            self.lbl_playback.config(text="")

    def stop_playback(self):
        if PLAYBACK_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass
        self.is_playing = False
        if PLAYBACK_AVAILABLE:
            self.btn_stop.config(state=tk.DISABLED)
        self.lbl_playback.config(text="")

    # -----------------------------------------------------------------------
    # NOISE PROFILE
    # -----------------------------------------------------------------------

    def _capture_from_region(self):
        """Use the current preview-region selection as the noise sample."""
        if self.region_start is None or self.region_end is None:
            messagebox.showinfo(
                "No selection",
                "Drag on the top waveform to select a quiet (no-music) region first."
            )
            return
        self.noise_start.set(round(self.region_start, 2))
        self.noise_end.set(round(self.region_end, 2))
        self.capture_noise_profile()

    def capture_noise_profile(self):
        if self.audio_data is None:
            messagebox.showwarning("No File", "Load a WAV file first.")
            return
        try:
            try:
                s0 = float(self.noise_start.get())
                s1 = float(self.noise_end.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Invalid Range",
                                     "Start/End must be numbers (seconds).")
                return
            dur = len(self.audio_data) / self.sample_rate

            if s0 < 0:
                s0 = 0.0
            s1 = min(s1, dur)
            if s0 >= s1:
                messagebox.showerror(
                    "Invalid Range",
                    f"Start ({s0:.2f}s) must be less than End ({s1:.2f}s).\n"
                    f"File duration is {dur:.2f}s."
                )
                return
            if s1 - s0 < 0.1:
                messagebox.showerror("Invalid Range",
                                     "Noise sample must be at least 0.1s long.")
                return

            data = self.audio_data
            si, ei = int(s0 * self.sample_rate), int(s1 * self.sample_rate)

            nperseg = 2048
            # Capture per-channel profiles for stereo
            if data.ndim > 1:
                profiles = []
                for c in range(data.shape[1]):
                    region = data[si:ei, c].astype(np.float64)
                    _, _, Zxx = stft(region, fs=self.sample_rate,
                                     nperseg=nperseg,
                                     noverlap=nperseg * 3 // 4)
                    profiles.append(np.mean(np.abs(Zxx), axis=1))
                self.noise_profile = profiles
            else:
                region = data[si:ei].astype(np.float64)
                _, _, Zxx = stft(region, fs=self.sample_rate,
                                 nperseg=nperseg,
                                 noverlap=nperseg * 3 // 4)
                self.noise_profile = [np.mean(np.abs(Zxx), axis=1)]

            self.noise_profile_range = (s0, s1)
            self.lbl_noise_profile.config(
                text=f"✓ Profile captured: {s0:.2f}s – {s1:.2f}s ({s1-s0:.2f}s)",
                style="Good.TLabel"
            )
            # Auto-enable noise reduction
            self.do_noise_reduce.set(True)
            self._update_summary()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture noise profile:\n{e}")

    # -----------------------------------------------------------------------
    # DSP
    # -----------------------------------------------------------------------

    def _butter_hp(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        # Clamp cutoff to a valid normalized range
        wn = max(1e-6, min(0.999, cutoff / nyq))
        b, a = butter(order, wn, btype="high", analog=False)
        return filtfilt(b, a, data)

    def _butter_lp(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        wn = max(1e-6, min(0.999, cutoff / nyq))
        b, a = butter(order, wn, btype="low", analog=False)
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
            return ch_data.copy(), 0

        # 1.4826 is the consistency factor that makes MAD equivalent to sigma
        # for a Gaussian distribution.
        threshold = sensitivity * mad * 1.4826
        mask = np.abs(d2) > threshold

        if not np.any(mask):
            return ch_data.copy(), 0

        # Dilate the mask to cover the full extent of each impulse
        dilation = max(1, int(self.sample_rate * window_ms / 1000.0))
        mask = binary_dilation(mask, structure=np.ones(dilation, dtype=bool))

        cleaned = ch.copy()
        indices = np.arange(len(ch))
        good = ~mask

        if np.sum(good) < 2:
            # If almost everything is masked the signal is probably corrupt;
            # return it untouched rather than interpolating garbage.
            return ch_data.copy(), 0

        # Count clicks (contiguous runs of True in the dilated mask)
        # Approximate by counting transitions from False to True
        n_clicks = int(np.sum(np.diff(mask.astype(np.int8)) == 1))
        if mask[0]:
            n_clicks += 1

        # Linear interpolation across each masked (click) region
        cleaned[mask] = np.interp(indices[mask], indices[good], ch[good])
        return cleaned, n_clicks

    def _spectral_nr(self, ch_data, noise_profile, alpha, beta):
        nperseg = 2048
        noverlap = nperseg * 3 // 4
        _, _, Zxx = stft(ch_data, fs=self.sample_rate,
                         nperseg=nperseg, noverlap=noverlap)
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
        max_gain = 10.0 ** (MAX_NORMALIZATION_GAIN_DB / 20.0)

        if dtype == np.int16:
            max_val = 32767.0
        elif dtype == np.int32:
            max_val = 2147483647.0
        else:
            max_val = 1.0

        if mode == "peak":
            current = float(np.max(np.abs(data)))
            if current == 0:
                return data
            scale = (target_lin * max_val) / current
        else:
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            if rms == 0:
                return data
            scale = (target_lin * max_val) / rms

        scale = min(scale, max_gain)
        return data * scale

    # -----------------------------------------------------------------------
    # PROCESSING PIPELINE
    # -----------------------------------------------------------------------

    def _run_pipeline(self, source_data, progress_offset=0, progress_span=90):
        """
        source_data: float64 array (mutable copy)
        progress_offset: starting % for the pipeline
        progress_span: % range available for this pipeline call
        """
        data = source_data
        fs = self.sample_rate
        is_stereo = data.ndim > 1
        n_ch = data.shape[1] if is_stereo else 1
        dtype = self.audio_data.dtype
        processed_channels = []
        total_clicks = 0

        # Build a list of enabled steps so we can report progress proportionally
        steps = []
        if self.do_dc_remove.get():
            steps.append("dc")
        if self.do_riaa.get():
            steps.append("riaa")
        if self.do_rumble.get():
            steps.append("rumble")
        if self.do_declick.get():
            for _ in range(max(1, int(self.declick_passes.get()))):
                steps.append("declick")
        if self.do_noise_reduce.get() and self.noise_profile is not None:
            steps.append("nr")
        if self.do_hiss.get():
            steps.append("hiss")

        total_units = max(1, len(steps) * n_ch)
        unit = progress_span / total_units
        progress = progress_offset

        for ch in range(n_ch):
            ch_data = data[:, ch].copy() if is_stereo else data.flatten().copy()
            label = f"ch {ch+1}/{n_ch}" if is_stereo else "mono"

            # Pick this channel's noise profile if NR is enabled
            np_for_ch = None
            if self.do_noise_reduce.get() and self.noise_profile is not None:
                if isinstance(self.noise_profile, list):
                    np_for_ch = self.noise_profile[min(ch, len(self.noise_profile) - 1)]
                else:
                    # legacy: single ndarray
                    np_for_ch = self.noise_profile

            for step in steps:
                if step == "dc":
                    self._update_progress(progress, f"DC offset removal ({label})...")
                    ch_data = ch_data - np.mean(ch_data)
                elif step == "riaa":
                    self._update_progress(progress, f"RIAA de-emphasis ({label})...")
                    ch_data = self._riaa_deemphasis(ch_data, fs)
                elif step == "rumble":
                    self._update_progress(progress, f"Rumble filter ({label})...")
                    ch_data = self._butter_hp(ch_data, self.rumble_freq.get(),
                                              fs, self.rumble_order.get())
                elif step == "declick":
                    self._update_progress(progress, f"De-click ({label})...")
                    ch_data, n_clicks = self._declick(
                        ch_data, self.click_sens.get(), self.click_window.get()
                    )
                    total_clicks += n_clicks
                elif step == "nr":
                    self._update_progress(progress,
                                          f"Spectral noise reduction ({label})...")
                    ch_data = self._spectral_nr(
                        ch_data, np_for_ch,
                        self.noise_alpha.get(), self.noise_beta.get()
                    )
                elif step == "hiss":
                    self._update_progress(progress, f"Hiss filter ({label})...")
                    ch_data = self._butter_lp(ch_data, self.hiss_freq.get(),
                                              fs, self.hiss_order.get())
                progress += unit

            processed_channels.append(ch_data)

        result = (np.column_stack(processed_channels)
                  if is_stereo else processed_channels[0])

        if self.do_normalize.get():
            self._update_progress(progress_offset + progress_span - 2, "Normalizing...")
            result = self._normalize(result, self.norm_mode.get(),
                                     self.norm_target.get(), dtype)

        # Clip to dtype range. For floats clip to [-1, 1] only when the input
        # was already in that convention.
        if dtype == np.int16:
            np.clip(result, -32768, 32767, out=result)
        elif dtype == np.int32:
            np.clip(result, -2147483648, 2147483647, out=result)
        elif dtype in (np.float32, np.float64):
            # Many float WAVs use [-1, 1]; if the original was already in that
            # range, preserve it. Otherwise leave it untouched.
            if np.max(np.abs(self.audio_data)) <= 1.0:
                np.clip(result, -1.0, 1.0, out=result)

        return result.astype(dtype), total_clicks

    def _check_steps(self):
        """Return True if processing should proceed, else False.
        Shows informational dialogs as needed."""
        any_enabled = any([
            self.do_dc_remove.get(), self.do_riaa.get(), self.do_rumble.get(),
            self.do_declick.get(), self.do_noise_reduce.get(),
            self.do_hiss.get(), self.do_normalize.get()
        ])
        if not any_enabled:
            messagebox.showinfo("Nothing to do", "Enable at least one processing step.")
            return False

        if self.do_noise_reduce.get() and self.noise_profile is None:
            return messagebox.askyesno(
                "No Noise Profile",
                "Spectral Noise Reduction is enabled but no noise profile has\n"
                "been captured.\n\n"
                "If you continue, the noise reduction step will be SKIPPED for\n"
                "this run. Other enabled steps will still apply.\n\n"
                "Continue anyway?"
            )

        # Sanity check: rumble/hiss cutoff vs. sample rate
        nyq = self.sample_rate / 2 if self.sample_rate else 0
        if self.do_rumble.get() and self.rumble_freq.get() >= nyq:
            messagebox.showerror(
                "Invalid Filter",
                f"Rumble cutoff ({self.rumble_freq.get():.0f} Hz) is at or above\n"
                f"the Nyquist frequency ({nyq:.0f} Hz). Lower the cutoff."
            )
            return False
        if self.do_hiss.get() and self.hiss_freq.get() >= nyq:
            messagebox.showerror(
                "Invalid Filter",
                f"Hiss cutoff ({self.hiss_freq.get():.0f} Hz) is at or above\n"
                f"the Nyquist frequency ({nyq:.0f} Hz). Lower the cutoff."
            )
            return False
        return True

    def _lock_ui(self):
        self._processing = True
        self.btn_preview.config(state=tk.DISABLED)
        self.btn_apply_full.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_save_preview.config(state=tk.DISABLED)
        self.btn_undo.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)

    def _unlock_ui_after(self, mode):
        self._processing = False
        if self.audio_data is not None:
            self.btn_apply_full.config(state=tk.NORMAL)
        if self.region_start is not None:
            self.btn_preview.config(state=tk.NORMAL)
        if mode == "preview" and self.preview_data is not None:
            self.btn_save_preview.config(state=tk.NORMAL)
            if PLAYBACK_AVAILABLE:
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
        if self.region_start is None or self.region_end is None:
            messagebox.showinfo(
                "No region",
                "Drag on the top waveform (or type a range) to select a "
                "preview region first."
            )
            return
        self._update_summary()
        self._lock_ui()
        self.progress["value"] = 0
        threading.Thread(target=self._process_preview, daemon=True).start()

    def _process_preview(self):
        try:
            s = self.region_start
            e = self.region_end
            si = int(s * self.sample_rate)
            ei = int(e * self.sample_rate)

            self._update_progress(5, f"Processing preview ({s:.2f}s – {e:.2f}s)...")
            # Work on an explicit copy so the original array is never touched
            segment = np.array(self.audio_data[si:ei], dtype=np.float64, copy=True)
            result, n_clicks = self._run_pipeline(segment,
                                                  progress_offset=5,
                                                  progress_span=85)
            self.preview_data = result
            self.root.after(0, lambda: self._finish_preview(n_clicks))

        except Exception as exc:
            err = f"{exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Preview Error", err))
            self.root.after(0, lambda: self._update_progress(0, "Preview failed."))
            self.root.after(0, lambda: self._unlock_ui_after("preview"))

    def _finish_preview(self, n_clicks=0):
        msg = "Preview done. Play and compare. If it sounds right, click 'Apply to Full Track'."
        if self.do_declick.get() and n_clicks > 0:
            msg = f"Preview done. {n_clicks} click(s) repaired in this region. " \
                  "Play and compare."
        self._update_progress(100, msg)
        self._refresh_plots()
        self._unlock_ui_after("preview")

    def start_full_processing(self):
        if self.audio_data is None or not self._check_steps():
            return
        # Estimate memory and warn if huge
        n_samples = self.audio_data.size
        bytes_needed = n_samples * 8 * 2  # float64 working copy + result
        if bytes_needed > 2 * 1024 ** 3:
            ok = messagebox.askyesno(
                "Large file",
                f"This file will use roughly {bytes_needed / 1024**3:.1f} GB of "
                "memory during processing.\n\nContinue?"
            )
            if not ok:
                return

        self._update_summary()
        self._save_undo()
        self._lock_ui()
        self.progress["value"] = 0
        threading.Thread(target=self._process_full, daemon=True).start()

    def _process_full(self):
        try:
            self._update_progress(5, "Processing full track...")
            data = np.array(self.audio_data, dtype=np.float64, copy=True)
            result, n_clicks = self._run_pipeline(data,
                                                  progress_offset=5,
                                                  progress_span=90)
            self.processed_data = result
            self.preview_data = None
            self.root.after(0, lambda: self._finish_full(n_clicks))
        except Exception as exc:
            err = f"{exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Processing Error", err))
            self.root.after(0, lambda: self._update_progress(0, "Processing failed."))
            self.root.after(0, lambda: self._unlock_ui_after("full"))

    def _finish_full(self, n_clicks=0):
        msg = "Full track done. Click 'Save Processed WAV' to export."
        if self.do_declick.get() and n_clicks > 0:
            msg = f"Full track done. {n_clicks} click(s) repaired total. Click 'Save Processed WAV' to export."
        self._update_progress(100, msg)
        self._refresh_plots()
        self._unlock_ui_after("full")

    def _update_progress(self, pct, text):
        """Thread-safe-ish progress update. Called from worker threads via
        small Tk operations; we marshal to main thread when possible."""
        def _do():
            if pct is not None:
                try:
                    self.progress["value"] = max(0, min(100, pct))
                except Exception:
                    pass
            try:
                self.lbl_status.config(text=text)
            except Exception:
                pass
        # If on main thread already, run directly; else schedule
        try:
            if threading.current_thread() is threading.main_thread():
                _do()
            else:
                self.root.after(0, _do)
        except Exception:
            pass

    def _save_undo(self):
        if self.processed_data is not None:
            self.undo_stack.append(self.processed_data.copy())
            if len(self.undo_stack) > UNDO_LEVELS:
                self.undo_stack.pop(0)

    def undo(self):
        if self._processing:
            return
        if self.undo_stack:
            self.processed_data = self.undo_stack.pop()
            self.btn_undo.config(state=tk.NORMAL if self.undo_stack else tk.DISABLED)
            self._refresh_plots()
            self._update_progress(self.progress["value"], "Undo applied.")
        else:
            self.btn_undo.config(state=tk.DISABLED)

    def reset_to_original(self):
        if self.audio_data is None or self._processing:
            return
        if not messagebox.askyesno(
            "Reset to original",
            "Discard all processed audio and return to the original?\n\n"
            "(This pushes the current result onto the undo stack first.)"
        ):
            return
        self._save_undo()
        self.processed_data = None
        self.preview_data = None
        self._refresh_plots()
        self.btn_save.config(state=tk.DISABLED)
        self.btn_save_preview.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        if PLAYBACK_AVAILABLE:
            self.btn_play_proc.config(state=tk.DISABLED)
            self.btn_play_preview.config(state=tk.DISABLED)
        self._update_progress(0, "Reset to original.")

    def _on_close(self):
        if self._processing:
            if not messagebox.askyesno(
                "Quit",
                "Processing is running. Quit anyway?"
            ):
                return
        try:
            self.stop_playback()
        except Exception:
            pass
        try:
            plt.close(self.fig)
        except Exception:
            pass
        self.root.destroy()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    _apply_dpi_awareness()
    root = tk.Tk()
    app = VinylCleanupApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
