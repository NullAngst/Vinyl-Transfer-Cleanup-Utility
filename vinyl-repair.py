"""
Vinyl Transfer Cleanup Utility
A desktop tool for cleaning up audio digitized from vinyl records.

Requires: numpy, scipy, matplotlib, sounddevice (optional)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, medfilt, stft, istft, bilinear_zpk, zpk2tf
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import traceback

try:
    import sounddevice as sd
    PLAYBACK_AVAILABLE = True
except ImportError:
    PLAYBACK_AVAILABLE = False


class VinylCleanupApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vinyl Transfer Cleanup Utility")
        self.root.geometry("1150x780")
        self.root.minsize(900, 620)

        # Audio state
        self.filepath = None
        self.audio_data = None
        self.sample_rate = None
        self.processed_data = None
        self.noise_profile = None
        self.undo_stack = []

        # Playback state
        self.is_playing = False

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # Top info bar
        info_bar = ttk.Frame(self.root, padding=(10, 4))
        info_bar.pack(side=tk.TOP, fill=tk.X)

        self.lbl_file_info = ttk.Label(info_bar, text="No file loaded", font=("Courier", 10))
        self.lbl_file_info.pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(info_bar, text="Ready", foreground="gray")
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # Bottom action bar
        bottom_bar = ttk.Frame(self.root, padding=(10, 6))
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(bottom_bar, mode="determinate", length=180)
        self.progress.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_process = ttk.Button(
            bottom_bar, text="Process Audio",
            command=self.start_processing, state=tk.DISABLED
        )
        self.btn_process.pack(side=tk.LEFT, padx=4)

        self.btn_save = ttk.Button(
            bottom_bar, text="Save Processed WAV",
            command=self.save_file, state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.btn_undo = ttk.Button(
            bottom_bar, text="Undo Last Process",
            command=self.undo, state=tk.DISABLED
        )
        self.btn_undo.pack(side=tk.LEFT, padx=4)

        self.btn_reset = ttk.Button(
            bottom_bar, text="Reset to Original",
            command=self.reset_to_original, state=tk.DISABLED
        )
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        # Main paned window
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel
        left_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_frame, weight=1)

        self._build_left_panel(left_frame)

        # Right panel
        right_frame = ttk.Frame(main_pane, padding=5)
        main_pane.add(right_frame, weight=3)

        self._build_right_panel(right_frame)

    def _build_left_panel(self, parent):
        # File section
        file_frame = ttk.LabelFrame(parent, text="File", padding=8)
        file_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(file_frame, text="Load WAV File", command=self.load_file).pack(fill=tk.X)

        # Playback section
        if PLAYBACK_AVAILABLE:
            pb_frame = ttk.LabelFrame(parent, text="Playback", padding=8)
            pb_frame.pack(fill=tk.X, pady=(0, 6))

            row1 = ttk.Frame(pb_frame)
            row1.pack(fill=tk.X)
            self.btn_play_orig = ttk.Button(
                row1, text="Play Original",
                command=self.play_original, state=tk.DISABLED
            )
            self.btn_play_orig.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
            self.btn_play_proc = ttk.Button(
                row1, text="Play Processed",
                command=self.play_processed, state=tk.DISABLED
            )
            self.btn_play_proc.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

            self.btn_stop = ttk.Button(
                pb_frame, text="Stop Playback",
                command=self.stop_playback, state=tk.DISABLED
            )
            self.btn_stop.pack(fill=tk.X, pady=(5, 0))
        else:
            ttk.Label(
                parent,
                text="Install 'sounddevice' for in-app playback.",
                foreground="gray", wraplength=200, justify=tk.LEFT
            ).pack(anchor=tk.W, pady=(0, 6))

        # Processing notebook
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)

        self._build_tab_clicks(nb)
        self._build_tab_filters(nb)
        self._build_tab_levels(nb)

    def _slider_row(self, parent, variable, from_, to, fmt, label_width=6):
        """Helper: returns a frame with a Scale and a live readout label."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)
        lbl = ttk.Label(frame, text=fmt.format(variable.get()), width=label_width, anchor=tk.E)
        lbl.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=from_, to=to,
            variable=variable, orient=tk.HORIZONTAL,
            command=lambda v: lbl.config(text=fmt.format(float(v)))
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_tab_clicks(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Clicks & Noise")

        # De-Click
        ttk.Label(tab, text="De-Click / De-Pop", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_declick = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Enable De-Click", variable=self.do_declick).pack(anchor=tk.W)

        ttk.Label(tab, text="Detection Sensitivity (lower = more aggressive)").pack(anchor=tk.W, pady=(6, 0))
        self.click_sens = tk.DoubleVar(value=5.0)
        self._slider_row(tab, self.click_sens, 1.0, 15.0, "{:.1f}")

        ttk.Label(tab, text="Repair Window (ms)").pack(anchor=tk.W, pady=(5, 0))
        self.click_window = tk.DoubleVar(value=2.0)
        self._slider_row(tab, self.click_window, 0.5, 10.0, "{:.1f}")

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        # Spectral Noise Reduction
        ttk.Label(tab, text="Spectral Noise Reduction", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_noise_reduce = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Enable Noise Reduction", variable=self.do_noise_reduce).pack(anchor=tk.W)

        ttk.Label(
            tab,
            text="Select a quiet passage (surface noise only)\nand capture its profile below.",
            foreground="gray", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(4, 2))

        time_frame = ttk.Frame(tab)
        time_frame.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(time_frame, text="Start (s):").pack(side=tk.LEFT)
        self.noise_start = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            time_frame, from_=0.0, to=9999.0, increment=0.1,
            textvariable=self.noise_start, width=6, format="%.2f"
        ).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(time_frame, text="End (s):").pack(side=tk.LEFT)
        self.noise_end = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            time_frame, from_=0.0, to=9999.0, increment=0.1,
            textvariable=self.noise_end, width=6, format="%.2f"
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(tab, text="Capture Noise Profile", command=self.capture_noise_profile).pack(fill=tk.X, pady=6)
        self.lbl_noise_profile = ttk.Label(tab, text="No profile captured", foreground="gray")
        self.lbl_noise_profile.pack(anchor=tk.W)

        ttk.Label(tab, text="Reduction Strength").pack(anchor=tk.W, pady=(8, 0))
        self.noise_alpha = tk.DoubleVar(value=2.0)
        self._slider_row(tab, self.noise_alpha, 0.5, 6.0, "{:.1f}")

        ttk.Label(tab, text="Spectral Floor (reduces musical noise)").pack(anchor=tk.W, pady=(5, 0))
        self.noise_beta = tk.DoubleVar(value=0.02)
        self._slider_row(tab, self.noise_beta, 0.001, 0.2, "{:.3f}", label_width=7)

    def _build_tab_filters(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Filters")

        # Rumble (high-pass)
        ttk.Label(tab, text="Rumble Filter (High-Pass)", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_rumble = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Enable Rumble Filter", variable=self.do_rumble).pack(anchor=tk.W)
        ttk.Label(
            tab, text="Removes turntable motor rumble and\nlow-frequency mechanical noise.",
            foreground="gray", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(2, 5))

        ttk.Label(tab, text="Cutoff Frequency (Hz)").pack(anchor=tk.W)
        self.rumble_freq = tk.DoubleVar(value=30.0)
        self._slider_row(tab, self.rumble_freq, 10.0, 150.0, "{:.0f}")

        ttk.Label(tab, text="Filter Order (higher = steeper rolloff)").pack(anchor=tk.W, pady=(5, 0))
        self.rumble_order = tk.IntVar(value=4)
        order_frame = ttk.Frame(tab)
        order_frame.pack(fill=tk.X)
        for o in (2, 4, 6, 8):
            ttk.Radiobutton(order_frame, text=str(o), variable=self.rumble_order, value=o).pack(side=tk.LEFT)

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        # Hiss (low-pass)
        ttk.Label(tab, text="Hiss Filter (Low-Pass)", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_hiss = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Enable Hiss Filter", variable=self.do_hiss).pack(anchor=tk.W)
        ttk.Label(
            tab,
            text="Blunt high-frequency cut. Use Spectral Noise\nReduction instead for better results.",
            foreground="gray", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(2, 5))

        ttk.Label(tab, text="Cutoff Frequency (Hz)").pack(anchor=tk.W)
        self.hiss_freq = tk.DoubleVar(value=14000.0)
        self._slider_row(tab, self.hiss_freq, 3000.0, 20000.0, "{:.0f}", label_width=7)

        ttk.Label(tab, text="Filter Order").pack(anchor=tk.W, pady=(5, 0))
        self.hiss_order = tk.IntVar(value=4)
        hiss_order_frame = ttk.Frame(tab)
        hiss_order_frame.pack(fill=tk.X)
        for o in (2, 4, 6, 8):
            ttk.Radiobutton(hiss_order_frame, text=str(o), variable=self.hiss_order, value=o).pack(side=tk.LEFT)

    def _build_tab_levels(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Levels")

        # DC Offset
        ttk.Label(tab, text="DC Offset Removal", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_dc_remove = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Remove DC Offset", variable=self.do_dc_remove).pack(anchor=tk.W)
        ttk.Label(
            tab,
            text="Removes constant DC bias common in\nolder playback equipment. Almost\nalways safe to leave on.",
            foreground="gray", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        # Normalization
        ttk.Label(tab, text="Normalization", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_normalize = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Enable Normalization", variable=self.do_normalize).pack(anchor=tk.W)

        self.norm_mode = tk.StringVar(value="peak")
        ttk.Radiobutton(tab, text="Peak (maximize loudness to target)", variable=self.norm_mode, value="peak").pack(anchor=tk.W, pady=(4, 0))
        ttk.Radiobutton(tab, text="RMS (match perceived loudness to target)", variable=self.norm_mode, value="rms").pack(anchor=tk.W)

        ttk.Label(tab, text="Target Level (dBFS)").pack(anchor=tk.W, pady=(5, 0))
        self.norm_target = tk.DoubleVar(value=-1.0)
        self._slider_row(tab, self.norm_target, -20.0, 0.0, "{:.1f}")

        ttk.Separator(tab).pack(fill=tk.X, pady=10)

        # RIAA
        ttk.Label(tab, text="RIAA De-Emphasis", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.do_riaa = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Apply RIAA De-Emphasis", variable=self.do_riaa).pack(anchor=tk.W)
        ttk.Label(
            tab,
            text="Only enable this if your phono preamp\ndid NOT apply RIAA equalization.\nIf the recording sounds bass-heavy and\ntreble-dull, this may be the cause.",
            foreground="gray", justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(2, 0))

    def _build_right_panel(self, parent):
        # View selector
        view_row = ttk.Frame(parent)
        view_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(view_row, text="View:").pack(side=tk.LEFT)
        self.view_mode = tk.StringVar(value="waveform")
        for label, val in (("Waveform", "waveform"), ("Spectrum", "spectrum"), ("Both", "both")):
            ttk.Radiobutton(
                view_row, text=label,
                variable=self.view_mode, value=val,
                command=self._refresh_plots
            ).pack(side=tk.LEFT, padx=5)

        self.fig = plt.Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._init_axes()

    # ------------------------------------------------------------------
    # PLOT MANAGEMENT
    # ------------------------------------------------------------------

    def _init_axes(self):
        self.fig.clear()
        mode = self.view_mode.get()

        if mode == "both":
            axes = self.fig.subplots(2, 2)
            self._ax_ow = axes[0][0]
            self._ax_pw = axes[1][0]
            self._ax_os = axes[0][1]
            self._ax_ps = axes[1][1]
        else:
            axes = self.fig.subplots(2, 1)
            self._ax_ow = axes[0]
            self._ax_pw = axes[1]
            self._ax_os = None
            self._ax_ps = None

        self._ax_ow.set_title("Original")
        self._ax_pw.set_title("Processed")
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def _refresh_plots(self):
        self._init_axes()
        mode = self.view_mode.get()

        if mode in ("waveform", "both"):
            if self.audio_data is not None:
                self._draw_waveform(self.audio_data, self._ax_ow, "Original - Waveform")
            if self.processed_data is not None:
                self._draw_waveform(self.processed_data, self._ax_pw, "Processed - Waveform")
        if mode in ("spectrum", "both"):
            target_orig = self._ax_os if mode == "both" else self._ax_ow
            target_proc = self._ax_ps if mode == "both" else self._ax_pw
            if self.audio_data is not None:
                lbl = "Original - Spectrum" if mode == "spectrum" else "Spectrum"
                self._draw_spectrum(self.audio_data, target_orig, lbl)
            if self.processed_data is not None:
                lbl = "Processed - Spectrum" if mode == "spectrum" else "Spectrum"
                self._draw_spectrum(self.processed_data, target_proc, lbl)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def _draw_waveform(self, data, ax, title):
        ax.clear()
        ax.set_title(title, fontsize=9)
        if data is None:
            return

        target_len = 60000
        step = max(1, len(data) // target_len)
        plot_data = data[::step]
        time_ax = np.linspace(0, len(data) / self.sample_rate, len(plot_data))

        if plot_data.ndim > 1:
            ax.plot(time_ax, plot_data[:, 0], color="#2196F3", alpha=0.7, linewidth=0.4, label="L")
            ax.plot(time_ax, plot_data[:, 1], color="#FF5722", alpha=0.7, linewidth=0.4, label="R")
            ax.legend(loc="upper right", fontsize=7)
        else:
            ax.plot(time_ax, plot_data, color="#2196F3", alpha=0.8, linewidth=0.4)

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    def _draw_spectrum(self, data, ax, title):
        ax.clear()
        ax.set_title(title, fontsize=9)
        if data is None or self.sample_rate is None:
            return

        ch = data[:, 0].astype(np.float64) if data.ndim > 1 else data.astype(np.float64)

        # Cap FFT length for performance
        fft_len = min(len(ch), 131072)
        ch_fft = ch[:fft_len]
        window = np.hanning(len(ch_fft))
        spectrum = np.abs(np.fft.rfft(ch_fft * window))
        freqs = np.fft.rfftfreq(len(ch_fft), 1.0 / self.sample_rate)

        spectrum_db = 20 * np.log10(np.maximum(spectrum / (spectrum.max() + 1e-10), 1e-10))

        ax.semilogx(freqs[1:], spectrum_db[1:], color="#2196F3", alpha=0.85, linewidth=0.5)
        ax.set_xlim(20, self.sample_rate / 2)
        ax.set_ylim(-80, 5)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Level (dB)", fontsize=8)
        ax.grid(True, alpha=0.25, which="both")
        ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # FILE I/O
    # ------------------------------------------------------------------

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
            self.noise_profile = None
            self.undo_stack = []

            duration = len(data) / rate
            channels = "Stereo" if data.ndim > 1 else "Mono"
            depth = str(data.dtype)

            self.lbl_file_info.config(
                text=(
                    f"{os.path.basename(filepath)}  |  "
                    f"{rate} Hz  |  {depth}  |  {channels}  |  "
                    f"{duration:.1f}s  ({duration/60:.1f} min)"
                )
            )

            self.btn_process.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)
            self.btn_reset.config(state=tk.DISABLED)
            self.lbl_noise_profile.config(text="No profile captured", foreground="gray")

            if PLAYBACK_AVAILABLE:
                self.btn_play_orig.config(state=tk.NORMAL)
                self.btn_play_proc.config(state=tk.DISABLED)

            self._refresh_plots()
            self.lbl_status.config(text=f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load file:\n{e}")
            self.lbl_status.config(text="Error loading file.")

    def save_file(self):
        if self.processed_data is None:
            return

        base = os.path.basename(self.filepath) if self.filepath else "audio.wav"
        suggested = "cleaned_" + base

        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=suggested
        )

        if filepath:
            try:
                wavfile.write(filepath, self.sample_rate, self.processed_data)
                self.lbl_status.config(text=f"Saved: {os.path.basename(filepath)}")
                messagebox.showinfo("Saved", f"File saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n{e}")

    # ------------------------------------------------------------------
    # PLAYBACK
    # ------------------------------------------------------------------

    def _to_float32(self, data):
        if data.dtype == np.int16:
            return (data.astype(np.float32) / 32768.0)
        if data.dtype == np.int32:
            return (data.astype(np.float32) / 2147483648.0)
        if data.dtype in (np.float32, np.float64):
            return data.astype(np.float32)
        return data.astype(np.float32)

    def play_original(self):
        self._play_audio(self.audio_data)

    def play_processed(self):
        if self.processed_data is not None:
            self._play_audio(self.processed_data)

    def _play_audio(self, data):
        if not PLAYBACK_AVAILABLE or data is None:
            return
        self.stop_playback()
        self.is_playing = True
        if PLAYBACK_AVAILABLE:
            self.btn_stop.config(state=tk.NORMAL)

        def _run():
            try:
                sd.play(self._to_float32(data), self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self.is_playing = False
                if PLAYBACK_AVAILABLE:
                    self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

        threading.Thread(target=_run, daemon=True).start()

    def stop_playback(self):
        if PLAYBACK_AVAILABLE:
            sd.stop()
        self.is_playing = False
        if PLAYBACK_AVAILABLE:
            self.btn_stop.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # NOISE PROFILE CAPTURE
    # ------------------------------------------------------------------

    def capture_noise_profile(self):
        if self.audio_data is None:
            messagebox.showwarning("No File", "Load a WAV file first.")
            return

        try:
            start_sec = float(self.noise_start.get())
            end_sec = float(self.noise_end.get())
            duration = len(self.audio_data) / self.sample_rate

            if start_sec >= end_sec:
                messagebox.showerror("Invalid Range", "Start must be less than end.")
                return
            if end_sec > duration:
                messagebox.showerror("Invalid Range", f"End time exceeds file duration ({duration:.2f}s).")
                return
            if (end_sec - start_sec) < 0.1:
                messagebox.showerror("Invalid Range", "Noise sample must be at least 0.1 seconds long.")
                return

            start_s = int(start_sec * self.sample_rate)
            end_s = int(end_sec * self.sample_rate)

            data = self.audio_data.astype(np.float64)
            region = np.mean(data[start_s:end_s], axis=1) if data.ndim > 1 else data[start_s:end_s]

            nperseg = 2048
            _, _, Zxx = stft(region, fs=self.sample_rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
            self.noise_profile = np.mean(np.abs(Zxx), axis=1)

            self.lbl_noise_profile.config(
                text=f"Profile captured: {start_sec:.2f}s to {end_sec:.2f}s",
                foreground="green"
            )
            self.do_noise_reduce.set(True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture noise profile:\n{e}")

    # ------------------------------------------------------------------
    # DSP FUNCTIONS
    # ------------------------------------------------------------------

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
        Median filter based de-click.

        Compares the signal to a median-filtered reference to isolate
        click/pop residuals. Detected transients are replaced with the
        median reference, which closely approximates the underlying audio.
        """
        # ~0.5ms kernel, minimum 5, forced odd
        kernel = max(5, int(self.sample_rate * 0.0005) | 1)
        ref = medfilt(ch_data.astype(np.float64), kernel_size=kernel)
        residual = ch_data - ref
        threshold = sensitivity * np.std(residual)

        if threshold == 0:
            return ch_data.copy()

        mask = np.abs(residual) > threshold

        # Dilate the mask to cover the full extent of each impulse
        dilation = max(1, int(self.sample_rate * window_ms / 1000.0))
        struct = np.ones(dilation, dtype=bool)
        mask = binary_dilation(mask, structure=struct)

        cleaned = ch_data.copy()
        cleaned[mask] = ref[mask]
        return cleaned

    def _spectral_nr(self, ch_data, noise_profile, alpha, beta):
        """
        STFT-domain spectral subtraction with a hard spectral floor.

        alpha: over-subtraction factor. Higher values remove more noise
               but risk audible artifacts ("musical noise").
        beta:  spectral floor as a fraction of the signal magnitude.
               Prevents over-subtraction and the tonal ringing artifacts
               that result from it.
        """
        nperseg = 2048
        noverlap = nperseg * 3 // 4

        _, _, Zxx = stft(ch_data, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Interpolate noise profile to match STFT bin count
        n_bins = mag.shape[0]
        if len(noise_profile) != n_bins:
            noise_profile = np.interp(
                np.linspace(0, 1, n_bins),
                np.linspace(0, 1, len(noise_profile)),
                noise_profile
            )

        clean_mag = np.maximum(mag - alpha * noise_profile[:, np.newaxis], beta * mag)
        _, result = istft(
            clean_mag * np.exp(1j * phase),
            fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Trim or pad to match original length
        n = len(ch_data)
        if len(result) >= n:
            return result[:n]
        return np.pad(result, (0, n - len(result)))

    def _riaa_deemphasis(self, ch_data, fs):
        """
        RIAA playback de-emphasis (IEC 60098).
        Time constants: t1=3180us, t2=318us, t3=75us.

        The transfer function is:
            H(s) = K * (1 + s*t2) / ((1 + s*t1)(1 + s*t3))

        Applied via bilinear transform to a digital filter.
        Only use this if your phono preamp did not apply RIAA equalization.
        """
        t1, t2, t3 = 3180e-6, 318e-6, 75e-6

        # Zero at -1/t2, poles at -1/t1 and -1/t3
        z_a = [-1.0 / t2]
        p_a = [-1.0 / t1, -1.0 / t3]
        k_a = t1 / t2  # DC normalization

        z_d, p_d, k_d = bilinear_zpk(z_a, p_a, k_a, fs=fs)
        b, a = zpk2tf(z_d, p_d, k_d)
        return filtfilt(b, a, ch_data)

    def _normalize(self, data, mode, target_dbfs):
        target_lin = 10.0 ** (target_dbfs / 20.0)
        dtype = self.audio_data.dtype

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
        else:  # rms
            rms = np.sqrt(np.mean(data ** 2))
            if rms == 0:
                return data
            scale = (target_lin * max_val) / rms

        return data * scale

    # ------------------------------------------------------------------
    # PROCESSING PIPELINE
    # ------------------------------------------------------------------

    def _save_undo(self):
        if self.processed_data is not None:
            self.undo_stack.append(self.processed_data.copy())
            if len(self.undo_stack) > 10:
                self.undo_stack.pop(0)
            self.btn_undo.config(state=tk.NORMAL)

    def undo(self):
        if self.undo_stack:
            self.processed_data = self.undo_stack.pop()
            if not self.undo_stack:
                self.btn_undo.config(state=tk.DISABLED)
            self._refresh_plots()
            self.lbl_status.config(text="Undo applied.")
        else:
            self.btn_undo.config(state=tk.DISABLED)

    def reset_to_original(self):
        if self.audio_data is None:
            return
        self._save_undo()
        self.processed_data = None
        self._refresh_plots()
        self.btn_save.config(state=tk.DISABLED)
        if PLAYBACK_AVAILABLE:
            self.btn_play_proc.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        self.lbl_status.config(text="Reset to original.")

    def start_processing(self):
        if self.audio_data is None:
            return

        steps_enabled = any([
            self.do_dc_remove.get(),
            self.do_riaa.get(),
            self.do_rumble.get(),
            self.do_declick.get(),
            self.do_noise_reduce.get(),
            self.do_hiss.get(),
            self.do_normalize.get(),
        ])

        if not steps_enabled:
            messagebox.showinfo("Nothing to do", "Enable at least one processing step.")
            return

        if self.do_noise_reduce.get() and self.noise_profile is None:
            if not messagebox.askyesno(
                "No Noise Profile",
                "Spectral noise reduction is enabled but no noise profile has been captured.\n\n"
                "Proceed anyway (noise reduction will be skipped)?"
            ):
                return

        self._save_undo()
        self.btn_process.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        if PLAYBACK_AVAILABLE:
            self.btn_play_proc.config(state=tk.DISABLED)
        self.progress["value"] = 0

        threading.Thread(target=self._process_audio, daemon=True).start()

    def _update_progress(self, pct, text):
        self.progress["value"] = pct
        self.lbl_status.config(text=text)
        self.root.update_idletasks()

    def _process_audio(self):
        try:
            # Always process from original for deterministic, non-destructive results.
            data = self.audio_data.astype(np.float64)
            fs = self.sample_rate
            is_stereo = data.ndim > 1
            n_ch = data.shape[1] if is_stereo else 1

            # Calculate steps for accurate progress
            per_ch_steps = sum([
                self.do_dc_remove.get(),
                self.do_riaa.get(),
                self.do_rumble.get(),
                self.do_declick.get(),
                self.do_noise_reduce.get() and self.noise_profile is not None,
                self.do_hiss.get(),
            ])
            total_steps = per_ch_steps * n_ch + self.do_normalize.get()
            step_pct = 90.0 / max(total_steps, 1)
            pct = 5.0

            self._update_progress(pct, "Starting...")
            processed_channels = []

            for ch in range(n_ch):
                ch_data = data[:, ch] if is_stereo else data.flatten()
                ch_label = f"ch {ch + 1}/{n_ch}"

                # Order matters:
                # 1. DC offset (before everything)
                # 2. RIAA (equalization before noise processing)
                # 3. Rumble (remove sub-bass before de-click to reduce false positives)
                # 4. De-click
                # 5. Spectral NR
                # 6. Hiss LP filter

                if self.do_dc_remove.get():
                    self._update_progress(pct, f"Removing DC offset ({ch_label})...")
                    ch_data = ch_data - np.mean(ch_data)
                    pct += step_pct

                if self.do_riaa.get():
                    self._update_progress(pct, f"Applying RIAA de-emphasis ({ch_label})...")
                    ch_data = self._riaa_deemphasis(ch_data, fs)
                    pct += step_pct

                if self.do_rumble.get():
                    self._update_progress(pct, f"Rumble filter ({ch_label})...")
                    ch_data = self._butter_hp(ch_data, self.rumble_freq.get(), fs, self.rumble_order.get())
                    pct += step_pct

                if self.do_declick.get():
                    self._update_progress(pct, f"De-clicking ({ch_label})...")
                    ch_data = self._declick(ch_data, self.click_sens.get(), self.click_window.get())
                    pct += step_pct

                if self.do_noise_reduce.get() and self.noise_profile is not None:
                    self._update_progress(pct, f"Spectral noise reduction ({ch_label})...")
                    ch_data = self._spectral_nr(
                        ch_data, self.noise_profile,
                        self.noise_alpha.get(), self.noise_beta.get()
                    )
                    pct += step_pct

                if self.do_hiss.get():
                    self._update_progress(pct, f"Hiss filter ({ch_label})...")
                    ch_data = self._butter_lp(ch_data, self.hiss_freq.get(), fs, self.hiss_order.get())
                    pct += step_pct

                processed_channels.append(ch_data)

            self._update_progress(90, "Reconstructing audio...")
            processed = np.column_stack(processed_channels) if is_stereo else processed_channels[0]

            if self.do_normalize.get():
                self._update_progress(93, "Normalizing...")
                processed = self._normalize(processed, self.norm_mode.get(), self.norm_target.get())
                pct += step_pct

            # Clip to dtype range and convert
            dtype = self.audio_data.dtype
            if dtype == np.int16:
                np.clip(processed, -32768, 32767, out=processed)
            elif dtype == np.int32:
                np.clip(processed, -2147483648, 2147483647, out=processed)
            elif dtype in (np.float32, np.float64):
                np.clip(processed, -1.0, 1.0, out=processed)

            self.processed_data = processed.astype(dtype)
            self.root.after(0, self._finish_processing)

        except Exception as e:
            err = f"{e}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Processing Error", err))
            self.root.after(0, lambda: self._update_progress(0, "Error during processing."))
            self.root.after(0, lambda: self.btn_process.config(state=tk.NORMAL))

    def _finish_processing(self):
        self._refresh_plots()
        self.btn_process.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.btn_reset.config(state=tk.NORMAL)
        if PLAYBACK_AVAILABLE:
            self.btn_play_proc.config(state=tk.NORMAL)
        self._update_progress(100, "Processing complete.")


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()

    # Windows high-DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = VinylCleanupApp(root)
    root.mainloop()
