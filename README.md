# Vinyl Transfer Cleanup Utility

A desktop GUI application for cleaning up audio digitized from vinyl records. Built with Python, tkinter, scipy, and matplotlib.

Handles the most common problems with vinyl transfers: clicks and pops, surface noise/hiss, turntable rumble, DC offset, and level normalization. Includes an STFT-based spectral noise reduction engine and a proper RIAA de-emphasis filter for edge cases where the phono preamp did not apply equalization.

All processing is non-destructive. The original file is never modified, and every run starts fresh from the original source data. An undo stack lets you roll back the last ten processing runs.

---

## Features

- **De-Click / De-Pop** using a second-order difference (discrete Laplacian) detector with a robust MAD-based threshold and dilation-based repair windows. Detected regions are repaired by linear interpolation across the gap, not replaced with a flat reference. More robust on high-frequency content than median-residual approaches.
- **Spectral Noise Reduction** via STFT spectral subtraction with a configurable over-subtraction factor and spectral floor. Per-channel noise profiles are captured for stereo files. Requires you to capture a noise profile from a quiet passage (pure surface noise) before processing.
- **Rumble Filter** (Butterworth high-pass), configurable cutoff from 10 to 150 Hz with selectable filter order.
- **Hiss Filter** (Butterworth low-pass), for a quick broadband cut above a chosen frequency. For best results, prefer the spectral noise reduction instead.
- **DC Offset Removal** to eliminate the constant bias introduced by some older equipment.
- **Normalization** with both peak and RMS modes, target level configurable in dBFS, with a hard +18 dB gain cap to prevent runaway gain on quiet RMS-normalized material.
- **RIAA De-Emphasis** for recordings made without a proper phono preamp (rarely needed but included).
- **Preview Region** processing — drag on the waveform to select a segment, dial in settings on that region in seconds, then commit to the full track.
- **Welch-averaged spectrum visualization** with a switchable view mode (Waveform / Spectrum / Both). Long files now show an accurate spectrum instead of just the first three seconds.
- **Live before/after stats** (peak and RMS in dBFS) shown at the top of the visualization.
- **Live processing summary** tab showing the exact step ordering and parameters that will be applied.
- **In-app playback** of original, preview, and processed audio with one-touch A/B comparison (requires `sounddevice`).
- **Undo stack** (10 levels), hard reset to original, and confirmation prompts for destructive actions.
- **Keyboard shortcuts** for all common actions (Ctrl+O, Ctrl+S, Ctrl+Z, F5, F6, Space).
- **Save Preview Region** to its own WAV file for sharing test snippets.
- **Windows high-DPI awareness** applied before window creation (so it actually takes effect).

---

## Requirements

- Python 3.8 or newer
- `numpy`
- `scipy`
- `matplotlib`
- `sounddevice` (optional, for in-app playback)
- `tkinter` (included with most Python distributions)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NullAngst/Vinyl-Transfer-Cleanup-Utility.git
cd Vinyl-Transfer-Cleanup-Utility
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not need in-app playback or are on a headless system, you can install without `sounddevice`:

```bash
pip install numpy scipy matplotlib
```

### 4. Run the application

```bash
python vinyl-repair.py
```

---

## Usage Guide

### Basic workflow

1. **File → Open WAV** (or `Ctrl+O`) and open your vinyl transfer.
2. The file info bar at the top shows the sample rate, bit depth, channel count, and duration.
3. **Drag on the top waveform** to select a 5–10 second region with representative noise.
4. Click **Process Preview Region** (or press `F6`). This is much faster than a full run and lets you tune settings interactively.
5. Use **▶ Original** and **▶ Preview** for direct A/B comparison. Press `Space` to toggle play/stop.
6. When the preview sounds right, click **Apply to Full Track** (or press `F5`).
7. Use **Undo** (`Ctrl+Z`) or **Reset to Original** if the result is not what you wanted.
8. Click **Save Processed WAV** (`Ctrl+S`) to export.

### Keyboard shortcuts

| Key | Action |
|---|---|
| Ctrl+O | Open WAV file |
| Ctrl+S | Save processed WAV |
| Ctrl+Z | Undo last full-track process |
| F5 | Apply to Full Track |
| F6 | Process Preview Region |
| Space | Play / Stop (when not editing a number) |

---

### Processing tabs

#### Clicks & Noise

**De-Click / De-Pop**

This is the most important step for most vinyl transfers. The algorithm computes the second-order difference (a discrete Laplacian) of the signal, which responds strongly to sample-to-sample discontinuities and weakly to smooth audio regardless of frequency. A robust MAD-based threshold (median absolute deviation, scaled by 1.4826 for Gaussian equivalence) flags impulses; the flagged region is then repaired by linear interpolation from the clean samples on either side.

This approach is deliberately *not* a median-filter residual method because (a) `scipy.signal.medfilt` has a known heap-corruption bug that can crash the process on long runs, and (b) for short kernels the median of a high-frequency sinusoid is approximately zero, which causes legitimate audio content to be flagged as clicks.

- **Detection Sensitivity**: Higher values flag fewer transients. Start at 10. If musical transients (snare hits, piano attacks) are being softened, raise the value. If light crackle is getting through, lower it.
- **Repair Window (ms)**: How wide a region around each detected click gets replaced. Keep this short (1-3 ms) for most surface noise. Increase it for heavy pops.
- **Passes**: Run the detector once (default) or twice for severely damaged records. The second pass catches secondary clicks that were masked by larger ones.

**Spectral Noise Reduction**

This addresses continuous surface noise (hiss, hum) that de-click cannot touch because it has no sharp transient character.

To use it:
1. Find a section of the recording with no music, just the surface noise. The first second or two before the music starts is ideal.
2. Enter the start and end times (in seconds) in the range fields, or select a region on the waveform and click **From Selection**.
3. Click **Capture Noise Profile**. The application analyzes the frequency-domain fingerprint of that noise. For stereo files, it captures a separate profile per channel.
4. Enable **Noise Reduction** (auto-enabled when you capture a profile) and set the strength.

- **Reduction Strength (alpha)**: How aggressively the noise floor is subtracted. Values of 1.5 to 3.0 are a reasonable starting point. High values (above 4.0) tend to produce "musical noise" artifacts (a warbling, metallic residual).
- **Spectral Floor (beta)**: The minimum retained fraction of each frequency bin. Raising this reduces musical noise at the expense of less noise removal. Keep it between 0.01 and 0.05 for most material.

#### Filters

**Rumble Filter (High-Pass)**

Removes sub-bass mechanical noise from the turntable motor and tonearm. The default 30 Hz cutoff is appropriate for most situations. If the recording still sounds muddy or has a woofer-pumping quality, try raising the cutoff to 50-80 Hz. Use order 4 unless you need an especially steep rolloff.

**Hiss Filter (Low-Pass)**

A blunt high-frequency cut. This will also cut high-frequency musical content, so it is disabled by default. If you use it, keep the cutoff above 12000 Hz to preserve most of the audio. For hiss removal, spectral noise reduction is almost always a better choice.

#### Levels

**DC Offset Removal**

Leave this on. It does no harm and corrects a constant bias that some older phono stages introduce, which can clip the waveform asymmetrically and cause subtle distortion.

**Normalization**

- **Peak**: Brings the loudest sample in the recording to the target level. Use -1.0 dBFS as the target to leave a 1 dB headroom margin.
- **RMS**: Normalizes to a target average loudness. More useful when you want consistent perceived volume across multiple sides of a record. A hard +18 dB gain cap prevents extreme gain on quiet recordings.

**RIAA De-Emphasis**

Leave this off unless you have a specific reason to enable it. RIAA de-emphasis is applied by every standard phono preamp. You only need this if you ran the cartridge directly into a line-level input without any phono stage, in which case the playback will sound extremely bass-heavy and dull. Enabling it in any other scenario will make the recording sound thin and harsh.

#### Summary

The Summary tab shows an ordered list of every step that will be applied with current settings, so you can verify the chain before pressing Apply. Click **Refresh** after changing settings.

---

### Tips for best results

- **Process order matters.** The application always applies steps in a fixed, sensible order regardless of the tab layout: DC offset removal first, then RIAA (if enabled), then rumble filter, then de-click, then spectral noise reduction, then hiss filter, and finally normalization. The Summary tab shows this in real time.

- **Always preview first.** The preview region runs the entire pipeline on just the selected segment in seconds, even on long files. Use it to tune sensitivity, alpha, and cutoff frequencies before committing to a full run.

- **Capture the noise profile before any clicks are repaired.** The profile capture uses the raw loaded file. This is by design; clicks in the noise sample would corrupt the profile. Select a region with no music and minimal pops.

- **Do not chain multiple processing runs.** Because every run starts from the original file, re-running with different settings simply replaces the previous result. You do not need to reset between attempts.

- **Use the spectrum view.** It is much easier to judge how much noise you have removed by looking at the frequency spectrum than by listening alone, especially in the 1 kHz to 15 kHz range where surface noise lives. The spectrum is computed via Welch's method (averaged periodograms), so it represents the entire file accurately rather than just the first segment.

- **Watch the level stats.** The peak/RMS readout at the top of the visualization shows whether normalization actually applied the gain you asked for.

- **High-order rumble filters can cause phase issues near the cutoff.** If you are doing critical listening after restoration, use order 4 and keep the cutoff well below 50 Hz.

- **For seriously degraded records**, run de-click first with a low sensitivity (3-5) and 2 passes to catch heavy damage before spectral noise reduction. Heavy clicks left in during spectral NR will smear across the spectrum and create odd artifacts.

---

## Troubleshooting

**The application window appears very small on Windows.**
The app sets per-monitor DPI awareness automatically before the window is created. If it still looks small, try right-clicking the Python executable, going to Properties → Compatibility, and enabling high-DPI override.

**`sounddevice` fails to install on Linux.**
You need PortAudio. On Debian/Ubuntu: `sudo apt install libportaudio2 portaudio19-dev`. Then re-run `pip install sounddevice`.

**`sounddevice` fails to install on macOS.**
Try `brew install portaudio` first, then `pip install sounddevice`.

**Processing is slow on long files.**
The de-click step uses `scipy.ndimage.binary_dilation` which can be slow on full album sides (20–30 minutes). This is normal. A 10-minute mono file at 44100 Hz typically takes 20–40 seconds on modern hardware. Spectral noise reduction adds another pass of similar cost. For tuning, always use the Preview Region first — it processes in a fraction of the time.

**The spectrum looks identical before and after noise reduction.**
The spectral floor (`beta`) may be set too high, or the noise profile may not have been captured from a representative region. Also check that the reduction strength is above 1.0 and that the noise profile status shows the green "✓ Profile captured" indicator.

**`No module named tkinter`.**
On Linux, tkinter is not always bundled with Python. Install it with: `sudo apt install python3-tk` (Debian/Ubuntu) or `sudo dnf install python3-tkinter` (Fedora).

**A very large file warning appears.**
For files that would use more than 2 GB of RAM during processing (roughly 45+ minutes of 24-bit stereo), the app will ask for confirmation before proceeding. This is informational; processing will still work if your machine has enough memory.

---

## File format notes

Input and output are WAV files only. The output bit depth matches the input (16-bit in, 16-bit out, etc.). Float32 and int32 WAV files are supported in addition to the standard int16 format. The sample rate is preserved unchanged.

For float WAV files whose sample values fall within [-1, 1], the output is clipped to that range. Float WAV files with values outside [-1, 1] (technically valid but uncommon) are not clipped, preserving any over-unity content.

---

## License

MIT License. See `LICENSE` for details.
