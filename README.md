# Vinyl Transfer Cleanup Utility

A desktop GUI application for cleaning up audio digitized from vinyl records. Built with Python, tkinter, and scipy.

Handles the most common problems with vinyl transfers: clicks and pops, surface noise/hiss, turntable rumble, DC offset, and level normalization. Includes an STFT-based spectral noise reduction engine and a proper RIAA de-emphasis filter for edge cases where the phono preamp did not apply equalization.

All processing is non-destructive. The original file is never modified, and every run starts fresh from the original source data. An undo stack lets you roll back the last ten processing runs.

---

## Features

- **De-Click / De-Pop** using a median filter residual detector with dilation-based repair windows. More robust than simple adjacent-sample differencing.
- **Spectral Noise Reduction** via STFT spectral subtraction with a configurable over-subtraction factor and spectral floor. Requires you to capture a noise profile from a quiet passage (pure surface noise) before processing.
- **Rumble Filter** (Butterworth high-pass), configurable cutoff from 10 to 150 Hz with selectable filter order.
- **Hiss Filter** (Butterworth low-pass), for a quick broadband cut above a chosen frequency. For best results, prefer the spectral noise reduction instead.
- **DC Offset Removal** to eliminate the constant bias introduced by some older equipment.
- **Normalization** with both peak and RMS modes, target level configurable in dBFS.
- **RIAA De-Emphasis** for recordings made without a proper phono preamp (rarely needed but included).
- **Waveform and frequency spectrum visualization** with a switchable view mode (Waveform / Spectrum / Both).
- **In-app playback** of both original and processed audio (requires `sounddevice`).
- **Undo stack** (10 levels) and a hard reset to original.
- **Windows high-DPI awareness** built in.

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
git clone https://github.com/NullAngst/vinyl-cleanup.git
cd vinyl-cleanup
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

1. Click **Load WAV File** and open your vinyl transfer (WAV format only).
2. The file info bar at the top shows the sample rate, bit depth, channel count, and duration.
3. Enable the processing steps you want using the tabs on the left panel.
4. Click **Process Audio** at the bottom.
5. Inspect the before/after waveform or spectrum in the right panel.
6. Use **Undo Last Process** or **Reset to Original** if the result is not what you wanted.
7. Click **Save Processed WAV** to export.

---

### Processing tabs

#### Clicks & Noise

**De-Click / De-Pop**

This is the most important step for most vinyl transfers. The algorithm computes a median-filtered version of the signal, subtracts it to isolate residual impulses, and replaces detected click regions with the smooth median reference.

- **Detection Sensitivity**: Lower values flag more transients as clicks. Start at 5.0. If musical transients (snare hits, piano attacks) are being softened, raise the value. If light crackle is getting through, lower it.
- **Repair Window (ms)**: How wide a region around each detected click gets replaced. Keep this short (1-3 ms) for most surface noise. Increase it for heavy pops.

**Spectral Noise Reduction**

This addresses continuous surface noise (hiss, hum) that de-click cannot touch because it has no sharp transient character.

To use it:
1. Find a section of the recording with no music, just the surface noise. Often the first second or two before the music starts is ideal.
2. Enter the start and end times (in seconds) in the range fields.
3. Click **Capture Noise Profile**. The application analyzes the frequency-domain fingerprint of that noise.
4. Enable **Noise Reduction** and set the strength.

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
- **RMS**: Normalizes to a target average loudness. More useful when you want consistent perceived volume across multiple sides of a record.

**RIAA De-Emphasis**

Leave this off unless you have a specific reason to enable it. RIAA de-emphasis is applied by every standard phono preamp. You only need this if you ran the cartridge directly into a line-level input without any phono stage, in which case the playback will sound extremely bass-heavy and dull. Enabling it in any other scenario will make the recording sound thin and harsh.

---

### Tips for best results

- **Process order matters.** The application always applies steps in a fixed, sensible order regardless of the tab layout: DC offset removal first, then RIAA (if enabled), then rumble filter, then de-click, then spectral noise reduction, then hiss filter, and finally normalization.

- **Capture the noise profile before any clicks are repaired.** The profile capture uses the raw loaded file. This is by design; clicks in the noise sample would corrupt the profile. Select a region with no music and minimal pops.

- **Do not chain multiple processing runs.** Because every run starts from the original file, re-running with different settings simply replaces the previous result. You do not need to reset between attempts.

- **Use the spectrum view.** It is much easier to judge how much noise you have removed by looking at the frequency spectrum than by listening alone, especially in the 1 kHz to 15 kHz range where surface noise lives.

- **High-order rumble filters can cause phase issues near the cutoff.** If you are doing critical listening after restoration, use order 4 and keep the cutoff well below 50 Hz.

- **For seriously degraded records**, run de-click first with a low sensitivity (2.0-3.0) to catch heavy damage before spectral noise reduction. Heavy clicks left in during spectral NR will smear across the spectrum and create odd artifacts.

---

## Troubleshooting

**The application window appears very small on Windows.**
The app sets DPI awareness automatically on Windows. If it still looks small, try right-clicking the Python executable, going to Properties, Compatibility, and enabling high-DPI override.

**`sounddevice` fails to install on Linux.**
You need PortAudio. On Debian/Ubuntu: `sudo apt install libportaudio2 portaudio19-dev`. Then re-run `pip install sounddevice`.

**`sounddevice` fails to install on macOS.**
Try `brew install portaudio` first, then `pip install sounddevice`.

**Processing is slow on long files.**
The de-click step uses `scipy.ndimage.binary_dilation` which can be slow on full album sides (20-30 minutes). This is normal. A 10-minute mono file at 44100 Hz typically takes 20-40 seconds on modern hardware. Spectral noise reduction adds another pass of similar cost.

**The spectrum looks identical before and after noise reduction.**
The spectral floor (`beta`) may be set too high, or the noise profile may not have been captured from a representative region. Also check that the reduction strength is above 1.0.

**`No module named tkinter`.**
On Linux, tkinter is not always bundled with Python. Install it with: `sudo apt install python3-tk` (Debian/Ubuntu) or `sudo dnf install python3-tkinter` (Fedora).

---

## File format notes

Input and output are WAV files only. The output bit depth matches the input (16-bit in, 16-bit out, etc.). Float32 and int32 WAV files are supported in addition to the standard int16 format. The sample rate is preserved unchanged.

---

## License

MIT License. See `LICENSE` for details.
