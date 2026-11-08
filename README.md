# Fibre Analysis

Analysis scripts for the Part II fibre interferometer practical. Every command
reads PicoScope CSV exports (time in s/ms/µs, volts in V/mV; "∞" over-range
samples are flagged) or plain `t,v` CSVs.

## Installation

You need [uv](https://docs.astral.sh/uv/). It also installs the right Python (3.13+)
for you, so nothing else is needed for analysis. Driving the scope directly needs
Pico's drivers as well (step 4).

**1. Install uv** (no admin rights needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Get the code:**

```bash
git clone https://github.com/Quarrinexus/fibre_analysis.git
cd fibre_analysis
```

**3. Install the Python packages.** For analysis only:

```bash
uv sync
```

To also capture from the PicoScope (`--pico`, `fibre-capture`):

```bash
uv sync --extra pico
```

Once you've added `--extra pico`, keep using it: a later plain `uv sync` removes the
Pico package again. `uv run …` never removes anything.

**4. Install Pico's drivers** (only for `--pico` / `fibre-capture`). The Python
package is only a wrapper; the drivers come from Pico Technology. The 2205A in this
lab uses the `ps2000` driver.

*Linux (Debian/Ubuntu):* add Pico's package repository, then install the driver:

```bash
wget -O- https://labs.picotech.com/Release.gpg.key | gpg --dearmor | sudo tee /usr/share/keyrings/picotech-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/picotech-archive-keyring.gpg] https://labs.picotech.com/picoscope7/debian/ picoscope main" | sudo tee /etc/apt/sources.list.d/picoscope7.list
sudo apt update
sudo apt install libps2000
```

The package puts the library in `/opt/picoscope/lib`, registers that path, and adds a
udev rule so you can use the scope without root. Unplug and replug the scope
afterwards. Other models need their own package (`libps2000a`, `libps3000a`,
`libps4000a`, `libps5000a`), or install `picoscope` (the app) to get them all.

*Windows / macOS:* install the 64-bit **PicoSDK** from
[picotech.com/downloads](https://www.picotech.com/downloads) (choose 2205A driver). If a capture then says the driver isn't found, add
the SDK's `lib` folder to `PATH` (Windows) or `DYLD_LIBRARY_PATH` (macOS).

**5. Check it works.**

```bash
uv run pytest                 # all tests pass, no scope needed
uv run fibre-synth demo       # fake lab data to try every command on
uv run fibre-capture test.csv # with the scope plugged in and the PicoScope app closed
```

`fibre-capture` should print the scope family (`ps2000`) and the voltages it read.
If it says *No PicoScope found*:

- `driver not installed` means step 4 isn't done.
- `no unit found` or `PICO_NOT_FOUND` means the scope is unplugged, or another
  program (usually the PicoScope app) is using it.

**Platforms.** Written to run on Linux, Windows and macOS: paths go through `pathlib`,
files are read and written as UTF-8, output is forced to UTF-8 (so µ, λ, Δ print even
when redirected on Windows), and European-locale PicoScope exports (`;` separators,
`,` decimals) are read correctly. Tested on Linux, including a simulated non-UTF-8
system; not yet run on real Windows or macOS machines. Scope capture there depends on
PicoSDK for that OS (on Apple-silicon Macs, check Pico supports your model).

## Commands

| Command | Handout section | What it does |
|---|---|---|
| `fibre-contrast trace.csv …` | 3–5 | Fringe contrast of one or more traces (`--dark` to subtract the detector offset) |
| `fibre-scan --pico --out scan.csv [--repeats N]` | 3–5 | Live logger driving the scope: type the delay (or gain) reading, it captures (N times with `--repeats`), logs and plots the contrast; `u` undoes a whole point |
| `fibre-scan --watch DIR --out scan.csv` | 3–5 | Live logger: type the delay (or gain) reading, save the trace in PicoScope, it logs the contrast and plots it |
| `fibre-envelope scan.csv --source sld` | 3 | Spectrum-shape fit to the contrast envelope → zero path difference, SLD bandwidth Δλ and shape, coherence length |
| `fibre-envelope scan.csv --source fp` | 3 | Contrast revivals → cavity round trip 2nL, mode spacing, mode envelope width; FFT spectrum |
| `fibre-calibrate trace.csv` | 3 | Fringes per modulator sweep → phase swing (π rad), µm of path difference per volt |
| `fibre-power 5 --i1 2 --i2 0.5` | 4 | Detector V → mW/dBm, best possible contrast for given arm intensities, noise figure |
| `fibre-vibro trace.csv` | 6 | Short-time FFT → speaker velocity v(t), displacement amplitude (fit, integration and fringe counting) |
| `fibre-spectrum scan.csv --source sld` | 3 | Reconstruct intensity vs wavelength from a delay scan (Appendix A), with the data sheet overlaid |
| `fibre-rescan scan.csv` | 3–5 | Re-derive a scan's contrast from its stored traces into a new `<scan>_period.csv` |
| `fibre-count --pico` | 3 | Micrometer calibration: counts fringes while you turn (one direction, pauses fine), from start/end readings → single vs double pass, scale factor; logs every run |
| `fibre-monitor --out monitor.csv` | 3 | Contrast against time at a fixed position (every 5 s for 5 min by default) |
| `fibre-capture out.csv` | any | Capture a trace straight from the PicoScope (see below) |
| `fibre-synth demo` | – | Fake lab data for trying everything out |

Every command has `--help`.

Collected data goes in `data/`: a bare filename such as `--out fp_scan.csv` is saved as
`data/fp_scan.csv`, and commands look in `data/` for files you name, so `fibre-envelope fp_scan.csv`
works. Run commands from this folder. Graphs are saved in `output/` (e.g. `output/fp_scan_spectrum.png`).
`fibre-scan` updates `output/<scan>.png` after every point.

## Using in the lab

**Contrast vs delay (SLD and FP):** set PicoScope to save CSVs into a folder, then

```bash
uv run fibre-scan --watch ~/pico --out sld_scan.csv --unit div --dark 0.002
```

Set the delay, type the screw reading, press Enter, save the trace. Use `u` to undo
and `q` to quit; re-running with the same `--out` carries on. Then

```bash
uv run fibre-envelope sld_scan.csv --source sld
```

- **SLD:** the envelope is only ~45 µm wide, about 4–5 screw divisions, so step in
  **quarter** divisions near zero path difference. Scan ±5 divisions to include the
  side lobes: they set the spectrum's shape in the fit.
- **FP:** scan several mm in coarser steps to catch two or more revivals.
- **Equal steps:** the FFT spectrum needs equal steps, otherwise it gets interpolated.

**Contrast vs gain (sections 4–5):** use the same logger with
`--quantity "pump current" --unit mA`.

**Vibrometer:** record the detector on channel A and the function generator on
channel B for a few drive periods. Choose the sample rate from the expected fringe
frequency f = 2v/λ (about 1 kHz for 40 µm at 2 Hz), so 10–20 kS/s is plenty.

```bash
uv run fibre-vibro trace.csv --signal A --drive B
```

The short-time FFT window is chosen automatically from the fringe rate, and the
result prints which one was used. Override it with `--window`: longer resolves
slower motion but blurs the velocity changes in time.

## Direct PicoScope capture (optional)

The scripts can drive the scope themselves, so you don't need to save from the
PicoScope app. This needs Pico's drivers and `uv sync --extra pico` (see
[Installation](#installation), steps 3–4), and the PicoScope app **closed**, because
only one program can use the scope at a time.

Then:

```bash
uv run fibre-scan --pico --out sld_scan.csv --unit div            # Enter = capture + log
uv run fibre-scan --pico --record B --out fp_scan.csv --unit div  # also save the drive
uv run fibre-capture vib.csv --channels A B --dt 50e-6 --samples 100000   # vibrometer, 5 s
```

- **Models:** the 2000/2000A/3000A/4000A/5000A series are detected automatically. Force one
  with `--family ps2000a` etc. if needed.
- **Inputs:** channels are DC-coupled.
- **Range:** chosen automatically. Clipping moves up a range, a small signal moves down,
  and the next point starts from the last range, so it's normally one capture per
  point. Fix it with `--range 5`.
- **Traces:** each point's trace is saved in `<scan>_traces/`.
- **Memory limits:** the scope's memory caps how many samples one capture can hold. An
  older 2204A/2205A holds only 8–16 k samples, too few for a 5 s vibrometer trace at
  20 kS/s. The script says so; then use a larger `--dt` or record that one in the
  PicoScope app.
- **Untested on hardware:** this is checked only against fake drivers, written to
  Pico's documented function signatures. Try one `fibre-capture` first and compare it
  with what the PicoScope app shows. The folder-watching mode (`--watch`) still works
  as a fallback.

## What the Setup A data sheets say

The data sheet values (`Setup_A_manuals_v2.pdf`) are in `setup_a.py`.
`fibre-envelope` uses them as wavelength defaults and prints them next to your
results. The spectra were traced off the scanned plots by
`tools/digitise_datasheets.py` (about one pixel accurate); the traced curves are in
`src/fibre_analysis/datasheets/` and are what `fibre-spectrum` overlays.

| | Data sheet | What to expect |
|---|---|---|
| SLD | Δλ = 67.7 nm (−3 dB), centre 1559 nm, flat-topped | contrast peak ≈ 33 µm FWHM; C > 0.05 only within ±35 µm |
| FP laser | 1547.7 nm, modes 1.158 nm apart, ~5.2 nm envelope, 1.83 mW | revivals every ≈ 2.07 mm (nL ≈ 1.03 mm) |
| Amplifiers | BOA1004P **semiconductor** boosters, 88.6 nm gain band, 15.2 dBm saturation | not erbium-fibre amplifiers as the handout describes |

- **Signals are small.** A rough loss budget gives ~5 mV per arm for the FP laser and
  ~50 mV for the SLD.
  - **Dark level:** always measure it and pass `--dark`. On fake data, forgetting a
    1.5 mV offset costs >10% of the contrast.
  - **Noise floor:** at these levels noise holds the contrast at about 0.02–0.03
    where it should be zero.
- **FP drift:** the FP spectrum shifts with current, temperature and back-reflections
  (data sheet note). Keep the current fixed during a scan.

## Assumptions worth checking

- Contrast is Eq. 19 applied to each whole modulator period and averaged; the error
  bar is the spread between periods (and between repeats at the same position).
  Periods are timed from the drive on channel B, which `fibre-scan --pico` records
  by default, or else from the detector signal.
- One screw division = 10 µm of path difference (single pass through the air gap).
  If not, pass `--um-per-unit` to `fibre-envelope`.
- **SLD bandwidth depends on the spectrum's shape.** The envelope width maps to Δλ
  as 0.88 λ²/Δλ for a Gaussian but ≈ 1.2 λ²/Δλ for the flat-topped SLD. So
  `fibre-envelope` fits a super-Gaussian spectrum with free width *and* shape
  (order 2 = Gaussian). On fake data shaped like the data sheet it recovers Δλ to
  ~4%, where a Gaussian-only fit reads ~25% low.
- **Modulator sweep:** contrast is measured over the modulator's ~5 µm sweep, which
  widens the envelope a little. `--sweep-um` (the swing from `fibre-calibrate`) models
  this. It moves Δλ by a few %, about the same as the fit's own systematic error, so
  quote roughly ±5%.
- The contrast only gives |FT of spectrum| (no phase). So the FFT spectrum is only
  right for roughly symmetric spectra.
- The vibrometer direction (sign of v) is taken from d(drive)/dt. This is valid well
  below the speaker resonance.
