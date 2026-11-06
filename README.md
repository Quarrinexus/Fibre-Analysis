# fibre-analysis

Analysis scripts for the Part II fibre interferometer practical. Every command
reads PicoScope CSV exports (time in s/ms/µs, volts in V/mV; "∞" over-range
samples are flagged) or plain `t,v` CSVs.

```bash
uv sync                    # install
uv run fibre-synth demo    # make fake lab data to try everything on
uv run pytest              # check the scripts recover known answers
ruff check . && ruff format . && ty check
```

## Commands

| Command | Handout section | What it does |
|---|---|---|
| `fibre-contrast trace.csv …` | 3–5 | Fringe contrast of one or more traces (`--dark` to subtract the detector offset) |
| `fibre-scan --watch DIR --out scan.csv` | 3–5 | Live logger: type the delay (or gain) reading, save the trace in PicoScope, it logs the contrast and plots it |
| `fibre-envelope scan.csv --source sld` | 3 | Spectrum-shape fit to the contrast envelope → zero path difference, SLD bandwidth Δλ and shape, coherence length |
| `fibre-envelope scan.csv --source fp` | 3 | Contrast revivals → cavity round trip 2nL, mode spacing, mode envelope width; FFT spectrum |
| `fibre-calibrate trace.csv` | 3 | Fringes per modulator sweep → phase swing (π rad), µm of path difference per volt |
| `fibre-power 5 --i1 2 --i2 0.5` | 4 | Detector V → mW/dBm, best possible contrast for given arm intensities, noise figure |
| `fibre-vibro trace.csv` | 6 | Short-time FFT → speaker velocity v(t), displacement amplitude (fit, integration and fringe counting) |
| `fibre-spectrum scan.csv --source sld` | 3 | Reconstruct intensity vs wavelength from a delay scan (Appendix A), with the data sheet overlaid |
| `fibre-rescan scan.csv` | 3–5 | Re-derive a scan's contrast from its stored traces into a new `<scan>_period.csv` |
| `fibre-capture out.csv` | any | Capture a trace straight from the PicoScope (see below) |
| `fibre-synth demo` | – | Fake lab data for trying everything out |

Every command has `--help`.

Collected data goes in `data/`: a bare filename such as `--out fp_scan.csv` is saved as
`data/fp_scan.csv`, and commands look in `data/` for files you name, so `fibre-envelope fp_scan.csv`
works. Run commands from this folder. Graphs are saved in `output/` (e.g. `output/fp_scan_spectrum.png`).
`fibre-scan` updates `output/<scan>.png` after every point.

## In the lab

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
PicoScope app. This needs:

1. Pico's drivers (**PicoSDK**, from picotech.com/downloads) installed on the lab PC.
2. The Python wrappers: `uv sync --extra pico`.
3. The PicoScope app **closed**, because only one program can use the scope at a time.

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

## First thing on lab day

Save one PicoScope trace and run `fibre-contrast` and `fibre-calibrate` on it.
Check that the time step and voltages look right before trusting anything else.
The loader understands a separate `(ms)` units row and `Time (ms)` style headers,
and it warns about units it doesn't recognise.

## What the Setup A data sheets say

The data sheet values (`Setup_A_manuals_v2.pdf`) are in `setup_a.py`.
`fibre-envelope` uses them as wavelength defaults and prints them next to your
results. They were read off scanned plots, so treat them as ±10%.

| | Data sheet | What to expect |
|---|---|---|
| SLD | 1560 nm, Δλ ≈ 62 nm, flat-topped | contrast envelope ≈ 45 µm FWHM, with small side lobes |
| FP laser | 1547.7 nm, modes 1.15 nm apart, ~4.5 nm envelope, 1.83 mW | revivals every ≈ 2.08 mm (nL ≈ 1.04 mm) |
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
