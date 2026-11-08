# AGENTS.md

Guidance for AI coding agents working in this repository.

There are two users doing this lab, one (Alex) is primarily responsible for developing and using the software,
the other (Jiahui) uses the software when Alex isn't there. The two users can be distinguished easily because
Alex is running a Debian Linux Machine whilst Jiahui is not.

The rest of this file primarily concerns Alex but may have applications with Jiahui.

## What this is

Acquisition and analysis scripts for a Part II physics practical: a fibre
Mach-Zehnder interferometer at 1550 nm (handout: `../Fibre_interferometer_M2026.pdf`,
manufacturer data: `../Setup_A_manuals_v2.pdf`). The scripts drive a PicoScope,
log fringe contrast against delay-line position, and extract source spectra, the
FP laser's cavity length, and vibrometer velocities.

It is used **live in the lab**. Keep commands working and backwards compatible, and
never touch collected data (see below).

The running lab report is `../REPORT.md`. When results or findings change, update it:
results, the issues log (I-1, I-2, …), the to-do ticks and "Last updated". Only
write numbers that come from data files or script output, never expected values.

## Commands

```bash
uv sync --extra pico          # install (the pico extra is the PicoScope wrappers)
uv run pytest                 # all tests; no hardware needed
ruff check . && ruff format . # lint and format (line length 120)
ty check                      # type check; keep it clean
uv run fibre-synth demo       # regenerate fake lab data in demo/
```

Run everything from this folder. All three checks (pytest, ruff, ty) must pass
before you call a change done.

## Layout

| Path | Contents |
|---|---|
| `src/fibre_analysis/io.py` | trace CSV load/save (PicoScope format), `data_path`, `find_data`, `graph_path` |
| `src/fibre_analysis/contrast.py` | fringe contrast of one trace (`fibre-contrast`) |
| `src/fibre_analysis/scan.py` | interactive logger (`fibre-scan`, `--repeats`), `load_scan`, re-derive from traces (`fibre-rescan`), contrast vs time (`fibre-monitor`) |
| `src/fibre_analysis/count.py` | micrometer calibration by fringe counting (`fibre-count`); runs logged to `data/fringecount.csv` |
| `src/fibre_analysis/capture.py` | direct PicoScope capture via picosdk ctypes (`fibre-capture`) |
| `src/fibre_analysis/envelope.py` | SLD bandwidth fit and FP cavity length (`fibre-envelope`) |
| `src/fibre_analysis/spectrum.py` | spectrum reconstruction by FFT (`fibre-spectrum`) |
| `src/fibre_analysis/calibrate.py` | phase-modulator calibration (`fibre-calibrate`) |
| `src/fibre_analysis/vibrometer.py` | short-time FFT velocity analysis (`fibre-vibro`) |
| `src/fibre_analysis/power.py` | volts ↔ mW/dBm, noise figure (`fibre-power`) |
| `src/fibre_analysis/setup_a.py` | data-sheet constants for this setup (defaults and comparisons) |
| `src/fibre_analysis/synth.py` | fake data shaped like the real setup (`fibre-synth`) |
| `src/fibre_analysis/datasheets/` | manufacturer spectra traced off the data sheets (FP, SLD); read via `setup_a.datasheet_spectrum` |
| `tools/digitise_datasheets.py` | re-traces those curves from `../Setup_A_manuals_v2.pdf` (needs `pdftoppm`); writes check images to `output/` |
| `tests/fake_pico.py` | fake picosdk drivers used by `tests/test_capture.py` |
| `data/` | **collected lab data, git-ignored**: scan CSVs, `<scan>_traces/` raw traces |
| `output/` | graphs, git-ignored |
| `demo/` | fake data from `fibre-synth`, git-ignored |

## Rules

- **Never modify, move or delete anything in `data/`** unless the user explicitly asks.
  It is irreplaceable lab data. Scan CSVs are append-only; repeated positions are
  averaged by `load_scan`, not overwritten.
- **Paths:** a bare `--out name.csv` saves to `data/name.csv` (`io.data_path`);
  readers fall back to `data/` (`io.find_data`); graphs go to `output/`
  (`io.graph_path`). Use these helpers for any new command.
- **No hardware in tests.** Anything touching the scope goes through
  `capture._load_driver`, which the tests replace with fakes from `tests/fake_pico.py`.
  Code that can't be run against real hardware must be checked against the picosdk
  function signatures in `.venv/lib/python*/site-packages/picosdk/`.
- **Test against known answers.** New analysis needs a test that recovers a known
  value from `synth` data shaped like the real setup (mV–V signals, noise, dark
  offset, non-integer fringes per sweep), not just a test that it runs.
- **Cross-platform (Linux, Windows, macOS):** use `pathlib`; pass `encoding="utf-8"`
  to every `open`/`read_text`/`write_text` (tests included); start every command's
  `main` with `utf8_output()`. Check with
  `PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 LC_ALL=C uv run pytest`, which simulates
  Windows' non-UTF-8 defaults.
- `setup_a.py` spectral values come from curves traced off the data sheets by
  `tools/digitise_datasheets.py` (in `src/fibre_analysis/datasheets/`). Use them as defaults
  and for comparison, never as measured results.
- Be honest about uncertainty: if a method has a systematic error, say so in the
  docstring and the printed output rather than hiding it.

## Physics and hardware notes

- **Scope:** PicoScope 2205A, `ps2000` driver. 8,064 samples per channel with two
  channels; sample intervals only in steps of 2× (10 ns × 2^n). Captures default to
  whatever fits in memory.
- **Wiring:** detector on **channel A**, function generator (modulator drive) on
  **channel B**. Inputs DC-coupled; the contrast needs the DC level.
- **Contrast** default is Eq. 19 (percentile max/min) **per modulator period**,
  averaged, with the spread as the error (`contrast_per_period`). Periods are timed
  from the drive on channel B (recorded by default) or else the detector. Don't go
  back to whole-trace contrast with fixed-chunk errors: chunks shorter than a fringe
  gave meaningless error bars, and single glitches dragged the whole-trace min. The
  whole-trace value is kept in `contrast_whole`. A cosine fit was tried and rejected
  (real fringes aren't sinusoidal at the current settings). `rms` is only for
  low-contrast comparison.
- `fibre-rescan` re-derives a scan from its stored traces into a new
  `<scan>_period.csv`; it refuses to overwrite the original.
- **Fringe counting** (`fibre-count`): the user can't turn the micrometer continuously and only
  gives start/end readings, so counting must tolerate pauses and uneven speed (hysteresis
  crossings) but assumes one direction. A still (drift) run takes its fringe size from the
  last moving run in the log.
- **Path difference** = change in delay-line air gap. The micrometer reads mm,
  0.01 mm (10 µm) per division; `--unit mm` or `div`.
- **SLD** (1559 nm, Δλ = 67.7 nm, flat-topped): contrast peak ≈ 33 µm FWHM, so fringes
  only within ~±0.02 mm of zero path difference. Its bandwidth comes from a
  super-Gaussian spectrum fit, because a Gaussian fit reads ~25% low for this shape.
- **FP laser** (1547.7 nm, modes 1.158 nm apart): contrast revives every 2nL ≈ 2.07 mm.
  It shows strong mode-partition noise: the contrast between revivals fluctuates
  with nothing moving, so revival *positions* are robust but shapes are not. The
  revival finder fits a regular comb for this reason; don't revert it to plain peak
  picking.
- **Detector** saturates at about 10–11 V. Flattened fringe tops mean the source must
  be turned down.

## Working with the user

- `!` commands typed in the chat run from `E1/`, not this folder, and can't take
  typed input. Interactive commands (`fibre-scan`) must run in a terminal tab
  opened in this folder.
- The user is in the lab, often mid-measurement: keep answers short and practical,
  and say exactly what to type.
