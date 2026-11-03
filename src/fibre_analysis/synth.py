"""Fake data that looks like the lab's, for testing the scripts before the lab.

    uv run fibre-synth demo/

writes PicoScope-style CSVs plus ready-made scan files:
    demo/sld/*.csv, demo/sld_scan.csv   SLD, Δλ = 50 nm, delay scan ±100 µm
    demo/fp/*.csv,  demo/fp_scan.csv    FP laser, 2nL = 2.4 mm, 3 nm envelope
    demo/calibration.csv                fringes + triangle drive, 3.6 fringes/sweep
    demo/vibrometer.csv                 speaker at 1.5 Hz, 40 µm amplitude
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import save_trace

C_LIGHT = 299_792_458.0
LAMBDA0 = 1550e-9


def sld_spectrum(fwhm_nm: float = 50.0, n: int = 801):
    """(optical frequencies in Hz, weights) for a Gaussian spectrum."""
    lam = LAMBDA0 + np.linspace(-3, 3, n) * fwhm_nm * 1e-9
    s = np.exp(-4 * np.log(2) * ((lam - LAMBDA0) / (fwhm_nm * 1e-9)) ** 2)
    return C_LIGHT / lam, s / s.sum()


def fp_spectrum(round_trip_mm: float = 2.4, envelope_nm: float = 3.0, linewidth_nm: float = 0.02, per_mode: int = 15):
    """Comb of modes spaced c / (2nL) under a Gaussian envelope."""
    nu0 = C_LIGHT / LAMBDA0
    dnu = C_LIGHT / (round_trip_mm * 1e-3)
    env_nu = C_LIGHT * envelope_nm * 1e-9 / LAMBDA0**2
    line_nu = C_LIGHT * linewidth_nm * 1e-9 / LAMBDA0**2
    m = np.arange(-int(2 * env_nu / dnu) - 1, int(2 * env_nu / dnu) + 2)
    nus, ws = [], []
    for k in m:
        sub = nu0 + k * dnu + np.linspace(-1.5, 1.5, per_mode) * line_nu
        w = np.exp(-4 * np.log(2) * ((nu0 + k * dnu - nu0) / env_nu) ** 2)
        w *= np.exp(-4 * np.log(2) * ((sub - nu0 - k * dnu) / line_nu) ** 2)
        nus.append(sub)
        ws.append(w)
    nu, s = np.concatenate(nus), np.concatenate(ws)
    return nu, s / s.sum()


def coherence(opd: np.ndarray, nu: np.ndarray, s: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Re[Σ S(ν) exp(-j2πν δ/c)] for each path difference δ (m)."""
    out = np.empty(opd.size)
    for i in range(0, opd.size, chunk):
        d = opd[i : i + chunk, None]
        out[i : i + chunk] = np.cos(2 * np.pi * nu[None, :] * d / C_LIGHT) @ s
    return out


def triangle(t: np.ndarray, f: float) -> np.ndarray:
    """Triangle wave between -1 and 1."""
    return 2 * np.abs(2 * ((t * f) % 1) - 1) - 1


def fringe_trace(
    opd0: float,
    nu: np.ndarray,
    s: np.ndarray,
    i1: float = 1.0,
    i2: float = 0.8,
    fringes_per_sweep: float = 4.0,
    mod_freq: float = 300.0,
    fs: float = 200e3,
    duration: float = 10e-3,
    noise: float = 0.01,
    dark: float = 0.0,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng()
    t = np.arange(int(fs * duration)) / fs
    drive = triangle(t, mod_freq)
    opd = opd0 + drive * fringes_per_sweep * LAMBDA0 / 2
    v = i1 + i2 + 2 * np.sqrt(i1 * i2) * coherence(opd, nu, s) + dark
    v += rng.normal(0, noise, t.size)
    return t, v, 5 * drive


def vibrometer_trace(
    amplitude: float = 40e-6,
    drive_freq: float = 1.5,
    fs: float = 20e3,
    duration: float = 2.0,
    contrast_: float = 0.7,
    lag: float = 0.2,
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng()
    t = np.arange(int(fs * duration)) / fs
    drive = np.sin(2 * np.pi * drive_freq * t)
    x = amplitude * np.sin(2 * np.pi * drive_freq * t - lag)  # speaker lags the drive slightly
    drift = 0.3 * np.sin(2 * np.pi * 0.2 * t)  # slow phase drift of the fibres
    v = 1 + contrast_ * np.cos(4 * np.pi * x / LAMBDA0 + 0.7 + drift) + rng.normal(0, noise, t.size)
    return t, v, 2 * drive


def _scan(folder: Path, name: str, positions_div: np.ndarray, nu, s, zero_div: float, rng) -> None:
    from .scan import _write_rows, make_row

    rows = []
    (folder / name).mkdir(parents=True, exist_ok=True)
    for x in positions_div:
        opd = (x - zero_div) * 10e-6
        # A non-integer, slightly varying fringe count per sweep, as in the lab.
        t, v, drive = fringe_trace(opd, nu, s, fringes_per_sweep=rng.uniform(3.3, 3.8), rng=rng)
        p = folder / name / f"{name}_{x:+08.2f}div.csv"
        save_trace(p, t, {"Channel A": v, "Channel B": drive})
        rows.append(make_row(x, "div", v, p))
    _write_rows(folder / f"{name}_scan.csv", rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="demo")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    nu, s = sld_spectrum()
    _scan(out, "sld", np.arange(-10, 10.01, 0.5) + 3.0, nu, s, zero_div=3.0, rng=rng)
    nu, s = fp_spectrum()
    _scan(out, "fp", np.arange(0, 600.1, 5.0), nu, s, zero_div=12.0, rng=rng)

    t, v, drive = fringe_trace(0.0, *sld_spectrum(), fringes_per_sweep=3.6, duration=20e-3, rng=rng)
    save_trace(out / "calibration.csv", t, {"Channel A": v, "Channel B": drive})
    t, v, drive = vibrometer_trace(rng=rng)
    save_trace(out / "vibrometer.csv", t, {"Channel A": v, "Channel B": drive})
    print(f"Fake data written to {out}/")


if __name__ == "__main__":
    main()
