"""Fake data that looks like the lab's, for testing the scripts before the lab.

    uv run fibre-synth demo/

Source spectra and powers follow the Setup A data sheets (see setup_a.py), and
signal levels follow a rough loss budget, so the detector sees mV, not V:

    demo/dark.csv                       detector with the source off (1.5 mV offset)
    demo/sld/*.csv, demo/sld_scan.csv   SLD: 1560 nm, 62 nm flat-topped, ~55 mV per arm
    demo/fp/*.csv,  demo/fp_scan.csv    FP laser: 1547.7 nm, modes 1.15 nm apart, ~5 mV per arm
    demo/calibration.csv                SLD fringes + triangle drive, 3.6 fringes/sweep
    demo/vibrometer.csv                 FP laser, speaker at 1.5 Hz, 40 µm amplitude, ~5 mV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import setup_a
from .io import save_trace

C_LIGHT = 299_792_458.0
LAMBDA0 = setup_a.FP_CENTRE_NM * 1e-9
SLD_LAMBDA = setup_a.SLD_CENTRE_NM * 1e-9

# Rough detector levels (V): source power x ~0.066 per arm (couplers, 3.3 dB modulator,
# delay line, connectors) x 0.95 A/W x 50 Ω.
SLD_ARMS = (0.060, 0.050)
FP_ARMS = (0.0060, 0.0050)
NOISE = 0.25e-3  # scope + detector noise, V rms
DARK = 1.5e-3  # detector/scope offset with the source off, V


def sld_spectrum(
    fwhm_nm: float = setup_a.SLD_FWHM_NM, centre_nm: float = setup_a.SLD_CENTRE_NM, order: float = 4.0, n: int = 801
):
    """(optical frequencies in Hz, weights). Super-Gaussian: order 2 = Gaussian, 4 = flat-topped like the data sheet."""
    lam0, w = centre_nm * 1e-9, fwhm_nm * 1e-9
    lam = lam0 + np.linspace(-1.5, 1.5, n) * w
    s = np.exp(-np.log(2) * np.abs(2 * (lam - lam0) / w) ** order)
    return C_LIGHT / lam, s / s.sum()


def fp_spectrum(
    round_trip_mm: float = setup_a.fp_round_trip_um() * 1e-3,
    envelope_nm: float = setup_a.FP_ENVELOPE_FWHM_NM,
    centre_nm: float = setup_a.FP_CENTRE_NM,
    linewidth_nm: float = 0.02,
    per_mode: int = 15,
):
    """Comb of modes spaced c / (2nL) under an uneven envelope (longer tail to short
    wavelengths, as on the data sheet)."""
    lam0 = centre_nm * 1e-9
    nu0 = C_LIGHT / lam0
    dnu = C_LIGHT / (round_trip_mm * 1e-3)
    env_nu = C_LIGHT * envelope_nm * 1e-9 / lam0**2
    line_nu = C_LIGHT * linewidth_nm * 1e-9 / lam0**2
    m = np.arange(-int(3 * env_nu / dnu) - 1, int(3 * env_nu / dnu) + 2)
    nus, ws = [], []
    for k in m:
        sub = nu0 + k * dnu + np.linspace(-1.5, 1.5, per_mode) * line_nu
        width = env_nu * (1.4 if k > 0 else 0.8)  # higher frequency = shorter wavelength: wider tail
        w = np.exp(-4 * np.log(2) * (k * dnu / width) ** 2)
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
    i1: float = FP_ARMS[0],
    i2: float = FP_ARMS[1],
    fringes_per_sweep: float = 4.0,
    mod_freq: float = 300.0,
    fs: float = 200e3,
    duration: float = 10e-3,
    noise: float = NOISE,
    dark: float = DARK,
    lambda0: float = LAMBDA0,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng()
    t = np.arange(int(fs * duration)) / fs
    drive = triangle(t, mod_freq)
    opd = opd0 + drive * fringes_per_sweep * lambda0 / 2
    v = i1 + i2 + 2 * np.sqrt(i1 * i2) * coherence(opd, nu, s) + dark
    v += rng.normal(0, noise, t.size)
    return t, v, 5 * drive


def vibrometer_trace(
    amplitude: float = 40e-6,
    drive_freq: float = 1.5,
    fs: float = 20e3,
    duration: float = 2.0,
    mean: float = 5e-3,
    contrast_: float = 0.5,
    lag: float = 0.2,
    noise: float = NOISE,
    dark: float = DARK,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng()
    t = np.arange(int(fs * duration)) / fs
    drive = np.sin(2 * np.pi * drive_freq * t)
    x = amplitude * np.sin(2 * np.pi * drive_freq * t - lag)  # speaker lags the drive slightly
    drift = 0.3 * np.sin(2 * np.pi * 0.2 * t)  # slow phase drift of the fibres
    v = mean * (1 + contrast_ * np.cos(4 * np.pi * x / LAMBDA0 + 0.7 + drift)) + dark
    v += rng.normal(0, noise, t.size)
    return t, v, 2 * drive


def _scan(folder: Path, name: str, positions_div: np.ndarray, nu, s, arms, lambda0, zero_div: float, rng) -> None:
    from .scan import _write_rows, make_row

    rows = []
    (folder / name).mkdir(parents=True, exist_ok=True)
    for x in positions_div:
        opd = (x - zero_div) * 10e-6
        # A non-integer, slightly varying fringe count per sweep, as in the lab.
        n = rng.uniform(3.3, 3.8)
        t, v, drive = fringe_trace(opd, nu, s, *arms, fringes_per_sweep=n, lambda0=lambda0, rng=rng)
        p = folder / name / f"{name}_{x:+08.2f}div.csv"
        save_trace(p, t, {"Channel A": v, "Channel B": drive})
        rows.append(make_row(x, "div", v, p, dark=DARK))
    _write_rows(folder / f"{name}_scan.csv", rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="demo")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    t = np.arange(20000) / 200e3
    save_trace(out / "dark.csv", t, {"Channel A": DARK + rng.normal(0, NOISE, t.size)})

    # SLD: zero path difference at 3 div; its envelope is only ~3 div wide, so step 0.25 div.
    nu, s = sld_spectrum()
    _scan(out, "sld", np.arange(-5, 5.01, 0.25) + 3.0, nu, s, SLD_ARMS, SLD_LAMBDA, zero_div=3.0, rng=rng)
    # FP: revivals every ~208 div; 0-600 div catches three.
    nu, s = fp_spectrum()
    _scan(out, "fp", np.arange(0, 600.1, 5.0), nu, s, FP_ARMS, LAMBDA0, zero_div=12.0, rng=rng)

    t, v, drive = fringe_trace(
        0.0, *sld_spectrum(), *SLD_ARMS, fringes_per_sweep=3.6, duration=20e-3, lambda0=SLD_LAMBDA, rng=rng
    )
    save_trace(out / "calibration.csv", t, {"Channel A": v, "Channel B": drive})
    t, v, drive = vibrometer_trace(rng=rng)
    save_trace(out / "vibrometer.csv", t, {"Channel A": v, "Channel B": drive})
    print(f"Fake data written to {out}/")


if __name__ == "__main__":
    main()
