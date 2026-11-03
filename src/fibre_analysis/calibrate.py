"""Phase-modulator calibration: how much path difference one sweep gives.

Record the detector on one channel and the function-generator output on the
other. Plotted against the drive voltage, each sweep traces the same cosine;
fitting it gives the number of fringes per sweep (fractions included), and
that times λ is the path-difference swing. That converts the scope's time axis into
path difference and checks the handout's "~4π for ~10 V" claim.

    uv run fibre-calibrate trace.csv --signal A --drive B
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .io import load_trace


@dataclass
class Calibration:
    fringes_per_sweep: float
    fringes_err: float
    opd_swing_um: float
    drive_pp_v: float
    n_sweeps: int

    @property
    def phase_swing_rad(self) -> float:
        return 2 * np.pi * self.fringes_per_sweep

    @property
    def um_per_volt(self) -> float:
        return self.opd_swing_um / self.drive_pp_v

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Sweeps analysed      {self.n_sweeps}",
                f"Fringes per sweep    {self.fringes_per_sweep:.2f} ± {self.fringes_err:.2f}",
                f"Phase swing          {self.phase_swing_rad / np.pi:.2f} π rad",
                f"Path-difference swing {self.opd_swing_um:.2f} µm",
                f"Drive peak-to-peak   {self.drive_pp_v:.2f} V  ->  {self.um_per_volt:.3f} µm/V",
            ]
        )


def sweep_edges(drive: np.ndarray, fs: float) -> np.ndarray:
    """Indices of the turning points of a triangle (or sawtooth) drive."""
    d = drive - np.mean(drive)
    # Rough period from the spectrum, then look for extrema at least ~1/3 period apart.
    spec = np.abs(np.fft.rfft(d))
    f = np.fft.rfftfreq(d.size, 1 / fs)
    f0 = f[1 + np.argmax(spec[1:])]
    dist = max(int(fs / f0 / 3), 1)
    prom = 0.5 * np.ptp(d)
    hi, _ = find_peaks(d, distance=dist, prominence=prom)
    lo, _ = find_peaks(-d, distance=dist, prominence=prom)
    return np.sort(np.concatenate([hi, lo]))


def _cos(u, a, b, n, phi):
    return a + b * np.cos(2 * np.pi * n * u + phi)


def _fit_profile(u: np.ndarray, v: np.ndarray, n0: float | None = None) -> float:
    """Fringes across u in [0, 1]: fit v = a + b cos(2π n u + φ)."""
    order = np.argsort(u)
    u, v = u[order], v[order]
    if n0 is None:
        grid = np.linspace(0, 1, 512)
        prof = np.interp(grid, u, v) - np.mean(v)
        spec = np.abs(np.fft.rfft(prof * np.hanning(grid.size), n=16 * grid.size))
        k = np.fft.rfftfreq(16 * grid.size, d=grid[1] - grid[0])
        n0 = float(k[1 + np.argmax(spec[1:])])
    best = None
    for phi0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        try:
            popt, _ = curve_fit(_cos, u, v, p0=[np.mean(v), np.std(v) * np.sqrt(2), n0, phi0], maxfev=5000)
        except RuntimeError:
            continue
        sse = np.sum((_cos(u, *popt) - v) ** 2)
        if best is None or sse < best[0]:
            best = (sse, abs(popt[2]))
    if best is None:
        raise RuntimeError("Fringe fit failed")
    return best[1]


def calibrate(signal: np.ndarray, drive: np.ndarray, fs: float, wavelength_nm: float = 1550.0) -> Calibration:
    """Plot detector signal against drive voltage: every sweep traces the same
    cosine, and the number of its periods across the drive range is the number
    of fringes per sweep. Fitted per sweep to get the scatter."""
    ok = np.isfinite(signal) & np.isfinite(drive)
    signal, drive = signal[ok], drive[ok]
    smooth = uniform_filter1d(drive, size=max(drive.size // 2000, 1))  # take the edge off scope noise
    lo, hi = float(np.min(smooth)), float(np.max(smooth))
    u = np.clip((drive - lo) / (hi - lo), 0, 1)
    n_all = _fit_profile(u, signal)

    edges = sweep_edges(drive, fs)
    counts = [_fit_profile(u[a:b], signal[a:b], n_all) for a, b in itertools.pairwise(edges) if b - a > 50]
    counts = np.array(counts) if counts else np.array([n_all])
    n = float(np.mean(counts))
    err = float(np.std(counts, ddof=1) / np.sqrt(len(counts))) if len(counts) > 1 else np.nan
    return Calibration(n, err, n * wavelength_nm * 1e-3, float(hi - lo), len(counts))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--signal", default="A", help="detector channel (default A)")
    p.add_argument("--drive", default="B", help="function generator channel (default B)")
    p.add_argument("--wavelength", type=float, default=1550.0, help="nm")
    a = p.parse_args(argv)
    tr = load_trace(a.trace)
    print(calibrate(tr.channel(a.signal), tr.channel(a.drive), tr.fs, a.wavelength))


if __name__ == "__main__":
    main()
