"""Source spectra from contrast-vs-delay scans (section 3, Appendix A).

Contrast vs path difference is |FT of the spectrum| (Eq. 20), so:

SLD (broad, smooth spectrum)
    One envelope around zero path difference. For a Gaussian spectrum of FWHM
    Δλ the envelope is Gaussian with FWHM  w = (4 ln2 / π) λ² / Δλ ≈ 0.88 λ²/Δλ,
    so Δλ = 0.88 λ² / w.

Fabry-Perot laser (comb of cavity modes)
    The contrast comes back every time the path difference is a multiple of the
    cavity round trip 2nL. Peak spacing D gives:
      optical cavity length  nL = D / 2
      mode spacing           Δν = c / D,  Δλ = λ² / D
    The width of each revival gives the width of the whole mode envelope (same
    formula as the SLD), and the decay of successive revivals the linewidth of
    the individual modes.

The FFT of the scan gives the spectrum shape directly (only correct if the
spectrum is roughly symmetric, because the scan only measures |FT|; the delay
steps must also be equal - the scan is resampled onto an even grid first).

Path difference = the change in air gap of the delay line. By default a screw
division is 10 µm; change it with --um-per-unit if your delay line differs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .scan import load_scan

C_LIGHT = 299_792_458.0
GAUSS_K = 4 * np.log(2) / np.pi  # FWHM(OPD) * FWHM(λ) = GAUSS_K * λ²


def gaussian(x, a, x0, w, b):
    """Gaussian with FWHM w, height a, baseline b."""
    return a * np.exp(-4 * np.log(2) * (x - x0) ** 2 / w**2) + b


@dataclass
class GaussFit:
    a: float
    x0: float
    w: float
    b: float
    a_err: float
    x0_err: float
    w_err: float


def fit_gaussian(x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None) -> GaussFit:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    s = None
    if sigma is not None:
        s = sigma[ok]
        if not np.all(np.isfinite(s) & (s > 0)):
            s = None
    i = int(np.argmax(y))
    half = y[i] / 2
    above = x[y > half + np.min(y) / 2]
    w0 = max(float(np.ptp(above)) if above.size > 1 else np.ptp(x) / 5, np.min(np.diff(x)))
    p0 = [y[i] - np.min(y), x[i], w0, np.min(y)]
    popt, pcov = curve_fit(gaussian, x, y, p0=p0, sigma=s, absolute_sigma=s is not None, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return GaussFit(popt[0], popt[1], abs(popt[2]), popt[3], perr[0], perr[1], perr[2])


def bandwidth_from_fwhm(w_opd: float, wavelength: float) -> float:
    """Spectral FWHM (same length unit as wavelength) from envelope FWHM in path difference."""
    return GAUSS_K * wavelength**2 / w_opd


@dataclass
class SLDResult:
    fit: GaussFit
    bandwidth_nm: float
    bandwidth_err_nm: float
    coherence_length_um: float

    def __str__(self) -> str:
        f = self.fit
        return "\n".join(
            [
                f"Zero path difference at {f.x0:.1f} ± {f.x0_err:.1f} µm",
                f"Envelope FWHM          {f.w:.1f} ± {f.w_err:.1f} µm",
                f"Peak contrast          {f.a + f.b:.3f}",
                f"Spectral FWHM Δλ       {self.bandwidth_nm:.1f} ± {self.bandwidth_err_nm:.1f} nm (Gaussian model)",
                f"Coherence length λ²/Δλ {self.coherence_length_um:.1f} µm",
            ]
        )


def analyse_sld(opd_um: np.ndarray, c: np.ndarray, c_err=None, wavelength_nm: float = 1550.0) -> SLDResult:
    fit = fit_gaussian(opd_um, c, c_err)
    lam_um = wavelength_nm * 1e-3
    bw_um = bandwidth_from_fwhm(fit.w, lam_um)
    return SLDResult(fit, bw_um * 1e3, bw_um * 1e3 * fit.w_err / fit.w, lam_um**2 / bw_um)


@dataclass
class FPResult:
    peaks_um: np.ndarray
    spacing_um: float
    spacing_err_um: float
    optical_length_um: float
    mode_spacing_nm: float
    mode_spacing_ghz: float
    envelope_fwhm_nm: float | None

    def __str__(self) -> str:
        lines = [
            f"Revival peaks at        {np.array2string(self.peaks_um, precision=0)} µm",
            f"Revival spacing 2nL     {self.spacing_um:.0f} ± {self.spacing_err_um:.0f} µm",
            f"Optical cavity length nL {self.optical_length_um:.0f} µm"
            f"  (L ≈ {self.optical_length_um / 3.5:.0f} µm if n ≈ 3.5 for InGaAsP)",
            f"Mode spacing            {self.mode_spacing_nm:.3f} nm = {self.mode_spacing_ghz:.1f} GHz",
        ]
        if self.envelope_fwhm_nm is not None:
            lines.append(f"Mode envelope FWHM      {self.envelope_fwhm_nm:.2f} nm (from width of central revival)")
        return "\n".join(lines)


def analyse_fp(
    opd_um: np.ndarray, c: np.ndarray, wavelength_nm: float = 1550.0, min_spacing_um: float = 200.0
) -> FPResult:
    step = float(np.median(np.diff(opd_um)))
    distance = max(int(min_spacing_um / step), 1)
    # Pad with the minimum so a revival at the very edge of the scan still counts as a peak.
    padded = np.concatenate([[np.nanmin(c)], c, [np.nanmin(c)]])
    peaks, _ = find_peaks(padded, distance=distance, prominence=0.1 * np.nanmax(c))
    peaks = peaks - 1
    if peaks.size < 2:
        raise ValueError("Found fewer than two contrast revivals - scan further or lower min_spacing_um")
    xp = opd_um[peaks]
    # Refine each peak with a local parabola, then fit position vs order.
    refined = []
    for i in peaks:
        lo, hi = max(i - 1, 0), min(i + 2, len(c))
        if hi - lo == 3:
            y0, y1, y2 = c[lo:hi]
            den = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            refined.append(opd_um[i] + shift * step)
        else:
            refined.append(opd_um[i])
    xp = np.array(refined)
    diffs = np.diff(xp)
    order = np.concatenate([[0], np.cumsum(np.round(diffs / np.median(diffs)))])
    if xp.size >= 3:
        (d, _), cov = np.polyfit(order, xp, 1, cov=True)
        d_err = float(np.sqrt(cov[0, 0]))
    else:
        d, d_err = float(diffs[0]), step / np.sqrt(2)
    lam_um = wavelength_nm * 1e-3

    env = None
    i0 = peaks[int(np.argmax(c[peaks]))]
    half = int(0.4 * d / step)
    sl = slice(max(i0 - half, 0), i0 + half + 1)
    if c[sl].size >= 5:
        try:
            g = fit_gaussian(opd_um[sl], c[sl])
            env = bandwidth_from_fwhm(g.w, lam_um) * 1e3
        except RuntimeError:
            pass

    return FPResult(
        peaks_um=xp,
        spacing_um=float(d),
        spacing_err_um=d_err,
        optical_length_um=float(d) / 2,
        mode_spacing_nm=lam_um**2 / d * 1e3,
        mode_spacing_ghz=C_LIGHT / (d * 1e-6) / 1e9,
        envelope_fwhm_nm=env,
    )


def spectrum_from_scan(opd_um: np.ndarray, c: np.ndarray, wavelength_nm: float = 1550.0, pad: int = 8):
    """Spectrum shape from an FFT of the contrast scan.

    Returns (wavelength offset from centre in nm, normalised spectral amplitude).
    """
    ok = np.isfinite(c)
    x, y = opd_um[ok], c[ok]
    step = float(np.median(np.diff(x)))
    grid = np.arange(x.min(), x.max() + step / 2, step)
    yi = np.interp(grid, x, y)
    yi = yi - np.min(yi)
    yi *= np.hanning(yi.size)
    n = pad * yi.size
    spec = np.abs(np.fft.rfft(yi, n=n))
    k = np.fft.rfftfreq(n, d=step)  # cycles per µm of path difference
    dlam_nm = (wavelength_nm * 1e-3) ** 2 * k * 1e3  # Δλ = λ² k
    return dlam_nm, spec / spec.max()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", help="CSV written by fibre-scan")
    p.add_argument("--source", choices=["sld", "fp"], required=True)
    p.add_argument("--wavelength", type=float, default=1550.0, help="centre wavelength in nm")
    p.add_argument("--um-per-unit", type=float, help="µm of path difference per unit of x (default from unit)")
    p.add_argument("--min-spacing", type=float, default=200.0, help="FP: smallest revival spacing to look for, µm")
    p.add_argument("--plot", help="save the figure to this file (default: <scan>_<source>.png)")
    p.add_argument("--show", action="store_true", help="open the figure window")
    a = p.parse_args(argv)

    import matplotlib

    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scan = load_scan(a.scan)
    x = scan.opd_um(a.um_per_unit)
    c = scan.contrast
    err = scan.contrast_err

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
    ax1.errorbar(x, c, yerr=np.nan_to_num(err), fmt="o", ms=3, label="data")
    ax1.set_xlabel("path difference [µm]")
    ax1.set_ylabel("contrast")

    if a.source == "sld":
        res = analyse_sld(x, c, err, a.wavelength)
        xx = np.linspace(x.min(), x.max(), 1000)
        f = res.fit
        ax1.plot(xx, gaussian(xx, f.a, f.x0, f.w, f.b), label=f"Gaussian, FWHM {f.w:.1f} µm")
    else:
        res = analyse_fp(x, c, a.wavelength, a.min_spacing)
        for xp in res.peaks_um:
            ax1.axvline(xp, color="C1", lw=0.8, ls="--")
    print(res)
    ax1.legend()

    dlam, spec = spectrum_from_scan(x, c, a.wavelength)
    keep = dlam <= (10 * res.bandwidth_nm if a.source == "sld" else 6 * res.mode_spacing_nm)
    ax2.plot(dlam[keep], spec[keep])
    ax2.set_xlabel("wavelength offset from centre [nm] (FFT of scan)")
    ax2.set_ylabel("spectral amplitude (norm.)")
    fig.tight_layout()

    out = a.plot or str(Path(a.scan).with_suffix("")) + f"_{a.source}.png"
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
