"""Source spectra from contrast-vs-delay scans (section 3, Appendix A).

Contrast vs path difference is |FT of the spectrum| (Eq. 20), so:

SLD (broad, smooth spectrum)
    One envelope around zero path difference. How its width maps to Δλ depends
    on the spectrum's shape: w ≈ 0.88 λ²/Δλ for a Gaussian, ≈ 1.2 λ²/Δλ for the
    flat-topped SLD on the data sheet. So the scan is fitted with |FT| of a
    super-Gaussian spectrum whose width Δλ and shape (order) are both free; the
    side lobes of the envelope pin down the shape. A Gaussian-only estimate is
    printed for comparison.

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

import numpy as np
from scipy.optimize import curve_fit

from . import setup_a
from .io import graph_path, utf8_output
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


def coherence_model(
    opd_um: np.ndarray, fwhm_nm: float, order: float, wavelength_nm: float, sweep_um: float = 0.0
) -> np.ndarray:
    """|FT of a super-Gaussian spectrum| at each path difference (normalised to 1 at zero).

    S(λ) = exp(-ln2 |2(λ-λ0)/Δλ|^order): order 2 is a Gaussian, larger orders are
    increasingly flat-topped (the SLD data sheet looks like ~4).

    sweep_um is the modulator's peak-to-peak path-difference swing: each measured
    point is the largest contrast within ±sweep/2 of the set delay, which widens
    the envelope slightly."""
    if sweep_um > 0:
        offsets = np.linspace(-sweep_um / 2, sweep_um / 2, 9)
        vals = coherence_model(np.add.outer(np.asarray(opd_um), offsets).ravel(), fwhm_nm, order, wavelength_nm)
        return vals.reshape((*np.shape(opd_um), offsets.size)).max(axis=-1)
    lam = (wavelength_nm + np.linspace(-2.0, 2.0, 401) * fwhm_nm) * 1e-3  # µm
    lam = lam[lam > 0]
    spec = np.exp(-np.log(2) * np.abs(2 * (lam - wavelength_nm * 1e-3) / (fwhm_nm * 1e-3)) ** order)
    phase = np.exp(-2j * np.pi * np.outer(opd_um, 1 / lam))
    return np.abs(phase @ spec) / spec.sum()


@dataclass
class SLDResult:
    x0: float
    x0_err: float
    peak: float
    fwhm_direct_um: float
    bandwidth_nm: float
    bandwidth_err_nm: float
    order: float
    order_err: float
    gauss: GaussFit
    gauss_bandwidth_nm: float
    wavelength_nm: float
    model_params: tuple[float, ...]
    sweep_um: float = 0.0

    @property
    def coherence_length_um(self) -> float:
        return (self.wavelength_nm * 1e-3) ** 2 / (self.bandwidth_nm * 1e-3)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Zero path difference     {self.x0:.1f} ± {self.x0_err:.1f} µm",
                f"Peak contrast            {self.peak:.3f}",
                f"Envelope FWHM (direct)   {self.fwhm_direct_um:.1f} µm",
                f"Spectral FWHM Δλ         {self.bandwidth_nm:.1f} ± {self.bandwidth_err_nm:.1f} nm"
                f"  (spectrum-shape fit, order {self.order:.1f} ± {self.order_err:.1f}; 2 = Gaussian)",
                f"  Gaussian-only estimate {self.gauss_bandwidth_nm:.1f} nm (central peak, for comparison)",
                f"Coherence length λ²/Δλ   {self.coherence_length_um:.1f} µm",
            ]
        )

    def model(self, x: np.ndarray) -> np.ndarray:
        a, x0, w, order, b = self.model_params
        return a * coherence_model(x - x0, w, order, self.wavelength_nm, self.sweep_um) + b


def _direct_fwhm(x: np.ndarray, c: np.ndarray) -> float:
    i0 = int(np.nanargmax(c))
    half = (c[i0] + np.nanmin(c)) / 2
    left = x[: i0 + 1][c[: i0 + 1] < half]
    right = x[i0:][c[i0:] < half]
    if left.size == 0 or right.size == 0:
        return np.nan
    j, k = np.searchsorted(x, left[-1]), np.searchsorted(x, right[0])
    xl = np.interp(half, [c[j], c[j + 1]], [x[j], x[j + 1]])
    xr = np.interp(half, [c[k], c[k - 1]], [x[k], x[k - 1]])
    return float(xr - xl)


def analyse_sld(
    opd_um: np.ndarray, c: np.ndarray, c_err=None, wavelength_nm: float = 1550.0, sweep_um: float = 0.0
) -> SLDResult:
    ok = np.isfinite(opd_um) & np.isfinite(c)
    x, y = opd_um[ok], c[ok]
    sig = None
    if c_err is not None:
        e = np.asarray(c_err)[ok]
        sig = e if np.all(np.isfinite(e) & (e > 0)) else None
    lam_um = wavelength_nm * 1e-3

    # Gaussian on the central peak only (side lobes would drag it about).
    i0 = int(np.argmax(y))
    main = np.abs(x - x[i0]) <= max(_direct_fwhm(x, y), 2 * np.median(np.diff(x)))
    gauss = fit_gaussian(x[main], y[main], None if sig is None else sig[main])
    gauss_bw = bandwidth_from_fwhm(gauss.w, lam_um) * 1e3

    def model(xx, a, x0, w, order, b):
        return a * coherence_model(xx - x0, w, order, wavelength_nm, sweep_um) + b

    best = None
    for order0 in (2.0, 4.0, 6.0):
        p0 = [gauss.a, gauss.x0, gauss_bw, order0, max(float(np.min(y)), 0.0)]
        try:
            popt, pcov = curve_fit(
                model,
                x,
                y,
                p0=p0,
                sigma=sig,
                absolute_sigma=sig is not None,
                bounds=([0, x.min(), 1.0, 1.0, -0.2], [1.5, x.max(), 1000.0, 12.0, 0.5]),
                maxfev=4000,
            )
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((model(x, *popt) - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        raise RuntimeError("Spectrum-shape fit failed")
    _, popt, pcov = best
    perr = np.sqrt(np.diag(pcov))
    return SLDResult(
        x0=float(popt[1]),
        x0_err=float(perr[1]),
        peak=float(popt[0] + popt[4]),
        fwhm_direct_um=_direct_fwhm(x, y),
        bandwidth_nm=float(popt[2]),
        bandwidth_err_nm=float(perr[2]),
        order=float(popt[3]),
        order_err=float(perr[3]),
        gauss=gauss,
        gauss_bandwidth_nm=float(gauss_bw),
        wavelength_nm=wavelength_nm,
        model_params=tuple(float(v) for v in popt),
        sweep_um=sweep_um,
    )


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
    opd_um: np.ndarray,
    c: np.ndarray,
    wavelength_nm: float = 1550.0,
    min_spacing_um: float = 0.6 * setup_a.fp_round_trip_um(),
) -> FPResult:
    ok = np.isfinite(opd_um) & np.isfinite(c)
    x, y = opd_um[ok], c[ok]
    step = float(np.median(np.diff(x)))

    # 1. The spacing that best fits a regular comb over the whole scan. Single high
    #    points between revivals (the laser's mode-partition noise) barely affect it.
    w = y - np.mean(y)
    cands = np.linspace(min_spacing_um, 2.2 * min_spacing_um, 4000)
    power = np.abs(np.exp(2j * np.pi * np.outer(1 / cands, x)) @ w)
    d0 = float(cands[np.argmax(power)])
    x0 = float(np.angle(np.exp(2j * np.pi * x / d0) @ w) / (2 * np.pi) * d0)

    # 2. Locate each revival near its comb position (contrast-weighted centroid),
    #    keeping only those that are inside the scan and reach a real peak.
    teeth, heights = [], []
    for m in range(int(np.ceil((x.min() - x0) / d0)), int(np.floor((x.max() - x0) / d0)) + 1):
        t = x0 + m * d0
        win = np.abs(x - t) <= 0.2 * d0
        if win.sum() < 3:
            continue
        i_max = np.flatnonzero(win)[np.argmax(y[win])]
        if i_max in (0, len(x) - 1) or y[i_max] < 0.5 * np.max(y):
            continue  # rising edge at the scan's end, or no revival here
        wt = y[win] ** 4
        teeth.append((m, float(np.sum(x[win] * wt) / np.sum(wt))))
        heights.append(float(y[i_max]))
    if len(teeth) < 2:
        raise ValueError("Found fewer than two contrast revivals - scan further or check --min-spacing")
    order = np.array([m for m, _ in teeth], dtype=float)
    xp = np.array([p for _, p in teeth])
    if xp.size >= 3:
        (d, _), cov = np.polyfit(order, xp, 1, cov=True)
        d_err = float(np.sqrt(cov[0, 0]))
    else:
        d, d_err = float((xp[1] - xp[0]) / (order[1] - order[0])), step / np.sqrt(2)
    centre_peak = xp[int(np.argmax(heights))]
    lam_um = wavelength_nm * 1e-3

    env = None
    sl = np.abs(x - centre_peak) <= 0.4 * d
    if sl.sum() >= 5:
        try:
            g = fit_gaussian(x[sl], y[sl])
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
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", help="CSV written by fibre-scan")
    p.add_argument("--source", choices=["sld", "fp"], required=True)
    p.add_argument("--wavelength", type=float, help="centre wavelength in nm (default: from the Setup A data sheet)")
    p.add_argument("--um-per-unit", type=float, help="µm of path difference per unit of x (default from unit)")
    p.add_argument(
        "--sweep-um",
        type=float,
        default=0.0,
        help="SLD: modulator path-difference swing from fibre-calibrate; models the slight widening it causes (~±5%%)",
    )
    p.add_argument(
        "--min-spacing",
        type=float,
        default=0.6 * setup_a.fp_round_trip_um(),
        help="FP: ignore peaks closer than this, µm (default 60%% of the data-sheet 2nL ≈ 1250 µm)",
    )
    p.add_argument("--plot", help="save the figure to this file (default: output/<scan>_<source>.png)")
    p.add_argument("--show", action="store_true", help="open the figure window")
    a = p.parse_args(argv)

    import matplotlib

    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.wavelength is None:
        a.wavelength = setup_a.SLD_CENTRE_NM if a.source == "sld" else setup_a.FP_CENTRE_NM
    scan = load_scan(a.scan)
    x = scan.opd_um(a.um_per_unit)
    c = scan.contrast
    err = scan.contrast_err

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
    ax1.errorbar(x, c, yerr=np.nan_to_num(err), fmt="o", ms=3, label="data")
    ax1.set_xlabel("path difference [µm]")
    ax1.set_ylabel("contrast")

    if a.source == "sld":
        res = analyse_sld(x, c, err, a.wavelength, a.sweep_um)
        xx = np.linspace(x.min(), x.max(), 1000)
        g = res.gauss
        ax1.plot(xx, res.model(xx), label=f"spectrum fit: Δλ = {res.bandwidth_nm:.1f} nm, order {res.order:.1f}")
        ax1.plot(xx, gaussian(xx, g.a, g.x0, g.w, g.b), "--", lw=1, label="Gaussian (central peak)")
    else:
        res = analyse_fp(x, c, a.wavelength, a.min_spacing)
        for xp in res.peaks_um:
            ax1.axvline(xp, color="C1", lw=0.8, ls="--")
    print(res)
    print(f"(centre wavelength {a.wavelength:g} nm)")
    if a.source == "sld":
        print(f"Data sheet: Δλ ≈ {setup_a.SLD_FWHM_NM:g} nm, flat-topped (order ≈ 4)")
    else:
        print(
            f"Data sheet: mode spacing ≈ {setup_a.FP_MODE_SPACING_NM} nm (2nL ≈ {setup_a.fp_round_trip_um():.0f} µm), "
            f"envelope ≈ {setup_a.FP_ENVELOPE_FWHM_NM} nm"
        )
    ax1.legend()

    dlam, spec = spectrum_from_scan(x, c, a.wavelength)
    keep = dlam <= (3 * res.bandwidth_nm if isinstance(res, SLDResult) else 6 * res.mode_spacing_nm)
    ax2.plot(dlam[keep], spec[keep])
    ax2.set_xlabel("wavelength offset from centre [nm] (FFT of scan: shape only, approximate)")
    ax2.set_ylabel("spectral amplitude (norm.)")
    fig.tight_layout()

    out = a.plot or graph_path(a.scan, f"_{a.source}.png")
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
