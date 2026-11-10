"""Reconstruct the source spectrum (intensity vs wavelength) from a delay scan.

Appendix A: contrast vs path difference δ is |FT of the spectrum S(ν)|, so the
inverse transform of the scan gives S back. Steps:

1. put zero path difference at δ = 0 (the highest-contrast point, or --zero);
2. mirror the scan, since C(-δ) = C(δ), so a one-sided scan still works;
3. subtract the noise floor and resample onto equal steps;
4. Fourier transform: spatial frequency k (cycles/µm) maps to wavenumber
   1/λ = 1/λ0 ± k, giving intensity vs wavelength.

    uv run fibre-spectrum sld_scan.csv --source sld
    uv run fibre-spectrum fp_scan.csv --source fp

Limits, printed for each run:
    resolution  ≈ λ² / (2 × largest |δ| scanned)   - scan further to resolve finer
    range       = ±λ² / (2 × step)                - step finer to see wider
Only |FT| is measured (no phase), so the result is symmetric about λ0: it
shows the width and the mode spacing, not lopsidedness.

Writes <scan>_spectrum.csv (wavelength_nm, intensity) and output/<section>/<scan>_spectrum.png
with the manufacturer spectrum (traced off the data sheet) overlaid. The
reconstruction's absolute wavelengths rest on the assumed centre wavelength, so
compare spacings and widths, not exact positions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from . import setup_a
from .io import find_data, graph_path, utf8_output
from .scan import load_scan


@dataclass
class Spectrum:
    wavelength_nm: np.ndarray
    intensity: np.ndarray  # normalised to 1 at the peak
    centre_nm: float
    resolution_nm: float
    range_nm: float
    zero_um: float

    def fwhm_nm(self) -> float:
        """Full width at half maximum of the main peak (interpolated)."""
        i = int(np.argmax(self.intensity))
        y, x = self.intensity, self.wavelength_nm
        lo = np.where(y[:i] < 0.5)[0]
        hi = np.where(y[i:] < 0.5)[0]
        if lo.size == 0 or hi.size == 0:
            return np.nan
        a, b = lo[-1], i + hi[0]
        xl = np.interp(0.5, [y[a], y[a + 1]], [x[a], x[a + 1]])
        xr = np.interp(0.5, [y[b], y[b - 1]], [x[b], x[b - 1]])
        return float(abs(xr - xl))

    def peaks_nm(self, min_height: float = 0.1) -> np.ndarray:
        idx, _ = find_peaks(self.intensity, height=min_height, prominence=0.05)
        return self.wavelength_nm[idx]


def reconstruct(
    opd_um: np.ndarray,
    contrast: np.ndarray,
    centre_nm: float,
    zero_um: float | None = None,
    window: str = "hann",
    pad: int = 16,
) -> Spectrum:
    ok = np.isfinite(opd_um) & np.isfinite(contrast)
    x, c = opd_um[ok], contrast[ok]
    if x.size < 4:
        raise ValueError("Need at least 4 scan points")
    if zero_um is None:
        zero_um = float(x[np.argmax(c)])
    d = x - zero_um
    step = float(np.median(np.abs(np.diff(np.sort(x)))))
    half = float(np.max(np.abs(d)))

    # Mirror (C is even in δ), then average onto an even grid.
    dd = np.concatenate([d, -d])
    cc = np.concatenate([c, c])
    order = np.argsort(dd)
    grid = np.arange(-half, half + step / 2, step)
    cg = np.interp(grid, dd[order], cc[order])
    cg = np.clip(cg - np.percentile(c, 5), 0, None)  # noise floor would add a spike at λ0
    if window == "hann":
        cg = cg * np.hanning(cg.size)

    # Zero-pad (smoother plot, not more resolution) with δ = 0 at index 0 and
    # negative δ wrapped to the end, so the transform of the even scan is real.
    n = pad * cg.size
    mid = int(np.argmin(np.abs(grid)))
    padded = np.zeros(n)
    padded[: cg.size - mid] = cg[mid:]
    if mid:
        padded[-mid:] = cg[:mid]
    spec = np.abs(np.fft.fft(padded))
    k = np.fft.fftfreq(n, d=step)  # cycles per µm of path difference
    order = np.argsort(k)
    k, spec = k[order], spec[order]
    sigma = 1 / (centre_nm * 1e-3) + k  # wavenumber, 1/µm
    keep = sigma > 0
    lam_nm = 1e3 / sigma[keep]
    inten = spec[keep]
    o = np.argsort(lam_nm)
    lam_nm, inten = lam_nm[o], inten[o]
    lam2 = (centre_nm * 1e-3) ** 2
    return Spectrum(
        wavelength_nm=lam_nm,
        intensity=inten / inten.max(),
        centre_nm=centre_nm,
        resolution_nm=lam2 / (2 * half) * 1e3,
        range_nm=lam2 / (2 * step) * 1e3,
        zero_um=zero_um,
    )


def datasheet_spectrum(source: str, lam_nm: np.ndarray) -> np.ndarray:
    """The manufacturer's spectrum traced off the data sheet, on the wavelengths
    `lam_nm` (zero outside the plotted range)."""
    lam, inten = setup_a.datasheet_spectrum(source)
    return np.interp(lam_nm, lam, inten, left=0.0, right=0.0)


def main(argv: list[str] | None = None) -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", help="CSV written by fibre-scan")
    p.add_argument("--source", choices=["sld", "fp"], required=True)
    p.add_argument("--wavelength", type=float, help="centre wavelength in nm (default: data sheet)")
    p.add_argument("--um-per-unit", type=float, help="µm of path difference per unit of x (default from unit)")
    p.add_argument("--zero", type=float, help="zero path difference, µm (default: highest-contrast point)")
    p.add_argument("--window", choices=["hann", "none"], default="hann", help="taper before the FFT")
    p.add_argument("--show", action="store_true", help="open the figure window")
    a = p.parse_args(argv)

    import matplotlib

    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centre = a.wavelength or (setup_a.SLD_CENTRE_NM if a.source == "sld" else setup_a.FP_CENTRE_NM)
    a.scan = str(find_data(a.scan))
    scan = load_scan(a.scan)
    x = scan.opd_um(a.um_per_unit)
    sp = reconstruct(x, scan.contrast, centre, a.zero, a.window)

    span = 2.5 * setup_a.SLD_FWHM_NM if a.source == "sld" else 4 * setup_a.FP_ENVELOPE_FWHM_NM
    view = np.abs(sp.wavelength_nm - centre) <= min(span, sp.range_nm)

    print(f"Zero path difference at {sp.zero_um:.1f} µm; centre wavelength {centre:g} nm")
    print(f"Resolution ≈ {sp.resolution_nm:.2f} nm, range ±{sp.range_nm:.1f} nm")
    if a.source == "sld":
        print(f"Reconstructed FWHM {sp.fwhm_nm():.1f} nm   (data sheet ≈ {setup_a.SLD_FWHM_NM:g} nm)")
        if sp.range_nm < setup_a.SLD_FWHM_NM:
            print("Warning: step too coarse to cover the SLD band - use smaller delay steps")
    else:
        pk = sp.peaks_nm()
        pk = pk[np.abs(pk - centre) <= span]
        if pk.size >= 2:
            print(f"Mode peaks at {np.array2string(pk, precision=2)} nm")
            print(f"Mean mode spacing {np.mean(np.diff(pk)):.3f} nm   (data sheet ≈ {setup_a.FP_MODE_SPACING_NM} nm)")
        if sp.resolution_nm > setup_a.FP_MODE_SPACING_NM / 2:
            print("Warning: scan too short to resolve the modes - scan over more revivals")

    stem = str(Path(a.scan).with_suffix(""))
    with open(stem + "_spectrum.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_nm", "intensity"])
        for lam, i in zip(sp.wavelength_nm[view], sp.intensity[view], strict=True):
            w.writerow([f"{lam:.4f}", f"{i:.5f}"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sp.wavelength_nm[view], sp.intensity[view], label="reconstructed from scan")
    ref = datasheet_spectrum(a.source, sp.wavelength_nm[view])
    cond = "650 mA" if a.source == "sld" else "24.1 mA, 25 °C"
    ax.plot(sp.wavelength_nm[view], ref, "--", lw=1, alpha=0.8, label=f"data sheet, digitised ({cond})")
    ax.set_xlabel(f"wavelength [nm]  (reconstruction centred on an assumed {centre:g} nm)")
    ax.set_ylabel("intensity (normalised)")
    ax.set_title(f"{a.source.upper()} spectrum: resolution ≈ {sp.resolution_nm:.2f} nm")
    ax.legend()
    fig.tight_layout()
    png = graph_path(a.scan, "_spectrum.png")
    fig.savefig(png, dpi=150)
    print(f"Saved {stem}_spectrum.csv and {png}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
