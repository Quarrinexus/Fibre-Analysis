"""Redraw the section 3 (Mach-Zehnder) report figures in the paper style (same data and analysis as the commands
that made them).

Run from fibre_analysis/:
    uv run python tools/section3_figures.py

Writes ../figures/fig02, fig03, fig04, fig06, fig07 and fig08 (main run). Fig05 was already made in this style,
fig01/fig01b are apparatus diagrams from tools/apparatus_figures.py, and the section 4 figure fig09 is in
tools/section4_figures.py, which reuses the plotting helpers here.
The style itself lives in tools/section6_figures.py (paper_style), so
all report figures share it. Prints the fitted numbers so they can be checked against REPORT.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fibre_analysis import setup_a
from fibre_analysis.count import count_crossings
from fibre_analysis.envelope import analyse_fp, analyse_sld, gaussian, spectrum_from_scan
from fibre_analysis.io import load_trace
from fibre_analysis.scan import load_scan
from fibre_analysis.spectrum import datasheet_spectrum, reconstruct

sys.path.insert(0, str(Path(__file__).resolve().parent))
from section6_figures import minor, paper_style

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
FIGS = HERE.parent / "figures"


def label(ax, text: str, x: float = 0.03, y: float = 0.95, ha: str = "left", va: str = "top") -> None:
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va)


def fig02_fringe_count() -> None:
    tr = load_trace(DATA / "fringecount_main_20261109_104324.csv")
    v, t = tr.channel("A"), tr.t - tr.t[0]
    c = count_crossings(v, 0.3)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, v, lw=0.6, color="C0", label="detector")
    ax.plot(
        t[c.crossing_idx],
        v[c.crossing_idx],
        "|",
        color="C3",
        ms=10,
        label=f"{c.crossings} crossings = {c.fringes:g} fringes",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Detector signal (V)")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.1, 3.4)
    ax.legend(loc="upper center", fontsize=10, ncol=2)
    minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig02_fringe_count.png")
    plt.close(fig)
    print(f"fig02: {c.fringes:g} fringes ({c.crossings} crossings) in {t[-1]:.1f} s")


def _scan(name: str):
    s = load_scan(DATA / name)
    return s, s.opd_um(), s.contrast, np.nan_to_num(s.contrast_err)


def fig03_fp_scan() -> None:
    _, x, c, err = _scan("fp_scan_main.csv")
    lam = setup_a.FP_CENTRE_NM
    res = analyse_fp(x, c, lam, 0.6 * setup_a.fp_round_trip_um())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7.5))
    for k, xp in enumerate(res.peaks_um):
        ax1.axvline(xp / 1e3, color="C1", lw=0.8, ls="--", label="revivals (comb fit)" if k == 0 else None)
    ax1.errorbar(x / 1e3, c, yerr=err, fmt="o", ms=3, color="C0", label="measured")
    ax1.set_xlabel("Main micrometer (mm)")
    ax1.set_ylabel("Fringe contrast")
    ax1.set_xlim(0, 25)
    ax1.set_ylim(0, 1.18)
    ax1.legend(loc="upper center", fontsize=10, ncol=2)
    dlam, spec = spectrum_from_scan(x, c, lam)
    keep = dlam <= 6 * res.mode_spacing_nm
    ax2.plot(dlam[keep], spec[keep], color="C0")
    ax2.set_xlabel("Wavelength offset from centre (nm)")
    ax2.set_ylabel("Spectral amplitude (normalised)")
    ax2.set_xlim(0, dlam[keep].max())
    ax2.set_ylim(0, 1.05)
    label(ax1, "(a)", y=0.97)
    label(ax2, "(b)", x=0.97, ha="right")
    for ax in (ax1, ax2):
        minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig03_fp_scan_revivals.png")
    plt.close(fig)
    print(f"fig03: {res}".replace("\n", " | "))


def _spectrum_fig(name: str, source: str, out: str, cond: str) -> None:
    _, x, c, _ = _scan(name)
    centre = setup_a.SLD_CENTRE_NM if source == "sld" else setup_a.FP_CENTRE_NM
    sp = reconstruct(x, c, centre, None, "hann")
    span = 2.5 * setup_a.SLD_FWHM_NM if source == "sld" else 4 * setup_a.FP_ENVELOPE_FWHM_NM
    view = np.abs(sp.wavelength_nm - centre) <= min(span, sp.range_nm)
    lam = sp.wavelength_nm[view]
    fig, ax = plt.subplots()
    ax.plot(lam, sp.intensity[view], color="C0", label="reconstructed from scan")
    ax.plot(lam, datasheet_spectrum(source, lam), "--", color="C1", lw=1.2, label=f"data sheet ({cond})")
    ax.set_xlabel(f"Wavelength (nm), centred on an assumed {centre:g} nm")
    ax.set_ylabel("Intensity (normalised)")
    ax.set_xlim(lam.min(), lam.max())
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left" if source == "fp" else "upper right", fontsize=10)
    ax.text(
        0.97 if source == "fp" else 0.03,
        0.95,
        f"resolution ≈ {sp.resolution_nm:.2g} nm",
        transform=ax.transAxes,
        ha="right" if source == "fp" else "left",
        va="top",
        fontsize=10,
    )
    minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / out)
    plt.close(fig)
    extra = (
        f"FWHM {sp.fwhm_nm():.1f} nm"
        if source == "sld"
        else f"peaks {np.round(sp.peaks_nm()[np.abs(sp.peaks_nm() - centre) <= span], 2)}"
    )
    print(f"{out[:5]}: resolution {sp.resolution_nm:.3f} nm, zero {sp.zero_um:.1f} µm, {extra}")


def fig06_contrast_vs_time() -> None:
    s = load_scan(DATA / "monitor_main_11p30mm.csv")  # x is time in s
    t, c, err = s.x, s.contrast, np.nan_to_num(s.contrast_err)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.errorbar(t, c, yerr=err, fmt="o-", ms=4, lw=0.8, color="C0")
    ax.axhline(np.mean(c), color="k", ls=":", lw=1, label=f"mean {np.mean(c):.3f}, SD {np.std(c, ddof=1):.3f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fringe contrast")
    ax.set_xlim(-5, 300)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10)
    minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig06_fp_contrast_vs_time.png")
    plt.close(fig)
    print(f"fig06: n={c.size}, mean {np.mean(c):.3f}, SD {np.std(c, ddof=1):.3f}")


def _sld_fig(name: str, out: str) -> None:
    _, x, c, err = _scan(name)
    lam = setup_a.SLD_CENTRE_NM
    res = analyse_sld(x, c, err, lam, 0.0)
    xx = np.linspace(x.min(), x.max(), 1000)
    g = res.gauss
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7.5))
    ax1.errorbar(x / 1e3, c, yerr=err, fmt="o", ms=3, color="C0", label="measured")
    ax1.plot(
        xx / 1e3,
        res.model(xx),
        color="C1",
        label=f"spectrum fit: Δλ = {res.bandwidth_nm:.1f} nm, order {res.order:.1f}",
    )
    ax1.plot(xx / 1e3, gaussian(xx, g.a, g.x0, g.w, g.b), "--", color="C2", lw=1, label="Gaussian (central peak)")
    ax1.set_xlabel("Main micrometer (mm)")
    ax1.set_ylabel("Fringe contrast")
    ax1.set_xlim(x.min() / 1e3, x.max() / 1e3)
    ax1.set_ylim(min(-0.02 * np.nanmax(c), ax1.get_ylim()[0]), 1.4 * np.nanmax(c))
    ax1.legend(loc="upper left", fontsize=9)
    dlam, spec = spectrum_from_scan(x, c, lam)
    keep = dlam <= 3 * res.bandwidth_nm
    ax2.plot(dlam[keep], spec[keep], color="C0")
    ax2.set_xlabel("Wavelength offset from centre (nm)")
    ax2.set_ylabel("Spectral amplitude (normalised)")
    ax2.set_xlim(0, dlam[keep].max())
    ax2.set_ylim(0, 1.05)
    label(ax1, "(a)", x=0.97, ha="right")
    label(ax2, "(b)", x=0.97, ha="right")
    for ax in (ax1, ax2):
        minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / out)
    plt.close(fig)
    print(f"{out[:5]}: {res}".replace("\n", " | "))


def main() -> None:
    paper_style()
    fig02_fringe_count()
    fig03_fp_scan()
    _spectrum_fig("fp_scan_main.csv", "fp", "fig04_fp_spectrum.png", "24.1 mA, 25 °C")
    fig06_contrast_vs_time()
    _sld_fig("sld_scan_main.csv", "fig07_sld_envelope.png")
    _spectrum_fig("sld_scan_main.csv", "sld", "fig08_sld_spectrum.png", "650 mA")


if __name__ == "__main__":
    main()
