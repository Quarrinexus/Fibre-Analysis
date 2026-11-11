"""Section 6 (vibrometer) analysis of the 2026-11-11 main run: tables and report figures.

Run from fibre_analysis/:
    uv run python tools/section6_figures.py

Writes output/section6/section6_summary.csv, output/section6/section6_windows.csv and the report
figures ../figures/fig15_* ... fig21_* (paper style, matching fig11).

Which captures are used, and why some are left out, is listed in CAPTURES / EXCLUDED below and in
REPORT.md §9.2. The wavelength is the FP default (1547.7 nm) until the bench laser's label is read.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, NullFormatter
from scipy.signal import ShortTimeFFT, butter, hilbert, sosfiltfilt
from scipy.signal.windows import gaussian

from fibre_analysis.io import load_trace
from fibre_analysis.vibrometer import analyse

LAM = 1547.7e-9
HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
OUT = HERE / "output" / "section6"
FIGS = HERE.parent / "figures"

# (group, drive Vpp on the display, drive frequency Hz, waveform, file stems). Disturbed or faded
# captures are left out (EXCLUDED).
CAPTURES = [
    ("C4", 0.2, 2.0, "sine", [f"vib_2Hz_0p2Vpp_main_{i}" for i in (1, 2, 3)]),
    ("C4", 0.5, 2.0, "sine", [f"vib_2Hz_0p5Vpp_main_{i}" for i in (1, 2, 8, 9)]),
    ("C4", 1.0, 2.0, "sine", [f"vib_2Hz_1p0Vpp_main_{i}" for i in (1, 2, 3)]),
    ("C4", 2.0, 2.0, "sine", [f"vib_2Hz_2p0Vpp_main_{i}" for i in (1, 2)]),
    ("C4", 3.0, 2.0, "sine", [f"vib_2Hz_3p0Vpp_main_{i}" for i in (1, 2, 3)]),
    ("C4", 4.0, 2.0, "sine", [f"vib_2Hz_4p0Vpp_main_{i}" for i in (1, 2, 3)] + ["vib_2Hz_4p0Vpp_fast_main_1"]),
    ("C4b", 5.0, 2.0, "sine", [f"vib_2Hz_5p0Vpp_high_main_{i}" for i in (1, 2)]),
    ("C4b", 6.0, 2.0, "sine", [f"vib_2Hz_6p0Vpp_high_main_{i}" for i in (1, 2)]),
    ("C4b", 8.0, 2.0, "sine", [f"vib_2Hz_8p0Vpp_high_main_{i}" for i in (1, 2)]),
    ("C4b", 8.0, 1.0, "sine", ["vib_1Hz_8p0Vpp_main_1", "vib_1Hz_8p0Vpp_main_2", "vib_1Hz_8p0Vpp_fast_main_1"]),
    ("C5", 2.0, 1.0, "sine", [f"vib_1Hz_2p0Vpp_main_{i}" for i in (4, 5, 8)]),
    ("C5", 2.0, 1.5, "sine", [f"vib_1p5Hz_2p0Vpp_main_{i}" for i in (1, 2, 3)]),
    ("C6", 2.0, 2.0, "sine", [f"vib_2Hz_2p0Vpp_rep_main_{i}" for i in range(1, 11)]),
]
EXCLUDED = {
    "vib_first_main": "first capture, motion not settled (5.21 µm against 5.02)",
    "vib_2Hz_0p5Vpp_main_3": "knock at 0.6 s",
    "vib_2Hz_0p5Vpp_main_4": "fringes faded (17 mV)",
    "vib_2Hz_0p5Vpp_main_5": "fringes faded (17 mV)",
    "vib_2Hz_0p5Vpp_main_6": "fibre moving after handling",
    "vib_2Hz_0p5Vpp_main_7": "fibre moving after handling",
    "vib_2Hz_2p0Vpp_main_3": "disturbed (integrated 72 µm)",
    "vib_1Hz_2p0Vpp_main_1..3": "fringes faded (17-26 mV)",
    "vib_1Hz_2p0Vpp_main_6, _7": "second harmonic of the fringes dominant",
    "vib_2Hz_8p0Vpp_24k_main, _49k_main": "< 1 drive period; used only for the fringe-rate test",
}


def paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "0.8",
            "errorbar.capsize": 3,
            "lines.markersize": 6,
        }
    )


def minor(ax) -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def drive_phase(b: np.ndarray, fs: float, fd: float) -> tuple[float, float]:
    """Amplitude and phase of a sine at fd fitted to the drive (linear least squares)."""
    t = np.arange(b.size) / fs
    m = np.column_stack([np.sin(2 * np.pi * fd * t), np.cos(2 * np.pi * fd * t), np.ones_like(t)])
    (s, c, _), *_ = np.linalg.lstsq(m, b, rcond=None)
    return float(np.hypot(s, c)), float(np.arctan2(c, s))


def analyse_file(stem: str, fd: float, window: float | None = None):
    tr = load_trace(DATA / f"{stem}.csv")
    a, b, fs = tr.channel("A"), tr.channel("B"), tr.fs
    r = analyse(a, fs, b, fd, LAM, window)
    return tr, r


def summary_rows() -> list[dict]:
    rows = []
    for group, vpp, fd, _wave, stems in CAPTURES:
        for stem in stems:
            tr, r = analyse_file(stem, fd)
            b = tr.channel("B")
            _, bph = drive_phase(b, tr.fs, fd)
            # displacement = X sin(wt + phase - pi/2); lag of displacement behind the drive, degrees
            lag = np.degrees((bph - (r.phase - np.pi / 2) + np.pi) % (2 * np.pi) - np.pi)
            rows.append(
                {
                    "group": group,
                    "file": stem,
                    "drive_Vpp": vpp,
                    "f_Hz": fd,
                    "B_Vpp": round(float(np.ptp(b)), 4),
                    "X_fit_um": round(r.amplitude * 1e6, 3),
                    "X_fit_err_um": round(r.amplitude_err * 1e6, 3),
                    "X_int_um": round(float(np.ptp(r.displacement)) / 2 * 1e6, 3),
                    "X_count_um": round(r.amplitude_counting * 1e6, 3),
                    "v_peak_mm_s": round(r.v_peak * 1e3, 4),
                    "window_ms": round(r.window * 1e3, 2),
                    "lag_deg": round(float(lag), 1),
                    "A_swing_V": round(float(np.ptp(tr.channel("A"))), 3),
                }
            )
    return rows


def group_stats(rows, key):
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def fig_example() -> None:
    stem = "vib_2Hz_2p0Vpp_rep_main_5"
    tr, r = analyse_file(stem, 2.0)
    t0 = 0.0
    fig, axs = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
    fm = r.spec_f <= 1300
    axs[0].pcolormesh(r.t + t0, r.spec_f[fm], np.log10(r.spectrogram[fm] + 1e-20), shading="auto", cmap="viridis")
    axs[0].plot(r.t + t0, r.freq, "w.", ms=2)
    axs[0].set_ylabel("Fringe frequency (Hz)")
    axs[1].plot(r.t, r.speed * 1e3, "o", ms=2.5, label="|v| from STFT peak")
    vm = np.abs(r.v_peak * np.sin(2 * np.pi * r.drive_freq * r.t + r.phase))
    axs[1].plot(r.t, vm * 1e3, "--", color="C1", label="fit |V sin(ωt + φ)|")
    axs[1].plot(r.t, r.velocity * 1e3, "-", color="C2", lw=1, label="signed v")
    axs[1].axhline(0, color="k", ls=":", lw=0.8)
    axs[1].set_ylabel("Velocity (mm/s)")
    axs[1].legend(loc="lower right", fontsize=10)
    axs[2].plot(r.t, r.displacement * 1e6, color="C0", label="mirror displacement")
    axs[2].set_ylabel("Displacement (µm)")
    axs[2].set_xlabel("Time (s)")
    ax2 = axs[2].twinx()
    ax2.plot(tr.t - tr.t[0], tr.channel("B"), color="C3", lw=0.8, alpha=0.6, label="drive (channel B)")
    ax2.set_ylabel("Drive (V)", color="C3")
    ax2.tick_params(direction="in", which="both")
    for ax in axs:
        minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig15_vibrometer_example.png")
    plt.close(fig)


def fig_drive(rows) -> dict:
    sel = [r for r in rows if r["f_Hz"] == 2.0 and r["group"] in ("C4", "C4b")]
    g = group_stats(sel, lambda r: r["drive_Vpp"])
    v = np.array(sorted(g))
    x = np.array([np.mean([r["X_fit_um"] for r in g[k]]) for k in v])
    xe = np.array([np.std([r["X_fit_um"] for r in g[k]], ddof=1) if len(g[k]) > 1 else 0 for k in v])
    xc = np.array([np.mean([r["X_count_um"] for r in g[k]]) for k in v])
    b = np.array([np.mean([r["B_Vpp"] for r in g[k]]) for k in v])
    lin = v <= 1.0
    slope = float(np.sum(x[lin] * v[lin]) / np.sum(v[lin] ** 2))  # through the origin, low drives
    fig, ax = plt.subplots()
    ax.errorbar(v, x, yerr=xe, fmt="o-", label="STFT fit (C4, C4b)")
    ax.plot(v, xc, "s", mfc="none", color="C1", label="fringe count")
    vv = np.linspace(0, 8.4, 50)
    ax.plot(vv, slope * vv, ":", color="k", label=f"linear from ≤ 1 V: {slope:.1f} µm/V")
    ax.axvspan(4.5, 8.4, color="0.92", zorder=0)
    ax.text(6.4, 15, "signal fading\n(§6.4)", ha="center", fontsize=10)
    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 290)
    ax.set_xlabel("Drive amplitude, generator display (V p-p)")
    ax.set_ylabel("Mirror amplitude X (µm)")
    ax.legend(loc="upper left")
    minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig16_amplitude_vs_drive.png")
    plt.close(fig)
    return {"V": v, "X": x, "Xsd": xe, "Xcount": xc, "B": b, "slope_low": slope}


def fig_frequency(rows) -> dict:
    sel = [r for r in rows if r["drive_Vpp"] == 2.0 and r["group"] in ("C4", "C5", "C6")]
    g = group_stats(sel, lambda r: (r["f_Hz"], r["group"] == "C6"))
    keys = sorted(g)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    res = {}
    for (f, rep), lst in [(k, g[k]) for k in keys]:
        xs = [r["X_fit_um"] for r in lst]
        vs = [r["v_peak_mm_s"] for r in lst]
        lags = [r["lag_deg"] for r in lst]
        lab = "repeat run, 10 captures (C6)" if rep else ("C4/C5, 2–3 captures" if f == 1.0 else None)
        mk = "D" if rep else "o"
        col = "C2" if rep else "C0"
        ax1.errorbar(f, np.mean(xs), yerr=np.std(xs, ddof=1), fmt=mk, color=col, label=lab)
        ax2.errorbar(f, np.mean(vs), yerr=np.std(vs, ddof=1), fmt=mk, color=col)
        res[(f, rep)] = (np.mean(xs), np.std(xs, ddof=1), np.mean(vs), np.mean(lags), len(lst))
    ff = np.linspace(0, 2.2, 20)
    xm = res[(1.0, False)][0]
    ax2.plot(ff, 2 * np.pi * ff * xm * 1e-3, ":", color="k", label=f"2πfX, X = {xm:.1f} µm")
    ax1.set_xlim(0.8, 2.2)
    ax1.set_ylim(55, 65)
    ax1.set_xlabel("Drive frequency (Hz)")
    ax1.set_ylabel("Mirror amplitude X (µm)")
    ax2.set_xlim(0, 2.2)
    ax2.set_ylim(0, 0.85)
    ax2.set_xlabel("Drive frequency (Hz)")
    ax2.set_ylabel("Peak velocity (mm/s)")
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.03, 0.95, lab, transform=ax.transAxes, va="top")
        minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig18_amplitude_vs_frequency.png")
    plt.close(fig)
    return res


WINDOWS_MS = [2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 60, 80]
WINDOW_CASES = [
    ("vib_2Hz_0p5Vpp_main_9", 2.0, "0.5 V p-p, 2 Hz"),
    ("vib_2Hz_2p0Vpp_rep_main_5", 2.0, "2 V p-p, 2 Hz"),
    ("vib_2Hz_4p0Vpp_main_2", 2.0, "4 V p-p, 2 Hz"),
    ("vib_1Hz_2p0Vpp_main_8", 1.0, "2 V p-p, 1 Hz"),
]


def window_study() -> list[dict]:
    rows = []
    for stem, fd, label in WINDOW_CASES:
        _, ref = analyse_file(stem, fd)
        xref = ref.amplitude
        rate = 2 * (2 * np.pi * fd) ** 2 * xref / LAM  # max |df/dt|, Hz/s
        topt = 1 / np.sqrt(rate)
        for w in WINDOWS_MS:
            try:
                _, r = analyse_file(stem, fd, w * 1e-3)
            except ValueError:
                rows.append({"file": stem, "label": label, "window_ms": w, "ok": False, "T_opt_ms": topt * 1e3})
                continue
            ok = np.isfinite(r.speed)
            if abs(r.amplitude / xref - 1) > 0.5:  # locked onto noise: the window can't resolve these fringes
                rows.append({"file": stem, "label": label, "window_ms": w, "ok": False, "T_opt_ms": topt * 1e3})
                continue
            # scatter about the reference fit, over the same part of every cycle (|v| > 0.6 V_peak), so
            # windows are compared like for like
            model = np.abs(ref.v_peak * np.sin(2 * np.pi * fd * r.t + ref.phase))
            mid = ok & (model > 0.6 * ref.v_peak)
            resid = r.speed[mid] - model[mid]
            rows.append(
                {
                    "file": stem,
                    "label": label,
                    "window_ms": w,
                    "ok": True,
                    "X_um": r.amplitude * 1e6,
                    "X_err_um": r.amplitude_err * 1e6,
                    "rms_resid_um_s": float(np.sqrt(np.mean(resid**2))) * 1e6,
                    "frac_resolved": float(np.mean(ok)),
                    "T_opt_ms": topt * 1e3,
                    "X_ref_um": xref * 1e6,
                }
            )
    return rows


def fig_windows(rows) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for k, (stem, _, label) in enumerate(WINDOW_CASES):
        rr = [r for r in rows if r["file"] == stem and r["ok"]]
        w = np.array([r["window_ms"] for r in rr])
        xref = rr[0]["X_ref_um"]
        dev = np.array([r["X_um"] / xref - 1 for r in rr]) * 100
        res = np.array([r["rms_resid_um_s"] for r in rr])
        fr = np.array([r["frac_resolved"] for r in rr]) * 100
        mk = "os^D"[k]
        ax1.plot(w, dev, mk + "-", color=f"C{k}", label=label)
        ax2.plot(w, res, mk + "-", color=f"C{k}")
        ax3.plot(w, fr, mk + "-", color=f"C{k}")
        topt = rr[0]["T_opt_ms"]
        for ax in (ax1, ax2, ax3):
            ax.axvline(topt, color=f"C{k}", ls=":", lw=1)
    ww = np.array(WINDOWS_MS, float)
    ax2.plot(ww, LAM / (2 * ww * 1e-3) * 1e6, "k--", lw=1, label="λ/(2T): one frequency bin")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("X − X(auto window) (%)")
    ax1.set_ylim(-1, 2.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Scatter of |v| (µm/s)")
    ax3.set_ylabel("Slices resolved (%)")
    ax3.set_xlabel("STFT window length T (ms)")
    ax3.set_xscale("log")
    ax3.set_xticks([2, 3, 5, 10, 20, 50, 80])
    ax3.set_xticklabels(["2", "3", "5", "10", "20", "50", "80"])
    ax3.xaxis.set_minor_formatter(NullFormatter())
    ax1.legend(loc="upper center", fontsize=10, ncol=2)
    ax2.legend(loc="upper right", fontsize=10)
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)"), (ax3, "(c)")):
        ax.text(
            0.02,
            0.05 if ax is ax3 else 0.93,
            lab,
            transform=ax.transAxes,
            va="bottom" if ax is ax3 else "top",
            ha="left",
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator()) if ax is not ax2 else None
    fig.tight_layout()
    fig.savefig(FIGS / "fig19_window_study.png")
    plt.close(fig)


def fringe_profile(stem: str, fd: float, nbins: int = 8):
    """Fringe peak-to-peak against normalised mirror position (x/X), fast parts of the cycle only."""
    tr, r = analyse_file(stem, fd)
    a, fs = tr.channel("A"), tr.fs
    s = sosfiltfilt(butter(4, 40, "highpass", fs=fs, output="sos"), a - a.mean())
    env = 2 * np.abs(hilbert(s))
    t = np.arange(a.size) / fs
    w = 2 * np.pi * r.drive_freq
    pos = np.sin(w * t + r.phase - np.pi / 2)  # x / X
    fast = np.abs(np.sin(w * t + r.phase)) * r.v_peak * 2 / LAM > 300  # fringes > 300 Hz
    edges = np.linspace(-1, 1, nbins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    prof = np.array(
        [
            np.median(env[fast & (pos >= lo) & (pos < hi)]) if np.sum(fast & (pos >= lo) & (pos < hi)) > 20 else np.nan
            for lo, hi in itertools.pairwise(edges)
        ]
    )
    return mid, prof, r


def fig_fading() -> dict:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    out = {}
    for k, (v, stem) in enumerate(
        [
            (3, "vib_2Hz_3p0Vpp_main_1"),
            (4, "vib_2Hz_4p0Vpp_fast_main_1"),
            (5, "vib_2Hz_5p0Vpp_high_main_1"),
            (6, "vib_2Hz_6p0Vpp_high_main_1"),
            (8, "vib_2Hz_8p0Vpp_high_main_1"),
        ]
    ):
        mid, prof, r = fringe_profile(stem, 2.0)
        ax1.plot(
            mid,
            prof / np.nanmax(prof),
            "os^Dv"[k] + "-",
            color=f"C{k}",
            label=f"{v} V p-p ({r.amplitude * 1e6:.0f} µm)",
        )
        out[v] = (mid, prof)
    for k, (f, stem) in enumerate(
        [
            (2.0, "vib_2Hz_8p0Vpp_high_main_1"),
            (2.0, "vib_2Hz_8p0Vpp_high_main_2"),
            (1.0, "vib_1Hz_8p0Vpp_main_1"),
            (1.0, "vib_1Hz_8p0Vpp_main_2"),
        ]
    ):
        mid, prof, r = fringe_profile(stem, f)
        col = "C4" if f == 2.0 else "C0"
        ax2.plot(
            mid,
            prof / np.nanmax(prof),
            ("o-" if k % 2 == 0 else "s--"),
            color=col,
            label=f"{f:.0f} Hz, peak fringe {r.v_peak * 2 / LAM / 1e3:.1f} kHz" if k % 2 == 0 else None,
        )
    ax1.set_xlabel("Mirror position x / X")
    ax2.set_xlabel("Mirror position x / X")
    ax1.set_ylabel("Fringe amplitude (relative)")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="lower right", fontsize=10)
    for ax, lab in ((ax1, "(a) 2 Hz"), (ax2, "(b) 8 V p-p")):
        ax.text(0.03, 0.95, lab, transform=ax.transAxes, va="top", ha="left")
        ax.set_xlim(-1, 1)
        minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig17_fringe_fading.png")
    plt.close(fig)
    return out


def triangle() -> tuple[list[tuple[float, float, float]], plt.Figure]:
    res = []
    fig, (axd, ax) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, gridspec_kw={"height_ratios": [1, 2.5]})
    for i in (1, 2, 3):
        tr = load_trace(DATA / f"vib_1Hz_tri_2p0Vpp_main_{i}.csv")
        a, b, fs = tr.channel("A"), tr.channel("B"), tr.fs
        t = np.arange(a.size) / fs
        db = np.gradient(sosfiltfilt(butter(2, 20, fs=fs, output="sos"), b), t)
        s = sosfiltfilt(butter(4, 60, "highpass", fs=fs, output="sos"), a - a.mean())
        n = int(0.10 * fs)
        st = ShortTimeFFT(gaussian(n, n / 6), hop=n // 8, fs=fs, mfft=8 * n)
        p = np.abs(st.stft(s)) ** 2
        f = st.f
        ts = st.t(s.size)
        p[f < 100] = 0
        idx = np.argmax(p, 0)
        fd = f[idx].astype(float)
        inner = (idx > 0) & (idx < f.size - 1)
        c = np.arange(p.shape[1])[inner]
        y0, y1, y2 = (np.log(p[idx[inner] + d, c] + 1e-300) for d in (-1, 0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            fd[inner] += np.nan_to_num(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * (f[1] - f[0])
        v = fd * LAM / 2
        ok = (ts > 0.05) & (ts < t[-1] - 0.05)
        dbs = np.interp(ts, t, db)
        tp = ts[np.where(np.diff(np.sign(dbs)) != 0)[0]]
        far = np.array([np.all(np.abs(tt - tp) > 0.12) for tt in ts])
        up, dn = ok & far & (dbs > 0), ok & far & (dbs < 0)
        res.append((float(np.median(v[up])), float(np.median(v[dn])), float(np.median(v[up | dn]) / (4 * 0.993))))
        if i == 1:
            ax.plot(ts[ok], v[ok] * 1e3, "o", ms=3, label="|v| from STFT (100 ms window)")
            axd.plot(t, b, color="C3", lw=1)
            axd.set_ylabel("Drive (V)")
    vmean = np.mean([r[2] for r in res]) * 4 * 0.993
    ax.axhline(vmean * 1e3, color="k", ls=":", lw=1, label=f"mean plateau {vmean * 1e3:.3f} mm/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|v| (mm/s)")
    ax.set_ylim(0.19, 0.30)
    ax.legend(loc="lower right", fontsize=10)
    minor(ax)
    minor(axd)
    fig.tight_layout()
    fig.savefig(FIGS / "fig20_triangle_drive.png")
    plt.close(fig)
    return res, fig


RING_WINDOWS = [(0.002, 0.030), (0.020, 0.055), (0.040, 0.080), (0.058, 0.130)]


def _ring_fit(t: np.ndarray, y: np.ndarray):
    """Damped cosine on a straight line (the sawtooth ramp); best of a few starting frequencies."""
    from scipy.optimize import curve_fit

    def m(tt, a, tau, f, ph, c0, c1):
        return a * np.exp(-tt / tau) * np.cos(2 * np.pi * f * tt + ph) + c0 + c1 * tt

    lin = np.polyfit(t, y, 1)
    best = None
    for f0 in (85, 88, 91, 94):
        try:
            p, cv = curve_fit(
                m, t, y, p0=[1.5 * np.std(y - np.polyval(lin, t)), 0.022, f0, 0, lin[1], lin[0]], maxfev=50000
            )
        except RuntimeError:
            continue
        r = float(np.sum((y - m(t, *p)) ** 2))
        if best is None or r < best[0]:
            best = (r, p, np.sqrt(np.abs(np.diag(cv))), m)
    assert best is not None
    return best[1], best[2], best[3]


def _mirror_ring(stem: str, start: float = 0.058, stop: float = 0.13):
    """Ring-down of the mirror from the fringe rate (channel A): |V e^(-t/tau) cos(2 pi f t + phi) + v_ramp|."""
    from scipy.optimize import curve_fit

    tr = load_trace(DATA / f"{stem}.csv")
    a, b, fs = tr.channel("A"), tr.channel("B"), tr.fs
    t = np.arange(a.size) / fs
    k = int(np.argmax(np.abs(np.diff(b))))
    s = sosfiltfilt(butter(4, 80, "highpass", fs=fs, output="sos"), a - a.mean())
    n = int(2e-3 * fs)
    st = ShortTimeFFT(gaussian(n, n / 6), hop=max(n // 8, 1), fs=fs, mfft=8 * n)
    p_ = np.abs(st.stft(s)) ** 2
    f = st.f
    ts = st.t(s.size)
    p_[f < 150] = 0
    fd = f[np.argmax(p_, 0)]
    m = (ts > t[k] + start) & (ts < min(t[k] + stop, t[-1] - 0.002))
    tt = ts[m] - t[k] - start
    v = fd[m] * LAM / 2

    def model(x, vv, tau, fr, ph, vr):
        return np.abs(vv * np.exp(-x / tau) * np.cos(2 * np.pi * fr * x + ph) + vr)

    p, cv = curve_fit(model, tt, v, p0=[v.max(), 0.025, 87, 0, 2.5e-4], maxfev=50000)
    return p, np.sqrt(np.diag(cv)), (ts[m] - t[k], v, model(tt, *p))


def ringing() -> list[dict]:
    """Speaker resonance from the sawtooth flyback: ringing of the drive voltage (B) and of the mirror (A)."""
    stems = ["vib_1Hz_saw_2p0Vpp_main_1", "vib_1Hz_saw_2p0Vpp_main_2"]
    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for j, stem in enumerate(stems):
        tr = load_trace(DATA / f"{stem}.csv")
        b, fs = tr.channel("B"), tr.fs
        t = np.arange(b.size) / fs
        k = int(np.argmax(np.abs(np.diff(b))))
        s = slice(k + 2, min(k + 2 + int(0.15 * fs), b.size))
        ts, y = t[s] - t[k], b[s]
        p, e, m = _ring_fit(ts - ts[0], y)
        q = np.pi * p[2] * abs(p[1])
        rows.append(
            {
                "file": stem,
                "window": "+2 to +150 ms",
                "signal": "B",
                "f": p[2],
                "f_err": e[2],
                "tau_ms": abs(p[1]) * 1e3,
                "Q": q,
            }
        )
        if j == 0:
            w = (t > t[k] - 0.02) & (t < t[k] + 0.15)
            ax1.plot((t[w] - t[k]) * 1e3, b[w], color="C3", lw=1, label="drive voltage (channel B)")
            ax1.plot(
                ts * 1e3,
                m(ts - ts[0], *p),
                "k--",
                lw=1,
                label=f"damped cosine: {p[2]:.1f} Hz, τ = {abs(p[1]) * 1e3:.1f} ms",
            )
        for lo, hi in RING_WINDOWS:
            s = slice(k + int(lo * fs), min(k + int(hi * fs), b.size))
            tw, yw = t[s] - t[s][0], b[s]
            p, e, _ = _ring_fit(tw, yw)
            rows.append(
                {
                    "file": stem,
                    "window": f"+{lo * 1e3:.0f} to +{hi * 1e3:.0f} ms",
                    "signal": "B",
                    "f": p[2],
                    "f_err": e[2],
                    "tau_ms": abs(p[1]) * 1e3,
                    "Q": np.pi * p[2] * abs(p[1]),
                    "t_mid_ms": (lo + hi) / 2 * 1e3,
                    "amp_mV": abs(p[0]) * 1e3,
                }
            )
        pa, ea, _ = _mirror_ring(stem)
        rows.append(
            {
                "file": stem,
                "window": "+58 to +130 ms",
                "signal": "A (mirror)",
                "f": pa[2],
                "f_err": ea[2],
                "tau_ms": pa[1] * 1e3,
                "Q": np.pi * pa[2] * pa[1],
                "t_mid_ms": 94.0,
            }
        )
    bw = [r for r in rows if r["signal"] == "B" and "t_mid_ms" in r]
    aw = [r for r in rows if r["signal"] == "A (mirror)"]
    ax2.errorbar(
        [r["t_mid_ms"] for r in bw],
        [r["f"] for r in bw],
        yerr=[r["f_err"] for r in bw],
        fmt="o",
        color="C3",
        label="drive voltage (B), 2 captures",
    )
    ax2.errorbar(
        [float(r["t_mid_ms"]) + 2 for r in aw],
        [r["f"] for r in aw],
        yerr=[r["f_err"] for r in aw],
        fmt="s",
        color="C0",
        label="mirror (A, fringe rate), 2 captures",
    )
    ax1.set_xlabel("Time after flyback (ms)")
    ax1.set_ylabel("Drive voltage (V)")
    ax1.legend(loc="lower right", fontsize=9)
    ax2.set_xlabel("Centre of fit window after flyback (ms)")
    ax2.set_ylabel("Ringing frequency (Hz)")
    ax2.set_ylim(85, 93)
    ax2.legend(loc="lower right", fontsize=9)
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(
            0.03,
            0.04 if ax is ax1 else 0.95,
            lab,
            transform=ax.transAxes,
            va="bottom" if ax is ax1 else "top",
            ha="left",
        )
        minor(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig21_speaker_ringing.png")
    plt.close(fig)
    return rows


def main() -> None:
    paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = summary_rows()
    with (OUT / "section6_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print("== per level ==")
    g = group_stats(rows, lambda r: (r["group"], r["f_Hz"], r["drive_Vpp"]))
    for k in sorted(g, key=lambda k: (k[0], k[1], k[2])):
        lst = g[k]
        col = lambda key, lst=lst: np.array([r[key] for r in lst])  # noqa: E731
        sd = lambda a: np.std(a, ddof=1) if a.size > 1 else 0.0  # noqa: E731
        print(
            f"{k[0]:4s} {k[1]:.1f} Hz {k[2]:4.1f} V (n={len(lst):2d}): B {col('B_Vpp').mean():.3f} V | "
            f"X fit {col('X_fit_um').mean():8.3f} ± {sd(col('X_fit_um')):.3f} | int {col('X_int_um').mean():8.3f} | "
            f"count {col('X_count_um').mean():8.3f} ± {sd(col('X_count_um')):.2f} | "
            f"v {col('v_peak_mm_s').mean():.4f} mm/s | "
            f"lag {col('lag_deg').mean():5.1f}° ± {sd(col('lag_deg')):.1f} | win {col('window_ms').mean():.1f} ms"
        )
    fig_example()
    d = fig_drive(rows)
    print("== drive ==", {k: (np.round(v, 3) if isinstance(v, np.ndarray) else round(v, 3)) for k, v in d.items()})
    fq = fig_frequency(rows)
    print("== frequency ==", {k: tuple(round(float(x), 4) for x in v) for k, v in fq.items()})
    wrows = window_study()
    with (OUT / "section6_windows.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in wrows for k in r})
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(wrows)
    print("== windows ==")
    for r in wrows:
        if r["ok"]:
            print(
                f"{r['label']:16s} T={r['window_ms']:3d} ms (T_opt {r['T_opt_ms']:.1f}): "
                f"X {r['X_um']:8.3f} ± {r['X_err_um']:.3f} "
                f"rms {r['rms_resid_um_s']:7.1f} µm/s resolved {r['frac_resolved'] * 100:3.0f}%"
            )
        else:
            print(f"{r['label']:16s} T={r['window_ms']:3d} ms: failed")
    fig_windows(wrows)
    fd = fig_fading()
    print("== fading profiles (mV) ==")
    for v, (_, prof) in fd.items():
        print(f"{v} V: " + " ".join(f"{p * 1e3:.0f}" for p in prof))
    tri, _ = triangle()
    print(
        "== triangle ==",
        [tuple(round(x * 1e3, 4) if j < 2 else round(x * 1e6, 2) for j, x in enumerate(r)) for r in tri],
    )
    print("== ringing (sawtooth flyback) ==")
    for r in ringing():
        print(
            f"{r['file']} {r['signal']:10s} {r['window']:16s}: f = {r['f']:.2f} ± {r['f_err']:.2f} Hz, "
            f"tau = {r['tau_ms']:.1f} ms, Q = {r['Q']:.2f}"
            + (f", amplitude {r['amp_mV']:.0f} mV" if "amp_mV" in r else "")
        )


if __name__ == "__main__":
    main()
