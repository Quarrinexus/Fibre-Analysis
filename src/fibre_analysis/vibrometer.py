"""Loudspeaker velocity and amplitude from the vibrometer trace (section 6).

The mirror is in one arm, so light travels to it and back: moving it by Δx
changes the path difference by 2Δx. Fringe frequency f = 2|v|/λ.

Steps:
1. high-pass the detector signal to remove slow drifts;
2. short-time FFT (scipy ShortTimeFFT, Gaussian window) -> fringe frequency
   in each time slice -> |v(t)| = f λ / 2;
3. fit |V sin(2π f_d t + φ)| to |v(t)| (f_d from the drive channel), giving the
   peak velocity V and displacement amplitude X = V / (2π f_d);
4. fringes give only speed, not direction: the sign is taken from the fit, with
   the overall direction chosen to match d(drive)/dt (true well below the
   speaker resonance, where displacement follows the drive voltage);
5. integrate v(t) for x(t);
6. independent check: count fringes (zero crossings with hysteresis).
   Each half cycle moves the mirror 2X -> 4X/λ fringes.

Near the turning points the speed is too low to resolve within one window
(f < 1/window); those slices are left out of the fit.

    uv run fibre-vibro trace.csv --signal A --drive B --window 0.01   # optional; chosen automatically otherwise
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import ShortTimeFFT, butter, sosfiltfilt
from scipy.signal.windows import gaussian

from . import setup_a
from .io import graph_path, load_trace, utf8_output


@dataclass
class VibroResult:
    t: np.ndarray  # STFT slice times, s
    freq: np.ndarray  # fringe frequency per slice, Hz (NaN where unresolved)
    speed: np.ndarray  # |v|, m/s
    velocity: np.ndarray  # signed v, m/s (gaps filled from fit)
    displacement: np.ndarray  # m
    drive_freq: float
    v_peak: float
    v_peak_err: float
    phase: float
    amplitude: float
    amplitude_err: float
    amplitude_counting: float
    spectrogram: np.ndarray
    spec_f: np.ndarray
    window: float

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Drive frequency          {self.drive_freq:.3f} Hz   (STFT window {self.window * 1e3:.1f} ms)",
                f"Peak velocity            {self.v_peak * 1e3:.4f} ± {self.v_peak_err * 1e3:.4f} mm/s",
                f"Amplitude (fit, V/2πf)   {self.amplitude * 1e6:.2f} ± {self.amplitude_err * 1e6:.2f} µm",
                f"Amplitude (integrated)   {np.ptp(self.displacement) / 2 * 1e6:.2f} µm",
                f"Amplitude (fringe count) {self.amplitude_counting * 1e6:.2f} µm",
            ]
        )


def _dominant_freq(x: np.ndarray, fs: float, fmin: float = 0.0) -> float:
    x = x - np.mean(x)
    n = 8 * x.size
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), n=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[f <= fmin] = 0
    return float(f[np.argmax(spec)])


def count_fringes(v: np.ndarray, hysteresis: float = 0.3) -> int:
    """Number of fringes = zero crossings / 2, ignoring wiggles smaller than hysteresis*std."""
    x = v - np.mean(v)
    th = hysteresis * np.std(x)
    state = 0
    crossings = 0
    for val in x:
        if state <= 0 and val > th:
            crossings += state != 0
            state = 1
        elif state >= 0 and val < -th:
            crossings += state != 0
            state = -1
    return crossings // 2


def analyse(
    signal: np.ndarray,
    fs: float,
    drive: np.ndarray | None = None,
    drive_freq: float | None = None,
    wavelength: float = setup_a.FP_CENTRE_NM * 1e-9,
    window: float | None = None,
    fmin: float | None = None,
    power_threshold: float = 0.05,
) -> VibroResult:
    signal = np.nan_to_num(signal - np.nanmean(signal))
    duration = signal.size / fs
    if drive_freq is None and drive is not None:
        drive_freq = _dominant_freq(np.nan_to_num(drive), fs)

    # Fringe count with only a light high-pass (slow drifts out, slow fringes kept).
    s_count = signal
    n_fringes = None
    if drive_freq is not None:
        sos_c = butter(2, max(3 * drive_freq, 0.5), btype="highpass", fs=fs, output="sos")
        s_count = sosfiltfilt(sos_c, signal)
        n_fringes = count_fringes(s_count)

    if window is None:
        window = 0.01
        if n_fringes and drive_freq is not None:
            # Peak fringe frequency of a sinusoidal motion is ~π/2 x the mean. The
            # window must hold enough fringes at well below the peak speed (fmin =
            # 2/window ~ peak/4), but stay short compared with the drive period.
            f_peak = np.pi / 2 * n_fringes / duration
            window = float(np.clip(8 / f_peak, 2e-3, 0.25 / drive_freq))
    nperseg = max(int(window * fs), 16)
    if fmin is None:
        fmin = 2.0 / window  # need ~2 fringes per window to resolve
    sos = butter(4, fmin / 2, btype="highpass", fs=fs, output="sos")
    s = sosfiltfilt(sos, signal)

    win = gaussian(nperseg, std=nperseg / 6, sym=True)
    stft = ShortTimeFFT(win, hop=max(nperseg // 8, 1), fs=fs, scale_to="magnitude")
    S = np.abs(stft.stft(s)) ** 2
    f = stft.f
    t = stft.t(s.size)
    keep_t = (t >= 0) & (t <= (s.size - 1) / fs)
    S, t = S[:, keep_t], t[keep_t]

    S_search = S.copy()
    S_search[f < fmin] = 0
    idx = np.argmax(S_search, axis=0)
    peak_pow = S_search[idx, np.arange(S.shape[1])]
    df = f[1] - f[0]
    freq = f[idx].astype(float)
    inner = (idx > 0) & (idx < len(f) - 1)
    cols = np.arange(S.shape[1])[inner]
    y0 = np.log(S_search[idx[inner] - 1, cols] + 1e-300)
    y1 = np.log(S_search[idx[inner], cols] + 1e-300)
    y2 = np.log(S_search[idx[inner] + 1, cols] + 1e-300)
    den = y0 - 2 * y1 + y2
    freq[inner] += np.where(den != 0, 0.5 * (y0 - y2) / den, 0) * df
    freq[peak_pow < power_threshold * np.max(peak_pow)] = np.nan
    # A peak stuck at the lowest allowed bin means the true frequency is below fmin (turning point).
    freq[idx <= np.searchsorted(f, fmin) + 1] = np.nan
    speed = freq * wavelength / 2
    if np.sum(np.isfinite(speed)) < 10:
        raise ValueError(
            f"Fringe frequency never above {fmin:.0f} Hz with a {window * 1e3:.0f} ms window - "
            "the motion is too slow for it; try a longer --window"
        )

    if drive_freq is None:
        filled = np.nan_to_num(speed)
        drive_freq = _dominant_freq(filled, 1 / (t[1] - t[0]), fmin=0.1) / 2

    w = 2 * np.pi * drive_freq

    def model(tt, vpk, phi):
        return np.abs(vpk * np.sin(w * tt + phi))

    ok = np.isfinite(speed)
    vmax = float(np.nanpercentile(speed, 95))
    phis = np.linspace(0, np.pi, 64, endpoint=False)
    sse = [np.sum((model(t[ok], vmax, ph) - speed[ok]) ** 2) for ph in phis]
    popt, pcov = curve_fit(model, t[ok], speed[ok], p0=[vmax, phis[int(np.argmin(sse))]])
    vpk, phi = abs(popt[0]), popt[1]
    vpk_err = float(np.sqrt(pcov[0, 0]))

    sign = np.sign(np.sin(w * t + phi))
    if drive is not None:
        tt_full = np.arange(drive.size) / fs
        dd = np.gradient(np.interp(t, tt_full, np.nan_to_num(drive)), t)
        if np.sum(sign * dd) < 0:
            sign = -sign
            phi += np.pi
    velocity = sign * np.where(ok, speed, model(t, vpk, phi))
    disp = np.concatenate([[0], np.cumsum(0.5 * (velocity[1:] + velocity[:-1]) * np.diff(t))])
    disp -= np.mean(disp)

    if n_fringes is None:
        sos_c = butter(2, max(3 * drive_freq, 0.5), btype="highpass", fs=fs, output="sos")
        n_fringes = count_fringes(sosfiltfilt(sos_c, signal))
    n_half_cycles = 2 * drive_freq * duration
    amp_count = n_fringes / n_half_cycles * wavelength / 4

    return VibroResult(
        t=t,
        freq=freq,
        speed=speed,
        velocity=velocity,
        displacement=disp,
        drive_freq=float(drive_freq),
        v_peak=float(vpk),
        v_peak_err=vpk_err,
        phase=float(phi),
        amplitude=float(vpk / w),
        amplitude_err=vpk_err / w,
        amplitude_counting=float(amp_count),
        spectrogram=S,
        spec_f=f,
        window=window,
    )


def main(argv: list[str] | None = None) -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--signal", default="A", help="detector channel (default A)")
    p.add_argument("--drive", default="B", help="function generator channel (default B; 'none' if not recorded)")
    p.add_argument("--drive-freq", type=float, help="Hz, if no drive channel")
    p.add_argument("--wavelength", type=float, default=setup_a.FP_CENTRE_NM, help="nm (default: FP laser)")
    p.add_argument("--window", type=float, help="STFT window length in s (default: chosen from the fringe rate)")
    p.add_argument("--plot", help="figure file (default: output/<section>/<trace>_vibro.png)")
    p.add_argument("--show", action="store_true")
    a = p.parse_args(argv)

    import matplotlib

    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = load_trace(a.trace)
    drive = None if a.drive.lower() == "none" else tr.channel(a.drive)
    res = analyse(tr.channel(a.signal), tr.fs, drive, a.drive_freq, a.wavelength * 1e-9, a.window)
    print(res)

    t0 = tr.t[0]
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fmax = np.nanmax(res.freq) * 1.3 if np.any(np.isfinite(res.freq)) else res.spec_f[-1]
    fm = res.spec_f <= fmax
    axs[0].pcolormesh(res.t + t0, res.spec_f[fm], np.log10(res.spectrogram[fm] + 1e-20), shading="auto")
    axs[0].plot(res.t + t0, res.freq, "w.", ms=2)
    axs[0].set_ylabel("fringe frequency [Hz]")
    axs[1].plot(res.t + t0, res.speed * 1e3, ".", ms=3, label="|v| from STFT")
    axs[1].plot(res.t + t0, res.velocity * 1e3, "-", lw=1, label="signed v")
    vm = np.abs(res.v_peak * np.sin(2 * np.pi * res.drive_freq * res.t + res.phase))
    axs[1].plot(res.t + t0, vm * 1e3, "--", lw=1, label="fit")
    axs[1].set_ylabel("velocity [mm/s]")
    axs[1].legend(fontsize=8)
    axs[2].plot(res.t + t0, res.displacement * 1e6)
    axs[2].set_ylabel("displacement [µm]")
    axs[2].set_xlabel("time [s]")
    if drive is not None:
        ax = axs[2].twinx()
        ax.plot(tr.t, drive, color="C3", alpha=0.4, lw=0.8)
        ax.set_ylabel("drive [V]", color="C3")
    fig.tight_layout()
    out = a.plot or graph_path(a.trace, "_vibro.png")
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
