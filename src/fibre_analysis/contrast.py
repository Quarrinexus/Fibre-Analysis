"""Fringe contrast from a single oscilloscope trace (handout Eq. 19).

Estimators:

* "period" (default when the sample rate is known, `contrast_per_period`): the
  trace is cut into whole modulator periods, Eq. 19 is applied to each, and the
  results are averaged. The error is their spread / sqrt(n). Each period covers a
  full sweep, so it is unaffected by drift or glitches elsewhere in the trace, and
  the error bar reflects real variation within the capture. The period comes from
  the drive signal (channel B) if given, otherwise from the detector signal itself
  (fine at ~1 fringe per sweep; less exact with several fringes per sweep).
* "peak": whole-trace smoothed (Imax - Imin) / (Imax + Imin) using percentiles
  rather than the raw extremes, so single noise spikes don't count. This is
  Eq. 19 and is right whatever the number of fringes per sweep, as long as each
  sweep covers at least one full fringe. Slightly biased upwards when the
  fringes are no bigger than the noise (C below ~0.02).
* "rms": amplitude from the standard deviation of the trace, with the
  white-noise variance (estimated from sample-to-sample differences) removed.
  For sinusoidal fringes A = sqrt(2 * var), C = A / mean. This is unbiased at
  low contrast, where max/min would just measure the noise. But it assumes a
  whole number of fringes in the trace: with e.g. 2.3 fringes per sweep the
  mean is off and C can be wrong by ~10%. Use it only as a low-contrast check.

For "peak" and "rms" the uncertainty is the scatter between 8 equal chunks of
the trace. That is unreliable when a chunk is shorter than one fringe, which is why
"period" is preferred. Subtract the detector dark level (`dark`, measured with the source off) first:
any offset changes the contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from .io import utf8_output


@dataclass
class ContrastResult:
    contrast: float
    error: float
    i_max: float
    i_min: float
    mean: float
    method: str
    n: int = 1  # number of periods averaged (period method)

    def __str__(self) -> str:
        return (
            f"C = {self.contrast:.4f} ± {self.error:.4f} ({self.method}); "
            f"Imax = {self.i_max:.4g} V, Imin = {self.i_min:.4g} V, mean = {self.mean:.4g} V"
        )


def _noise_var(v: np.ndarray) -> float:
    # White noise adds 2σ² to the variance of first differences; a smooth fringe
    # signal adds little if it is well sampled. Use the median-based estimate so
    # fast fringe edges don't dominate.
    d = np.diff(v)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    return float(sigma**2)


def _contrast_rms(v: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(v))
    var = max(float(np.var(v)) - _noise_var(v), 0.0)
    amp = np.sqrt(2 * var)
    return amp / mean, mean + amp, mean - amp, mean


def _contrast_peak(v: np.ndarray, smooth: int, pct: float) -> tuple[float, float, float, float]:
    vs = uniform_filter1d(v, size=max(smooth, 1), mode="nearest") if smooth > 1 else v
    i_max = float(np.percentile(vs, 100 - pct))
    i_min = float(np.percentile(vs, pct))
    return (i_max - i_min) / (i_max + i_min), i_max, i_min, float(np.mean(v))


def contrast(
    v: np.ndarray,
    method: str = "peak",
    dark: float = 0.0,
    smooth: int = 5,
    pct: float = 0.5,
    n_chunks: int = 8,
) -> ContrastResult:
    """Fringe contrast of a trace `v` (volts) containing several fringes."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)] - dark
    if v.size < 10:
        raise ValueError("Trace too short")

    def one(x: np.ndarray) -> tuple[float, float, float, float]:
        if method == "rms":
            return _contrast_rms(x)
        if method == "peak":
            return _contrast_peak(x, smooth, pct)
        raise ValueError(f"Unknown method {method!r}")

    c, i_max, i_min, mean = one(v)
    err = np.nan
    if n_chunks > 1 and v.size >= 50 * n_chunks:
        cs = [one(chunk)[0] for chunk in np.array_split(v, n_chunks)]
        err = float(np.std(cs, ddof=1) / np.sqrt(n_chunks))
    return ContrastResult(float(c), err, i_max, i_min, mean, method)


def sweep_period(fs: float, v: np.ndarray | None = None, drive: np.ndarray | None = None) -> float:
    """Modulator period in s: from the drive's fundamental, else the detector's
    strongest frequency (the fringe frequency)."""
    sig = drive if drive is not None else v
    if sig is None:
        raise ValueError("need a drive or detector signal")
    x = np.nan_to_num(np.asarray(sig, dtype=float))
    x = x - x.mean()
    n = 32 * x.size
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[f < 5] = 0
    return float(1 / f[np.argmax(spec)])


def contrast_per_period(
    v: np.ndarray,
    fs: float,
    drive: np.ndarray | None = None,
    dark: float = 0.0,
    smooth: int = 5,
    pct: float = 0.5,
) -> ContrastResult:
    """Eq. 19 in each whole modulator period, averaged; error = spread / sqrt(n).
    Falls back to the whole-trace estimate if fewer than 2 periods fit."""
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)) or (drive is not None and not np.all(np.isfinite(drive))):
        return contrast(v, "peak", dark=dark, smooth=smooth, pct=pct)
    v = v - dark
    n_per = round(sweep_period(fs, v, drive) * fs)
    n = v.size // n_per if n_per > 0 else 0
    if n < 2 or n_per < 20:
        whole = contrast(v, "peak", smooth=smooth, pct=pct)
        whole.method = "peak (fewer than 2 periods)"
        return whole
    vs = uniform_filter1d(v, size=max(smooth, 1), mode="nearest") if smooth > 1 else v
    cs, his, los = [], [], []
    for k in range(n):
        w = vs[k * n_per : (k + 1) * n_per]
        hi, lo = float(np.percentile(w, 100 - pct)), float(np.percentile(w, pct))
        cs.append((hi - lo) / (hi + lo))
        his.append(hi)
        los.append(lo)
    c = np.array(cs)
    return ContrastResult(
        contrast=float(c.mean()),
        error=float(c.std(ddof=1) / np.sqrt(n)),
        i_max=float(np.mean(his)),
        i_min=float(np.mean(los)),
        mean=float(np.mean(v)),
        method="period",
        n=n,
    )


def expected_max_contrast(i1: float, i2: float) -> float:
    """Best possible contrast for arm intensities I1, I2 (Eq. 20 with |FT| = 1)."""
    return 2 * np.sqrt(i1 * i2) / (i1 + i2)


def main(argv: list[str] | None = None) -> None:
    utf8_output()
    import argparse

    from .io import load_trace

    p = argparse.ArgumentParser(description="Fringe contrast of one or more scope traces.")
    p.add_argument("files", nargs="+")
    p.add_argument("-c", "--channel", help="channel name or letter (default: first)")
    p.add_argument("--dark", type=float, default=0.0, help="detector dark level in V")
    p.add_argument("--drive", help="drive channel for the period method (default: B if present)")
    p.add_argument("--method", choices=["period", "peak", "rms", "all"], default="all")
    a = p.parse_args(argv)

    methods = ["period", "peak", "rms"] if a.method == "all" else [a.method]
    for f in a.files:
        tr = load_trace(f)
        v = tr.channel(a.channel)
        drive = None
        if a.drive:
            drive = tr.channel(a.drive)
        elif len(tr.channels) > 1 and (a.channel or "A").upper()[-1] != "B":
            try:
                drive = tr.channel("B")
            except KeyError:
                drive = None
        for m in methods:
            if m == "period":
                r = contrast_per_period(v, tr.fs, drive, dark=a.dark)
                src = "drive" if drive is not None else "detector"
                print(f"{f}: {r}  [{r.n} periods, timed from {src}]")
            else:
                print(f"{f}: {contrast(v, method=m, dark=a.dark)}")


if __name__ == "__main__":
    main()
