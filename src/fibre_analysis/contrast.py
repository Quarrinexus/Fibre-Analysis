"""Fringe contrast from a single oscilloscope trace (handout Eq. 19).

Two estimators:

* "peak" (default): smoothed (Imax - Imin) / (Imax + Imin) using percentiles
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

The uncertainty is the scatter between equal chunks of the trace (std / sqrt(n)).
Subtract the detector dark level (`dark`, measured with the source off) first:
any offset changes the contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class ContrastResult:
    contrast: float
    error: float
    i_max: float
    i_min: float
    mean: float
    method: str

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


def expected_max_contrast(i1: float, i2: float) -> float:
    """Best possible contrast for arm intensities I1, I2 (Eq. 20 with |FT| = 1)."""
    return 2 * np.sqrt(i1 * i2) / (i1 + i2)


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .io import load_trace

    p = argparse.ArgumentParser(description="Fringe contrast of one or more scope traces.")
    p.add_argument("files", nargs="+")
    p.add_argument("-c", "--channel", help="channel name or letter (default: first)")
    p.add_argument("--dark", type=float, default=0.0, help="detector dark level in V")
    p.add_argument("--method", choices=["rms", "peak", "both"], default="both")
    a = p.parse_args(argv)

    methods = ["rms", "peak"] if a.method == "both" else [a.method]
    for f in a.files:
        v = load_trace(f).channel(a.channel)
        for m in methods:
            print(f"{f}: {contrast(v, method=m, dark=a.dark)}")


if __name__ == "__main__":
    main()
