"""Calibrate the delay-line micrometer by counting fringes (`fibre-count`).

With the modulator OFF and the FP laser on, turn the micrometer while the
detector is recorded. Every fringe that passes is one wavelength of path
difference, so

    fringes = (path change) / λ,   path change = s × (micrometer change),

where s = 1 for a single pass through the delay line's air gap (what the analysis
assumes) and s = 2 for a double pass. This measures s, i.e. the scale of every
path difference you quote.

Turn in ONE direction only. Pauses, stops and uneven speed are fine: fringes are
counted as hysteresis crossings of the mid level, so a still trace adds nothing.
Reversing would count fringes twice (the direction can't be seen in the signal).

    uv run fibre-count --pico                         # 40 s recording; asks for start/end readings
    uv run fibre-count recording.csv --start 12.60 --end 12.80

Each run is appended to data/fringecount.csv; the summary averages all runs.
Work near an FP contrast peak (e.g. 12.6-12.8 mm) so the fringes stay large.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import setup_a
from .io import data_path, find_data, graph_path, load_trace, save_trace, utf8_output

LOG_COLUMNS = [
    "when", "file", "start_mm", "end_mm", "fringes", "seconds", "pass_factor", "um_per_fringe", "lo_v", "hi_v",
]  # fmt: skip


@dataclass
class Count:
    crossings: int
    fringes: float  # crossings / 2, so half-fringe resolution
    threshold: float
    crossing_idx: np.ndarray
    smallest_swing: float  # smallest local fringe swing seen, as a fraction of the overall swing
    lo: float  # fringe minimum and maximum used for the mid level
    hi: float


def count_crossings(
    v: np.ndarray, hysteresis: float = 0.3, smooth: int = 3, ref: tuple[float, float] | None = None
) -> Count:
    """Fringes as mid-level crossings with hysteresis.

    The level must go from below mid - h to above mid + h (or back) to count, with
    h = hysteresis × half the fringe swing, so noise and slow drift during pauses
    are not counted. The swing is taken from the recording itself, or from `ref`
    (min, max of a recording that did move) - needed for a still recording, whose
    own swing is just its drift."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    vs = uniform_filter1d(v, size=smooth, mode="nearest") if smooth > 1 else v
    lo, hi = ref if ref is not None else (np.percentile(vs, 1), np.percentile(vs, 99))
    mid, h = (lo + hi) / 2, hysteresis * (hi - lo) / 2
    state, idx = 0, []
    for i, val in enumerate(vs):
        if state <= 0 and val > mid + h:
            if state != 0:
                idx.append(i)
            state = 1
        elif state >= 0 and val < mid - h:
            if state != 0:
                idx.append(i)
            state = -1
    idx_arr = np.array(idx, dtype=int)
    # How small did the fringes get between crossings? (warns if contrast collapsed)
    swings = [np.ptp(vs[a:b]) for a, b in itertools.pairwise(idx_arr) if b > a]
    smallest = float(min(swings) / (hi - lo)) if swings else 1.0
    return Count(len(idx), len(idx) / 2, h, idx_arr, smallest, float(lo), float(hi))


def drift_fringes(v: np.ndarray, lo: float, hi: float, smooth: int = 25) -> float:
    """Fringes of drift in a still recording (less than about half a fringe), from
    the phase implied by the signal level within the known fringe swing. A lower
    bound: the direction of the phase change can't be seen."""
    vs = uniform_filter1d(np.asarray(v, dtype=float), size=smooth, mode="nearest")
    phase = np.arccos(np.clip(2 * (vs - (lo + hi) / 2) / (hi - lo), -1, 1))
    return float(np.ptp(phase) / (2 * np.pi))


def _last_reference(log: Path) -> tuple[float, float] | None:
    """Fringe min/max from the most recent run in which the micrometer moved."""
    if not log.exists():
        return None
    with log.open(encoding="utf-8") as f:
        moving = [r for r in csv.DictReader(f) if r.get("lo_v") and np.isfinite(float(r["pass_factor"]))]
    return (float(moving[-1]["lo_v"]), float(moving[-1]["hi_v"])) if moving else None


def pass_factor(fringes: float, start_mm: float, end_mm: float, wavelength_nm: float) -> float:
    """Path change per micrometer change: 1 = single pass, 2 = double pass."""
    return fringes * wavelength_nm * 1e-6 / abs(end_mm - start_mm)


def _ask_float(prompt: str) -> float:
    while True:
        s = input(prompt).strip()
        try:
            return float(s)
        except ValueError:
            print("  type the micrometer reading in mm, e.g. 12.600")


def _append_log(row: dict[str, str]) -> Path:
    log = data_path("fringecount.csv")
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return log


def _summary(log: Path, wavelength_nm: float) -> None:
    with log.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    s = np.array([float(r["pass_factor"]) for r in rows])
    s = s[np.isfinite(s)]  # drift runs (micrometer not moved) have no pass factor
    if s.size == 0:
        return
    err = s.std(ddof=1) / np.sqrt(s.size) if s.size > 1 else np.nan
    print(f"\nAll {s.size} run(s) in {log}: pass factor = {s.mean():.4f} ± {err:.4f}")
    print(f"  per division (10 µm): {s.mean() * 10 / (wavelength_nm * 1e-3):.2f} fringes")
    verdict = "single pass" if abs(s.mean() - 1) < abs(s.mean() - 2) else "DOUBLE pass"
    print(f"  -> {verdict}; path differences should be scaled by {s.mean():.4f} (analysis assumes 1)")


def main(argv: list[str] | None = None) -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", nargs="?", help="an existing recording (omit with --pico)")
    p.add_argument("--pico", action="store_true", help="record from the PicoScope now")
    p.add_argument("--duration", type=float, default=40.0, help="recording length in s (--pico; default 40)")
    p.add_argument("--start", type=float, help="micrometer reading before turning, mm (asked if omitted)")
    p.add_argument("--end", type=float, help="micrometer reading after turning, mm (asked if omitted)")
    p.add_argument("-c", "--channel", default="A", help="detector channel (default A)")
    p.add_argument("--wavelength", type=float, default=setup_a.FP_CENTRE_NM, help="nm (default: FP laser)")
    p.add_argument("--hysteresis", type=float, default=0.3, help="fraction of the half swing (default 0.3)")
    a = p.parse_args(argv)
    if not a.pico and not a.trace:
        p.error("give a recording file, or --pico to record now")

    if a.pico:
        from .capture import PicoError, PicoScope, autorange_capture

        start = a.start if a.start is not None else _ask_float("Micrometer reading now (mm): ")
        try:
            with PicoScope() as scope:
                # Set the range from a quick capture, so the long one only runs once.
                quick = autorange_capture(scope, [a.channel.upper()], 2000, 1e-4)
                scope.configure(quick.ranges)
                # Sample intervals come in fixed steps: pick one long enough for the whole duration.
                _, dt, max_n = scope.find_timebase(2 * a.duration / 8000)
                n = min(max_n, int(np.ceil(a.duration / dt)))
                input(
                    f"Recording will last {n * dt:.0f} s. Press Enter, then turn in ONE direction "
                    "(pauses are fine; finish before the time is up)..."
                )
                t0 = time.time()
                cap = scope.capture(n, dt)
                print(f"  recorded {cap.t[-1] + cap.dt:.1f} s ({time.time() - t0:.0f} s wall clock)")
        except PicoError as e:
            raise SystemExit(f"error: {e}") from None
        end = a.end if a.end is not None else _ask_float("Micrometer reading now (mm): ")
        path = data_path(f"fringecount_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        tr = cap.trace()
        save_trace(path, tr.t, tr.channels)
        v, t = cap.volts[a.channel.upper()], cap.t
        if cap.clipped:
            print("  warning: the signal clipped during the recording; fringes near the top may be miscounted")
    else:
        path = find_data(a.trace)
        tr = load_trace(path)
        v, t = tr.channel(a.channel), tr.t - tr.t[0]
        start = a.start if a.start is not None else _ask_float("Start reading (mm): ")
        end = a.end if a.end is not None else _ask_float("End reading (mm): ")

    still = abs(end - start) < 1e-9
    ref = _last_reference(data_path("fringecount.csv")) if still else None
    c = count_crossings(v, a.hysteresis, ref=ref)
    print(f"{path.name}: {c.fringes:g} fringes ({c.crossings} crossings), micrometer {start:g} -> {end:g} mm")
    if c.crossing_idx.size > 1 and np.min(np.diff(c.crossing_idx)) < 4:
        print(
            "  warning: at times the fringes came faster than the sample rate can follow; "
            "turn more slowly (or use a shorter --duration) and repeat"
        )
    if c.smallest_swing < 0.3:
        print(
            f"  warning: fringes shrank to {c.smallest_swing:.0%} of full size somewhere; "
            "some may be missed. Stay nearer an FP contrast peak."
        )
    if still:
        if ref is None:
            print("  micrometer didn't move, but there's no moving run in the log yet to size a fringe against;")
            print("  do a moving run first, then repeat this drift recording")
        else:
            d = drift_fringes(v, *ref) if c.fringes == 0 else c.fringes
            print(f"  micrometer didn't move: about {d:.2f} fringes of drift in {t[-1]:.0f} s")
        s, um = np.nan, np.nan
    else:
        s = pass_factor(c.fringes, start, end, a.wavelength)
        um = abs(end - start) * 1000 / c.fringes if c.fringes else np.nan
        frac = 0.5 / c.fringes if c.fringes else np.nan
        print(f"  {um:.4f} µm of micrometer per fringe; pass factor {s:.4f} (±{s * frac:.4f} from ±½ fringe)")
        print("  single pass expects 1, double pass 2")

    log = _append_log(
        {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file": str(path),
            "start_mm": f"{start:g}",
            "end_mm": f"{end:g}",
            "fringes": f"{c.fringes:g}",
            "seconds": f"{t[-1]:.2f}",
            "pass_factor": f"{s:.6g}",
            "um_per_fringe": f"{um:.6g}",
            "lo_v": f"{c.lo:.6g}",
            "hi_v": f"{c.hi:.6g}",
        }
    )
    _summary(log, a.wavelength)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, v, lw=0.6)
    if c.crossing_idx.size:
        ax.plot(t[c.crossing_idx], v[c.crossing_idx], "r|", ms=12, label=f"{c.crossings} crossings")
        ax.legend(loc="upper right")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("detector [V]")
    ax.set_title(f"{c.fringes:g} fringes for {start:g} → {end:g} mm")
    fig.tight_layout()
    png = graph_path(path, ".png")
    fig.savefig(png, dpi=130)
    print(f"Plot with each counted crossing marked: {png}")


if __name__ == "__main__":
    main()
