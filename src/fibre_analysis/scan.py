"""Interactive logger: contrast vs. something you change by hand.

Use it for delay-line scans (sections 3) and for gain scans (sections 4-5).

Typical use with the PicoScope software saving CSVs into a folder:

    uv run fibre-scan --watch ~/pico_exports --out sld_scan.csv --unit div

For each point: set the delay line, type its reading, press Enter, then save
the trace from PicoScope. The script picks up the newest CSV in the folder,
computes the contrast, appends a row to the output file and updates the plot.

With --pico the script drives the scope itself (PicoScope app closed): press
Enter and it captures, saves the trace into <out>_traces/ and logs the point.

    uv run fibre-scan --pico --out sld_scan.csv --unit div

Without --watch or --pico you're asked for the trace path each time instead.
Commands at the prompt: "u" removes the last point, "q" quits.
Re-running with the same --out file carries on from where you left off.

Stepped scans: give --step (and --start, --stop) and the prompt shows the next
position; set the micrometer there and press Enter alone to take it. Typing a
number instead takes that position, and the steps carry on from it.

    uv run fibre-scan --pico --out fp_scan.csv --unit mm --start 24.5 --step -0.1 --stop 0.5
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contrast import contrast, contrast_per_period
from .io import Trace, data_path, find_data, graph_path, load_trace, utf8_output

COLUMNS = [
    "x", "unit", "contrast", "contrast_err", "n_periods", "contrast_whole", "contrast_rms",
    "i_max", "i_min", "mean", "file",
]  # fmt: skip
UM_PER_UNIT = {"div": 10.0, "um": 1.0, "mm": 1000.0}


@dataclass
class Scan:
    x: np.ndarray
    unit: str
    contrast: np.ndarray
    contrast_err: np.ndarray
    contrast_rms: np.ndarray
    mean: np.ndarray
    n: np.ndarray  # number of measurements averaged at each position

    def opd_um(self, um_per_unit: float | None = None) -> np.ndarray:
        """Path difference in µm (relative to the zero of the screw)."""
        k = um_per_unit if um_per_unit is not None else UM_PER_UNIT.get(self.unit)
        if k is None:
            raise ValueError(f"Unknown unit {self.unit!r}; pass um_per_unit")
        return self.x * k


def load_scan(path: str | Path) -> Scan:
    with find_data(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows])

    # Repeated positions are averaged; their spread (if larger) becomes the error bar.
    x_all = np.round(col("x"), 6)
    xs = np.unique(x_all)
    c_all, e_all = col("contrast"), col("contrast_err")
    contrast_, err, n = [], [], []
    for xv in xs:
        m = x_all == xv
        cv, ev = c_all[m], e_all[m]
        k = int(m.sum())
        stat = float(np.sqrt(np.nanmean(ev**2)) / np.sqrt(k)) if np.any(np.isfinite(ev)) else np.nan
        spread = float(np.std(cv, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
        contrast_.append(float(np.mean(cv)))
        err.append(np.nanmax([stat, spread]) if k > 1 else stat)
        n.append(k)

    def avg(name: str) -> np.ndarray:
        v = col(name)
        return np.array([np.nanmean(v[x_all == xv]) for xv in xs])

    return Scan(
        x=xs,
        unit=rows[0]["unit"],
        contrast=np.array(contrast_),
        contrast_err=np.array(err),
        contrast_rms=avg("contrast_rms"),
        mean=avg("mean"),
        n=np.array(n),
    )


def make_row(
    x: float,
    unit: str,
    v: np.ndarray,
    path: Path | str,
    dark: float = 0.0,
    fs: float | None = None,
    drive: np.ndarray | None = None,
) -> dict[str, str]:
    """One scan row.

    "contrast" is Eq. 19 per modulator period, averaged, with the spread as the
    error ("n_periods" of them); this needs the sample rate `fs`, and uses the
    drive signal for the period if given. Without `fs` it is the whole-trace
    estimate. "contrast_whole" (whole-trace Eq. 19) and "contrast_rms" are kept
    for comparison."""
    whole = contrast(v, "peak", dark=dark)
    best = contrast_per_period(v, fs, drive, dark=dark) if fs else whole
    rms = contrast(v, "rms", dark=dark)
    return {
        "x": f"{x:g}",
        "unit": unit,
        "contrast": f"{best.contrast:.6g}",
        "contrast_err": f"{best.error:.3g}",
        "n_periods": str(best.n),
        "contrast_whole": f"{whole.contrast:.6g}",
        "contrast_rms": f"{rms.contrast:.6g}",
        "i_max": f"{best.i_max:.6g}",
        "i_min": f"{best.i_min:.6g}",
        "mean": f"{best.mean:.6g}",
        "file": str(path),
    }


def row_from_trace(
    x: float, unit: str, tr: Trace, path: Path | str, dark: float = 0.0, channel: str | None = None
) -> dict[str, str]:
    """Scan row from a whole trace: detector `channel` (default first), with channel B
    used as the drive when it is present and isn't the detector."""
    v = tr.channel(channel)
    drive = None
    if (channel or "A").upper()[-1] != "B":
        try:
            drive = tr.channel("B")
        except KeyError:
            drive = None
    return make_row(x, unit, v, path, dark, fs=tr.fs, drive=drive)


def _wait_for_new_csv(folder: Path, since: float, timeout: float = 600) -> Path | None:
    """Wait for a CSV modified after `since` whose size has stopped changing."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        new = [p for p in folder.glob("*.csv") if p.stat().st_mtime > since]
        if new:
            p = max(new, key=lambda q: q.stat().st_mtime)
            s1 = p.stat().st_size
            time.sleep(0.5)
            if p.stat().st_size == s1 and s1 > 0:
                return p
        time.sleep(0.3)
    return None


def _read_rows(out: Path) -> list[dict[str, str]]:
    if not out.exists():
        return []
    with out.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(out: Path, rows: list[dict[str, str]]) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


class PicoSource:
    """Captures one trace per point from the PicoScope and saves it next to the scan."""

    def __init__(self, a: argparse.Namespace, out: Path):
        from .capture import PicoScope

        self.a = a
        self.det = (a.channel or "A").upper()[-1]
        # The drive (B) is recorded by default: it times the modulator periods for the contrast.
        extra = [] if self.det == "B" else ["B"]
        self.record = sorted({self.det, *extra, *(c.upper()[-1] for c in (a.record or []))})
        self.ranges = {c: a.range for c in self.record} if a.range else None
        self.dir = out.with_name(out.stem + "_traces")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem = out.stem
        self.scope = PicoScope(a.family, a.resolution)
        print(f"Connected to a {self.scope.family} scope; recording channel(s) {', '.join(self.record)}")

    def get(self, x: float) -> tuple[Path, Trace]:
        from .capture import autorange_capture
        from .io import save_trace

        a = self.a
        cap = autorange_capture(self.scope, self.record, a.samples, a.dt, self.ranges, auto=not a.range)
        self.ranges = cap.ranges  # start the next point from here: usually one capture
        stamp = time.strftime("%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"  # unique even for fast repeats
        path = self.dir / f"{self.stem}_{x:g}{a.unit}_{stamp}.csv"
        tr = cap.trace()
        save_trace(path, tr.t, tr.channels)
        print(f"  captured {len(cap.t)} samples, range ±{cap.ranges[self.det]:g} V")
        if cap.clipped:
            print(f"  warning: clipped on {sorted(cap.clipped)} even at the largest range")
        return path, tr

    def close(self) -> None:
        self.scope.close()


def next_position(prev: float | None, start: float | None, step: float | None, stop: float | None) -> float | None:
    """Next position of a stepped scan, or None when there is none to suggest.

    With no point taken yet it's ``start``; after that, the last position plus ``step``.
    Returns None once the next position would pass ``stop``. Rounded so that repeated
    steps of 0.1 don't drift (12.299999 -> 12.3).
    """
    if step is None:
        return None
    if prev is None:
        nxt = start
    else:
        nxt = prev + step
    if nxt is None:
        return None
    nxt = round(nxt, 9)
    if stop is not None and (nxt - stop) * step > 1e-9:
        return None
    return nxt


def main(argv: list[str] | None = None) -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="CSV file the scan is written to")
    p.add_argument("--watch", help="folder PicoScope saves CSVs into")
    p.add_argument("--unit", default="div", help="unit of what you type: div, mm, um, or anything (e.g. 'mA')")
    p.add_argument("--quantity", default="delay", help="label for the x axis (e.g. 'pump current')")
    p.add_argument("-c", "--channel", help="detector channel (default: first, or A with --pico)")
    p.add_argument("--dark", type=float, default=0.0, help="detector dark level in V")
    p.add_argument("--no-plot", action="store_true")
    s_g = p.add_argument_group("stepped scan (press Enter alone to take the suggested position)")
    s_g.add_argument("--step", type=float, help="change between points, e.g. -0.1 to go down in 0.1 steps")
    s_g.add_argument("--start", type=float, help="first position (default: carry on from the last point in --out)")
    s_g.add_argument("--stop", type=float, help="last position; no more suggestions past it")
    g = p.add_argument_group("direct PicoScope capture (--pico)")
    g.add_argument("--pico", action="store_true", help="capture from the PicoScope directly")
    g.add_argument("--family", default="auto", help="driver family, e.g. ps2000a (default: try all)")
    g.add_argument("--dt", type=float, default=5e-6, help="sample interval in s (default 5 µs)")
    g.add_argument("--samples", type=int, help="samples per trace (default 10000 = 50 ms, or the scope memory)")
    g.add_argument("--record", nargs="+", help="extra channels to save (B, the drive, is always saved)")
    g.add_argument("--range", type=float, help="fixed range in V (default: automatic)")
    g.add_argument("--resolution", type=int, default=12, help="bits, 5000 series only")
    g.add_argument(
        "--repeats", type=int, default=1, help="captures per Enter press (repeats are averaged in the analysis)"
    )
    g.add_argument("--repeat-wait", type=float, default=2.0, help="seconds between repeats (default 2)")
    a = p.parse_args(argv)
    if a.pico and a.watch:
        p.error("use either --pico or --watch, not both")
    if a.step == 0:
        p.error("--step can't be 0")
    if (a.start is not None or a.stop is not None) and a.step is None:
        p.error("--start and --stop need --step")

    out = data_path(a.out)
    rows = _read_rows(out)
    if rows:
        print(f"Resuming {out} ({len(rows)} points)")
    watch = Path(a.watch).expanduser() if a.watch else None
    if watch and not watch.is_dir():
        p.error(f"{watch} is not a folder")
    source = None
    if a.pico:
        from .capture import PicoError

        try:
            source = PicoSource(a, out)
        except PicoError as e:
            p.exit(1, f"error: {e}\n")

    plt = None
    if not a.no_plot:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel(f"{a.quantity} [{a.unit}]")
        ax.set_ylabel("contrast")
        (line,) = ax.plot([], [], "o-")

    def redraw() -> None:
        if plt is None:
            return
        pts = sorted((float(r["x"]), float(r["contrast"])) for r in rows)
        line.set_data([q[0] for q in pts], [q[1] for q in pts])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)
        # The window can't repaint while waiting for typed input, so keep a PNG too.
        if rows:
            fig.savefig(graph_path(out, ".png"), dpi=120)

    redraw()
    groups: list[int] = [1] * len(rows)  # captures per Enter press, so "u" undoes a whole point
    # Stepped scan: a fresh --start wins; otherwise carry on from the last point in the file.
    if a.start is not None:
        nxt = next_position(None, a.start, a.step, a.stop)
    else:
        nxt = next_position(float(rows[-1]["x"]) if rows else None, None, a.step, a.stop)
    if a.step is not None and nxt is None and a.start is None and not rows:
        print("No --start and no earlier points: type the first position; the steps follow from it.")
    try:
        while True:
            if nxt is not None:
                prompt = f"{a.quantity}: set {nxt:g} {a.unit}, then Enter (or type another value, u = undo, q = quit): "
            elif a.step is not None and (a.start is not None or rows):
                prompt = f"{a.quantity}: end of range reached (type a value, u = undo, q = quit): "
            else:
                prompt = f"{a.quantity} [{a.unit}] (u = undo, q = quit): "
            s = input(prompt).strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            if s.lower() == "u":
                if rows:
                    n_undo = groups.pop() if groups else 1
                    x_undo = rows[-1]["x"]
                    del rows[-n_undo:]
                    print(f"Removed x = {x_undo} ({n_undo} capture{'s' if n_undo > 1 else ''})")
                    _write_rows(out, rows)
                    redraw()
                    if a.step is not None:
                        nxt = float(x_undo)  # take the undone position again next
                continue
            if s == "" and nxt is not None:
                x = nxt
            else:
                try:
                    x = float(s)
                except ValueError:
                    print("Type a number, u or q" + (" (or Enter alone for the suggested position)" if nxt else ""))
                    continue

            n_rep = max(a.repeats, 1) if source is not None else 1
            added = 0
            for k in range(n_rep):
                if k:
                    time.sleep(a.repeat_wait)
                try:  # any failure skips this capture but keeps the session alive
                    if source is not None:
                        path, tr = source.get(x)
                        channel = source.det
                    else:
                        if watch:
                            print("  waiting for a new CSV - save the trace from PicoScope now...")
                            path = _wait_for_new_csv(watch, since=time.time() - 1)
                            if path is None:
                                print("  no new file appeared; point skipped")
                                continue
                        else:
                            path = Path(input("  trace file: ").strip().strip("'\"")).expanduser()
                        tr = load_trace(path)
                        channel = a.channel
                    row = row_from_trace(x, a.unit, tr, path, a.dark, channel)
                except Exception as e:
                    print(f"  capture skipped: {e}")
                    continue
                rows.append(row)
                added += 1
                _write_rows(out, rows)
                tag = f" [{k + 1}/{n_rep}]" if n_rep > 1 else ""
                print(
                    f"  {path.name}{tag}: C = {row['contrast']} ± {row['contrast_err']} ({row['n_periods']} periods; "
                    f"whole-trace {row['contrast_whole']})"
                )
            if added and a.step is not None:
                nxt = next_position(x, None, a.step, a.stop)
            if added:
                groups.append(added)
                if added > 1:
                    cs = np.array([float(r["contrast"]) for r in rows[-added:]])
                    print(
                        f"  -> mean C = {cs.mean():.4f} ± {cs.std(ddof=1) / np.sqrt(added):.4f} over {added} captures"
                    )
                redraw()
    finally:
        if source is not None:
            source.close()
    print(f"Saved {len(rows)} points to {out}")


if __name__ == "__main__":
    main()


def rescan_main(argv: list[str] | None = None) -> None:
    """Re-derive a scan's contrast values from its stored traces into a NEW file.

    The original scan file is never modified.

        uv run fibre-rescan fp_scan.csv            # writes data/fp_scan_period.csv
    """
    utf8_output()
    p = argparse.ArgumentParser(description=rescan_main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", help="existing scan CSV (rows must point to their trace files)")
    p.add_argument("--out", help="new scan CSV (default: <scan>_period.csv next to it)")
    p.add_argument("-c", "--channel", help="detector channel (default: first)")
    p.add_argument("--dark", type=float, default=0.0, help="detector dark level in V")
    a = p.parse_args(argv)

    src = find_data(a.scan)
    out = data_path(a.out) if a.out else src.with_name(src.stem + "_period.csv")
    if out.resolve() == src.resolve():
        p.error("--out must differ from the input: the original scan is never overwritten")
    old = _read_rows(src)
    new, changes, missing = [], [], 0
    for r in old:
        trace = Path(r["file"])
        if not trace.exists():
            missing += 1
            continue
        row = row_from_trace(float(r["x"]), r["unit"], load_trace(trace), trace, a.dark, a.channel)
        new.append(row)
        d = float(row["contrast"]) - float(r["contrast"])
        changes.append((abs(d), d, r["x"], r["contrast"], row["contrast"]))
    _write_rows(out, new)
    print(
        f"Re-derived {len(new)} of {len(old)} rows -> {out}" + (f" ({missing} trace files missing)" if missing else "")
    )
    big = sorted(changes, reverse=True)[:8]
    if big:
        print("Largest changes (position: old -> new):")
        for _, d, x, c_old, c_new in big:
            print(f"  {x}: {float(c_old):.3f} -> {float(c_new):.3f}  ({d:+.3f})")


def monitor_main(argv: list[str] | None = None) -> None:
    """Log contrast against time at a fixed delay (`fibre-monitor`).

    Leave the micrometer alone; the scope captures every --interval seconds for
    --duration seconds (or until Ctrl-C). The log is an ordinary scan file with
    x = seconds since the start, so every scan tool reads it.

        uv run fibre-monitor --out monitor_14p50mm.csv --interval 5 --duration 300
    """
    utf8_output()
    p = argparse.ArgumentParser(description=monitor_main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="log CSV (put the position in the name, e.g. monitor_14p50mm.csv)")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between captures (default 5)")
    p.add_argument("--duration", type=float, default=300.0, help="seconds to run; 0 = until Ctrl-C (default 300)")
    p.add_argument("-c", "--channel", help="detector channel (default A)")
    p.add_argument("--dark", type=float, default=0.0, help="detector dark level in V")
    p.add_argument("--family", default="auto")
    p.add_argument("--dt", type=float, default=5e-6, help="sample interval in s (default 5 µs)")
    p.add_argument("--samples", type=int, help="samples per trace (default: what fits)")
    p.add_argument("--record", nargs="+", help="extra channels to save")
    p.add_argument("--range", type=float, help="fixed range in V (default: automatic)")
    p.add_argument("--resolution", type=int, default=12)
    a = p.parse_args(argv)
    a.unit = "s"

    from .capture import PicoError

    out = data_path(a.out)
    rows = _read_rows(out)
    if rows:
        p.error(f"{out} already exists: choose a new --out for each monitoring run")
    try:
        source = PicoSource(a, out)
    except PicoError as e:
        p.exit(1, f"error: {e}\n")
    print(f"Logging every {a.interval:g} s " + (f"for {a.duration:g} s" if a.duration else "until Ctrl-C") + " ...")
    t0 = time.time()
    try:
        while not a.duration or time.time() - t0 <= a.duration:
            t_next = time.time() + a.interval
            elapsed = round(time.time() - t0, 1)
            try:
                path, tr = source.get(elapsed)
                row = row_from_trace(elapsed, "s", tr, path, a.dark, source.det)
            except Exception as e:  # keep logging through a bad capture
                print(f"  capture skipped: {e}")
            else:
                rows.append(row)
                _write_rows(out, rows)
                print(f"  t = {elapsed:6.1f} s: C = {row['contrast']} ± {row['contrast_err']}")
            time.sleep(max(0.0, t_next - time.time()))
    except KeyboardInterrupt:
        print("  stopped")
    finally:
        source.close()

    if not rows:
        return
    t = np.array([float(r["x"]) for r in rows])
    c = np.array([float(r["contrast"]) for r in rows])
    sd = c.std(ddof=1) if len(c) > 1 else 0.0
    print(
        f"\n{len(c)} captures over {t[-1]:.0f} s: mean C = {c.mean():.4f}, std {sd:.4f}, "
        f"range {c.min():.4f}-{c.max():.4f}"
    )
    if len(c) > 2 and np.std(c[:-1]) > 0 and np.std(c[1:]) > 0:
        r1 = float(np.corrcoef(c[:-1], c[1:])[0, 1])
        print(f"  correlation between consecutive captures {r1:+.2f} (near 0: changes faster than {a.interval:g} s)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    err = np.array([float(r["contrast_err"]) for r in rows])
    ax.errorbar(t, c, yerr=np.nan_to_num(err), fmt="o-", ms=3, lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("contrast")
    fig.tight_layout()
    png = graph_path(out, ".png")
    fig.savefig(png, dpi=130)
    print(f"Saved {out} and {png}")
