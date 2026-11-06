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
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contrast import contrast
from .io import data_path, find_data, graph_path, load_trace

COLUMNS = ["x", "unit", "contrast", "contrast_err", "contrast_rms", "i_max", "i_min", "mean", "file"]
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
    with find_data(path).open() as f:
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


def make_row(x: float, unit: str, v: np.ndarray, path: Path | str, dark: float = 0.0) -> dict[str, str]:
    """One scan row. "contrast" is the max/min estimate (Eq. 19), which is right
    whatever the number of fringes per sweep; "contrast_rms" is kept for
    comparison at very low contrast."""
    rp = contrast(v, "peak", dark=dark)
    r = contrast(v, "rms", dark=dark)
    return {
        "x": f"{x:g}",
        "unit": unit,
        "contrast": f"{rp.contrast:.6g}",
        "contrast_err": f"{rp.error:.3g}",
        "contrast_rms": f"{r.contrast:.6g}",
        "i_max": f"{rp.i_max:.6g}",
        "i_min": f"{rp.i_min:.6g}",
        "mean": f"{rp.mean:.6g}",
        "file": str(path),
    }


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
    with out.open() as f:
        return list(csv.DictReader(f))


def _write_rows(out: Path, rows: list[dict[str, str]]) -> None:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


class PicoSource:
    """Captures one trace per point from the PicoScope and saves it next to the scan."""

    def __init__(self, a: argparse.Namespace, out: Path):
        from .capture import PicoScope

        self.a = a
        self.det = (a.channel or "A").upper()[-1]
        self.record = sorted({self.det, *(c.upper()[-1] for c in (a.record or []))})
        self.ranges = {c: a.range for c in self.record} if a.range else None
        self.dir = out.with_name(out.stem + "_traces")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem = out.stem
        self.scope = PicoScope(a.family, a.resolution)
        print(f"Connected to a {self.scope.family} scope; recording channel(s) {', '.join(self.record)}")

    def get(self, x: float) -> tuple[Path, np.ndarray]:
        from .capture import autorange_capture
        from .io import save_trace

        a = self.a
        cap = autorange_capture(self.scope, self.record, a.samples, a.dt, self.ranges, auto=not a.range)
        self.ranges = cap.ranges  # start the next point from here: usually one capture
        path = self.dir / f"{self.stem}_{x:g}{a.unit}_{time.strftime('%H%M%S')}.csv"
        tr = cap.trace()
        save_trace(path, tr.t, tr.channels)
        print(f"  captured {len(cap.t)} samples, range ±{cap.ranges[self.det]:g} V")
        if cap.clipped:
            print(f"  warning: clipped on {sorted(cap.clipped)} even at the largest range")
        return path, cap.volts[self.det]

    def close(self) -> None:
        self.scope.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="CSV file the scan is written to")
    p.add_argument("--watch", help="folder PicoScope saves CSVs into")
    p.add_argument("--unit", default="div", help="unit of what you type: div, mm, um, or anything (e.g. 'mA')")
    p.add_argument("--quantity", default="delay", help="label for the x axis (e.g. 'pump current')")
    p.add_argument("-c", "--channel", help="detector channel (default: first, or A with --pico)")
    p.add_argument("--dark", type=float, default=0.0, help="detector dark level in V")
    p.add_argument("--no-plot", action="store_true")
    g = p.add_argument_group("direct PicoScope capture (--pico)")
    g.add_argument("--pico", action="store_true", help="capture from the PicoScope directly")
    g.add_argument("--family", default="auto", help="driver family, e.g. ps2000a (default: try all)")
    g.add_argument("--dt", type=float, default=5e-6, help="sample interval in s (default 5 µs)")
    g.add_argument("--samples", type=int, help="samples per trace (default 10000 = 50 ms, or the scope memory)")
    g.add_argument("--record", nargs="+", help="extra channels to save, e.g. B for the drive")
    g.add_argument("--range", type=float, help="fixed range in V (default: automatic)")
    g.add_argument("--resolution", type=int, default=12, help="bits, 5000 series only")
    a = p.parse_args(argv)
    if a.pico and a.watch:
        p.error("use either --pico or --watch, not both")

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
    try:
        while True:
            s = input(f"{a.quantity} [{a.unit}] (u = undo, q = quit): ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            if s.lower() == "u":
                if rows:
                    print(f"Removed x = {rows.pop()['x']}")
                    _write_rows(out, rows)
                    redraw()
                continue
            try:
                x = float(s)
            except ValueError:
                print("Type a number, u or q")
                continue

            try:  # any failure skips this point but keeps the session alive
                if source is not None:
                    path, v = source.get(x)
                else:
                    if watch:
                        print("  waiting for a new CSV - save the trace from PicoScope now...")
                        path = _wait_for_new_csv(watch, since=time.time() - 1)
                        if path is None:
                            print("  no new file appeared; point skipped")
                            continue
                    else:
                        path = Path(input("  trace file: ").strip().strip("'\"")).expanduser()
                    v = load_trace(path).channel(a.channel)
                row = make_row(x, a.unit, v, path, a.dark)
            except Exception as e:
                print(f"  point skipped: {e}")
                continue

            rows.append(row)
            _write_rows(out, rows)
            print(f"  {path.name}: C = {row['contrast']} ± {row['contrast_err']} (rms estimate {row['contrast_rms']})")
            redraw()
    finally:
        if source is not None:
            source.close()
    print(f"Saved {len(rows)} points to {out}")


if __name__ == "__main__":
    main()
