"""Load and save oscilloscope traces.

Reads PicoScope CSV exports, which look like:

    Time,Channel A,Channel B
    (ms),(V),(mV)

    -10.0,0.512,3.1
    ...

Headers with units in the names ("Time (ms),Channel A (mV)") and plain CSVs
with a single header row ("t,v", assumed seconds and volts) also work. Over-range samples ("∞") become NaN.
"""

from __future__ import annotations

import csv
import re
import sys
import warnings
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path

import numpy as np

TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ns": 1e-9}
VOLT_UNITS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6, "μv": 1e-6}


@dataclass
class Trace:
    t: np.ndarray  # seconds
    channels: dict[str, np.ndarray]  # volts
    path: Path | None = field(default=None)

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    def channel(self, name: str | None = None) -> np.ndarray:
        """Return a channel by name, letter ("A"), or the first one if name is None."""
        if name is None:
            return next(iter(self.channels.values()))
        if name in self.channels:
            return self.channels[name]
        for key, value in self.channels.items():
            if key.lower() in (name.lower(), f"channel {name.lower()}"):
                return value
        raise KeyError(f"No channel {name!r}; available: {list(self.channels)}")


def _to_float(s: str) -> float:
    s = s.strip()
    if s in ("∞", "+∞", "inf", "+inf"):
        return np.inf
    if s in ("-∞", "-inf"):
        return -np.inf
    try:
        return float(s)
    except ValueError:
        return np.nan


def _is_number(s: str) -> bool:
    return not np.isnan(_to_float(s)) or s.strip().lower() == "nan"


def load_trace(path: str | Path) -> Trace:
    path = find_data(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines()]
    sample = "\n".join(lines[:20])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    rows = [r for r in csv.reader(lines, delimiter=delimiter)]
    if delimiter != ",":
        # European-locale PicoScope exports: "0,512" with ";" separators.
        rows = [[c.replace(",", ".") for c in r] for r in rows]

    names: list[str] | None = None
    units: list[str] | None = None
    data_rows: list[list[str]] = []
    for row in rows:
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if all(_is_number(c) or c in ("∞", "-∞", "+∞") for c in cells if c):
            data_rows.append(cells)
        elif cells[0].startswith("(") and names is not None and not data_rows:
            units = [c.strip("() ").lower() for c in cells]
        elif not data_rows:
            names = cells

    if not data_rows:
        raise ValueError(f"No numeric data found in {path}")
    ncol = max(len(r) for r in data_rows)
    arr = np.full((len(data_rows), ncol), np.nan)
    for i, r in enumerate(data_rows):
        arr[i, : len(r)] = [_to_float(c) for c in r]

    if names is None or len(names) < ncol:
        names = ["Time"] + [f"Channel {chr(65 + i)}" for i in range(ncol - 1)]
    if units is None:
        # Units may sit inside the header names instead, e.g. "Time (ms)".
        found = [re.search(r"\(([^)]*)\)\s*$", n) for n in names[:ncol]]
        if any(found):
            units = [m.group(1).strip().lower() if m else ("s" if i == 0 else "v") for i, m in enumerate(found)]
            names = [re.sub(r"\s*\([^)]*\)\s*$", "", n) for n in names]
        else:
            units = ["s"] + ["v"] * (ncol - 1)
    unknown = [u for u in units[:1] if u not in TIME_UNITS] + [u for u in units[1:ncol] if u not in VOLT_UNITS]
    if unknown:
        warnings.warn(f"{path.name}: unrecognised units {unknown}, treated as s / V", stacklevel=2)

    t = arr[:, 0] * TIME_UNITS.get(units[0], 1.0)
    channels: dict[str, np.ndarray] = {}
    for j in range(1, ncol):
        v = arr[:, j] * VOLT_UNITS.get(units[j] if j < len(units) else "v", 1.0)
        n_over = int(np.sum(~np.isfinite(v)))
        if n_over:
            warnings.warn(
                f"{path.name}: {names[j]} has {n_over} over-range/blank samples (set to NaN). "
                "Consider a larger scope range.",
                stacklevel=2,
            )
            v = np.where(np.isfinite(v), v, np.nan)
        channels[names[j]] = v
    return Trace(t=t, channels=channels, path=path)


def utf8_output() -> None:
    """Make printing µ, λ, Δ, π … safe on every OS. Windows otherwise uses a legacy
    encoding when output is redirected to a file or pipe, and the print crashes."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper) and stream.encoding.lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


DATA_DIR = Path("data")


def data_path(name: str | Path) -> Path:
    """Where new data is saved: a bare filename goes into data/; a path with a
    folder is used as given."""
    p = Path(name)
    if p.parent == Path("."):
        DATA_DIR.mkdir(exist_ok=True)
        return DATA_DIR / p
    return p


def find_data(name: str | Path) -> Path:
    """An existing data file: as given, or else inside data/."""
    p = Path(name)
    if not p.exists() and (DATA_DIR / p).exists():
        return DATA_DIR / p
    return p


def output_section(name: str) -> str:
    """Which output/ subfolder a graph belongs in, from its data file's name.

    The lab's naming convention already says which handout section a file is for:
    amp2_* = section 5 (an amplifier in each arm), amp1_* = section 4 (one amplifier),
    vib* = section 6 (vibrometer), anything containing "_trial" = the trial run.
    Everything else (FP and SLD scans, fringe counts, checks) is section 3.
    """
    stem = Path(name).stem.lower()
    if "_trial" in stem:
        return "trial"
    if stem.startswith("amp2_"):
        return "section5"
    if stem.startswith("amp1_"):
        return "section4"
    if stem.startswith("vib"):
        return "section6"
    return "section3"


def graph_path(data_file: str | Path, suffix: str) -> Path:
    """Where a graph made from `data_file` goes: output/<section>/ beside the data
    (beside data/ itself for files kept in data/). The section comes from
    `output_section`."""
    data_file = Path(data_file)
    base = data_file.parent.parent if data_file.parent.name == DATA_DIR.name else data_file.parent
    folder = base / "output" / output_section(data_file.name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (data_file.stem + suffix)


def save_trace(path: str | Path, t: np.ndarray, channels: dict[str, np.ndarray]) -> None:
    """Save in PicoScope CSV format (time in ms, volts in V)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(channels)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Time", *names])
        w.writerow(["(ms)", *["(V)"] * len(names)])
        w.writerow([])
        cols = [t * 1e3, *[channels[n] for n in names]]
        for row in zip(*cols, strict=True):
            w.writerow([f"{x:.9g}" for x in row])
