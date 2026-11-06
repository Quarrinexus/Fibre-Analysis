"""Capture traces straight from a PicoScope (block mode), no PicoScope app needed.

Needs Pico's drivers (PicoSDK, from picotech.com/downloads) installed on the
computer, and the Python wrappers:  uv sync --extra pico
Close the PicoScope app first: only one program can use the scope at a time.

Supported driver families (the model number says which):
    ps2000   2204A, 2205A (older 2000 series)
    ps2000a  2206B-2208B, 2405A-2408B
    ps3000a  3203D-3406D and other 3000 A/B/D
    ps4000a  4224A, 4262, 4444, 4824
    ps5000a  5242D, 5444D and other 5000 A/B/D

    uv run fibre-capture trace.csv --channels A B --dt 2e-6 --samples 10000
    uv run fibre-capture vib.csv --channels A B --dt 50e-6 --samples 100000

Inputs are DC-coupled (the contrast needs the DC level). The range is chosen
automatically unless you give --range: a capture that clips moves up a range,
one using under ~40% of it moves down, and the capture is repeated.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import time
import warnings
from dataclasses import dataclass

import numpy as np

from .io import Trace, data_path, save_trace, utf8_output

FAMILIES = ("ps2000a", "ps3000a", "ps4000a", "ps5000a", "ps2000")
CHANNELS = "ABCDEFGH"
RESOLUTION_5000A = {8: 0, 12: 1, 14: 2, 15: 3, 16: 4}
PICO_OK = 0
DEFAULT_SAMPLES = 10_000
POWER_STATUSES = (0x11A, 0x11E)  # no external PSU / USB3 scope on a USB2 port: carry on on USB power


class PicoError(RuntimeError):
    pass


def _tag(status: int) -> str:
    try:
        from picosdk.constants import pico_tag

        return pico_tag(status)
    except Exception:  # picosdk missing or unknown code
        return hex(status)


def _load_driver(family: str):
    """The picosdk ctypes wrapper object for a driver family (raises if the C library is missing)."""
    module = importlib.import_module(f"picosdk.{family}")
    return getattr(module, family)


@dataclass
class Capture:
    t: np.ndarray
    volts: dict[str, np.ndarray]
    ranges: dict[str, float]
    clipped: set[str]
    dt: float

    def trace(self) -> Trace:
        return Trace(self.t, {f"Channel {c}": v for c, v in self.volts.items()})


class PicoScope:
    """One open scope. Use as a context manager so it is always closed."""

    def __init__(self, family: str = "auto", resolution_bits: int = 12):
        self.family, self.lib, self.handle = self._open(family, resolution_bits)
        self.old_api = self.family == "ps2000"
        vr = self.lib.PICO_VOLTAGE_RANGE  # {enum value: volts}
        self.ranges = sorted(((v, k) for k, v in vr.items() if 0.005 <= v <= 50), key=lambda p: p[0])
        # Unused channels are switched off (frees memory); missing ones just fail quietly.
        self.n_channels = 2 if self.old_api else 8 if self.family == "ps4000a" else 4
        self._enabled: dict[str, tuple[float, int]] = {}
        if self.old_api:
            self.max_adc = 32767
        else:
            m = ctypes.c_int16()
            self._check(self.lib._maximum_value(ctypes.c_int16(self.handle), ctypes.byref(m)), "MaximumValue")
            self.max_adc = m.value

    # ---------------------------------------------------------------- opening
    def _open(self, family: str, resolution_bits: int):
        tried = []
        for fam in FAMILIES if family == "auto" else (family,):
            try:
                lib = _load_driver(fam)
            except ModuleNotFoundError:
                raise PicoError("picosdk is not installed - run: uv sync --extra pico") from None
            except Exception as e:
                tried.append(f"{fam}: driver not installed ({type(e).__name__})")
                continue
            try:
                handle = self._open_with(lib, fam, resolution_bits)
            except PicoError as e:
                tried.append(f"{fam}: {e}")
                continue
            return fam, lib, handle
        raise PicoError(
            "No PicoScope found. Is it plugged in, the PicoScope app closed, and PicoSDK installed?\n  "
            + "\n  ".join(tried)
        )

    @staticmethod
    def _open_with(lib, fam: str, resolution_bits: int) -> int:
        if fam == "ps2000":
            h = int(lib._open_unit())
            if h <= 0:
                raise PicoError("no unit found" if h == 0 else "unit failed to open")
            return h
        handle = ctypes.c_int16()
        if fam == "ps5000a":
            if resolution_bits not in RESOLUTION_5000A:
                raise PicoError(f"resolution must be one of {list(RESOLUTION_5000A)}")
            status = lib._open_unit(ctypes.byref(handle), None, RESOLUTION_5000A[resolution_bits])
        else:
            status = lib._open_unit(ctypes.byref(handle), None)
        if status in POWER_STATUSES and hasattr(lib, "_change_power_source"):
            status = lib._change_power_source(ctypes.c_int16(handle.value), status)
        if status != PICO_OK or handle.value <= 0:
            raise PicoError(f"open failed ({_tag(status)})")
        return handle.value

    def close(self) -> None:
        if self.handle:
            self.lib._stop(ctypes.c_int16(self.handle))
            self.lib._close_unit(ctypes.c_int16(self.handle))
            self.handle = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _check(self, status: int, what: str) -> None:
        if status != PICO_OK:
            raise PicoError(f"{self.family} {what} failed ({_tag(status)})")

    # --------------------------------------------------------------- channels
    def _set_channel(self, ch: str, enabled: bool, enum: int = 0) -> bool:
        h, idx = ctypes.c_int16(self.handle), CHANNELS.index(ch)
        if self.old_api:
            return bool(self.lib._set_channel(h, idx, int(enabled), 1, enum))
        status = self.lib._set_channel(h, idx, int(enabled), 1, enum, ctypes.c_float(0.0))  # 1 = DC
        return status == PICO_OK

    def configure(self, ranges: dict[str, float]) -> dict[str, float]:
        """Enable these channels at the smallest range >= the requested volts; disable the rest.
        Returns the ranges actually set."""
        self._enabled = {}
        for ch in CHANNELS[: self.n_channels]:
            if ch not in ranges:
                self._set_channel(ch, False)  # may fail on 2-channel models; harmless
                continue
            for volts, enum in self.ranges:
                if volts >= ranges[ch] * 0.999 and self._set_channel(ch, True, enum):
                    self._enabled[ch] = (volts, enum)
                    break
            else:
                raise PicoError(f"Channel {ch}: no range up to {ranges[ch]} V accepted")
        unknown = set(ranges) - set(CHANNELS[: self.n_channels])
        if unknown:
            raise PicoError(f"{self.family} has no channel(s) {sorted(unknown)}")
        return {c: v for c, (v, _) in self._enabled.items()}

    def range_steps(self) -> list[float]:
        return [v for v, _ in self.ranges]

    # --------------------------------------------------------------- timebase
    def _timebase(self, tb: int) -> tuple[float, int] | None:
        """(sample interval in s, max samples) for timebase id tb, or None if invalid."""
        h = ctypes.c_int16(self.handle)
        mx = ctypes.c_int32()
        if self.old_api:
            if tb > 32767:
                return None
            dt_ns, units = ctypes.c_int32(), ctypes.c_int16()
            ok = self.lib._get_timebase(h, tb, 0, ctypes.byref(dt_ns), ctypes.byref(units), 1, ctypes.byref(mx))
            return (dt_ns.value * 1e-9, mx.value) if ok else None
        dt_ns = ctypes.c_float()
        if self.family in ("ps2000a", "ps3000a"):
            status = self.lib._get_timebase2(h, tb, 0, ctypes.byref(dt_ns), 1, ctypes.byref(mx), 0)
        else:
            status = self.lib._get_timebase2(h, tb, 0, ctypes.byref(dt_ns), ctypes.byref(mx), 0)
        return (dt_ns.value * 1e-9, mx.value) if status == PICO_OK else None

    def find_timebase(self, dt: float) -> tuple[int, float, int]:
        """Slowest timebase with interval <= dt: (id, interval s, max samples)."""
        lo = next((tb for tb in range(64) if self._timebase(tb)), None)
        first = self._timebase(lo) if lo is not None else None
        if lo is None or first is None:
            raise PicoError("no valid timebase - too many channels enabled for this model?")
        if first[0] > dt:
            warnings.warn(f"Fastest sampling is {first[0]:.3g} s, slower than requested", stacklevel=2)
            return lo, *first
        # Interval grows with the id: double until too slow/invalid, then bisect.
        good, good_info, step = lo, first, 1
        while True:
            cand = good + step
            info = self._timebase(cand)
            if info is None or info[0] > dt:
                bad = cand
                break
            good, good_info, step = cand, info, step * 2
        while bad - good > 1:
            mid = (good + bad) // 2
            info = self._timebase(mid)
            if info is not None and info[0] <= dt:
                good, good_info = mid, info
            else:
                bad = mid
        return good, *good_info

    # ---------------------------------------------------------------- capture
    def capture(self, n: int | None, dt: float, timeout: float | None = None) -> Capture:
        """Block capture of n samples (None: up to DEFAULT_SAMPLES, or as many as the memory holds)."""
        if not self._enabled:
            raise PicoError("call configure() first")
        tb, real_dt, max_samples = self.find_timebase(dt)
        if n is None:
            n = min(DEFAULT_SAMPLES, max_samples)
        if n > max_samples:
            raise PicoError(
                f"Scope memory holds {max_samples} samples with these channels; asked for {n}. "
                "Use fewer samples or a larger --dt (or record this one in the PicoScope app)."
            )
        h = ctypes.c_int16(self.handle)
        busy = ctypes.c_int32()
        if self.old_api:
            if not self.lib._run_block(h, n, tb, 1, ctypes.byref(busy)):
                raise PicoError("run_block failed")
        elif self.family in ("ps2000a", "ps3000a"):
            self._check(self.lib._run_block(h, 0, n, tb, 1, ctypes.byref(busy), 0, None, None), "RunBlock")
        else:
            self._check(self.lib._run_block(h, 0, n, tb, ctypes.byref(busy), 0, None, None), "RunBlock")

        deadline = time.time() + (timeout if timeout is not None else n * real_dt * 2 + 5)
        while True:
            if self.old_api:
                ready = self.lib._ready(h) > 0
            else:
                r = ctypes.c_int16()
                self._check(self.lib._is_ready(h, ctypes.byref(r)), "IsReady")
                ready = bool(r.value)
            if ready:
                break
            if time.time() > deadline:
                self.lib._stop(h)
                raise PicoError("capture timed out")
            time.sleep(min(0.01, n * real_dt / 10))

        bufs = {c: np.zeros(n, dtype=np.int16) for c in self._enabled}
        ptr = {c: b.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)) for c, b in bufs.items()}
        overflow = ctypes.c_int16()
        if self.old_api:
            got = self.lib._get_values(h, ptr.get("A"), ptr.get("B"), None, None, ctypes.byref(overflow), n)
            if got <= 0:
                raise PicoError("get_values returned no data")
            n_got = int(got)
        else:
            for c, p in ptr.items():
                self._check(self.lib._set_data_buffers(h, CHANNELS.index(c), p, None, n, 0, 0), "SetDataBuffers")
            count = ctypes.c_uint32(n)
            self._check(self.lib._get_values(h, 0, ctypes.byref(count), 1, 0, 0, ctypes.byref(overflow)), "GetValues")
            n_got = int(count.value)
        self.lib._stop(h)

        volts = {c: bufs[c][:n_got].astype(float) / self.max_adc * self._enabled[c][0] for c in bufs}
        clipped = {c for c in bufs if overflow.value & (1 << CHANNELS.index(c))}
        # Some models don't flag overflow reliably; samples at full scale count too.
        clipped |= {c for c in bufs if np.any(np.abs(bufs[c][:n_got]) >= self.max_adc)}
        return Capture(
            t=np.arange(n_got) * real_dt,
            volts=volts,
            ranges={c: v for c, (v, _) in self._enabled.items()},
            clipped=clipped,
            dt=real_dt,
        )


def autorange_capture(
    scope: PicoScope,
    channels: list[str],
    n: int | None,
    dt: float,
    ranges: dict[str, float] | None = None,
    auto: bool = True,
    max_tries: int = 6,
) -> Capture:
    """Capture, adjusting each channel's range until nothing clips and the signal
    fills a reasonable part of the range. `ranges` is the starting point (e.g. the
    previous point's ranges) - with auto=False they are used as given."""
    steps = scope.range_steps()
    want = {c: (ranges or {}).get(c, 10.0 if 10.0 in steps else steps[-1]) for c in channels}
    set_ranges = scope.configure(want)
    cap = scope.capture(n, dt)
    for _ in range(max_tries - 1 if auto else 0):
        changed = False
        for c in channels:
            r = set_ranges[c]
            i = steps.index(r)
            peak = float(np.max(np.abs(cap.volts[c]))) if cap.volts[c].size else 0.0
            if c in cap.clipped:
                if i + 1 < len(steps):
                    want[c] = steps[i + 1]
                    changed = True
            elif peak < 0.4 * r and i > 0:
                target = next(s for s in steps if s >= peak / 0.8)  # leave 20% headroom
                if target < r:
                    want[c] = target
                    changed = True
        if not changed:
            break
        set_ranges = scope.configure(want)
        cap = scope.capture(n, dt)
    for c in sorted(cap.clipped):
        warnings.warn(f"Channel {c} clipped at the {cap.ranges[c]:g} V range", stacklevel=2)
    return cap


def main(argv: list[str] | None = None) -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="CSV file to write")
    p.add_argument("--channels", nargs="+", default=["A", "B"])
    p.add_argument("--dt", type=float, default=5e-6, help="sample interval in s (default 5 µs; nearest faster used)")
    p.add_argument("--samples", type=int, help="default: 10000, or as many as the scope memory holds")
    p.add_argument("--range", type=float, help="fixed range in V for all channels (default: automatic)")
    p.add_argument("--family", default="auto", choices=["auto", *FAMILIES])
    p.add_argument("--resolution", type=int, default=12, help="bits, 5000 series only (8/12/14/15/16)")
    a = p.parse_args(argv)

    channels = [c.upper()[-1] for c in a.channels]
    fixed = {c: a.range for c in channels} if a.range else None
    try:
        with PicoScope(a.family, a.resolution) as scope:
            cap = autorange_capture(scope, channels, a.samples, a.dt, fixed, auto=fixed is None)
    except PicoError as e:
        raise SystemExit(f"error: {e}") from None
    tr = cap.trace()
    out = data_path(a.out)
    save_trace(out, tr.t, tr.channels)
    print(f"{scope.family}: {len(cap.t)} samples at {cap.dt:.3g} s ({cap.t[-1] + cap.dt:.4g} s) -> {out}")
    for c in channels:
        v = cap.volts[c]
        print(f"  {c}: range ±{cap.ranges[c]:g} V, min {v.min():.4g} V, max {v.max():.4g} V")


if __name__ == "__main__":
    main()
