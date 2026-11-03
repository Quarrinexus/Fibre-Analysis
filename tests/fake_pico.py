"""Stand-ins for the picosdk ctypes drivers, so capture code runs without a scope.

They take the same arguments as the real C functions (handles, byref() pointers,
buffer pointers) and fill buffers with a known signal:
    A: detector, 1.8 V mean, 1.5 V fringes at 2.4 kHz   -> contrast 0.8333
    B: function generator, ±5 V triangle at 300 Hz
"""

import ctypes

import numpy as np

RANGES = {i: v for i, v in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])}
TRUE_CONTRAST = 1.5 / 1.8


def signal(ch: int, t: np.ndarray) -> np.ndarray:
    if ch == 0:
        return 1.8 + 1.5 * np.cos(2 * np.pi * 2400 * t)
    return 5 * (2 * np.abs(2 * ((t * 300) % 1) - 1) - 1)


def _set(ref, value):
    ref._obj.value = value


class FakeNewApi:
    """ps2000a / ps3000a / ps4000a / ps5000a style (PICO_STATUS return codes)."""

    def __init__(self, family="ps2000a", open_status=0, min_range=0, max_samples=500_000):
        self.family = family
        self.PICO_VOLTAGE_RANGE = dict(RANGES)
        self.open_status = open_status
        self.min_range = min_range
        self.max_samples = max_samples
        self.channels = {}
        self.buffers = {}
        self.power_changed = False
        self.closed = False

    def _open_unit(self, handle, serial, *resolution):
        _set(handle, 1)
        return self.open_status

    def _change_power_source(self, handle, status):
        self.power_changed = True
        return 0

    def _maximum_value(self, handle, value):
        _set(value, 32512)
        return 0

    def _set_channel(self, handle, ch, enabled, coupling, rng, offset):
        if enabled and rng < self.min_range:
            return 0x0C  # PICO_INVALID_VOLTAGE_RANGE
        if enabled:
            self.channels[ch] = rng
        else:
            self.channels.pop(ch, None)
        return 0

    def _interval_ns(self, tb):
        return 2.0**tb if tb < 3 else 8.0 * (tb - 2)

    def _get_timebase2(self, handle, tb, n, interval, *rest):
        if tb == 0 or tb > 2**31:  # lowest id invalid, as on real scopes with 2+ channels
            return 0x0E
        _set(interval, self._interval_ns(tb))
        max_ref = rest[1] if self.family in ("ps2000a", "ps3000a") else rest[0]
        _set(max_ref, self.max_samples // max(len(self.channels), 1))
        return 0

    def _run_block(self, handle, pre, post, tb, *rest):
        self.n, self.dt = post, self._interval_ns(tb) * 1e-9
        return 0

    def _is_ready(self, handle, ready):
        _set(ready, 1)
        return 0

    def _set_data_buffers(self, handle, ch, ptr, ptr_min, n, seg, mode):
        self.buffers[ch] = (ptr, n)
        return 0

    def _get_values(self, handle, start, count, ratio, mode, seg, overflow):
        t = np.arange(self.n) * self.dt
        flags = 0
        for ch, (ptr, n) in self.buffers.items():
            v = signal(ch, t)
            rng = RANGES[self.channels[ch]]
            if np.any(np.abs(v) > rng):
                flags |= 1 << ch
            arr = np.ctypeslib.as_array(ptr, shape=(n,))
            arr[:] = np.clip(np.round(v / rng * 32512), -32512, 32512).astype(np.int16)
        _set(overflow, flags)
        return 0

    def _stop(self, handle):
        return 0

    def _close_unit(self, handle):
        self.closed = True
        return 0


class FakePs2000:
    """Old ps2000 API (2204A/2205A): 0 = failure return codes, 8 kS memory."""

    def __init__(self):
        self.PICO_VOLTAGE_RANGE = {k: v for k, v in RANGES.items() if k >= 2}  # 50 mV - 20 V
        self.channels = {}
        self.closed = False

    def _open_unit(self):
        return 1

    def _set_channel(self, handle, ch, enabled, dc, rng):
        if enabled:
            if rng < 2:
                return 0
            self.channels[ch] = rng
        else:
            self.channels.pop(ch, None)
        return 1

    def _get_timebase(self, handle, tb, n, interval, units, oversample, max_samples):
        if tb > 20:
            return 0
        _set(interval, int(10 * 2**tb))
        _set(max_samples, 8000 // max(len(self.channels), 1))
        return 1

    def _run_block(self, handle, n, tb, oversample, busy):
        self.n, self.dt = n, 10 * 2**tb * 1e-9
        return 1

    def _ready(self, handle):
        return 1

    def _get_values(self, handle, a, b, c, d, overflow, n):
        t = np.arange(n) * self.dt
        for ch, ptr in ((0, a), (1, b)):
            if ptr is None:
                continue
            rng = RANGES[self.channels[ch]]
            arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int16)), shape=(n,))
            arr[:] = np.clip(np.round(signal(ch, t) / rng * 32767), -32767, 32767).astype(np.int16)
        _set(overflow, 0)
        return n

    def _stop(self, handle):
        return 1

    def _close_unit(self, handle):
        self.closed = True
        return 1
