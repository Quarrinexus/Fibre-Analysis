"""Detector voltage <-> optical power, dBm, noise figure, best possible contrast.

    uv run fibre-power 5.0                  # V -> W and dBm
    uv run fibre-power --dbm 3              # dBm -> W and expected detector V
    uv run fibre-power --i1 2.1 --i2 0.4    # best contrast for these arm intensities
    uv run fibre-power --snr-in 1000 --snr-out 180   # noise figure in dB

Detector (DET01CFC): responsivity 0.95 A/W into a 50 Ω load, saturates at 10 V.
"""

from __future__ import annotations

import argparse

import numpy as np

from .contrast import expected_max_contrast

RESPONSIVITY = 0.95  # A/W
LOAD = 50.0  # ohm
SATURATION_V = 10.0


def volts_to_watts(v, responsivity: float = RESPONSIVITY, load: float = LOAD):
    return np.asarray(v) / (load * responsivity)


def watts_to_volts(p, responsivity: float = RESPONSIVITY, load: float = LOAD):
    return np.asarray(p) * load * responsivity


def watts_to_dbm(p):
    return 10 * np.log10(np.asarray(p) / 1e-3)


def dbm_to_watts(dbm):
    return 1e-3 * 10 ** (np.asarray(dbm) / 10)


def noise_figure_db(snr_in, snr_out):
    return 10 * np.log10(np.asarray(snr_in) / np.asarray(snr_out))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volts", nargs="*", type=float, help="detector voltage(s)")
    p.add_argument("--dbm", type=float, nargs="*", help="power(s) in dBm")
    p.add_argument("--i1", type=float, help="arm 1 intensity (other arm blocked), any unit")
    p.add_argument("--i2", type=float, help="arm 2 intensity, same unit")
    p.add_argument("--snr-in", type=float)
    p.add_argument("--snr-out", type=float)
    p.add_argument("--dark", type=float, default=0.0, help="detector dark level in V")
    a = p.parse_args(argv)

    for v in a.volts:
        w = float(volts_to_watts(v - a.dark))
        flag = "  (at/above saturation!)" if v >= SATURATION_V else ""
        print(f"{v:g} V -> {w * 1e3:.4g} mW = {float(watts_to_dbm(w)):.2f} dBm{flag}")
    for d in a.dbm or []:
        w = float(dbm_to_watts(d))
        print(f"{d:g} dBm -> {w * 1e3:.4g} mW -> detector {float(watts_to_volts(w)):.3g} V")
    if a.i1 is not None and a.i2 is not None:
        i1, i2 = a.i1 - a.dark, a.i2 - a.dark
        print(f"I1 = {i1:g}, I2 = {i2:g} -> max contrast 2√(I1 I2)/(I1+I2) = {expected_max_contrast(i1, i2):.4f}")
    if a.snr_in is not None and a.snr_out is not None:
        print(f"Noise figure = {float(noise_figure_db(a.snr_in, a.snr_out)):.2f} dB")
    if not (a.volts or a.dbm or a.i1 is not None or a.snr_in is not None):
        p.print_help()


if __name__ == "__main__":
    main()
