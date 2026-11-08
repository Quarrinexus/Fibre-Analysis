"""Values from the Setup A manufacturer data sheets (Setup_A_manuals_v2.pdf).

Two kinds of value:
* printed on the sheets (centre wavelength, powers, amplifier specs): exact;
* spectral shapes, derived from the plots traced by tools/digitise_datasheets.py
  (curves in datasheets/*.csv). The tracing is accurate to ~1 pixel (0.06 nm for
  the FP plot, 0.2 nm / 0.07 dB for the SLD plot); the sheets' own measurement
  conditions differ from ours (current, temperature), which matters more.

Used as defaults and printed next to results for comparison, never as results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATASHEET_DIR = Path(__file__).parent / "datasheets"

# SLD: Thorlabs SLD1005 (S/N SLD-16063), spectrum at 650 mA. Flat-topped, not Gaussian.
SLD_CENTRE_NM = 1559.0  # midpoint of the -3 dB points (peak at 1563.5 nm)
SLD_FWHM_NM = 67.7  # -3 dB points at 1525.1 and 1592.8 nm

# Fabry-Perot laser: Thorlabs LPS-PM1550-FC (S/N 230906-39) in an S1FC1550PM, 24.1 mA.
FP_CENTRE_NM = 1547.7  # printed "Center Wavelength"; power-weighted centre of the plot is 1547.9
FP_MODE_SPACING_NM = 1.158  # ± 0.003, fit to 11 traced mode peaks -> 2nL = λ²/Δλ ≈ 2.07 mm
FP_ENVELOPE_FWHM_NM = 5.2  # Gaussian fit to the mode heights; uneven, tallest mode at 1549.7 nm
FP_POWER_MW = 1.83

# Amplifiers: S9FC1004P benchtop units containing BOA1004P semiconductor boosters.
AMP_3DB_BANDWIDTH_NM = 88.6
AMP_SATURATION_DBM = 15.2

# Other components
MODULATOR_LOSS_DB = 3.32  # LN65
POLARISER_LOSS_DB = 0.90  # ILP1550PM-APC
POLARISER_EXTINCTION_DB = 28.0


def fp_round_trip_um(centre_nm: float = FP_CENTRE_NM, spacing_nm: float = FP_MODE_SPACING_NM) -> float:
    return (centre_nm * 1e-3) ** 2 / (spacing_nm * 1e-3)


def datasheet_spectrum(source: str) -> tuple[np.ndarray, np.ndarray]:
    """The manufacturer's spectrum, traced off the data sheet: (wavelength nm,
    linear intensity normalised to 1). The SLD plot is in dB and is converted."""
    name = {"sld": "sld_spectrum.csv", "fp": "fp_spectrum.csv"}[source]
    d = np.loadtxt(DATASHEET_DIR / name, delimiter=",", comments="#", encoding="utf-8")
    lam, y = d[:, 0], d[:, 1]
    if source == "sld":
        y = 10 ** (y / 10)
    return lam, np.clip(y, 0, None) / np.max(y)
