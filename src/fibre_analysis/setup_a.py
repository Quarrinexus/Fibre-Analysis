"""Values from the Setup A manufacturer data sheets (Setup_A_manuals_v2.pdf).

Spectral values are read by eye off scanned plots, so treat them as ±10%.
They are used as defaults and printed next to your results for comparison.
"""

# SLD: Thorlabs SLD1005 (S/N SLD-16063), spectrum at 650 mA. Flat-topped, not Gaussian.
SLD_CENTRE_NM = 1560.0
SLD_FWHM_NM = 62.0

# Fabry-Perot laser: Thorlabs LPS-PM1550-FC (S/N 230906-39) in an S1FC1550PM, 24.1 mA.
FP_CENTRE_NM = 1547.7
FP_MODE_SPACING_NM = 1.15  # -> round trip 2nL = λ²/Δλ ≈ 2.08 mm
FP_ENVELOPE_FWHM_NM = 4.5  # uneven, with a tail to short wavelengths
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
