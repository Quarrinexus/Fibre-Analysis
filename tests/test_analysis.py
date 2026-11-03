"""Check each script recovers known values from fake data."""

import numpy as np
import pytest

from fibre_analysis import calibrate, contrast, envelope, io, power, scan, synth, vibrometer
from fibre_analysis.scan import load_scan

RNG = np.random.default_rng(0)


def sine_trace(c, mean=2.0, noise=0.0, n=20000, fringes=20):
    t = np.linspace(0, 1, n)
    return mean * (1 + c * np.cos(2 * np.pi * fringes * t)) + RNG.normal(0, noise, n)


@pytest.mark.parametrize("c", [0.05, 0.3, 0.9])
def test_contrast_rms_unbiased_with_noise(c):
    r = contrast.contrast(sine_trace(c, noise=0.05), method="rms")
    assert r.contrast == pytest.approx(c, abs=0.01)


def test_contrast_peak_clean():
    r = contrast.contrast(sine_trace(0.6), method="peak")
    assert r.contrast == pytest.approx(0.6, abs=0.01)


@pytest.mark.parametrize("n", [2.3, 3.6])
@pytest.mark.parametrize("opd", [0.0, 0.2e-6, 0.4e-6, 0.6e-6])
def test_contrast_non_integer_fringes(n, opd):
    # Real sweeps never hold a whole number of fringes; the default must not care.
    nu, s = synth.sld_spectrum(fwhm_nm=0.001)  # ~monochromatic: true C = 2√(I1 I2)/(I1+I2)
    _, v, _ = synth.fringe_trace(opd, nu, s, i1=1.0, i2=0.8, fringes_per_sweep=n, noise=0.01, rng=RNG)
    assert contrast.contrast(v).contrast == pytest.approx(contrast.expected_max_contrast(1.0, 0.8), rel=0.01)


@pytest.mark.parametrize("c", [0.05, 0.3, 0.9])
def test_contrast_peak_default(c):
    assert contrast.contrast(sine_trace(c, noise=0.02)).contrast == pytest.approx(c, abs=0.01)


def test_contrast_dark_offset():
    v = sine_trace(0.5) + 0.3
    assert contrast.contrast(v, dark=0.3).contrast == pytest.approx(0.5, abs=0.01)


def test_expected_max_contrast():
    assert contrast.expected_max_contrast(1, 1) == pytest.approx(1)
    assert contrast.expected_max_contrast(4, 1) == pytest.approx(0.8)


def test_io_roundtrip_and_picoscope_units(tmp_path):
    t = np.linspace(0, 1e-3, 100)
    io.save_trace(tmp_path / "a.csv", t, {"Channel A": np.sin(t), "Channel B": np.cos(t)})
    tr = io.load_trace(tmp_path / "a.csv")
    assert np.allclose(tr.t, t)
    assert np.allclose(tr.channel("B"), np.cos(t))

    (tmp_path / "b.csv").write_text("Time,Channel A\n(us),(mV)\n\n0,100\n1,∞\n2,300\n")
    with pytest.warns(UserWarning, match="over-range"):
        tr = io.load_trace(tmp_path / "b.csv")
    assert tr.dt == pytest.approx(1e-6)
    assert tr.channel()[0] == pytest.approx(0.1)
    assert np.isnan(tr.channel()[1])

    (tmp_path / "c.csv").write_text("Time (ms),Channel A (mV)\n0,100\n1,200\n")
    tr = io.load_trace(tmp_path / "c.csv")
    assert tr.dt == pytest.approx(1e-3)
    assert tr.channel("A")[1] == pytest.approx(0.2)


def test_power_prep_question():
    w = power.volts_to_watts(5.0)
    assert w == pytest.approx(5 / (50 * 0.95))
    assert power.watts_to_dbm(w) == pytest.approx(20.22, abs=0.01)
    assert power.dbm_to_watts(0) == pytest.approx(1e-3)


def test_calibration():
    t, v, drive = synth.fringe_trace(0.0, *synth.sld_spectrum(), fringes_per_sweep=3.6, duration=20e-3, noise=0.005)
    cal = calibrate.calibrate(v, drive, fs=1 / (t[1] - t[0]))
    assert cal.fringes_per_sweep == pytest.approx(3.6, rel=0.01)
    assert cal.opd_swing_um == pytest.approx(3.6 * 1.55, rel=0.01)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    synth.main([str(out)])
    return out


def test_sld_bandwidth(demo):
    scan = load_scan(demo / "sld_scan.csv")
    res = envelope.analyse_sld(scan.opd_um(), scan.contrast, scan.contrast_err)
    assert res.bandwidth_nm == pytest.approx(50, rel=0.15)
    assert res.fit.x0 == pytest.approx(30, abs=3)  # zero set at 3 div = 30 µm


def test_fp_cavity(demo):
    scan = load_scan(demo / "fp_scan.csv")
    res = envelope.analyse_fp(scan.opd_um(), scan.contrast)
    assert res.peaks_um.size == 3  # including the one near the scan edge at 120 µm
    assert res.spacing_um == pytest.approx(2400, rel=0.03)
    assert res.optical_length_um == pytest.approx(1200, rel=0.03)
    assert res.envelope_fwhm_nm == pytest.approx(3.0, rel=0.25)


def test_vibrometer(demo):
    tr = io.load_trace(demo / "vibrometer.csv")
    res = vibrometer.analyse(tr.channel("A"), tr.fs, tr.channel("B"))
    assert res.drive_freq == pytest.approx(1.5, rel=0.02)
    assert res.amplitude == pytest.approx(40e-6, rel=0.05)
    assert res.amplitude_counting == pytest.approx(40e-6, rel=0.05)
    assert np.ptp(res.displacement) / 2 == pytest.approx(40e-6, rel=0.1)


@pytest.mark.parametrize(("amp", "f"), [(5e-6, 1.0), (300e-6, 1.5)])
def test_vibrometer_amplitude_range(amp, f):
    t, v, d = synth.vibrometer_trace(amplitude=amp, drive_freq=f, rng=RNG)
    res = vibrometer.analyse(v, 1 / (t[1] - t[0]), d)
    assert res.amplitude == pytest.approx(amp, rel=0.08)
    assert res.amplitude_counting == pytest.approx(amp, rel=0.08)


def test_scan_logger(demo, tmp_path, monkeypatch):
    traces = sorted((demo / "sld").glob("*.csv"))[:2]
    answers = iter(["1.5", str(traces[0]), "2", str(traces[1]), "u", "abc", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = tmp_path / "scan.csv"
    scan.main(["--out", str(out), "--no-plot"])
    rows = scan.load_scan(out)
    assert list(rows.x) == [1.5]  # second point undone
    # Resuming keeps the existing point.
    answers = iter(["3", str(traces[1]), "q"])
    scan.main(["--out", str(out), "--no-plot"])
    assert list(scan.load_scan(out).x) == [1.5, 3.0]


def test_clis_run(demo, tmp_path):
    envelope.main([str(demo / "sld_scan.csv"), "--source", "sld", "--plot", str(tmp_path / "s.png")])
    envelope.main([str(demo / "fp_scan.csv"), "--source", "fp", "--plot", str(tmp_path / "f.png")])
    vibrometer.main([str(demo / "vibrometer.csv"), "--plot", str(tmp_path / "v.png")])
    calibrate.main([str(demo / "calibration.csv")])
    contrast.main([str(next((demo / "sld").glob("*.csv")))])
    power.main(["5"])
    assert (tmp_path / "v.png").exists()
