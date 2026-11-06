"""Check each script recovers known values from fake data."""

from pathlib import Path

import numpy as np
import pytest

from fibre_analysis import calibrate, contrast, envelope, io, power, scan, setup_a, synth, vibrometer
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
    t, v, drive = synth.fringe_trace(0.0, *synth.fp_spectrum(), fringes_per_sweep=3.6, duration=20e-3)
    cal = calibrate.calibrate(v, drive, fs=1 / (t[1] - t[0]), wavelength_nm=setup_a.FP_CENTRE_NM)
    assert cal.fringes_per_sweep == pytest.approx(3.6, rel=0.01)
    assert cal.opd_swing_um == pytest.approx(3.6 * 1.5477, rel=0.01)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    synth.main([str(out)])
    return out


def test_sld_bandwidth(demo):
    scan = load_scan(demo / "sld_scan.csv")
    res = envelope.analyse_sld(scan.opd_um(), scan.contrast, scan.contrast_err, setup_a.SLD_CENTRE_NM)
    assert res.bandwidth_nm == pytest.approx(setup_a.SLD_FWHM_NM, rel=0.07)
    assert res.order == pytest.approx(4, abs=1)  # flat-topped, as on the data sheet
    assert res.gauss_bandwidth_nm < 0.85 * setup_a.SLD_FWHM_NM  # why a Gaussian-only fit misleads
    assert res.x0 == pytest.approx(30, abs=3)  # zero set at 3 div = 30 µm


def test_fp_cavity(demo):
    scan = load_scan(demo / "fp_scan.csv")
    res = envelope.analyse_fp(scan.opd_um(), scan.contrast, setup_a.FP_CENTRE_NM)
    assert res.peaks_um.size == 3  # the one near the start counts; the rise at the end doesn't
    assert res.spacing_um == pytest.approx(setup_a.fp_round_trip_um(), rel=0.02)
    assert res.mode_spacing_nm == pytest.approx(setup_a.FP_MODE_SPACING_NM, rel=0.02)
    assert res.envelope_fwhm_nm == pytest.approx(setup_a.FP_ENVELOPE_FWHM_NM, rel=0.25)


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


def test_demo_signal_levels(demo):
    # mV-level signals like the real setup, with a dark offset that matters.
    dark = io.load_trace(demo / "dark.csv").channel()
    assert np.mean(dark) == pytest.approx(synth.DARK, abs=1e-4)
    fp = load_scan(demo / "fp_scan.csv")
    assert np.max(fp.mean) < 0.03


def test_dark_offset_matters_at_mv_levels():
    _t, v, _ = synth.fringe_trace(0.0, *synth.fp_spectrum(), fringes_per_sweep=3.5)
    true = contrast.expected_max_contrast(*synth.FP_ARMS)
    assert contrast.contrast(v, dark=synth.DARK).contrast == pytest.approx(true, abs=0.01)
    assert contrast.contrast(v).contrast < true - 0.1  # forgetting --dark costs >10%


def test_clis_run(demo, tmp_path):
    envelope.main([str(demo / "sld_scan.csv"), "--source", "sld", "--plot", str(tmp_path / "s.png")])
    envelope.main([str(demo / "fp_scan.csv"), "--source", "fp", "--plot", str(tmp_path / "f.png")])
    vibrometer.main([str(demo / "vibrometer.csv"), "--plot", str(tmp_path / "v.png")])
    calibrate.main([str(demo / "calibration.csv")])
    contrast.main([str(next((demo / "sld").glob("*.csv")))])
    power.main(["5"])
    assert (tmp_path / "v.png").exists()


def test_spectrum_reconstruction(demo):
    from fibre_analysis import spectrum

    s = load_scan(demo / "sld_scan.csv")
    sp = spectrum.reconstruct(s.opd_um(), s.contrast, setup_a.SLD_CENTRE_NM)
    assert sp.wavelength_nm[np.argmax(sp.intensity)] == pytest.approx(setup_a.SLD_CENTRE_NM, abs=2)
    assert sp.fwhm_nm() == pytest.approx(setup_a.SLD_FWHM_NM, rel=0.15)  # limited by ~24 nm resolution

    f = load_scan(demo / "fp_scan.csv")
    sp = spectrum.reconstruct(f.opd_um(), f.contrast, setup_a.FP_CENTRE_NM)
    pk = sp.peaks_nm(min_height=0.4)
    assert np.mean(np.diff(pk)) == pytest.approx(setup_a.FP_MODE_SPACING_NM, rel=0.02)


def test_spectrum_cli(demo, tmp_path):
    import shutil

    from fibre_analysis import spectrum

    shutil.copy(demo / "fp_scan.csv", tmp_path / "fp.csv")
    spectrum.main([str(tmp_path / "fp.csv"), "--source", "fp"])
    assert (tmp_path / "fp_spectrum.csv").exists()
    assert (tmp_path / "output" / "fp_spectrum.png").exists()


def test_repeated_points_are_averaged(tmp_path):
    rows = [
        {
            "x": "1",
            "unit": "mm",
            "contrast": "0.9",
            "contrast_err": "0.01",
            "contrast_rms": "0.9",
            "i_max": "",
            "i_min": "",
            "mean": "1",
            "file": "",
        },
        {
            "x": "2",
            "unit": "mm",
            "contrast": "0.3",
            "contrast_err": "0.01",
            "contrast_rms": "0.3",
            "i_max": "",
            "i_min": "",
            "mean": "1",
            "file": "",
        },
        {
            "x": "2",
            "unit": "mm",
            "contrast": "0.7",
            "contrast_err": "0.01",
            "contrast_rms": "0.7",
            "i_max": "",
            "i_min": "",
            "mean": "1",
            "file": "",
        },
    ]
    scan._write_rows(tmp_path / "s.csv", rows)
    s = load_scan(tmp_path / "s.csv")
    assert list(s.x) == [1, 2]
    assert list(s.n) == [1, 2]
    assert s.contrast[1] == pytest.approx(0.5)
    assert s.contrast_err[1] == pytest.approx(0.2)  # spread of the repeats, not the tiny per-trace error


def test_data_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert io.data_path("x.csv") == Path("data/x.csv")
    assert (tmp_path / "data").is_dir()
    assert io.data_path("sub/x.csv") == Path("sub/x.csv")
    (tmp_path / "data" / "y.csv").write_text("t,v\n0,1\n1,2\n")
    assert io.find_data("y.csv") == Path("data/y.csv")
    assert io.load_trace("y.csv").channel()[1] == 2
    assert io.graph_path("data/y.csv", ".png") == Path("output/y.png")


def _fp_trace(n_fringes, duration=20e-3, mod_freq=100.0, **kw):
    nu, s = synth.fp_spectrum()
    return synth.fringe_trace(
        0.0, nu, s, 4.0, 3.5, fringes_per_sweep=n_fringes, mod_freq=mod_freq, fs=390e3, duration=duration,
        noise=0.01, dark=0.0, rng=RNG, **kw,
    )  # fmt: skip


@pytest.mark.parametrize("n_fringes", [0.99, 1.7, 3.6])
def test_per_period_contrast_known_answer(n_fringes):
    t, v, drive = _fp_trace(n_fringes)
    true = contrast.expected_max_contrast(4.0, 3.5)
    fs = 1 / (t[1] - t[0])
    r = contrast.contrast_per_period(v, fs, drive)
    assert r.method == "period" and r.n >= 2
    assert r.contrast == pytest.approx(true, abs=0.01)
    assert r.error < 0.01  # honest: periods agree, so the error is small


def test_per_period_timed_from_detector_at_one_fringe_per_sweep():
    t, v, _ = _fp_trace(1.0)
    fs = 1 / (t[1] - t[0])
    # Triangle drive at 100 Hz: each 5 ms ramp holds one fringe, so one fringe period = one ramp = a full 2π.
    assert contrast.sweep_period(fs, v) == pytest.approx(5e-3, rel=0.02)
    r = contrast.contrast_per_period(v, fs)
    assert r.contrast == pytest.approx(contrast.expected_max_contrast(4.0, 3.5), abs=0.01)


def test_per_period_resists_a_glitch():
    nu, s = synth.fp_spectrum()
    t, v, drive = synth.fringe_trace(
        0.0, nu, s, 4.0, 1.0, fringes_per_sweep=2.0, mod_freq=100.0, fs=390e3, duration=25e-3, noise=0.01, dark=0.0,
        rng=RNG,
    )  # fmt: skip
    fs = 1 / (t[1] - t[0])
    clean = contrast.contrast_per_period(v, fs, drive).contrast
    assert clean == pytest.approx(0.8, abs=0.01)  # minimum ~1 V, like the real 14.5 mm trace
    v = v.copy()
    v[-300:] = 0.1  # brief dropout in the last, incomplete period
    assert contrast.contrast(v).contrast > clean + 0.02  # whole-trace estimate is dragged
    assert contrast.contrast_per_period(v, fs, drive).contrast == pytest.approx(clean, abs=0.005)


def test_rescan_writes_new_file_and_keeps_original(demo, tmp_path, monkeypatch):
    import shutil

    monkeypatch.chdir(tmp_path)
    shutil.copy(demo / "fp_scan.csv", tmp_path / "fp.csv")
    before = (tmp_path / "fp.csv").read_text()
    scan.rescan_main([str(tmp_path / "fp.csv"), "--dark", str(synth.DARK)])
    assert (tmp_path / "fp.csv").read_text() == before
    new = load_scan(tmp_path / "fp_period.csv")
    old = load_scan(tmp_path / "fp.csv")
    assert np.allclose(new.contrast, old.contrast, atol=0.02)
