"""PicoScope capture code against fake drivers (no hardware needed)."""

import numpy as np
import pytest
from fake_pico import TRUE_CONTRAST, FakeNewApi, FakePs2000

from fibre_analysis import capture, contrast, io, scan


def use_drivers(monkeypatch, drivers):
    """drivers: {family: fake}; any other family behaves as 'driver not installed'."""

    def load(family):
        if family not in drivers:
            raise OSError("PicoSDK not found")
        return drivers[family]

    monkeypatch.setattr(capture, "_load_driver", load)


def test_autodetect_and_power_source(monkeypatch):
    fake = FakeNewApi("ps5000a", open_status=0x11A)
    use_drivers(monkeypatch, {"ps5000a": fake})
    with capture.PicoScope() as scope:
        assert scope.family == "ps5000a"
        assert fake.power_changed
    assert fake.closed


def test_no_scope_message(monkeypatch):
    use_drivers(monkeypatch, {})
    with pytest.raises(capture.PicoError, match="No PicoScope found"):
        capture.PicoScope()


@pytest.mark.parametrize("family", ["ps2000a", "ps4000a"])
def test_timebase_is_slowest_within_request(monkeypatch, family):
    use_drivers(monkeypatch, {family: FakeNewApi(family)})
    with capture.PicoScope(family) as scope:
        scope.configure({"A": 5})
        tb, dt, _ = scope.find_timebase(2e-6)
        assert dt <= 2e-6
        assert scope._timebase(tb + 1)[0] > 2e-6


@pytest.mark.parametrize("family", ["ps2000a", "ps3000a", "ps4000a", "ps5000a"])
def test_capture_autorange_and_scaling(monkeypatch, family):
    use_drivers(monkeypatch, {family: FakeNewApi(family)})
    with capture.PicoScope(family) as scope:
        cap = capture.autorange_capture(scope, ["A", "B"], 10_000, 2e-6)
    assert cap.ranges == {"A": 5, "B": 10}  # A: 3.3 V peak -> 5 V; B: ±5 V needs 10 V
    assert not cap.clipped
    assert cap.volts["A"].max() == pytest.approx(3.3, abs=0.01)
    assert contrast.contrast(cap.volts["A"]).contrast == pytest.approx(TRUE_CONTRAST, abs=0.005)
    assert np.ptp(cap.volts["B"]) == pytest.approx(10, abs=0.05)


def test_autorange_steps_up_from_clipping(monkeypatch):
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi()})
    with capture.PicoScope() as scope:
        cap = capture.autorange_capture(scope, ["A"], 5000, 2e-6, ranges={"A": 1})
    assert cap.ranges["A"] == 5
    assert not cap.clipped


def test_fixed_range_warns_when_clipped(monkeypatch):
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi()})
    with capture.PicoScope() as scope, pytest.warns(UserWarning, match="clipped"):
        cap = capture.autorange_capture(scope, ["A"], 5000, 2e-6, ranges={"A": 2}, auto=False)
    assert "A" in cap.clipped


def test_range_below_model_minimum_steps_up(monkeypatch):
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi(min_range=2)})  # smallest range 50 mV
    with capture.PicoScope() as scope:
        assert scope.configure({"A": 0.01}) == {"A": 0.05}


def test_too_many_samples(monkeypatch):
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi(max_samples=8000)})
    with capture.PicoScope() as scope:
        scope.configure({"A": 5, "B": 10})
        with pytest.raises(capture.PicoError, match="memory holds 4000"):
            scope.capture(10_000, 2e-6)


def test_old_ps2000_api(monkeypatch):
    fake = FakePs2000()
    use_drivers(monkeypatch, {"ps2000": fake})
    with capture.PicoScope() as scope:
        assert scope.family == "ps2000"
        cap = capture.autorange_capture(scope, ["A", "B"], 3000, 2e-6)
    assert cap.dt <= 2e-6
    assert cap.ranges == {"A": 5, "B": 10}
    assert contrast.contrast(cap.volts["A"]).contrast == pytest.approx(TRUE_CONTRAST, abs=0.005)
    assert fake.closed


def test_capture_cli(monkeypatch, tmp_path):
    use_drivers(monkeypatch, {"ps3000a": FakeNewApi("ps3000a")})
    out = tmp_path / "t.csv"
    capture.main([str(out), "--channels", "A", "B", "--samples", "4000"])
    tr = io.load_trace(out)
    assert tr.dt <= 2e-6
    assert contrast.contrast(tr.channel("A")).contrast == pytest.approx(TRUE_CONTRAST, abs=0.005)


def test_scan_with_pico(monkeypatch, tmp_path):
    fake = FakeNewApi()
    use_drivers(monkeypatch, {"ps2000a": fake})
    answers = iter(["0", "0.5", "u", "1", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    out = tmp_path / "scan.csv"
    scan.main(["--out", str(out), "--pico", "--record", "B", "--no-plot"])
    s = scan.load_scan(out)
    assert list(s.x) == [0, 1]
    assert s.contrast == pytest.approx([TRUE_CONTRAST] * 2, abs=0.005)
    traces = sorted((tmp_path / "scan_traces").glob("*.csv"))
    assert len(traces) == 3  # the undone point's trace is kept on disk
    assert set(io.load_trace(traces[0]).channels) == {"Channel A", "Channel B"}
    assert fake.closed
