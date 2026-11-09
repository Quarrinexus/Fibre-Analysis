"""fibre-count, fibre-scan --repeats and fibre-monitor: known answers and fake-scope runs."""

import csv

import numpy as np
import pytest
from fake_pico import TRUE_CONTRAST, FakeNewApi

from fibre_analysis import capture, count, scan, setup_a
from fibre_analysis.io import save_trace

RNG = np.random.default_rng(7)
LAM_UM = setup_a.FP_CENTRE_NM * 1e-3


def stop_start_recording(distance_um: float, passes: float = 1.0, fs: float = 200.0, seconds: float = 40.0):
    """Detector signal while the micrometer is turned by hand: moves at uneven speed
    with pauses, never reverses. Slow thermal drift (a fraction of a fringe) runs
    throughout, and there is noise."""
    t = np.arange(int(fs * seconds)) / fs
    # Speed profile: bursts of turning separated by pauses.
    speed = np.zeros_like(t)
    for start in np.arange(2.0, seconds - 4, 4.0):
        burst = (t >= start) & (t < start + RNG.uniform(1.0, 2.5))
        speed[burst] = RNG.uniform(0.5, 2.0)
    x = np.cumsum(speed) / fs
    x *= distance_um / x[-1]  # micrometer movement in µm
    drift = 0.15 * np.sin(2 * np.pi * t / 50)  # fringes of drift
    phase = 2 * np.pi * (passes * x / LAM_UM + drift)
    v = 4.0 * (1 + 0.9 * np.cos(phase)) + RNG.normal(0, 0.03, t.size)
    return t, v


@pytest.mark.parametrize("passes", [1.0, 2.0])
def test_count_stop_start_turning(passes):
    distance = 200.0  # µm = 20 divisions
    _, v = stop_start_recording(distance, passes)
    c = count.count_crossings(v)
    expected = passes * distance / LAM_UM
    assert c.fringes == pytest.approx(expected, abs=0.75)  # half-fringe resolution, plus drift
    s = count.pass_factor(c.fringes, 12.6, 12.8, setup_a.FP_CENTRE_NM)
    assert s == pytest.approx(passes, rel=0.01)


def test_count_ignores_noise_while_still():
    t = np.arange(8000) / 200.0
    v = 4.0 * (1 + 0.9 * np.cos(2 * np.pi * 0.1 * np.sin(2 * np.pi * t / 60))) + RNG.normal(0, 0.03, t.size)
    ref = (0.4, 7.6)  # fringe min/max from a moving run
    assert count.count_crossings(v, ref=ref).fringes == 0
    assert count.drift_fringes(v, *ref) == pytest.approx(0.1, abs=0.03)  # a tenth of a fringe of drift


def test_count_cli_file_and_log(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for k in range(2):
        t, v = stop_start_recording(200.0)
        save_trace(tmp_path / f"rec{k}.csv", t, {"Channel A": v})
        count.main([str(tmp_path / f"rec{k}.csv"), "--start", "12.6", "--end", "12.8"])
    out = capsys.readouterr().out
    assert "All 2 run(s)" in out and "single pass" in out
    rows = list(csv.DictReader((tmp_path / "data" / "fringecount.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert float(rows[0]["pass_factor"]) == pytest.approx(1.0, rel=0.01)
    assert (tmp_path / "output" / "rec0.png").exists()


def test_count_cli_drift_run_is_not_averaged(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    t, v = stop_start_recording(200.0)
    save_trace(tmp_path / "rec.csv", t, {"Channel A": v})
    count.main([str(tmp_path / "rec.csv"), "--start", "12.6", "--end", "12.8"])
    still_t = np.arange(8000) / 200.0
    still_v = 4.0 * (1 + 0.9 * np.cos(2 * np.pi * (0.3 + 0.1 * np.sin(2 * np.pi * still_t / 60))))
    save_trace(tmp_path / "still.csv", still_t, {"Channel A": still_v})
    count.main([str(tmp_path / "still.csv"), "--start", "12.8", "--end", "12.8"])  # micrometer not moved
    out = capsys.readouterr().out
    assert "didn't move: about 0." in out
    assert "All 1 run(s)" in out.split("didn't move")[1]


def use_drivers(monkeypatch, drivers):
    def load(family):
        if family not in drivers:
            raise OSError("PicoSDK not found")
        return drivers[family]

    monkeypatch.setattr(capture, "_load_driver", load)


def test_count_cli_with_scope(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi()})
    monkeypatch.setattr("builtins.input", lambda _="": "")
    count.main(["--pico", "--duration", "0.01", "--start", "12.6", "--end", "12.8"])
    assert len(list((tmp_path / "data").glob("fringecount_*.csv"))) == 1
    assert (tmp_path / "data" / "fringecount.csv").exists()


def test_scan_repeats_and_whole_point_undo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi()})
    answers = iter(["1", "u", "2", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    scan.main(["--out", "rep.csv", "--pico", "--repeats", "3", "--repeat-wait", "0", "--no-plot"])
    s = scan.load_scan("rep.csv")
    assert list(s.x) == [2.0]  # all 3 captures at x=1 were undone together
    assert list(s.n) == [3]
    assert s.contrast[0] == pytest.approx(TRUE_CONTRAST, abs=0.01)
    assert len(list((tmp_path / "data" / "rep_traces").glob("*.csv"))) == 6  # unique names, none overwritten


def test_next_position_steps_without_drift():
    x = scan.next_position(None, 24.5, -0.1, 0.5)
    for _ in range(100):
        x = scan.next_position(x, None, -0.1, 0.5)
    assert x == 14.5  # exact, not 14.499999999
    assert scan.next_position(0.5, None, -0.1, 0.5) is None  # past --stop
    assert scan.next_position(0.6, None, -0.1, 0.5) == 0.5  # --stop itself is included
    assert scan.next_position(3.0, None, None, None) is None  # no --step: nothing suggested


def test_scan_enter_takes_suggested_position(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi()})
    # 12.3, 12.2, undo 12.2, 12.2 again, 12.1, 12.0, then past --stop: Enter alone is refused.
    answers = iter(["", "", "u", "", "", "", "", "q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    args = ["--out", "step.csv", "--pico", "--unit", "mm", "--no-plot", "--step", "-0.1"]
    scan.main([*args, "--start", "12.3", "--stop", "12.0"])
    assert list(scan.load_scan("step.csv").x) == pytest.approx([12.0, 12.1, 12.2, 12.3])

    # Re-run without --start: carries on from the last point; a typed value restarts the steps.
    answers = iter(["", "5", "", "q"])
    scan.main(args)
    assert list(scan.load_scan("step.csv").x) == pytest.approx([4.9, 5.0, 11.9, 12.0, 12.1, 12.2, 12.3])


def test_monitor_logs_against_time(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    use_drivers(monkeypatch, {"ps2000a": FakeNewApi()})
    scan.monitor_main(["--out", "mon.csv", "--interval", "0.05", "--duration", "0.2"])
    s = scan.load_scan("mon.csv")
    assert s.unit == "s"
    assert len(s.x) >= 3 and np.all(np.diff(s.x) > 0)
    assert np.allclose(s.contrast, TRUE_CONTRAST, atol=0.01)
    assert (tmp_path / "output" / "mon.png").exists()
    with pytest.raises(SystemExit):  # refuses to append to an existing run
        scan.monitor_main(["--out", "mon.csv", "--interval", "0.05", "--duration", "0.1"])
