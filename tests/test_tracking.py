"""control tests for the experiment tracker (src/tracking.py)."""

import json

import pytest

from src import tracking


def _history(folder, run_id):
    return [json.loads(l) for l in (folder / f"{run_id}.history.jsonl").read_text().splitlines()]


def test_run_lifecycle_and_thinning(tmp_path):
    exp = tracking.Experiment("t", title="test", config={"a": 1}, root=tmp_path)
    exp.start(n_runs=1)
    with exp.run("r0", params={"method": "HEA"}, reference=-1.0) as run:
        for step in range(1235):
            run.log(step, -1.0 + 1.0 / (step + 1), grad_norm=0.1, evaluations=step)
        run.finish(dE_mHa=0.5, converged=True)
    exp.finish()

    folder = tmp_path / "t"
    meta = json.loads((folder / "experiment.json").read_text())
    rec = json.loads((folder / "r0.json").read_text())
    steps = [p["step"] for p in _history(folder, "r0")]

    assert meta["status"] == "done" and meta["n_runs"] == 1 and meta["config"] == {"a": 1}
    assert rec["status"] == "done" and rec["metrics"]["dE_mHa"] == 0.5
    assert exp.is_done("r0")
    assert steps[:100] == list(range(100))           # every step up to 100
    assert 110 in steps and 115 not in steps         # every 10th up to 1000
    assert 1200 in steps and 1210 not in steps       # every 100th after that
    assert steps[-1] == 1234                          # final step always kept
    assert not list(folder.glob("*.tmp"))             # atomic writes leave nothing behind


def test_failed_run_is_marked(tmp_path):
    exp = tracking.Experiment("t", title="test", root=tmp_path)
    with pytest.raises(RuntimeError):
        with exp.run("bad", params={}):
            raise RuntimeError("boom")
    rec = json.loads((tmp_path / "t" / "bad.json").read_text())
    assert rec["status"] == "failed" and "boom" in rec["error"]
    assert not exp.is_done("bad")


def test_restart_keeps_start_date(tmp_path):
    exp = tracking.Experiment("t", title="test", root=tmp_path)
    exp.start(n_runs=2)
    first = json.loads((tmp_path / "t" / "experiment.json").read_text())["started_at"]
    exp.start(n_runs=2)                              # resumed after an interruption
    meta = json.loads((tmp_path / "t" / "experiment.json").read_text())
    assert meta["started_at"] == first and meta["status"] == "running"
