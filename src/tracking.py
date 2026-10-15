"""
tracking: a small experiment tracker (like MLflow, but plain files in the repo).

layout, one folder per experiment:
    results/runs/<experiment>/experiment.json        title, description, config, status, figures
    results/runs/<experiment>/<run_id>.json          params, status, final metrics
    results/runs/<experiment>/<run_id>.history.jsonl training history, one line per logged step

every run writes only its own files, so parallel worker processes never collide.
files are replaced atomically (write to tmp, then rename), so the notebook UI never reads
a half-written file while an experiment is running.

the history is thinned so long trainings stay small: every step up to 100, then every 10th
up to 1000, then every 100th, plus the final step.

usage:
    exp = tracking.Experiment("exp02_h4", title="...", description="...", config=CONFIG,
                              figures=["fig_exp02_h4.png"])
    exp.start(n_runs=279)
    with exp.run("d1.50_hea_L3_s0", params={...}, reference=e_fci) as run:
        res = run_vqe(..., callback=run.log)
        run.finish(error_mHa=..., converged=...)
    exp.finish()

view:  python tools/notebook.py
"""

import json
import os
import time
from pathlib import Path

ROOT = Path("results/runs")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float))
    os.replace(tmp, path)


def _keep(step):
    return step < 100 or (step < 1000 and step % 10 == 0) or step % 100 == 0


class Experiment:
    def __init__(self, name, title, description="", config=None, figures=(), root=ROOT):
        self.name = name
        self.dir = Path(root) / name
        self.meta = {"name": name, "title": title, "description": description,
                     "config": config or {}, "figures": list(figures)}

    def _save(self, **update):
        self.meta.update(update)
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.dir / "experiment.json", self.meta)

    def start(self, n_runs=None):
        prev = self.dir / "experiment.json"
        started = json.loads(prev.read_text()).get("started_at") if prev.exists() else None
        self._save(status="running", started_at=started or _now(), finished_at=None,
                   n_runs=n_runs)

    def finish(self):
        self._save(status="done", finished_at=_now())

    def is_done(self, run_id):
        f = self.dir / f"{run_id}.json"
        return f.exists() and json.loads(f.read_text()).get("status") == "done"

    def run(self, run_id, params, reference=None):
        return Run(self.dir, run_id, params, reference)


class Run:
    def __init__(self, folder, run_id, params, reference=None):
        self.folder, self.id = Path(folder), run_id
        self.record = {"run_id": run_id, "params": params, "reference": reference,
                       "status": "running", "started_at": _now(), "metrics": {}}
        self._history = self.folder / f"{run_id}.history.jsonl"
        self._last = None
        self._t0 = time.time()

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self._history.write_text("")
        _write_json(self.folder / f"{self.id}.json", self.record)
        return self

    def log(self, step, energy, grad_norm=None, evaluations=None):
        """training callback: one point of the history (thinned for long trainings)."""
        point = {"step": int(step), "energy": float(energy),
                 "grad_norm": None if grad_norm is None else float(grad_norm),
                 "evaluations": evaluations, "t": round(time.time() - self._t0, 3)}
        self._last = point
        if _keep(step):
            with open(self._history, "a") as f:
                f.write(json.dumps(point) + "\n")

    def finish(self, **metrics):
        if self._last is not None and not _keep(self._last["step"]):
            with open(self._history, "a") as f:
                f.write(json.dumps(self._last) + "\n")
        self.record.update(status="done", finished_at=_now(),
                           duration_s=round(time.time() - self._t0, 2), metrics=metrics)
        _write_json(self.folder / f"{self.id}.json", self.record)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.record.update(status="failed", finished_at=_now(), error=repr(exc))
            _write_json(self.folder / f"{self.id}.json", self.record)
        return False
