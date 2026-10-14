"""
runlog: save results together with everything needed to reproduce them.

every experiment writes  results/<name>_results.csv  (one row per run)
and                      results/<name>_meta.json    (config, library versions, timing).
"""

import csv
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

PACKAGES = ["qiskit", "qiskit-nature", "pyscf", "numpy", "scipy"]


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def save(name, rows, config, started_at, out_dir="results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "experiment": name,
        "config": config,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in PACKAGES},
        "git_commit": _git_commit(),
        "runtime_s": round(time.time() - started_at, 1),
        "rows": len(rows),
    }
    with open(out / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved: {out / f'{name}_results.csv'}, {out / f'{name}_meta.json'}")
