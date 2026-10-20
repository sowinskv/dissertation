"""
lab notebook: a local web UI for the experiment logs in results/runs/ (see src/tracking.py).

run (from repo root):  python tools/notebook.py [--port 8765] [--no-browser]
then open http://localhost:8765. the page refreshes itself while an experiment is running.

only the python standard library is used; the server listens on localhost only.

api:
  GET /api/experiments                      all experiments with run counts
  GET /api/experiments/<name>               experiment metadata + all runs (no histories)
  GET /api/experiments/<name>/history?ids=a,b,c   training histories of the listed runs
  GET /results/<file>                       figures and other files from results/
  everything else is served from notebook/
"""

import argparse
import json
import mimetypes
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
RESULTS = REPO / "results"
STATIC = REPO / "notebook"


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None   # file is being replaced right now; the next poll will get it


def _runs(folder):
    out = []
    for f in sorted(folder.glob("*.json")):
        if f.name == "experiment.json":
            continue
        rec = _read_json(f)
        if rec is None:
            continue
        hist = folder / f"{rec['run_id']}.history.jsonl"
        rec["has_history"] = hist.exists() and hist.stat().st_size > 0
        out.append(rec)
    return out


def _counts(runs):
    c = {"done": 0, "running": 0, "failed": 0}
    for r in runs:
        c[r.get("status", "done")] = c.get(r.get("status", "done"), 0) + 1
    return c


def _extras(name):
    """links the page shows next to an experiment: its CSV in results/, if there is one."""
    csv = RESULTS / f"{name}_results.csv"
    return {"csv": csv.name if csv.exists() else None}


def _labels():
    """experiment numbers as shown on the pages: 01, 02, then 02b, 02c for follow-ups that share
    a number (ordered by start date)."""
    groups = {}
    for folder in RUNS.glob("*"):
        meta = _read_json(folder / "experiment.json") if folder.is_dir() else None
        m = re.match(r"exp(\d+)", folder.name)
        if meta and m:
            groups.setdefault(m.group(1), []).append((meta.get("started_at") or "", folder.name))
    out = {}
    for num, items in groups.items():
        for i, (_, name) in enumerate(sorted(items)):
            out[name] = num + ("" if i == 0 else "bcdefghij"[i - 1])
    return out


def list_experiments():
    items = []
    for folder in sorted(p for p in RUNS.glob("*") if p.is_dir()):
        meta = _read_json(folder / "experiment.json")
        if meta is None:
            continue
        runs = _runs(folder)
        meta.pop("config", None)
        items.append({**meta, **_extras(meta["name"]), "counts": _counts(runs), "n_logged": len(runs)})
    labels = _labels()
    for x in items:
        x["label"] = labels.get(x["name"], "")
    return sorted(items, key=lambda x: (x["label"], x["name"]))


def get_experiment(name):
    folder = RUNS / name
    meta = _read_json(folder / "experiment.json")
    if meta is None:
        return None
    runs = _runs(folder)
    for r in runs:
        r.get("metrics", {}).pop("theta", None)   # parameters are large and not shown
    return {**meta, **_extras(name), "label": _labels().get(name, ""), "runs": runs,
            "counts": _counts(runs)}


def get_history(name, ids):
    folder = RUNS / name
    out = {}
    for rid in ids:
        f = folder / f"{Path(rid).name}.history.jsonl"
        if not f.exists():
            continue
        points = []
        for line in f.read_text().splitlines():
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                break   # last line still being written
        out[rid] = points
    return out


def _safe(base, rel):
    path = (base / unquote(rel)).resolve()
    return path if base.resolve() in path.parents or path == base.resolve() else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path):
        if path is None or not path.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def do_GET(self):
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if parts[:2] == ["api", "experiments"]:
            if len(parts) == 2:
                return self._send(200, list_experiments())
            name = Path(unquote(parts[2])).name
            if len(parts) == 3:
                exp = get_experiment(name)
                return self._send(200, exp) if exp else self._send(404, {"error": "not found"})
            if len(parts) == 4 and parts[3] == "history":
                ids = [i for i in parse_qs(url.query).get("ids", [""])[0].split(",") if i]
                return self._send(200, get_history(name, ids))
            return self._send(404, {"error": "not found"})
        if parts[:1] == ["results"]:
            return self._file(_safe(RESULTS, "/".join(parts[1:])))
        return self._file(_safe(STATIC, "/".join(parts) or "index.html"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    mimetypes.add_type("font/woff2", ".woff2")
    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"lab notebook: {url}  (ctrl+c to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
