"""
control test for the lab notebook texts: every logged experiment has its title, description,
config names and figure captions in English and Polish (experiments/<folder>/notebook.json).

run:  uv run pytest -q tests/test_notebook_texts.py
"""

import json
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
LANGS = ("en", "pl")


def _logged():
    """(experiment folder, logged experiment.json) for every experiment in the notebook."""
    return [(m.parents[3], json.loads(m.read_text()))
            for m in sorted(EXPERIMENTS.glob("*/results/runs/*/experiment.json"))]


@pytest.mark.parametrize("folder,meta", _logged(), ids=lambda x: x["name"] if isinstance(x, dict) else "")
def test_texts_complete(folder, meta):
    path = folder / "notebook.json"
    assert path.exists(), f"missing {path.relative_to(EXPERIMENTS.parent)}"
    texts = json.loads(path.read_text()).get(meta["name"])
    assert texts, f"{path.name} has no entry for {meta['name']}"
    missing = []
    for field in ("title", "description"):
        missing += [f"{field}.{l}" for l in LANGS if not texts.get(field, {}).get(l)]
    for key in meta.get("config", {}):
        entry = texts.get("config", {}).get(key, {})
        missing += [f"config.{key}.label.{l}" for l in LANGS if not entry.get("label", {}).get(l)]
        if "value" in entry:   # a translated value must exist in both languages
            missing += [f"config.{key}.value.{l}" for l in LANGS if not entry["value"].get(l)]
    for fig in meta.get("figures", []):
        missing += [f"figures.{fig}.{l}" for l in LANGS if not texts.get("figures", {}).get(fig, {}).get(l)]
    assert not missing, f"{meta['name']}: missing texts: {', '.join(missing)}"


def test_no_stale_entries():
    """texts for experiments, config keys or figures that no longer exist are flagged."""
    logged = {meta["name"]: meta for _, meta in _logged()}
    stale = []
    for path in EXPERIMENTS.glob("*/notebook.json"):
        for name, texts in json.loads(path.read_text()).items():
            if name not in logged:
                stale.append(name)
                continue
            stale += [f"{name}: config.{k}" for k in texts.get("config", {}) if k not in logged[name].get("config", {})]
            stale += [f"{name}: figures.{f}" for f in texts.get("figures", {}) if f not in logged[name].get("figures", [])]
    assert not stale, "texts without a logged counterpart: " + ", ".join(stale)
