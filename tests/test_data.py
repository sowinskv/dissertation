"""
control tests for the data preparation (experiments/exp00_data/prepare.py).

run:  uv run pytest -q tests/test_data.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_")[:-3], EXPERIMENTS / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load("exp00_data/prepare.py")


def test_splits_match_the_experiments():
    """the geometries listed in exp00 are the ones the experiment scripts actually use."""
    s = {k: v[2] for k, v in prep.SPLITS.items()}
    assert s["exp01 H2 curve"] == _load("exp01_h2_curve/run.py").CONFIG["bond_lengths_A"]
    assert s["exp02 H4 spacings"] == _load("exp02_h4/ansatze.py").CONFIG["spacings_A"]
    exp03 = _load("exp03_conditional/conditional.py")
    assert s["exp03 test grid"] == exp03.GRID
    assert prep.EXP03_TRAIN_RANGE_A == exp03.CONFIG["train_range_A"]
    for m in exp03.CONFIG["n_train"]:
        assert s[f"exp03 train M={m}"] == exp03.train_spacings(m)
    exp04 = _load("exp04_hardware/hardware.py").CONFIG
    assert s["exp04 simulator"] == exp04["bond_lengths_A"]
    assert s["exp04 hardware"] == exp04["hardware_bond_lengths_A"]


@pytest.mark.parametrize("mol,d", [("H2", 0.735), ("H4", 1.5)])
def test_describe_passes_checks(mol, d):
    row = prep.describe(mol, d)
    assert row["fci_check_Ha"] < prep.FCI_TOL_HA
    assert row["rebuild_diff"] < 1e-12
    assert row["qubits"] == row["qubits_before"] - 2
    assert 0 < row["hf_weight"] <= 1
    assert row["E_corr_mHa"] < 0


def test_h2_hamiltonian_size():
    """H2 after parity reduction: 2 qubits, Pauli terms IZ, ZI, ZZ, XX, two measurement groups."""
    row = prep.describe("H2", 0.735)
    assert (row["qubits"], row["pauli_terms"], row["measurement_groups"]) == (2, 4, 2)
    assert np.isclose(row["E_FCI_Ha"], -1.137306, atol=1e-6)
