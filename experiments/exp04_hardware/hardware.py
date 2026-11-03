"""
experiment 04: H2 on a noisy quantum processor, with and without error mitigation (RQ4).

the circuits are trained on the exact simulator (as everywhere in this thesis); here the
trained circuits are only EVALUATED on a noisy device. the energy is estimated from shots
(src/hardware.py), so it carries two kinds of error: shot noise (finite repetitions) and
hardware noise (gate errors, decoherence, readout errors).

  methods:     VQE-UCCSD (3 angles) and VQE-HEA (2 layers, 12 angles, best of 5 starts)
  bond length: 0.5, 0.735 (equilibrium), 1.0, 1.5, 2.0, 2.5 A
  arms:        raw | readout mitigation | ZNE | readout mitigation + ZNE
               (ZNE: two-qubit gates folded x1, x3, x5, linear fit to zero noise; the
                (1, 3) linear fit and the (1, 3, 5) Richardson fit are saved too)
  backends:    a noisy simulator of the IBM Fez processor (156-qubit Heron r2, noise model from
               its calibration data), 5 shot seeds  -- runs anywhere
               a real IBM processor                  -- needs an IBM Quantum account

every run saves the raw counts (results/counts/exp04_h2_counts_<label>.json), so the mitigation can
be recomputed later without touching the hardware again.

run (from repo root):
  python experiments/exp04_hardware/hardware.py                        # noisy simulator, 5 seeds
  python experiments/exp04_hardware/hardware.py --backend ibm          # least busy real IBM processor
  python experiments/exp04_hardware/hardware.py --backend ibm_fez      # a named one
options: --shots N, --bond-lengths 0.735 1.5, --dry-run (build and estimate QPU time, submit nothing)

IBM account, either way (never commit the key):
  a) a .env file in the repository root (ignored by git; template: .env.example):
       QISKIT_IBM_TOKEN=<API key>
       QISKIT_IBM_INSTANCE=<instance CRN or name>   (optional)
     read by this script into the environment; qiskit-ibm-runtime prefers it over a saved account
  b) saved once in ~/.qiskit/qiskit-ibm.json:
       python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<API key>', set_as_default=True)"

output: results/exp04_h2_params.json (trained angles, reused by every backend),
        results/exp04_h2_results.csv (one row per backend run x method x bond length), _meta.json
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(__file__).resolve().parent / "results"   # this experiment's outputs
from src import ansatze, hamiltonians as hm, hardware as hw, runlog, tracking  # noqa: E402
from src.classical import reference_energies  # noqa: E402
from src.fastsim import FastCircuit  # noqa: E402
from src.train import run_vqe  # noqa: E402

warnings.filterwarnings("ignore")

CONFIG = {
    "molecule": "H2",
    "basis": hm.BASIS,
    "qubits": 2,
    "bond_lengths_A": [0.5, 0.735, 1.0, 1.5, 2.0, 2.5],
    "methods": {"UCCSD": "from HF (theta = 0)", "HEA_L2": "2 layers, best of 5 random starts uniform(-0.1, 0.1)"},
    "hardware_bond_lengths_A": [0.735, 1.5, 2.5],   # default on real hardware: ~3 min of QPU time
    "fake_backend": "FakeFez",
    "simulator_seeds": [0, 1, 2, 3, 4],
    "shots": 16384,
    "fold_factors": [1, 3, 5],
    "zne_fit": "linear least squares over all fold factors, value at 0",
    "transpile": "optimization_level 1 (keeps the circuit as designed), one fixed pair of physical qubits for every circuit",
    "readout_calibration": "all 4 basis states, same qubits and shots, full 4x4 assignment matrix inverted",
}
NAME = "exp04_h2_hardware"
PARAMS = OUT / "exp04_h2_params.json"
RESULTS = OUT / f"{NAME}_results.csv"
EXP = tracking.Experiment(
    NAME, title="H₂ on a noisy IBM processor: readout mitigation and zero-noise extrapolation",
    description="Trained UCCSD and HEA circuits for H₂ evaluated from shots on a noisy simulator of "
                "IBM Fez and on real IBM hardware, raw and with readout mitigation and zero-noise "
                "extrapolation. Research question 4.",
    config=CONFIG, root=OUT / "runs", figures=["fig_exp04_h2_noise.png"])


def circuit_for(method, qp):
    return ansatze.uccsd(qp) if method == "UCCSD" else ansatze.hea(qp, 2)


# ---------------- 1. training (exact simulator, cheap for H2) ----------------
def train_all():
    out = {}
    for d in CONFIG["bond_lengths_A"]:
        atom = hm.h2(d)
        qp = hm.build(atom)
        ref = reference_energies(atom)
        e_fci, _, _ = hm.exact_ground_state(qp)
        assert abs(e_fci - ref["E_FCI_Ha"]) < 1e-8, f"qubit hamiltonian != FCI at d={d}"
        entry = {"E_HF_Ha": ref["E_HF_Ha"], "E_CCSD_Ha": ref["E_CCSD_Ha"], "E_FCI_Ha": e_fci}
        for method in CONFIG["methods"]:
            circ = circuit_for(method, qp)
            sim = FastCircuit(circ)
            starts = ([("hf", np.zeros(circ.num_parameters))] if method == "UCCSD" else
                      [(f"s{s}", np.random.default_rng(s).uniform(-0.1, 0.1, circ.num_parameters))
                       for s in range(5)])
            best = None
            for tag, theta0 in starts:
                rid = f"d{d:.3f}_{method}" + (f"_{tag}" if method != "UCCSD" else "")
                with EXP.run(rid, {"d_A": d, "method": method, "start": tag, "stage": "training"},
                             reference=e_fci) as run:
                    res = run_vqe(sim, qp.matrix(), qp.e_shift, theta0, maxiter=10000, gtol=1e-9,
                                  callback=run.log)
                    run.finish(E_Ha=res.energy, dE_mHa=1e3 * (res.energy - e_fci),
                               evaluations=res.evaluations, theta=[float(t) for t in res.theta])
                if best is None or res.energy < best[0]:
                    best = (res.energy, res.theta, tag)
            entry[method] = {"theta": [float(t) for t in best[1]], "E_vqe_Ha": best[0], "start": best[2],
                             **ansatze.resources(circ)}
        out[f"{d}"] = entry
        print(f"trained d={d}: UCCSD {1e3 * (entry['UCCSD']['E_vqe_Ha'] - e_fci):.2e} mHa, "
              f"HEA {1e3 * (entry['HEA_L2']['E_vqe_Ha'] - e_fci):.2e} mHa")
    PARAMS.write_text(json.dumps({"config": CONFIG, "params": out}, indent=1))
    print(f"saved: {PARAMS}")
    return out


def load_params():
    if PARAMS.exists():
        saved = json.loads(PARAMS.read_text())
        if [float(k) for k in saved["params"]] == CONFIG["bond_lengths_A"]:
            return saved["params"]
    return train_all()


# ---------------- 2. circuits for one backend ----------------
def best_pair(backend):
    """the connected pair of physical qubits with the lowest two-qubit gate error + readout errors
    (from the backend's latest calibration)."""
    p = backend.properties()
    two_q = next(g for g in ("cz", "ecr", "cx") if g in backend.operation_names)
    best = None
    for a, b in backend.coupling_map.get_edges():
        try:
            if not (p.is_qubit_operational(a) and p.is_qubit_operational(b)):
                continue
            score = p.gate_error(two_q, [a, b]) + p.readout_error(a) + p.readout_error(b)
        except Exception:
            continue
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, [a, b])
    return best[1]


def build(params, backend, bond_lengths):
    """bound, measured, transpiled (and folded) circuits on one fixed pair of physical qubits."""
    jobs, info, layout = [], {}, best_pair(backend)
    for d in bond_lengths:
        qp = hm.build(hm.h2(d))
        const, groups = hw.measurement_groups(qp.hamiltonian)
        for method in CONFIG["methods"]:
            bound = circuit_for(method, qp).assign_parameters(params[f"{d}"][method]["theta"])
            e_ideal = float(Statevector(bound).expectation_value(qp.hamiltonian).real) + qp.e_shift
            assert abs(e_ideal - params[f"{d}"][method]["E_vqe_Ha"]) < 1e-8, "saved angles do not reproduce"
            for basis, g in groups:
                pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                                  initial_layout=layout, seed_transpiler=0)
                isa = pm.run(hw.with_measurement(bound, basis))
                assert isa.layout.final_index_layout() == layout, "routing moved the qubits"
                n2 = sum(1 for i in isa.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
                info[(method, d)] = {"two_qubit_gates": n2, "depth": isa.depth(), "e_ideal": e_ideal,
                                     "const": const + qp.e_shift, "groups": groups}
                for f in CONFIG["fold_factors"]:
                    jobs.append(((method, d, basis, f), hw.fold_two_qubit_gates(isa, f)))
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=layout,
                                      seed_transpiler=0)
    cal = [pm.run(c) for c in hw.readout_calibration_circuits(2)]
    return jobs, cal, info, layout


def qpu_seconds_estimate(backend, n_circuits, shots):
    rep = getattr(backend.configuration(), "default_rep_delay", 250e-6) or 250e-6
    return n_circuits * shots * (rep + 5e-6) + 5.0


def qubit_properties(backend, layout):
    try:
        p = backend.properties()
        two_q = next((g for g in ("cz", "ecr", "cx") if g in backend.operation_names), None)
        return {"physical_qubits": layout,
                "readout_error": [p.readout_error(q) for q in layout],
                "t1_us": [1e6 * p.t1(q) for q in layout], "t2_us": [1e6 * p.t2(q) for q in layout],
                "two_qubit_gate": two_q,
                "two_qubit_error": p.gate_error(two_q, layout) if two_q else None}
    except Exception as e:     # some backends do not publish every property
        return {"physical_qubits": layout, "error": repr(e)}


def load_env(path):
    """QISKIT_IBM_* variables from a .env file into the environment (already set ones win).
    nothing is printed: the file holds the API key."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (x.strip() for x in line.split("=", 1))
        if key.startswith("QISKIT_IBM_") and value:
            os.environ.setdefault(key, value.strip("'\""))


# ---------------- 3. run and evaluate ----------------
def run_sampler(backend, circuits, shots, seed=None, hardware=False):
    if hardware:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = shots
        job = sampler.run(circuits)
    else:
        # Aer directly, with the fake backend's noise model: the local mode of qiskit-ibm-runtime
        # ignores the simulator seed, Aer's sampler is reproducible
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        sampler = AerSampler.from_backend(AerSimulator.from_backend(backend), seed=seed)
        job = sampler.run(circuits, shots=shots)
    result = job.result()
    meta = {"job_id": job.job_id() if hardware else ""}
    if hardware:
        try:
            meta["usage_s"] = job.usage()   # quantum seconds billed
        except Exception:
            pass
    return [r.data.meas.get_counts() for r in result], meta


def evaluate(keys, counts, cal_counts, info, bond_lengths):
    a_inv = np.linalg.inv(hw.assignment_matrix(cal_counts, 2))
    by_key = dict(zip(keys, counts))
    factors = CONFIG["fold_factors"]
    rows = []
    for d in bond_lengths:
        for method in CONFIG["methods"]:
            inf = info[(method, d)]
            raw, ro = [], []
            for f in factors:
                e_raw = e_ro = inf["const"]
                for basis, g in inf["groups"]:
                    p = hw.probabilities(by_key[(method, d, basis, f)], 2)
                    e_raw += hw.group_energy(g, p)
                    e_ro += hw.group_energy(g, a_inv @ p)
                raw.append(e_raw)
                ro.append(e_ro)
            rows.append({"method": method, "d_A": d, "E_ideal_Ha": inf["e_ideal"],
                         "two_qubit_gates": inf["two_qubit_gates"], "depth": inf["depth"],
                         "E_raw_Ha": raw[0], "E_readout_Ha": ro[0],
                         "E_zne_Ha": hw.extrapolate(factors, raw), "E_readout_zne_Ha": hw.extrapolate(factors, ro),
                         "E_readout_zne13_Ha": hw.extrapolate(factors[:2], ro[:2]),
                         "E_readout_richardson_Ha": hw.extrapolate(factors, ro, "richardson"),
                         **{f"E_raw_x{f}_Ha": e for f, e in zip(factors, raw)},
                         **{f"E_readout_x{f}_Ha": e for f, e in zip(factors, ro)}})
    return rows


def save_rows(new_rows, label):
    old = list(csv.DictReader(RESULTS.open())) if RESULTS.exists() else []
    rows = [r for r in old if r["run_label"] != label] + new_rows
    rows.sort(key=lambda r: (r["kind"], r["run_label"], r["method"], float(r["d_A"])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="fake", help="fake | ibm (least busy) | an IBM backend name")
    parser.add_argument("--shots", type=int, default=CONFIG["shots"])
    parser.add_argument("--bond-lengths", type=float, nargs="+", default=None,
                        help="default: all 6 on the simulator, 0.735 1.5 2.5 on hardware")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="submit to hardware without asking")
    args = parser.parse_args()
    started = time.time()
    EXP.start()
    params = load_params()

    if args.backend == "fake":
        from qiskit_ibm_runtime import fake_provider
        backend, kind, seeds = getattr(fake_provider, CONFIG["fake_backend"])(), "simulator", CONFIG["simulator_seeds"]
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        load_env(Path(__file__).resolve().parents[2] / ".env")
        service = QiskitRuntimeService()
        backend = (service.least_busy(operational=True, simulator=False, min_num_qubits=2)
                   if args.backend == "ibm" else service.backend(args.backend))
        kind, seeds = "hardware", [None]
    wanted = args.bond_lengths or CONFIG["bond_lengths_A" if kind == "simulator" else "hardware_bond_lengths_A"]
    bond_lengths = [d for d in CONFIG["bond_lengths_A"] if any(abs(d - x) < 1e-9 for x in wanted)]
    jobs, cal, info, layout = build(params, backend, bond_lengths)
    keys, circuits = [k for k, _ in jobs], [c for _, c in jobs] + cal
    est = qpu_seconds_estimate(backend, len(circuits), args.shots)
    print(f"backend {backend.name}, qubits {layout}, {len(circuits)} circuits x {args.shots} shots, "
          f"estimated QPU time {est:.0f} s")
    for (method, d), inf in sorted(info.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  {method:7s} d={d:<5}  two-qubit gates {inf['two_qubit_gates']}, depth {inf['depth']}")
    if args.dry_run:
        return
    if kind == "hardware" and not args.yes:
        if input(f"submit to {backend.name}? this uses about {est / 60:.1f} min of your QPU time [y/N] ").lower() != "y":
            return

    date = time.strftime("%Y-%m-%d")
    props = qubit_properties(backend, layout)
    all_new = []
    for seed in seeds:
        label = f"{backend.name}_s{seed}" if kind == "simulator" else f"{backend.name}_{time.strftime('%Y%m%d_%H%M')}"
        counts, meta = run_sampler(backend, circuits, args.shots, seed, hardware=kind == "hardware")
        cal_counts = counts[len(keys):]
        rows = evaluate(keys, counts[:len(keys)], cal_counts, info, bond_lengths)
        (OUT / "counts").mkdir(parents=True, exist_ok=True)
        (OUT / "counts" / f"exp04_h2_counts_{label}.json").write_text(json.dumps({
            "backend": backend.name, "kind": kind, "date": date, "seed": seed, "shots": args.shots,
            "layout": layout, "qubit_properties": props, **meta,
            "counts": {f"{m}|{d}|{b}|x{f}": c for (m, d, b, f), c in zip(keys, counts)},
            "calibration_counts": cal_counts}, indent=1))
        for r in rows:
            e_fci = params[f"{r['d_A']}"]["E_FCI_Ha"]
            r.update({"backend": backend.name, "kind": kind, "run_label": label, "seed": "" if seed is None else seed,
                      "date": date, "shots": args.shots, "qubits": "-".join(map(str, layout)),
                      "E_FCI_Ha": e_fci, **meta})
            for arm in ("ideal", "raw", "readout", "zne", "readout_zne", "readout_zne13", "readout_richardson"):
                r[f"dE_{arm}_mHa"] = 1e3 * (r[f"E_{arm}_Ha"] - e_fci)
            rid = f"{label}_d{r['d_A']:.3f}_{r['method']}"
            with EXP.run(rid, {"d_A": r["d_A"], "method": r["method"], "backend": backend.name,
                               "stage": "evaluation", "seed": seed}, reference=e_fci) as run:
                run.finish(**{k: v for k, v in r.items() if k.startswith(("dE_", "E_")) or k in
                              ("two_qubit_gates", "depth", "shots", "qubits", "job_id", "usage_s")})
        all_new += rows
        worst = max(abs(r["dE_readout_zne_mHa"]) for r in rows)
        print(f"{label}: worst |error| raw {max(abs(r['dE_raw_mHa']) for r in rows):.1f} mHa, "
              f"readout + ZNE {worst:.1f} mHa" + (f", QPU {meta['usage_s']} s" if "usage_s" in meta else ""))

    columns = ["kind", "backend", "run_label", "seed", "date", "method", "d_A", "shots", "qubits",
               "two_qubit_gates", "depth", "E_FCI_Ha", "E_ideal_Ha", "E_raw_Ha", "E_readout_Ha", "E_zne_Ha",
               "E_readout_zne_Ha", "E_readout_zne13_Ha", "E_readout_richardson_Ha",
               *[f"E_{a}_x{f}_Ha" for a in ("raw", "readout") for f in CONFIG["fold_factors"]],
               "dE_ideal_mHa", "dE_raw_mHa", "dE_readout_mHa", "dE_zne_mHa", "dE_readout_zne_mHa",
               "dE_readout_zne13_mHa", "dE_readout_richardson_mHa", "job_id", "usage_s"]
    new = [{c: r.get(c, "") for c in columns} for r in all_new]
    labels = {r["run_label"] for r in new}
    old = list(csv.DictReader(RESULTS.open())) if RESULTS.exists() else []
    rows = [r for r in old if r["run_label"] not in labels] + new
    rows.sort(key=lambda r: (r["kind"], r["run_label"], r["method"], float(r["d_A"])))
    runlog.save(NAME, rows, {**CONFIG, "qiskit-ibm-runtime": version("qiskit-ibm-runtime"),
                             "qiskit-aer": version("qiskit-aer"), "last_backend": backend.name,
                             "last_shots": args.shots}, started, out_dir=OUT)
    EXP.finish()


if __name__ == "__main__":
    main()
