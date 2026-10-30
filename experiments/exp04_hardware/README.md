# exp04: H₂ on a noisy IBM processor, with and without error mitigation (RQ4)

The trained H₂ circuits (UCCSD, and HEA with 2 layers) are evaluated from shots, on a noisy simulator of IBM Fez and on real IBM hardware. Four variants are compared: raw, readout mitigation, zero-noise extrapolation (ZNE), and readout mitigation + ZNE. The mitigation is implemented in `src/hardware.py`, and the same code runs on the simulator and on the hardware.

## Run (from the repository root)

```
uv run python experiments/exp04_hardware/hardware.py               # noisy simulator, 5 seeds, ~2 min
uv run python experiments/exp04_hardware/hardware.py --backend ibm --dry-run   # checks the IBM account, no QPU time
uv run python experiments/exp04_hardware/hardware.py --backend ibm             # real hardware, ~3 min QPU, asks first
uv run python experiments/exp04_hardware/plot.py
```

### IBM account

This is done once, on your own machine. The key is saved in `~/.qiskit/qiskit-ibm.json`, outside the repository:

```
uv run python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<API key>', set_as_default=True)"
```

## Files in `results/`

- `exp04_h2_params.json`: trained angles, reused by every backend
- `exp04_h2_hardware_results.csv`: one row per backend run × method × bond length (`kind` = simulator / hardware)
- `counts/`: raw counts of every run, so the mitigation can be recomputed without the hardware
- `fig_exp04_h2_noise`, `fig_exp04_h2_zne` (`.pdf` / `.png`)
- `runs/exp04_h2_hardware/`: logs for the lab notebook

## Result (noisy simulator)

Raw error is 7–37 mHa. Readout mitigation alone helps little. Readout + ZNE brings the mean error to 0.04–3.8 mHa, with a shot-noise spread of 1–4 mHa. HEA (2 two-qubit gates) beats UCCSD (4) on noisy hardware, although both are exact without noise.
