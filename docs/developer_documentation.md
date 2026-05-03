# GroverLab Developer Documentation

## Architecture

GroverLab is organized as a modular Python package. The core quantum engine does
not import Streamlit or FastAPI. The UI and API layers call shared configuration,
runner, analysis, export, and education modules.

High-level layers:

- Input and presentation: `app.py`, `api.py`
- Simulation orchestration: `grover_runner.py`
- Quantum circuit construction and execution: `grover_core.py`, `grover_oracle.py`
- Data preparation: `grover_data.py`
- Analysis and visualization: `grover_analysis.py`, `grover_plots.py`
- Education and reporting: `grover_education.py`, `grover_export.py`
- Privacy-preserving logging: `grover_logging.py`

## Module Responsibilities

- `grover_config.py`: dataclasses for run configuration, noise settings, dataset
  mapping, simulation results, and experiment logs.
- `grover_data.py`: parses text/CSV input, cleans items, calculates qubits and
  padding, maps targets to binary states, and handles missing-target modes.
- `grover_oracle.py`: applies target-state phase oracle logic to Qiskit circuits.
- `grover_core.py`: builds circuits, applies superposition and diffuser steps,
  creates Aer noise models, runs ideal/noisy simulations, and decodes counts.
- `grover_runner.py`: converts a `GroverConfig` into a complete `GroverResult`.
- `grover_analysis.py`: computes success probabilities, sweeps iterations/noise,
  compares classical vs Grover search, and produces research metrics.
- `grover_plots.py`: creates matplotlib figures for counts, sweeps, comparisons,
  and heatmaps.
- `grover_education.py`: generates beginner-friendly explanations, warnings,
  quiz questions, and activities.
- `grover_export.py`: writes JSON, CSV, Markdown reports, and research-log files.
- `grover_logging.py`: creates anonymous research logs without raw identifiers.
- `app.py`: Streamlit interface.
- `api.py`: FastAPI backend.

## Data Flow

1. User enters comma-separated text or uploads CSV data.
2. `grover_data.py` cleans items and creates a `DatasetMapping`.
3. `GroverConfig` carries dataset, target, shots, iteration, noise, and
   missing-target settings.
4. `grover_runner.py` builds the mapping, chooses iterations if needed, calls
   `grover_core.py`, decodes counts, and returns `GroverResult`.
5. UI/API/reporting layers consume the same `GroverResult`.

## How The Grover Circuit Is Built

For a valid target:

1. Create a `QuantumCircuit` with `n_qubits`.
2. Apply Hadamard gates to all qubits to create equal superposition.
3. Repeat oracle plus diffuser for the configured iteration count.
4. Add measurement operations.
5. Run with `qiskit_aer.AerSimulator`.

For a missing target in practical mode, the runner stops before circuit
construction. In experimental no-solution mode, the circuit applies only
superposition and measurement; no oracle or diffuser is applied by default.

## How The Oracle Works

The oracle marks one encoded target state. For target bits equal to `0`, the
oracle applies `X` gates to map the target to the all-ones state, applies a phase
flip, then uncomputes the `X` gates.

For one qubit, the phase flip is a `Z` gate. For multiple qubits, GroverLab uses:

1. `H` on the last qubit
2. multi-controlled `X`
3. `H` on the last qubit

This implements a multi-controlled `Z` around the all-ones state.

## How The Diffuser Works

The diffuser performs amplitude amplification:

1. Apply `H` to all qubits.
2. Apply `X` to all qubits.
3. Apply multi-controlled `Z` around `|111...1>`.
4. Apply `X` to all qubits.
5. Apply `H` to all qubits.

This is the inversion-about-the-mean step that increases the marked state's
measurement probability when a valid oracle exists.

## How Noise Is Modelled

Noise is optional and configured with `NoiseConfig`.

- Depolarizing/gate noise is modelled with Qiskit Aer `depolarizing_error`.
- Measurement noise is modelled with `ReadoutError`.
- Noisy execution uses `AerSimulator(noise_model=...)`.
- If noise is disabled, `run_noisy_simulation` returns `None`.

## How Tests Are Run

From the repository root:

```bash
python3 -m pytest
```

For focused checks:

```bash
python3 -m pytest tests/test_core.py
python3 -m pytest tests/test_runner.py
python3 -m pytest tests/test_api.py
```

Use `python3 -m pytest` rather than the bare `pytest` launcher on systems where
the local package is not automatically placed on `PYTHONPATH`.

## Extension Points

- Add new oracle types in `grover_oracle.py`.
- Add new circuit modes in `grover_core.py`.
- Add additional sweep studies in `grover_analysis.py`.
- Add classroom materials in `grover_education.py`.
- Add report formats in `grover_export.py`.
- Add deployment authentication, persistence, or LMS integration in `api.py`.

## Searching For An Item That Is Not In The Dataset

GroverLab supports two missing-target modes:

- `missing_target_mode="stop"`: practical mode. No quantum circuit is built.
- `missing_target_mode="experimental"`: no-solution demonstration. A
  superposition-only circuit runs and produces near-uniform measurements.

Example:

```text
Dataset: apple, banana, pineapple
Target: abc
```

Expected practical result: the simulator reports that the item is not present
and stops before quantum execution.

