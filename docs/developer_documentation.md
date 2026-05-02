# GroverLab Developer Documentation

## Architecture

GroverLab separates concerns into independent modules:

- `grover_data.py`: dataset cleaning, indexing, padding, and bitstring mapping
- `grover_oracle.py`: oracle construction contracts
- `grover_core.py`: Grover circuit construction
- `grover_runner.py`: simulation orchestration
- `grover_analysis.py`: measurement and comparison analysis
- `grover_plots.py`: plotting utilities
- `grover_education.py`: educational text and warnings
- `grover_export.py`: JSON, CSV, and Markdown exports
- `grover_logging.py`: anonymous research logging
- `app.py`: Streamlit UI
- `api.py`: future FastAPI backend

## Design Rule

The quantum engine must remain independent from Streamlit and FastAPI.

## Searching For An Item That Is Not In The Dataset

GroverLab supports two explicit missing-target modes through `GroverConfig`:

- `missing_target_mode="stop"`: practical mode. The runner creates a
  `DatasetMapping` with `target_found=False`, returns a `GroverResult` with
  `stopped_before_quantum_execution=True`, leaves `ideal_counts={}`, and does
  not construct a quantum circuit.
- `missing_target_mode="experimental"`: no-solution demonstration mode. The
  core builds a superposition-only circuit with no oracle and no diffuser. The
  resulting measurements should be near-uniform, and `success_probability`
  remains `0.0` because no valid target state exists.

Grover requires a valid oracle. If the target is absent, no marked state exists.
No marked state means no phase inversion and no amplitude amplification.

Example:

```text
Dataset: apple, banana, pineapple
Target: abc
```

Expected practical result: the simulator reports that the item is not present
and stops before quantum execution.
