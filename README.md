# GroverLab

GroverLab is a research-grade educational simulator for Grover's Algorithm.

The project is intentionally modular. The quantum engine is separate from the
Streamlit UI, future FastAPI backend, export layer, logging layer, and
educational explanations.

## Current Status

This repository currently contains the project skeleton and initial contracts.
The next step is implementing the Qiskit circuit construction and Aer simulation
engine.

## Educational Warning

Grover's algorithm does not directly search raw CSV text. GroverLab maps dataset
entries to quantum-searchable indices and searches those encoded indices.

## Planned Workflow

1. Enter comma-separated data or upload a CSV file.
2. Select or type a target item.
3. Clean, index, and pad the dataset.
4. Map the target item to a binary quantum-searchable index.
5. Build and run ideal or noisy Grover simulations.
6. Decode measurements back to dataset items.
7. Explain each step and export results for research use.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run groverlab/app.py
```

## Tests

```bash
pytest
```

