# GroverLab

GroverLab is a research-grade educational simulator for Grover's Algorithm. It
helps learners explore how a classical dataset is cleaned, indexed, encoded into
binary states, searched with Grover-style amplitude amplification, and interpreted
through probabilistic measurement.

The project is intentionally modular. The quantum engine is independent from the
Streamlit interface, FastAPI backend, plotting layer, export system, and
anonymous research logging.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Streamlit Run Command

From the repository root:

```bash
streamlit run groverlab/app.py
```

If you are already inside `groverlab/`:

```bash
streamlit run app.py
```

## FastAPI Run Command

From the repository root:

```bash
uvicorn groverlab.api:app --reload
```

If you are already inside `groverlab/`:

```bash
uvicorn api:app --reload
```

## Example Dataset

Dataset:

```text
apple, mango, banana, orange
```

Target:

```text
banana
```

Expected output:

- target index: `2`
- target binary state: `10`
- number of qubits: `2`
- padded search size: `4`
- ideal simulation: `10` should dominate the measurement counts
- decoded item: `banana`
- success probability: high in the ideal simulation

## Searching For An Item That Is Not In The Dataset

Grover requires a valid oracle. If the target is absent, no marked state exists.
Without a marked state, no phase inversion occurs and no useful amplitude
amplification occurs.

Example:

```text
Dataset: apple, banana, pineapple
Target: abc
```

Expected practical result: the simulator reports that the item is not present
and stops before quantum execution. Experimental mode can run a no-solution
demonstration where no oracle state is marked and measurement probabilities
remain near-uniform.

## Educational Purpose

GroverLab teaches:

- classical-to-quantum index encoding
- equal superposition
- oracle phase marking
- diffuser/amplitude amplification
- probabilistic measurement
- over-rotation from too many iterations
- noise effects and loss of quantum advantage

It also warns that Grover's Algorithm does not directly search raw CSV text.
Dataset entries are mapped to quantum-searchable indices first.

## Research Purpose

GroverLab is designed for TOCE/SIGCSE-style educational research. It supports
controlled experiments, iteration and noise sweeps, exportable reports, and
anonymous interaction logging for studying how students understand quantum search
concepts.

## Privacy Note

Research logs are anonymous. GroverLab does not store names, emails, student IDs,
IP addresses, raw dataset values, or full dataset item names in research logs.
Logged data is limited to aggregate simulation metadata such as dataset size,
qubit count, iteration count, noise settings, success probability, runtime,
circuit depth, and gate counts.

## Deployment Note

The Streamlit app can be deployed as a standalone educational interface. The
FastAPI backend is available for future online deployments, remote experiments,
or integration with other learning platforms. Keep secrets and deployment
settings in environment variables rather than committing them to the repository.

## Tests

```bash
python3 -m pytest
```

