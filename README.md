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

## Local Deployment

Install the dependencies, then run the Streamlit app locally.

From the repository root:

```bash
streamlit run groverlab/app.py
```

If you are already inside `groverlab/`:

```bash
streamlit run app.py
```

The app runs on Streamlit's default port, usually:

```text
http://localhost:8501
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

## Deploying To Streamlit Community Cloud

Streamlit Community Cloud deploys directly from GitHub. GroverLab includes
public-demo safeguards so the free online version remains stable for small
educational demonstrations.

1. Push the repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Select the GroverLab repository.
4. Select the `main` branch.
5. Set the main file path to `groverlab/app.py`. If deploying with `groverlab/` as the app root, use `app.py`.
6. Click Deploy.

The current app does not require private credentials for basic simulation. Keep
any future secrets out of the repository and configure them through Streamlit
Cloud settings.

## Public Demo Limitations

Free Streamlit deployments can hit resource limits when apps process large data
or run expensive simulations. GroverLab uses caching, upload limits, and public
simulation caps to keep the demo stable.

Public demo mode is enabled in `deployment_config.py`:

```python
PUBLIC_DEMO_MODE = True
```

The online demo is intended for:

- small educational demonstrations
- limited dataset size
- limited shots
- limited iteration and noise sweeps

Current public limits:

- recommended dataset size: `4-256` items
- maximum public demo dataset size: `1024` items
- maximum public demo qubits: `10`
- maximum public shots: `2048`
- maximum iteration sweep points: `20`
- maximum noise sweep points: `8`

Large simulations should be run locally. If an online dataset exceeds the public
limits, GroverLab shows theoretical scaling information instead of building and
running the quantum circuit.

## Local Vs Online Mode

Online mode is designed for small public demonstrations on free cloud resources.
It limits dataset size, shots, qubit count, upload size, and sweep size.

Local mode is better for:

- larger datasets
- larger shot counts
- heavier noisy simulations
- research development
- FastAPI backend work

To disable public-demo limits locally, edit `deployment_config.py`:

```python
PUBLIC_DEMO_MODE = False
```

Then run the app locally with:

```bash
streamlit run groverlab/app.py
```

## Docker Deployment

Build the image from the repository root:

```bash
docker build -t groverlab-streamlit .
```

Run the container:

```bash
docker run --rm -p 8501:8501 groverlab-streamlit
```

Open:

```text
http://localhost:8501
```

The Dockerfile uses a Python base image, installs `requirements.txt`, copies the
project files, exposes port `8501`, and runs:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## Future Google Cloud VM Deployment

For a future Google Cloud VM deployment:

- Create a VM with Python and Docker support.
- Clone the GroverLab repository onto the VM.
- Build the Docker image on the VM or pull it from a container registry.
- Run the container with port `8501` exposed.
- Configure firewall rules to allow inbound traffic to the chosen public port.
- Place the app behind HTTPS and authentication if it is used in a class or research study.
- Store any future credentials or deployment-specific settings outside the repository.

For research deployments, verify that anonymous logging paths, retention
policies, consent language, and access controls match the approved study design.

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

The Streamlit app is the primary standalone educational interface. The FastAPI
backend is available for future online deployments, remote experiments, or
integration with other learning platforms. Keep secrets and deployment settings
in environment variables or managed platform settings rather than committing them
to the repository.

## Tests

```bash
python3 -m pytest
```
