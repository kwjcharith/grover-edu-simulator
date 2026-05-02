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

