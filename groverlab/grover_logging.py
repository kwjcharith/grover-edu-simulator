"""Anonymous research logging utilities."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from groverlab.grover_config import ExperimentLog, GroverResult
from groverlab.grover_core import recommended_iterations


DEFAULT_LOG_PATH = "outputs/experiment_logs/grover_logs.jsonl"

ALLOWED_LOG_KEYS = {
    "session_id",
    "timestamp",
    "dataset_size",
    "target_index",
    "n_qubits",
    "padded_size",
    "unused_states",
    "iterations",
    "shots",
    "noise_enabled",
    "depolar_prob",
    "measurement_error_prob",
    "success_probability",
    "decoded_correctly",
    "runtime_seconds",
    "circuit_depth",
    "gate_counts",
}


def create_session_id() -> str:
    """Create an anonymous random session identifier."""

    return uuid.uuid4().hex


def create_experiment_log(
    result: GroverResult,
    session_id: str | None = None,
) -> ExperimentLog:
    """Create an anonymous experiment log from a GroverResult."""

    resolved_session_id = session_id or create_session_id()
    iterations = result.config.iterations
    if iterations is None:
        iterations = recommended_iterations(result.mapping.padded_size)

    return ExperimentLog(
        session_id=resolved_session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_size=result.mapping.n_items,
        target_index=result.mapping.target_index,
        n_qubits=result.mapping.n_qubits,
        padded_size=result.mapping.padded_size,
        unused_states=result.mapping.unused_states,
        iterations=iterations,
        shots=result.config.shots,
        noise_enabled=result.config.noise_config.noise_enabled,
        depolar_prob=result.config.noise_config.depolar_prob,
        measurement_error_prob=result.config.noise_config.measurement_error_prob,
        success_probability=result.success_probability,
        decoded_correctly=result.found,
        runtime_seconds=result.runtime_seconds,
        circuit_depth=result.circuit_depth,
        gate_counts=dict(result.gate_counts),
    )


def append_log_jsonl(
    log: ExperimentLog,
    path: str | Path = DEFAULT_LOG_PATH,
) -> Path:
    """Append one anonymous ExperimentLog record to a JSONL file."""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(_log_to_allowed_dict(log), sort_keys=True) + "\n")
    return log_path


def anonymize_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only approved anonymous evaluation fields from a dictionary."""

    return {key: payload[key] for key in ALLOWED_LOG_KEYS if key in payload}


def append_research_log(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Compatibility helper for appending an anonymous JSONL dictionary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **anonymize_log_payload(payload),
    }
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def _log_to_allowed_dict(log: ExperimentLog) -> dict[str, Any]:
    """Convert an ExperimentLog to a privacy-approved dictionary."""

    if not is_dataclass(log):
        raise TypeError("log must be an ExperimentLog dataclass instance.")
    return anonymize_log_payload(asdict(log))

