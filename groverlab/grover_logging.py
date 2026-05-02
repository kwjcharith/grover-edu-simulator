"""Anonymous research logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_LOG_KEYS = {
    "dataset_size",
    "target_present",
    "num_qubits",
    "iterations",
    "shots",
    "noisy",
    "success",
    "duration_ms",
}


def anonymize_log_payload(payload: dict) -> dict:
    """Keep only approved anonymous evaluation fields."""

    return {key: payload[key] for key in ALLOWED_LOG_KEYS if key in payload}


def append_research_log(payload: dict, output_path: str | Path) -> Path:
    """Append one anonymous JSONL research-log record."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **anonymize_log_payload(payload),
    }
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")
    return path

