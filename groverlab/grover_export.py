"""Export GroverLab results to JSON, CSV, and Markdown reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def export_json(payload: dict, output_path: str | Path) -> Path:
    """Write a JSON export and return the path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_csv(rows: list[dict], output_path: str | Path) -> Path:
    """Write rows to a CSV export and return the path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_markdown_report(title: str, sections: dict[str, str], output_path: str | Path) -> Path:
    """Write a simple Markdown report and return the path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", body, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

