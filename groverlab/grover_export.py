"""Export GroverLab results to JSON, CSV, Markdown, and research logs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from groverlab.grover_config import GroverResult
from groverlab.grover_core import decoherence_probabilities, recommended_iterations
from groverlab.grover_education import generate_full_explanation
from groverlab.grover_logging import append_log_jsonl, create_experiment_log


def export_result_json(result: GroverResult, path: str | Path) -> Path:
    """Export a complete GroverResult as JSON."""

    return export_json(_json_safe(asdict(result)), path)


def export_counts_csv(counts: dict[str, int], path: str | Path) -> Path:
    """Export measurement counts to CSV."""

    rows = [
        {"bitstring": bitstring, "count": count}
        for bitstring, count in sorted(counts.items())
    ]
    return export_csv(rows, path)


def export_experiment_summary_csv(results: list[GroverResult | dict[str, Any]], path: str | Path) -> Path:
    """Export a compact experiment summary table to CSV."""

    rows = [_summary_row(result) for result in results]
    return export_csv(rows, path)


def export_student_report_markdown(result: GroverResult, path: str | Path) -> Path:
    """Export a beginner-friendly Markdown report for a GroverLab run."""

    explanation = generate_full_explanation(result)
    sections = explanation["sections"]
    quiz_lines = []
    for index, item in enumerate(explanation["quiz_questions"], start=1):
        quiz_lines.extend(
            [
                f"{index}. {item['question']}",
                f"   Answer: {item['answer']}",
            ]
        )

    lines = [
        "# GroverLab Student Report",
        "",
        "## Run Summary",
        "",
        f"- Dataset size: {result.mapping.n_items}",
        f"- Target item: {result.mapping.target_item}",
        f"- Target index: {result.mapping.target_index}",
        f"- Target binary state: |{result.mapping.target_binary}>",
        f"- Number of qubits: {result.mapping.n_qubits}",
        f"- Padded states: {result.mapping.padded_size}",
        f"- Unused padded states: {result.mapping.unused_states}",
        f"- Grover iterations: {_resolved_iterations(result)}",
        f"- Shots: {result.config.shots}",
        f"- Depolarising noise: {result.config.noise_config.depolar_prob}",
        f"- Measurement error: {result.config.noise_config.measurement_error_prob}",
        f"- T1 relaxation time: {result.config.noise_config.t1_relaxation_us} µs",
        f"- T2 coherence time: {result.config.noise_config.t2_coherence_us} µs",
        f"- Derived Tphi: {_format_t_phi(result)}",
        "",
        "## Ideal Result",
        "",
        f"- Most likely bitstring: {result.measured_bitstring}",
        f"- Decoded item: {result.decoded_item}",
        f"- Found target: {result.found}",
        f"- Success probability: {result.success_probability:.3f}",
        "",
    ]

    if result.noisy_counts is not None:
        lines.extend(
            [
                "## Noisy Result",
                "",
                f"- Noisy success probability: {result.noisy_success_probability:.3f}",
                f"- Probability degradation: {result.success_probability - (result.noisy_success_probability or 0.0):.3f}",
                f"- Noisy counts: `{json.dumps(result.noisy_counts, sort_keys=True)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Circuit",
            "",
            f"- Circuit depth: {result.circuit_depth}",
            f"- Gate counts: `{json.dumps(result.gate_counts, sort_keys=True)}`",
            "",
            "## Step-by-Step Explanation",
            "",
        ]
    )
    for heading, body in sections.items():
        lines.extend([f"### {heading.replace('_', ' ').title()}", "", body, ""])

    lines.extend(["## Misconception Warnings", ""])
    for warning in explanation["warnings"]:
        lines.append(f"- {warning}")

    lines.extend(["", "## Short Quiz", "", *quiz_lines, ""])

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def export_research_log(result: GroverResult, path: str | Path) -> Path:
    """Export a privacy-preserving anonymous research log as JSONL."""

    log = create_experiment_log(result)
    return append_log_jsonl(log, path)


def export_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON export and return the path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return path


def export_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write rows to a CSV export and return the path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
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


def _summary_row(result: GroverResult | dict[str, Any]) -> dict[str, Any]:
    """Create one CSV summary row from a result or analysis dictionary."""

    if isinstance(result, dict):
        return dict(result)

    return {
        "dataset_size": result.mapping.n_items,
        "target_index": result.mapping.target_index,
        "target_binary": result.mapping.target_binary,
        "n_qubits": result.mapping.n_qubits,
        "padded_size": result.mapping.padded_size,
        "iterations": _resolved_iterations(result),
        "shots": result.config.shots,
        "success_probability": result.success_probability,
        "noisy_success_probability": result.noisy_success_probability,
        "probability_degradation": (
            result.success_probability - result.noisy_success_probability
            if result.noisy_success_probability is not None
            else None
        ),
        "depolar_prob": result.config.noise_config.depolar_prob,
        "measurement_error_prob": result.config.noise_config.measurement_error_prob,
        "t1_relaxation_us": result.config.noise_config.t1_relaxation_us,
        "t2_coherence_us": result.config.noise_config.t2_coherence_us,
        "t_phi_us": decoherence_probabilities(result.config.noise_config)["t_phi_us"],
        "circuit_depth": result.circuit_depth,
        "total_gates": sum(int(value) for value in result.gate_counts.values()),
        "decoded_correctly": result.found,
    }


def _resolved_iterations(result: GroverResult) -> int:
    """Return the explicit or automatically selected iteration count."""

    if result.config.iterations is not None:
        return result.config.iterations
    return recommended_iterations(result.mapping.padded_size)


def _format_t_phi(result: GroverResult) -> str:
    """Format derived Tphi for Markdown reports."""

    t_phi = decoherence_probabilities(result.config.noise_config)["t_phi_us"]
    return "infinite" if t_phi == float("inf") else f"{t_phi:.3f} µs"


def _json_safe(value: Any) -> Any:
    """Convert dataclasses and non-JSON scalar values into JSON-safe values."""

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
