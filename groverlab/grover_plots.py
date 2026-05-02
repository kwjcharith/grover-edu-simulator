"""Matplotlib plotting utilities for GroverLab experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_counts_histogram(
    counts: dict[str, int],
    target_binary: str,
    save_path: str | Path | None = None,
):
    """Plot measurement counts and highlight the target bitstring."""

    if not counts:
        raise ValueError("Measurement counts cannot be empty.")
    labels = sorted(counts)
    values = [counts[label] for label in labels]
    colors = [
        "#D1495B" if target_binary and label == target_binary else "#2E86AB"
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_title("Measurement Counts")
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Counts")
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2E86AB", label="Measured states")]
    if target_binary:
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, color="#D1495B", label="Target"))
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_iteration_sweep(results: list[dict[str, Any]], save_path: str | Path | None = None):
    """Plot success probability across Grover iteration counts."""

    _validate_results(results)
    iterations = [_require_key(row, "iterations") for row in results]
    success = [_require_key(row, "success_probability") for row in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, success, marker="o", color="#2E86AB", linewidth=2)
    ax.set_title("Grover Iteration Sweep")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Target success probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_noise_sweep(results: list[dict[str, Any]], save_path: str | Path | None = None):
    """Plot ideal and noisy success probabilities across noise values."""

    _validate_results(results)
    noise_values = [_require_key(row, "noise_value") for row in results]
    ideal_success = [_require_key(row, "success_probability") for row in results]
    noisy_success = [
        row.get("noisy_success_probability", row.get("success_probability"))
        for row in results
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(noise_values, ideal_success, marker="o", label="Ideal", color="#2E86AB")
    ax.plot(noise_values, noisy_success, marker="s", label="Noisy", color="#D1495B")
    ax.set_title("Noise Sweep")
    ax.set_xlabel("Noise probability")
    ax.set_ylabel("Target success probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_classical_vs_grover(
    results: list[dict[str, Any]] | dict[str, Any],
    save_path: str | Path | None = None,
):
    """Plot classical query counts against Grover iteration estimates."""

    rows = results if isinstance(results, list) else [results]
    _validate_results(rows)

    sizes = [_first_present(row, ["n_items", "dataset_size", "padded_size"]) for row in rows]
    classical_worst = [
        _first_present(row, ["classical_worst_case_queries", "classical_worst_case"])
        for row in rows
    ]
    classical_average = [
        _first_present(row, ["classical_average_queries", "classical_average"])
        for row in rows
    ]
    grover_iterations = [
        _first_present(row, ["grover_recommended_iterations", "iterations"])
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, classical_worst, marker="o", label="Classical worst case", color="#D1495B")
    ax.plot(sizes, classical_average, marker="s", label="Classical average", color="#F4A261")
    ax.plot(sizes, grover_iterations, marker="^", label="Grover recommended", color="#2E86AB")
    ax.set_title("Classical Linear Search vs Grover Search")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Queries or iterations")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_success_probability_heatmap(
    data: list[list[float]] | dict[str, Any],
    save_path: str | Path | None = None,
):
    """Plot a heatmap of success probabilities."""

    values, x_labels, y_labels = _normalize_heatmap_data(data)
    if not values or not values[0]:
        raise ValueError("Heatmap data must contain at least one value.")

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Success Probability Heatmap")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Dataset or iteration")
    ax.set_xticks(range(len(values[0])))
    ax.set_yticks(range(len(values)))
    ax.set_xticklabels(x_labels or [str(index) for index in range(len(values[0]))])
    ax.set_yticklabels(y_labels or [str(index) for index in range(len(values))])
    fig.colorbar(image, ax=ax, label="Success probability")

    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            text_color = "white" if value < 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_measurement_counts(
    counts: dict[str, int],
    output_path: str | Path | None = None,
):
    """Compatibility wrapper for the initial plotting placeholder."""

    target_binary = max(counts, key=counts.get) if counts else ""
    return plot_counts_histogram(counts, target_binary=target_binary, save_path=output_path)


def _save_if_requested(fig, save_path: str | Path | None) -> None:
    """Save a figure if a path is supplied."""

    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _validate_results(results: list[dict[str, Any]]) -> None:
    """Validate a non-empty list of result rows."""

    if not results:
        raise ValueError("Results must contain at least one row.")


def _require_key(row: dict[str, Any], key: str) -> Any:
    """Read a required key with a clear error."""

    if key not in row:
        raise ValueError(f"Result row is missing required key '{key}'.")
    return row[key]


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    """Return the first available key from a row."""

    for key in keys:
        if key in row:
            return row[key]
    joined = "', '".join(keys)
    raise ValueError(f"Result row must contain one of '{joined}'.")


def _normalize_heatmap_data(
    data: list[list[float]] | dict[str, Any],
) -> tuple[list[list[float]], list[str] | None, list[str] | None]:
    """Normalize heatmap inputs into values and optional labels."""

    if isinstance(data, dict):
        values = data.get("values")
        if values is None:
            raise ValueError("Heatmap dictionary data must include 'values'.")
        return (
            _coerce_matrix(values),
            data.get("x_labels"),
            data.get("y_labels"),
        )
    return _coerce_matrix(data), None, None


def _coerce_matrix(values: Any) -> list[list[float]]:
    """Convert a matrix-like object into a list of float rows."""

    try:
        matrix = [[float(value) for value in row] for row in values]
    except TypeError as exc:
        raise ValueError("Heatmap values must be a 2D matrix.") from exc

    if matrix:
        width = len(matrix[0])
        if any(len(row) != width for row in matrix):
            raise ValueError("Heatmap rows must all have the same length.")
    return matrix
