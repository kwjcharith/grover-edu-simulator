"""Matplotlib plotting utilities for GroverLab experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


TARGET_COLOR = "#D1495B"
MEASURED_STATE_COLOR = "#2E86AB"
NOISY_MOST_LIKELY_COLOR = "#F4A261"


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
        TARGET_COLOR if target_binary and label == target_binary else MEASURED_STATE_COLOR
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_title("Measurement Counts")
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Counts")
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MEASURED_STATE_COLOR, label="Measured states")]
    if target_binary:
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, color=TARGET_COLOR, label="Target"))
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_iteration_sweep(results: list[dict[str, Any]], save_path: str | Path | None = None):
    """Plot ideal and optional noisy success probabilities across iterations."""

    _validate_results(results)
    iterations = [_require_key(row, "iterations") for row in results]
    success = [_require_key(row, "success_probability") for row in results]
    noisy_success = [row.get("noisy_success_probability") for row in results]
    has_noisy_curve = any(value is not None for value in noisy_success)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, success, marker="o", color="#2E86AB", linewidth=2, label="Ideal")
    if has_noisy_curve:
        ax.plot(
            iterations,
            [0.0 if value is None else value for value in noisy_success],
            marker="s",
            color="#D1495B",
            linewidth=2,
            label="Noisy",
        )
        ax.legend(loc="best")
    ax.set_title("Ideal vs Noisy Grover Iteration Sweep")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Target success probability")
    _set_adaptive_xlim(ax, iterations)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_counts_probability_comparison(
    ideal_counts: dict[str, int],
    noisy_counts: dict[str, int],
    target_binary: str,
    save_path: str | Path | None = None,
):
    """Plot side-by-side ideal and noisy measurement probabilities."""

    if not ideal_counts:
        raise ValueError("Ideal measurement counts cannot be empty.")
    if not noisy_counts:
        raise ValueError("Noisy measurement counts cannot be empty.")

    labels = _comparison_labels(ideal_counts, noisy_counts, target_binary)
    ideal_probabilities = _probabilities_for_labels(ideal_counts, labels)
    noisy_probabilities = _probabilities_for_labels(noisy_counts, labels)
    noisy_most_likely = max(noisy_counts, key=noisy_counts.get)
    ideal_colors = _comparison_bar_colors(labels, target_binary)
    noisy_colors = _comparison_bar_colors(labels, target_binary, noisy_most_likely)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, title, probabilities, colors in (
        (axes[0], "Ideal Simulation", ideal_probabilities, ideal_colors),
        (axes[1], "Noisy Simulation (Adaptive Y-Axis)", noisy_probabilities, noisy_colors),
    ):
        ax.bar(labels, probabilities, color=colors)
        ax.set_title(title)
        ax.set_xlabel("Measured bitstring")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, _adaptive_probability_upper_limit(noisy_probabilities))
    axes[0].set_ylabel("Measurement probability")
    axes[1].set_ylabel("Measurement probability")
    axes[1].legend(
        handles=_comparison_legend_handles(target_binary, noisy_most_likely),
        loc="best",
    )
    fig.suptitle("Final Measurement Probability Comparison")
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
    _set_adaptive_xlim(ax, noise_values)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_decoherence_iteration_overlay(
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
):
    """Plot target probability across iterations for multiple T1/T2 scenarios."""

    _validate_results(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    scenarios = list(dict.fromkeys(row["scenario"] for row in results))
    for scenario in scenarios:
        rows = [row for row in results if row["scenario"] == scenario]
        iterations = [_require_key(row, "iterations") for row in rows]
        probabilities = [_require_key(row, "target_probability") for row in rows]
        ax.plot(iterations, probabilities, marker="o", linewidth=2, label=scenario)
        peak_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        ax.scatter(
            [iterations[peak_index]],
            [probabilities[peak_index]],
            s=55,
            zorder=4,
        )
        ax.annotate(
            f"k={iterations[peak_index]}\nP={probabilities[peak_index]:.2f}",
            (iterations[peak_index], probabilities[peak_index]),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=8,
        )

    ax.set_title("Grover Amplification Under Increasing Decoherence")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Target-state probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_iteration_overlay_curves(
    results: list[dict[str, Any]],
    title: str,
    adaptive_y: bool = False,
    save_path: str | Path | None = None,
):
    """Plot multiple target-probability curves over Grover iterations."""

    _validate_results(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    all_probabilities: list[float] = []
    recommended_iterations = sorted(
        {
            row["recommended_iteration"]
            for row in results
            if row.get("recommended_iteration") is not None
        }
    )
    scenarios = list(dict.fromkeys(row["scenario"] for row in results))
    for scenario in scenarios:
        rows = [row for row in results if row["scenario"] == scenario]
        iterations = [_require_key(row, "iterations") for row in rows]
        probabilities = [_require_key(row, "target_probability") for row in rows]
        all_probabilities.extend(float(value) for value in probabilities)
        ax.plot(iterations, probabilities, marker="o", linewidth=2, label=scenario)
        _annotate_peak(ax, iterations, probabilities)

    for recommended in recommended_iterations[:5]:
        ax.axvline(recommended, linestyle="--", linewidth=1.2, color="#444444", alpha=0.55)
        ax.text(
            recommended,
            0.98 if not adaptive_y else _adaptive_overlay_ymax(all_probabilities) * 0.92,
            "Recommended k",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#444444",
        )

    ax.set_title(title)
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Target-state probability")
    if adaptive_y:
        ax.set_ylim(0, _adaptive_overlay_ymax(all_probabilities))
    else:
        ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_probability_loss(
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
):
    """Plot Delta P = P_ideal - P_noisy across iterations."""

    _validate_results(results)
    iterations = [_require_key(row, "iterations") for row in results]
    losses = [_require_key(row, "probability_loss") for row in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, losses, marker="o", color="#D1495B", linewidth=2)
    ax.axhline(0, color="#222222", linewidth=1, alpha=0.5)
    ax.set_title("Probability Loss From Ideal")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Delta P = Pideal - Pnoisy")
    _set_adaptive_xlim(ax, iterations)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_coherence_sweep(
    results: list[dict[str, Any]],
    x_key: str,
    title: str,
    x_label: str,
    save_path: str | Path | None = None,
):
    """Plot target probability while sweeping a physical coherence time."""

    _validate_results(results)
    x_values = [_require_key(row, x_key) for row in results]
    noisy_success = [
        row.get("noisy_success_probability", row.get("success_probability"))
        for row in results
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_values, noisy_success, marker="o", color="#2E86AB", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Target-state probability")
    _set_adaptive_xlim(ax, x_values)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_scalability_growth(
    results: list[dict[str, Any]],
    y_key: str,
    title: str,
    y_label: str,
    save_path: str | Path | None = None,
):
    """Plot a scalability-growth metric against dataset size."""

    _validate_results(results)
    sizes = [_require_key(row, "dataset_size") for row in results]
    values = [_require_key(row, y_key) for row in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, values, marker="o", color="#2E86AB", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
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
    ax.set_xlabel("Dataset size (log scale)")
    ax.set_ylabel("Queries or iterations")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both")
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


def _comparison_labels(
    ideal_counts: dict[str, int],
    noisy_counts: dict[str, int],
    target_binary: str,
    top_n: int = 14,
) -> list[str]:
    """Choose readable comparison states, always retaining the target."""

    all_labels = set(ideal_counts) | set(noisy_counts)
    ranked = sorted(
        all_labels,
        key=lambda label: max(ideal_counts.get(label, 0), noisy_counts.get(label, 0)),
        reverse=True,
    )
    labels: list[str] = []
    if target_binary:
        labels.append(target_binary)
    for label in ranked:
        if label not in labels:
            labels.append(label)
        if len(labels) >= top_n:
            break
    return sorted(labels)


def _probabilities_for_labels(counts: dict[str, int], labels: list[str]) -> list[float]:
    """Convert counts into probabilities for a fixed label order."""

    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Measurement counts must sum to a positive value.")
    return [counts.get(label, 0) / total for label in labels]


def _comparison_bar_colors(
    labels: list[str],
    target_binary: str,
    noisy_most_likely: str | None = None,
) -> list[str]:
    """Color target and optional noisy most-likely states distinctly."""

    colors: list[str] = []
    for label in labels:
        if target_binary and label == target_binary:
            colors.append(TARGET_COLOR)
        elif noisy_most_likely and label == noisy_most_likely:
            colors.append(NOISY_MOST_LIKELY_COLOR)
        else:
            colors.append(MEASURED_STATE_COLOR)
    return colors


def _comparison_legend_handles(target_binary: str, noisy_most_likely: str) -> list:
    """Build a legend for target and noisy most-likely highlighting."""

    handles = [plt.Rectangle((0, 0), 1, 1, color=MEASURED_STATE_COLOR, label="Measured states")]
    if target_binary:
        handles.insert(0, plt.Rectangle((0, 0), 1, 1, color=TARGET_COLOR, label="Target state"))
    if noisy_most_likely != target_binary:
        handles.insert(
            1 if target_binary else 0,
            plt.Rectangle((0, 0), 1, 1, color=NOISY_MOST_LIKELY_COLOR, label="Noisy most likely"),
        )
    return handles


def _annotate_peak(ax, iterations: list[Any], probabilities: list[Any]) -> None:
    """Mark and label the peak of one iteration curve."""

    peak_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    peak_iteration = iterations[peak_index]
    peak_probability = probabilities[peak_index]
    ax.scatter([peak_iteration], [peak_probability], s=55, zorder=4)
    ax.annotate(
        f"Peak k={peak_iteration}, P={peak_probability:.2f}",
        (peak_iteration, peak_probability),
        textcoords="offset points",
        xytext=(4, 5),
        fontsize=8,
    )


def _adaptive_overlay_ymax(probabilities: list[float]) -> float:
    """Return adaptive y-axis max for noisy overlay plots."""

    if not probabilities:
        return 0.05
    return max(0.05, max(probabilities) * 1.15)


def _adaptive_probability_upper_limit(probabilities: list[float]) -> float:
    """Return a readable y-axis upper limit for flattened noisy distributions."""

    max_probability = max(probabilities, default=0.0)
    if max_probability <= 0:
        return 1.0
    if max_probability >= 0.8:
        return 1.05
    return min(1.05, max(0.01, max_probability * 1.25))


def _set_adaptive_xlim(ax, values: list[Any]) -> None:
    """Set x-axis limits from actual data with a readable margin."""

    numeric_values = [float(value) for value in values]
    min_value = min(numeric_values)
    max_value = max(numeric_values)
    if min_value == max_value:
        margin = 1.0 if min_value == 0 else abs(min_value) * 0.1
    else:
        margin = (max_value - min_value) * 0.08
    ax.set_xlim(min_value - margin, max_value + margin)


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
