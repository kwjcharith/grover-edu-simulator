import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from groverlab.grover_plots import (
    NOISY_MOST_LIKELY_COLOR,
    TARGET_COLOR,
    plot_classical_vs_grover,
    plot_counts_histogram,
    plot_counts_probability_comparison,
    plot_iteration_sweep,
    plot_noise_sweep,
    plot_success_probability_heatmap,
)


def test_plot_counts_histogram_returns_figure_and_saves(tmp_path):
    output = tmp_path / "counts.png"

    fig = plot_counts_histogram({"00": 4, "10": 12}, "10", save_path=output)

    assert isinstance(fig, Figure)
    assert output.exists()


def test_plot_iteration_sweep_returns_figure():
    fig = plot_iteration_sweep(
        [
            {"iterations": 0, "success_probability": 0.25},
            {"iterations": 1, "success_probability": 1.0},
        ]
    )

    assert isinstance(fig, Figure)


def test_plot_iteration_sweep_includes_noisy_curve_when_available():
    fig = plot_iteration_sweep(
        [
            {
                "iterations": 0,
                "success_probability": 0.25,
                "noisy_success_probability": 0.24,
            },
            {
                "iterations": 1,
                "success_probability": 1.0,
                "noisy_success_probability": 0.82,
            },
        ]
    )

    assert len(fig.axes[0].lines) == 2


def test_plot_iteration_sweep_x_axis_adapts_to_large_values():
    fig = plot_iteration_sweep(
        [
            {"iterations": 0, "success_probability": 0.1},
            {"iterations": 25, "success_probability": 0.8},
            {"iterations": 50, "success_probability": 0.2},
        ]
    )

    left, right = fig.axes[0].get_xlim()
    assert left <= 0
    assert right >= 50


def test_plot_noise_sweep_returns_figure():
    fig = plot_noise_sweep(
        [
            {
                "noise_value": 0.0,
                "success_probability": 1.0,
                "noisy_success_probability": 1.0,
            },
            {
                "noise_value": 0.02,
                "success_probability": 1.0,
                "noisy_success_probability": 0.9,
            },
        ]
    )

    assert isinstance(fig, Figure)


def test_plot_counts_probability_comparison_returns_figure():
    fig = plot_counts_probability_comparison(
        ideal_counts={"00": 1, "10": 99},
        noisy_counts={"00": 20, "01": 15, "10": 65},
        target_binary="10",
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_counts_probability_comparison_uses_adaptive_noisy_y_axis():
    fig = plot_counts_probability_comparison(
        ideal_counts={"00": 1, "10": 999},
        noisy_counts={"00": 3, "01": 2, "10": 4, "11": 1},
        target_binary="10",
    )

    ideal_top = fig.axes[0].get_ylim()[1]
    noisy_top = fig.axes[1].get_ylim()[1]
    assert ideal_top == 1.05
    assert noisy_top < ideal_top
    assert noisy_top >= 0.4


def test_plot_counts_probability_comparison_highlights_noisy_most_likely_state():
    fig = plot_counts_probability_comparison(
        ideal_counts={"00": 1, "10": 99},
        noisy_counts={"00": 70, "10": 30},
        target_binary="10",
    )

    noisy_bar_colors = [patch.get_facecolor() for patch in fig.axes[1].patches]
    assert to_rgba(NOISY_MOST_LIKELY_COLOR) in noisy_bar_colors
    assert to_rgba(TARGET_COLOR) in noisy_bar_colors


def test_plot_noise_sweep_x_axis_adapts_to_large_values():
    fig = plot_noise_sweep(
        [
            {
                "noise_value": 0.0,
                "success_probability": 1.0,
                "noisy_success_probability": 1.0,
            },
            {
                "noise_value": 0.6,
                "success_probability": 1.0,
                "noisy_success_probability": 0.2,
            },
            {
                "noise_value": 1.0,
                "success_probability": 1.0,
                "noisy_success_probability": 0.0,
            },
        ]
    )

    left, right = fig.axes[0].get_xlim()
    assert left <= 0
    assert right >= 1.0


def test_plot_classical_vs_grover_returns_figure():
    fig = plot_classical_vs_grover(
        [
            {
                "n_items": 4,
                "classical_worst_case_queries": 4,
                "classical_average_queries": 2.5,
                "grover_recommended_iterations": 1,
            },
            {
                "n_items": 8,
                "classical_worst_case_queries": 8,
                "classical_average_queries": 4.5,
                "grover_recommended_iterations": 2,
            },
        ]
    )

    assert isinstance(fig, Figure)


def test_plot_success_probability_heatmap_returns_figure():
    fig = plot_success_probability_heatmap(
        {
            "values": [[0.25, 0.8], [0.4, 0.95]],
            "x_labels": ["low", "high"],
            "y_labels": ["small", "large"],
        }
    )

    assert isinstance(fig, Figure)
