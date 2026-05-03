import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from groverlab.grover_plots import (
    plot_classical_vs_grover,
    plot_counts_histogram,
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
