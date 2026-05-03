from groverlab.grover_analysis import (
    analyse_research_metrics,
    build_research_summary,
    calculate_success_probability,
    classical_linear_search_steps,
    compare_classical_vs_grover,
    detect_over_rotation,
    get_top_measurements,
    ideal_grover_sweep_curves,
    most_likely_bitstring,
    probability_loss_by_iteration,
    recommended_iterations,
    run_decoherence_iteration_overlay,
    run_iteration_sweep,
    run_noisy_condition_iteration_overlay,
    run_noise_sweep,
    run_t1_iteration_overlay,
    run_t1_sweep,
    run_t2_iteration_overlay,
    scalability_analysis,
)
from groverlab.grover_config import GroverConfig, NoiseConfig
from groverlab.grover_runner import run_grover_simulation


def test_recommended_iteration_calculation_uses_padded_space():
    assert recommended_iterations(4) == 1
    assert recommended_iterations(5) == 2
    assert recommended_iterations(16) == 3


def test_success_probability_calculation():
    counts = {"00": 3, "10": 7}

    assert calculate_success_probability(counts, "10") == 0.7
    assert calculate_success_probability(counts, "11") == 0.0


def test_get_top_measurements_sorts_by_count():
    assert get_top_measurements({"00": 3, "11": 9, "10": 4}, top_n=2) == [
        ("11", 9),
        ("10", 4),
    ]


def test_iteration_sweep_returns_list():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=64,
        seed=42,
    )

    results = run_iteration_sweep(config, max_iterations=2)

    assert isinstance(results, list)
    assert [row["iterations"] for row in results] == [0, 1, 2]
    assert all("success_probability" in row for row in results)


def test_noise_sweep_returns_list():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=64,
        iterations=1,
        seed=42,
    )

    results = run_noise_sweep(config, noise_values=[0.0, 0.01])

    assert isinstance(results, list)
    assert [row["noise_value"] for row in results] == [0.0, 0.01]
    assert all(row["noisy_success_probability"] is not None for row in results)


def test_classical_vs_grover_comparison():
    comparison = compare_classical_vs_grover(4)

    assert comparison["classical_worst_case_queries"] == 4
    assert comparison["classical_average_queries"] == 2.5
    assert comparison["grover_recommended_iterations"] == 1
    assert comparison["n_qubits"] == 2


def test_detect_over_rotation_reports_best_iteration():
    result = detect_over_rotation(
        [
            {"iterations": 0, "success_probability": 0.25},
            {"iterations": 1, "success_probability": 1.0},
            {"iterations": 2, "success_probability": 0.25},
        ]
    )

    assert result["over_rotation_detected"] is True
    assert result["optimal_observed_iteration_count"] == 1


def test_analyse_research_metrics_contains_requested_fields():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=64,
        iterations=1,
        seed=42,
    )
    result = run_grover_simulation(config)

    metrics = analyse_research_metrics(result)

    assert metrics["success_probability"] == 1.0
    assert metrics["circuit_depth"] > 0
    assert metrics["total_gates"] > 0
    assert metrics["n_qubits"] == 2
    assert metrics["shots"] == 64
    assert metrics["grover_recommended_iterations"] == 1


def test_most_likely_bitstring_returns_max_count():
    assert most_likely_bitstring({"00": 3, "11": 9, "10": 4}) == "11"


def test_classical_linear_search_steps_returns_one_based_step():
    assert classical_linear_search_steps(["a", "b", "c"], "b") == 2
    assert classical_linear_search_steps(["a", "b", "c"], "z") is None


def test_decoherence_iteration_overlay_returns_scenario_rows():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=32,
        iterations=1,
        seed=42,
    )

    rows = run_decoherence_iteration_overlay(config, max_iterations=1)

    assert {row["scenario"] for row in rows} >= {"Ideal", "Medium decoherence"}
    assert all("target_probability" in row for row in rows)


def test_t1_sweep_and_probability_loss_return_research_rows():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=32,
        iterations=1,
        seed=42,
        noise_config=NoiseConfig(noise_enabled=True),
    )

    t1_rows = run_t1_sweep(config, [50.0, 120.0])
    iteration_rows = run_iteration_sweep(config, max_iterations=1)
    loss_rows = probability_loss_by_iteration(iteration_rows)
    result = run_grover_simulation(config)
    summary = build_research_summary(result, iteration_rows)

    assert [row["t1_relaxation_us"] for row in t1_rows] == [50.0, 120.0]
    assert all("probability_loss" in row for row in loss_rows)
    assert "dominant_noise_type" in summary


def test_scalability_analysis_contains_growth_metrics():
    rows = scalability_analysis([4, 16])

    assert rows[0]["n_qubits"] == 2
    assert rows[1]["estimated_circuit_depth"] > rows[0]["estimated_circuit_depth"]


def test_ideal_grover_sweep_curves_return_multiple_dataset_lines():
    rows = ideal_grover_sweep_curves([16, 32], max_iterations=2)

    assert {row["scenario"] for row in rows} == {"N=16", "N=32"}
    assert all(0 <= row["target_probability"] <= 1 for row in rows)


def test_noisy_and_coherence_iteration_overlays_return_curve_rows():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=32,
        iterations=1,
        seed=42,
        noise_config=NoiseConfig(noise_enabled=True),
    )

    noisy_rows = run_noisy_condition_iteration_overlay(config, max_iterations=1)
    t1_rows = run_t1_iteration_overlay(config, [500.0, 120.0], max_iterations=1)
    t2_rows = run_t2_iteration_overlay(config, [80.0, 30.0], max_iterations=1)

    assert {"Low noise", "Selected setting"}.issubset({row["scenario"] for row in noisy_rows})
    assert {row["scenario"] for row in t1_rows} == {"T1=500 µs", "T1=120 µs"}
    assert {row["scenario"] for row in t2_rows} == {"T2=80 µs", "T2=30 µs"}
