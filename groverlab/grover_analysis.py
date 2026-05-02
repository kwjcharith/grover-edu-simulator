"""Analysis helpers for GroverLab experiments and research metrics."""

from __future__ import annotations

from dataclasses import replace
from math import floor, pi, sqrt
from typing import Any

from groverlab.grover_config import GroverConfig, GroverResult, NoiseConfig
from groverlab.grover_data import calculate_padded_size, calculate_required_qubits


def recommended_iterations(n_items: int, n_solutions: int = 1) -> int:
    """Return floor((pi / 4) * sqrt(N / M)) using padded searchable space N."""

    if n_items < 1:
        raise ValueError("n_items must be positive.")
    if n_solutions < 1:
        raise ValueError("n_solutions must be positive.")

    n_qubits = calculate_required_qubits(n_items)
    searchable_states = calculate_padded_size(n_qubits)
    if n_solutions > searchable_states:
        raise ValueError("n_solutions cannot exceed the searchable state space.")

    return floor((pi / 4) * sqrt(searchable_states / n_solutions))


def calculate_success_probability(counts: dict[str, int], target_binary: str) -> float:
    """Calculate the measured probability of the target bitstring."""

    if not counts:
        raise ValueError("Measurement counts cannot be empty.")
    if not target_binary or any(bit not in {"0", "1"} for bit in target_binary):
        raise ValueError("target_binary must contain only 0 and 1 characters.")

    total_counts = sum(counts.values())
    if total_counts <= 0:
        raise ValueError("Measurement counts must sum to a positive value.")
    return counts.get(target_binary, 0) / total_counts


def get_top_measurements(counts: dict[str, int], top_n: int = 5) -> list[tuple[str, int]]:
    """Return the top measurements sorted by count descending."""

    if top_n < 1:
        return []
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]


def run_iteration_sweep(
    config: GroverConfig,
    max_iterations: int | None = None,
) -> list[dict[str, Any]]:
    """Run the same configuration over a range of Grover iteration counts."""

    from groverlab.grover_runner import run_grover_simulation

    if max_iterations is None:
        recommended = recommended_iterations(len(config.dataset_items))
        max_iterations = max(1, (2 * recommended) + 2)
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative.")

    results: list[dict[str, Any]] = []
    for iteration in range(max_iterations + 1):
        run_config = replace(config, iterations=iteration)
        result = run_grover_simulation(run_config)
        results.append(_result_summary(result, {"iterations": iteration}))
    return results


def run_noise_sweep(config: GroverConfig, noise_values: list[float]) -> list[dict[str, Any]]:
    """Run noisy simulations for each supplied depolarizing-noise value."""

    from groverlab.grover_runner import run_grover_simulation

    results: list[dict[str, Any]] = []
    for noise_value in noise_values:
        if noise_value < 0 or noise_value > 1:
            raise ValueError("Noise values must be between 0 and 1.")

        noise_config = NoiseConfig(
            noise_enabled=True,
            depolar_prob=noise_value,
            measurement_error_prob=config.noise_config.measurement_error_prob,
            gate_error_prob=config.noise_config.gate_error_prob,
        )
        run_config = replace(config, noise_config=noise_config)
        result = run_grover_simulation(run_config)
        results.append(_result_summary(result, {"noise_value": noise_value}))
    return results


def run_size_sweep(
    base_items: list[str],
    target_item: str,
    sizes: list[int],
) -> list[dict[str, Any]]:
    """Run Grover simulations across dataset sizes derived from base_items."""

    from groverlab.grover_runner import run_grover_simulation

    if not base_items:
        raise ValueError("base_items must not be empty.")
    if not target_item or not target_item.strip():
        raise ValueError("target_item must not be empty.")

    results: list[dict[str, Any]] = []
    for size in sizes:
        if size < 1:
            raise ValueError("Sweep sizes must be positive.")
        items = _items_for_size(base_items, target_item, size)
        config = GroverConfig(dataset_items=items, target_item=target_item, shots=256)
        result = run_grover_simulation(config)
        results.append(_result_summary(result, {"dataset_size": size}))
    return results


def compare_classical_vs_grover(n_items: int) -> dict[str, float | int]:
    """Compare classical linear-search query counts with Grover iterations."""

    if n_items < 1:
        raise ValueError("n_items must be positive.")

    n_qubits = calculate_required_qubits(n_items)
    padded_size = calculate_padded_size(n_qubits)
    return {
        "n_items": n_items,
        "padded_size": padded_size,
        "n_qubits": n_qubits,
        "classical_worst_case_queries": n_items,
        "classical_average_queries": (n_items + 1) / 2,
        "grover_recommended_iterations": recommended_iterations(n_items),
    }


def detect_over_rotation(iteration_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect whether success falls after the best observed iteration count."""

    if not iteration_results:
        raise ValueError("iteration_results must not be empty.")

    best = max(iteration_results, key=lambda row: row.get("success_probability", 0.0))
    best_index = iteration_results.index(best)
    later_results = iteration_results[best_index + 1 :]
    later_drop = any(
        row.get("success_probability", 0.0) < best.get("success_probability", 0.0)
        for row in later_results
    )

    return {
        "over_rotation_detected": later_drop,
        "optimal_observed_iteration_count": best.get("iterations"),
        "max_success_probability": best.get("success_probability", 0.0),
    }


def analyse_research_metrics(result: GroverResult) -> dict[str, Any]:
    """Summarize result-level metrics for educational research analysis."""

    total_gates = sum(int(value) for value in result.gate_counts.values())
    ideal_noisy_gap = None
    if result.noisy_success_probability is not None:
        ideal_noisy_gap = result.success_probability - result.noisy_success_probability

    comparison = compare_classical_vs_grover(result.mapping.n_items)
    return {
        "success_probability": result.success_probability,
        "circuit_depth": result.circuit_depth,
        "total_gates": total_gates,
        "n_qubits": result.mapping.n_qubits,
        "shots": result.config.shots,
        "runtime_seconds": result.runtime_seconds,
        "ideal_vs_noisy_success_gap": ideal_noisy_gap,
        "classical_worst_case_queries": comparison["classical_worst_case_queries"],
        "classical_average_queries": comparison["classical_average_queries"],
        "grover_recommended_iterations": comparison["grover_recommended_iterations"],
    }


def most_likely_bitstring(counts: dict[str, int]) -> str:
    """Return the most frequently measured bitstring."""

    if not counts:
        raise ValueError("Measurement counts cannot be empty.")
    return get_top_measurements(counts, top_n=1)[0][0]


def classical_linear_search_steps(items: list[str], target_item: str) -> int | None:
    """Return one-based linear-search steps, or None when the target is absent."""

    for step, item in enumerate(items, start=1):
        if item == target_item:
            return step
    return None


def _result_summary(result: GroverResult, extra_fields: dict[str, Any]) -> dict[str, Any]:
    """Convert a GroverResult to a compact sweep row."""

    row = {
        **extra_fields,
        "success_probability": result.success_probability,
        "noisy_success_probability": result.noisy_success_probability,
        "circuit_depth": result.circuit_depth,
        "total_gates": sum(int(value) for value in result.gate_counts.values()),
        "n_qubits": result.mapping.n_qubits,
        "shots": result.config.shots,
        "runtime_seconds": result.runtime_seconds,
        "decoded_item": result.decoded_item,
        "found": result.found,
    }
    return row


def _items_for_size(base_items: list[str], target_item: str, size: int) -> list[str]:
    """Build a deterministic dataset of size with target_item included."""

    items: list[str] = []
    index = 0
    while len(items) < size:
        candidate = base_items[index % len(base_items)]
        if candidate not in items:
            items.append(candidate)
        else:
            items.append(f"{candidate}_{index}")
        index += 1

    if target_item not in items:
        items[-1] = target_item
    return items

