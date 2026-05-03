"""Analysis helpers for GroverLab experiments and research metrics."""

from __future__ import annotations

from dataclasses import replace
from math import asin, floor, pi, sin, sqrt
from typing import Any

from groverlab.grover_config import GroverConfig, GroverResult, NoiseConfig
from groverlab.grover_data import calculate_padded_size, calculate_required_qubits
from groverlab.grover_decoherence import decoherence_probabilities


DECOHERENCE_ITERATION_PRESETS = {
    "Ideal": {"t1_relaxation_us": 500.0, "t2_coherence_us": 300.0},
    "Low decoherence": {"t1_relaxation_us": 250.0, "t2_coherence_us": 150.0},
    "Medium decoherence": {"t1_relaxation_us": 120.0, "t2_coherence_us": 80.0},
    "High decoherence": {"t1_relaxation_us": 50.0, "t2_coherence_us": 30.0},
    "Extreme decoherence": {"t1_relaxation_us": 15.0, "t2_coherence_us": 10.0},
}


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
            t1_relaxation_us=config.noise_config.t1_relaxation_us,
            t2_coherence_us=config.noise_config.t2_coherence_us,
        )
        run_config = replace(config, noise_config=noise_config)
        result = run_grover_simulation(run_config)
        results.append(_result_summary(result, {"noise_value": noise_value}))
    return results


def run_decoherence_iteration_overlay(
    config: GroverConfig,
    max_iterations: int,
) -> list[dict[str, Any]]:
    """Run iteration sweeps for standard T1/T2 decoherence scenarios."""

    rows: list[dict[str, Any]] = []
    for scenario, values in DECOHERENCE_ITERATION_PRESETS.items():
        noise_config = NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.0,
            measurement_error_prob=0.0,
            gate_error_prob=0.0,
            t1_relaxation_us=values["t1_relaxation_us"],
            t2_coherence_us=values["t2_coherence_us"],
        )
        scenario_config = replace(config, noise_config=noise_config)
        for row in run_iteration_sweep(scenario_config, max_iterations=max_iterations):
            rows.append(
                {
                    **row,
                    "scenario": scenario,
                    "t1_relaxation_us": values["t1_relaxation_us"],
                    "t2_coherence_us": values["t2_coherence_us"],
                    "target_probability": (
                        row["success_probability"]
                        if scenario == "Ideal"
                        else row.get("noisy_success_probability", row["success_probability"])
                    ),
                }
            )
    return rows


def ideal_grover_sweep_curves(
    sizes: list[int],
    max_iterations: int,
) -> list[dict[str, Any]]:
    """Generate analytical ideal Grover probability curves for dataset sizes."""

    rows: list[dict[str, Any]] = []
    for size in sizes:
        if size < 1:
            raise ValueError("Sweep sizes must be positive.")
        n_qubits = calculate_required_qubits(size)
        search_space = calculate_padded_size(n_qubits)
        theta = asin(sqrt(1 / search_space))
        recommended = recommended_iterations(size)
        for iteration in range(max_iterations + 1):
            probability = sin(((2 * iteration) + 1) * theta) ** 2
            rows.append(
                {
                    "scenario": f"N={search_space}",
                    "dataset_size": size,
                    "n_qubits": n_qubits,
                    "padded_size": search_space,
                    "iterations": iteration,
                    "target_probability": probability,
                    "recommended_iteration": recommended,
                }
            )
    return rows


def run_noisy_condition_iteration_overlay(
    config: GroverConfig,
    max_iterations: int,
) -> list[dict[str, Any]]:
    """Run noisy iteration overlays for low-to-extreme NISQ conditions."""

    scenarios = {
        "Low noise": NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.0003,
            measurement_error_prob=0.005,
            t1_relaxation_us=250.0,
            t2_coherence_us=150.0,
        ),
        "Medium noise": NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.001,
            measurement_error_prob=0.01,
            t1_relaxation_us=120.0,
            t2_coherence_us=80.0,
        ),
        "High noise": NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.005,
            measurement_error_prob=0.02,
            t1_relaxation_us=50.0,
            t2_coherence_us=30.0,
        ),
        "Extreme noise": NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.03,
            measurement_error_prob=0.05,
            t1_relaxation_us=15.0,
            t2_coherence_us=10.0,
        ),
        "Selected setting": replace(config.noise_config, noise_enabled=True),
    }
    return _run_noise_overlay_scenarios(config, scenarios, max_iterations)


def run_t1_iteration_overlay(
    config: GroverConfig,
    t1_values: list[float],
    max_iterations: int,
) -> list[dict[str, Any]]:
    """Run iteration overlays that vary T1 and keep other noise settings fixed."""

    scenarios: dict[str, NoiseConfig] = {}
    for t1 in t1_values:
        t2 = min(config.noise_config.t2_coherence_us, min(300.0, 2 * t1))
        scenarios[f"T1={t1:g} µs"] = replace(
            config.noise_config,
            noise_enabled=True,
            t1_relaxation_us=t1,
            t2_coherence_us=t2,
        )
    return _run_noise_overlay_scenarios(config, scenarios, max_iterations)


def run_t2_iteration_overlay(
    config: GroverConfig,
    t2_values: list[float],
    max_iterations: int,
) -> list[dict[str, Any]]:
    """Run iteration overlays that vary T2 and keep other noise settings fixed."""

    scenarios: dict[str, NoiseConfig] = {}
    max_t2 = min(300.0, 2 * config.noise_config.t1_relaxation_us)
    for t2 in t2_values:
        adjusted_t2 = min(t2, max_t2)
        scenarios[f"T2={adjusted_t2:g} µs"] = replace(
            config.noise_config,
            noise_enabled=True,
            t2_coherence_us=adjusted_t2,
        )
    return _run_noise_overlay_scenarios(config, scenarios, max_iterations)


def run_t1_sweep(config: GroverConfig, t1_values: list[float]) -> list[dict[str, Any]]:
    """Vary T1 while keeping T2, depolarising, and measurement settings fixed."""

    return _run_coherence_sweep(config, t1_values, varying="t1")


def run_t2_sweep(config: GroverConfig, t2_values: list[float]) -> list[dict[str, Any]]:
    """Vary T2 while keeping T1, depolarising, and measurement settings fixed."""

    return _run_coherence_sweep(config, t2_values, varying="t2")


def probability_loss_by_iteration(iteration_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate Delta P = P_ideal - P_noisy for iteration results."""

    rows: list[dict[str, Any]] = []
    for row in iteration_results:
        noisy = row.get("noisy_success_probability")
        if noisy is None:
            continue
        ideal = row["success_probability"]
        rows.append(
            {
                "iterations": row["iterations"],
                "probability_loss": ideal - noisy,
                "success_probability": ideal,
                "noisy_success_probability": noisy,
            }
        )
    return rows


def build_research_summary(
    result: GroverResult,
    iteration_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact NISQ robustness summary table row."""

    ideal_peak = max(iteration_results, key=lambda row: row.get("success_probability", 0.0))
    noisy_candidates = [
        row for row in iteration_results if row.get("noisy_success_probability") is not None
    ]
    noisy_peak = (
        max(noisy_candidates, key=lambda row: row.get("noisy_success_probability", 0.0))
        if noisy_candidates
        else None
    )
    noisy_probability = (
        noisy_peak.get("noisy_success_probability", 0.0)
        if noisy_peak is not None
        else result.noisy_success_probability
    )
    ideal_probability = ideal_peak.get("success_probability", result.success_probability)
    probabilities = decoherence_probabilities(result.config.noise_config)

    return {
        "dataset_size": result.mapping.n_items,
        "qubits": result.mapping.n_qubits,
        "padded_states": result.mapping.padded_size,
        "circuit_depth": result.circuit_depth,
        "total_gates": sum(int(value) for value in result.gate_counts.values()),
        "ideal_peak_iteration": ideal_peak.get("iterations"),
        "noisy_peak_iteration": noisy_peak.get("iterations") if noisy_peak else None,
        "ideal_peak_probability": ideal_probability,
        "noisy_peak_probability": noisy_probability,
        "probability_degradation": (
            ideal_probability - noisy_probability if noisy_probability is not None else None
        ),
        "t1_relaxation_us": result.config.noise_config.t1_relaxation_us,
        "t2_coherence_us": result.config.noise_config.t2_coherence_us,
        "t_phi_us": probabilities["t_phi_us"],
        "dominant_noise_type": dominant_noise_type(result.config.noise_config),
    }


def dominant_noise_type(noise_config: NoiseConfig) -> str:
    """Return a simple dominant-noise label for interpretation."""

    probabilities = decoherence_probabilities(noise_config)
    candidates = {
        "Depolarising": noise_config.depolar_prob,
        "Measurement error": noise_config.measurement_error_prob,
        "T1 relaxation": probabilities["amplitude_damping_probability"],
        "T2 pure dephasing": probabilities["pure_dephasing_probability"],
    }
    return max(candidates, key=candidates.get)


def scalability_analysis(sizes: list[int]) -> list[dict[str, Any]]:
    """Estimate qubit, depth, and gate growth for representative dataset sizes."""

    rows: list[dict[str, Any]] = []
    for size in sizes:
        n_qubits = calculate_required_qubits(size)
        padded_size = calculate_padded_size(n_qubits)
        iterations = recommended_iterations(size)
        estimated_depth = 1 + iterations * ((4 * n_qubits) + 3)
        estimated_total_gates = n_qubits + iterations * ((6 * n_qubits) + 4) + n_qubits
        rows.append(
            {
                "dataset_size": size,
                "n_qubits": n_qubits,
                "padded_size": padded_size,
                "recommended_iterations": iterations,
                "estimated_circuit_depth": estimated_depth,
                "estimated_total_gates": estimated_total_gates,
            }
        )
    return rows


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


def _run_coherence_sweep(
    config: GroverConfig,
    values: list[float],
    varying: str,
) -> list[dict[str, Any]]:
    """Shared implementation for T1 and T2 sweeps."""

    from groverlab.grover_runner import run_grover_simulation

    rows: list[dict[str, Any]] = []
    for value in values:
        if varying == "t1":
            t1 = value
            t2 = min(config.noise_config.t2_coherence_us, min(300.0, 2 * t1))
        else:
            t1 = config.noise_config.t1_relaxation_us
            t2 = min(value, min(300.0, 2 * t1))

        noise_config = replace(
            config.noise_config,
            noise_enabled=True,
            t1_relaxation_us=t1,
            t2_coherence_us=t2,
        )
        result = run_grover_simulation(replace(config, noise_config=noise_config))
        key = "t1_relaxation_us" if varying == "t1" else "t2_coherence_us"
        rows.append(
            _result_summary(
                result,
                {
                    key: value,
                    "t1_relaxation_us": t1,
                    "t2_coherence_us": t2,
                },
            )
        )
    return rows


def _run_noise_overlay_scenarios(
    config: GroverConfig,
    scenarios: dict[str, NoiseConfig],
    max_iterations: int,
) -> list[dict[str, Any]]:
    """Run noisy iteration curves for named noise scenarios."""

    rows: list[dict[str, Any]] = []
    for scenario, noise_config in scenarios.items():
        scenario_config = replace(config, noise_config=noise_config)
        for row in run_iteration_sweep(scenario_config, max_iterations=max_iterations):
            rows.append(
                {
                    **row,
                    "scenario": scenario,
                    "target_probability": row.get(
                        "noisy_success_probability",
                        row["success_probability"],
                    ),
                    "recommended_iteration": recommended_iterations(
                        len(config.dataset_items)
                    ),
                    "t1_relaxation_us": noise_config.t1_relaxation_us,
                    "t2_coherence_us": noise_config.t2_coherence_us,
                    "depolar_prob": noise_config.depolar_prob,
                    "measurement_error_prob": noise_config.measurement_error_prob,
                }
            )
    return rows
