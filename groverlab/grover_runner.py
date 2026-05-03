"""Simulation orchestration independent from Streamlit or FastAPI."""

from __future__ import annotations

import time
from dataclasses import dataclass

from groverlab.grover_config import GroverConfig, GroverResult, SimulationConfig
from groverlab.grover_core import (
    build_grover_circuit,
    decode_result,
    get_gate_counts,
    recommended_iterations,
    run_ideal_simulation,
    run_noisy_simulation,
)
from groverlab.grover_data import create_dataset_mapping, index_dataset, index_to_bitstring
from groverlab.grover_education import CORE_WARNING, MISCONCEPTION_WARNINGS


@dataclass(frozen=True)
class GroverRunRequest:
    """Compatibility request used by the initial runner scaffold."""

    items: list[str]
    target_item: str
    config: SimulationConfig = SimulationConfig()


@dataclass(frozen=True)
class PreparedGroverRun:
    """Prepared data needed before circuit construction and simulation."""

    target_item: str
    target_index: int
    target_bitstring: str
    num_qubits: int
    search_space_size: int
    iterations: int
    padded_items: list[str | None]


def run_grover_simulation(config: GroverConfig) -> GroverResult:
    """Run a complete Grover simulation and return a structured result."""

    start_time = time.perf_counter()
    mapping = create_dataset_mapping(
        config.dataset_items,
        config.target_item,
        missing_target_mode=config.missing_target_mode,
    )
    iterations = config.iterations
    if iterations is None:
        iterations = recommended_iterations(mapping.padded_size)

    if not mapping.target_found and config.missing_target_mode == "stop":
        warnings = _generate_warnings(mapping, config, iterations)
        explanation_steps = _generate_explanation_steps(mapping, iterations, config.education_mode)
        runtime_seconds = time.perf_counter() - start_time
        return GroverResult(
            config=config,
            mapping=mapping,
            circuit_depth=0,
            gate_counts={},
            ideal_counts={},
            noisy_counts=None,
            success_probability=0.0,
            noisy_success_probability=None,
            measured_bitstring="",
            decoded_item=None,
            found=False,
            target_found=False,
            stopped_before_quantum_execution=True,
            missing_target_explanation=mapping.missing_target_explanation,
            runtime_seconds=runtime_seconds,
            explanation_steps=explanation_steps,
            warnings=warnings,
        )

    circuit = build_grover_circuit(
        mapping,
        iterations=iterations,
        measure=True,
        allow_no_target=config.missing_target_mode == "experimental",
    )
    ideal_counts = run_ideal_simulation(circuit, shots=config.shots, seed=config.seed)
    noisy_counts = run_noisy_simulation(
        circuit,
        shots=config.shots,
        noise_config=config.noise_config,
        seed=config.seed,
    )

    decoded = decode_result(ideal_counts, mapping)
    noisy_decoded = decode_result(noisy_counts, mapping) if noisy_counts else None
    warnings = _generate_warnings(mapping, config, iterations)
    explanation_steps = _generate_explanation_steps(mapping, iterations, config.education_mode)
    runtime_seconds = time.perf_counter() - start_time

    return GroverResult(
        config=config,
        mapping=mapping,
        circuit_depth=circuit.depth(),
        gate_counts=get_gate_counts(circuit),
        ideal_counts=ideal_counts,
        noisy_counts=noisy_counts,
        success_probability=decoded["success_probability"],
        noisy_success_probability=(
            noisy_decoded["success_probability"] if noisy_decoded is not None else None
        ),
        measured_bitstring=decoded["measured_bitstring"],
        decoded_item=decoded["decoded_item"],
        found=decoded["found"],
        target_found=mapping.target_found,
        stopped_before_quantum_execution=False,
        missing_target_explanation=mapping.missing_target_explanation,
        runtime_seconds=runtime_seconds,
        explanation_steps=explanation_steps,
        warnings=warnings,
    )


def prepare_run(request: GroverRunRequest) -> PreparedGroverRun:
    """Clean data, validate the target, map it to a bitstring, and choose iterations."""

    dataset = index_dataset(request.items)
    target = request.target_item.strip()
    if target not in dataset.item_to_index:
        raise ValueError("Target item was not found in the cleaned dataset.")

    target_index = dataset.item_to_index[target]
    search_space_size = len(dataset.padded_items)
    iterations = request.config.manual_iterations
    if iterations is None:
        iterations = recommended_iterations(search_space_size)

    return PreparedGroverRun(
        target_item=target,
        target_index=target_index,
        target_bitstring=index_to_bitstring(target_index, dataset.num_qubits),
        num_qubits=dataset.num_qubits,
        search_space_size=search_space_size,
        iterations=iterations,
        padded_items=dataset.padded_items,
    )


def run_simulation(request: GroverRunRequest) -> GroverResult:
    """Compatibility wrapper around run_grover_simulation."""

    config = GroverConfig(
        dataset_items=request.items,
        target_item=request.target_item,
        shots=request.config.shots,
        iterations=request.config.manual_iterations,
        seed=request.config.seed,
    )
    return run_grover_simulation(config)


def _generate_explanation_steps(
    mapping,
    iterations: int,
    education_mode: bool,
) -> list[str]:
    """Generate concise educational steps for the completed run."""

    if not education_mode:
        return []

    if not mapping.target_found:
        return [
            CORE_WARNING,
            mapping.missing_target_explanation or "",
            (
                f"The cleaned dataset has {mapping.n_items} items encoded into "
                f"{mapping.n_qubits} qubits and {mapping.padded_size} padded states."
            ),
            "No valid marked state exists, so a meaningful Grover oracle cannot be constructed.",
            "In experimental no-solution mode, the circuit demonstrates near-uniform measurement rather than a successful search.",
        ]

    return [
        CORE_WARNING,
        (
            f"The cleaned dataset has {mapping.n_items} items and is encoded into "
            f"{mapping.n_qubits} qubits."
        ),
        (
            f"The target item '{mapping.target_item}' maps to index "
            f"{mapping.target_index} and bitstring {mapping.target_binary}."
        ),
        (
            f"The search space is padded to {mapping.padded_size} basis states, "
            f"leaving {mapping.unused_states} unused states."
        ),
        "Hadamard gates create an equal superposition over encoded indices.",
        "The phase oracle marks the target state by flipping its phase.",
        f"The oracle and diffuser are repeated {iterations} time(s).",
        "Measurement is probabilistic, so counts should be interpreted statistically.",
    ]


def _generate_warnings(mapping, config: GroverConfig, iterations: int) -> list[str]:
    """Collect mapping, educational, and runtime warnings for a result."""

    warnings = [*mapping.warnings, *MISCONCEPTION_WARNINGS]
    if mapping.unused_states > 0:
        warnings.append(
            f"{mapping.unused_states} padded state(s) do not correspond to dataset items."
        )
    if config.noise_config.noise_enabled:
        warnings.append(
            "Noisy simulation is enabled; depolarising noise, measurement error, T1 relaxation, and T2 dephasing can reduce measured success."
        )
    if not mapping.target_found and config.missing_target_mode == "experimental":
        warnings.append(
            "This run is an experimental no-solution demonstration. Any measured item is random and should not be interpreted as a successful search."
        )
    if iterations == 0:
        warnings.append("Zero Grover iterations were requested; this only samples superposition.")
    return _deduplicate(warnings)


def _deduplicate(items: list[str]) -> list[str]:
    """Preserve order while removing duplicate warnings."""

    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item and item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items
