"""Quantum circuit construction and simulation for GroverLab."""

from __future__ import annotations

from math import floor, pi, sqrt
from typing import Any

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from groverlab.grover_config import DatasetMapping, NoiseConfig
from groverlab.grover_data import decode_bitstring
from groverlab.grover_oracle import apply_multi_controlled_z, apply_phase_oracle


def create_initial_circuit(n_qubits: int) -> QuantumCircuit:
    """Create an empty quantum circuit with n_qubits quantum wires."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    return QuantumCircuit(n_qubits)


def apply_superposition(qc: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
    """Apply Hadamard gates to all search qubits."""

    _validate_qubit_count(qc, n_qubits)
    qc.h(range(n_qubits))
    return qc


def apply_diffuser(qc: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
    """Apply Grover's diffuser, also called inversion about the mean."""

    _validate_qubit_count(qc, n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    apply_multi_controlled_z(qc, n_qubits)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(
    mapping: DatasetMapping | str,
    iterations: int | None = None,
    measure: bool = True,
    allow_no_target: bool = False,
    apply_diffuser_without_oracle: bool = False,
) -> QuantumCircuit:
    """Build a Grover circuit from a DatasetMapping or target bitstring.

    Passing a raw bitstring is retained for compatibility with the initial
    scaffold; production code should pass DatasetMapping.
    """

    if isinstance(mapping, DatasetMapping) and not mapping.target_found:
        if not allow_no_target:
            raise ValueError("Cannot build Grover circuit because no target state exists.")
        qc = create_initial_circuit(mapping.n_qubits)
        apply_superposition(qc, mapping.n_qubits)
        if iterations is None:
            iterations = recommended_iterations(mapping.padded_size)
        if iterations < 0:
            raise ValueError("iterations must be non-negative.")
        if apply_diffuser_without_oracle:
            for _ in range(iterations):
                apply_diffuser(qc, mapping.n_qubits)
        if measure:
            _add_measurements(qc, mapping.n_qubits)
        return qc

    target_binary, n_qubits, search_space_size = _extract_mapping_fields(mapping)
    if iterations is None:
        iterations = recommended_iterations(search_space_size)
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")

    qc = create_initial_circuit(n_qubits)
    apply_superposition(qc, n_qubits)
    for _ in range(iterations):
        apply_phase_oracle(qc, target_binary)
        apply_diffuser(qc, n_qubits)

    if measure:
        _add_measurements(qc, n_qubits)
    return qc


def build_noise_model(noise_config: NoiseConfig | None) -> NoiseModel | None:
    """Build an Aer noise model from GroverLab noise settings."""

    if noise_config is None or not noise_config.noise_enabled:
        return None

    noise_model = NoiseModel()
    gate_probability = _combined_probability(
        noise_config.depolar_prob,
        noise_config.gate_error_prob,
    )

    if gate_probability > 0:
        one_qubit_error = depolarizing_error(gate_probability, 1)
        two_qubit_error = depolarizing_error(gate_probability, 2)
        three_qubit_error = depolarizing_error(gate_probability, 3)
        noise_model.add_all_qubit_quantum_error(one_qubit_error, ["x", "h", "z"])
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
        noise_model.add_all_qubit_quantum_error(three_qubit_error, ["ccx"])

    if noise_config.measurement_error_prob > 0:
        p = noise_config.measurement_error_prob
        readout_error = ReadoutError([[1 - p, p], [p, 1 - p]])
        noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def run_ideal_simulation(
    qc: QuantumCircuit,
    shots: int,
    seed: int | None = None,
) -> dict[str, int]:
    """Run an ideal shot-based simulation and return measurement counts."""

    _validate_shots(shots)
    measured_qc = _ensure_measured(qc)
    simulator = _make_simulator(seed=seed)
    compiled = transpile(measured_qc, simulator, seed_transpiler=seed)
    result = simulator.run(compiled, shots=shots).result()
    return dict(result.get_counts())


def run_noisy_simulation(
    qc: QuantumCircuit,
    shots: int,
    noise_config: NoiseConfig,
    seed: int | None = None,
) -> dict[str, int] | None:
    """Run a noisy shot-based simulation, or return None when noise is disabled."""

    _validate_shots(shots)
    if noise_config.noise_enabled and _has_zero_noise(noise_config):
        return run_ideal_simulation(qc, shots=shots, seed=seed)

    noise_model = build_noise_model(noise_config)
    if noise_model is None:
        return None

    measured_qc = _ensure_measured(qc)
    simulator = _make_simulator(seed=seed, noise_model=noise_model)
    compiled = transpile(measured_qc, simulator, seed_transpiler=seed)
    result = simulator.run(compiled, shots=shots).result()
    return dict(result.get_counts())


def get_gate_counts(qc: QuantumCircuit) -> dict[str, int]:
    """Return circuit operation counts as a standard Python dictionary."""

    return dict(qc.count_ops())


def decode_result(counts: dict[str, int], mapping: DatasetMapping) -> dict[str, Any]:
    """Decode the most likely bitstring from counts back to the dataset item."""

    if not counts:
        raise ValueError("Measurement counts cannot be empty.")

    measured_bitstring = max(counts, key=counts.get)
    decoded_item = decode_bitstring(measured_bitstring, mapping)
    total_shots = sum(counts.values())
    if mapping.target_found:
        success_probability = counts.get(mapping.target_binary, 0) / total_shots if total_shots else 0.0
        found = decoded_item == mapping.target_item
    else:
        success_probability = 0.0
        found = False

    return {
        "measured_bitstring": measured_bitstring,
        "decoded_item": decoded_item,
        "found": found,
        "success_probability": success_probability,
    }


def get_statevector_probabilities(qc: QuantumCircuit) -> dict[str, float]:
    """Return exact basis-state probabilities for a non-noisy circuit."""

    circuit = qc.remove_final_measurements(inplace=False)
    statevector = Statevector.from_instruction(circuit)
    return {
        str(state): float(probability)
        for state, probability in statevector.probabilities_dict().items()
    }


def recommended_iterations(search_space_size: int, marked_states: int = 1) -> int:
    """Return the standard approximate number of Grover iterations."""

    if search_space_size < 1:
        raise ValueError("Search space size must be positive.")
    if marked_states < 1:
        raise ValueError("Marked states must be positive.")
    if marked_states > search_space_size:
        raise ValueError("Marked states cannot exceed the search space size.")
    return max(1, floor((pi / 4) * sqrt(search_space_size / marked_states)))


def _extract_mapping_fields(mapping: DatasetMapping | str) -> tuple[str, int, int]:
    """Extract target bitstring, qubit count, and search-space size."""

    if isinstance(mapping, str):
        if not mapping or any(bit not in {"0", "1"} for bit in mapping):
            raise ValueError("Target bitstring must contain only 0 and 1.")
        return mapping, len(mapping), 2 ** len(mapping)

    if not mapping.target_found:
        raise ValueError("Cannot extract a target bitstring because no target state exists.")
    return mapping.target_binary, mapping.n_qubits, mapping.padded_size


def _validate_qubit_count(qc: QuantumCircuit, n_qubits: int) -> None:
    """Ensure the circuit has enough qubits for an operation."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    if qc.num_qubits < n_qubits:
        raise ValueError("Circuit does not contain enough qubits.")


def _add_measurements(qc: QuantumCircuit, n_qubits: int) -> None:
    """Add one classical bit per search qubit and measure in matching order."""

    if qc.num_clbits < n_qubits:
        qc.add_register(ClassicalRegister(n_qubits - qc.num_clbits, "c"))
    qc.measure(range(n_qubits), range(n_qubits))


def _ensure_measured(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a measured circuit, adding measurements to a copy if needed."""

    if qc.num_clbits > 0 and "measure" in qc.count_ops():
        return qc

    measured_qc = qc.copy()
    _add_measurements(measured_qc, measured_qc.num_qubits)
    return measured_qc


def _make_simulator(
    seed: int | None = None,
    noise_model: NoiseModel | None = None,
) -> AerSimulator:
    """Create an AerSimulator with optional seed and noise model."""

    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["seed_simulator"] = seed
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    return AerSimulator(**kwargs)


def _validate_shots(shots: int) -> None:
    """Validate shot count for Aer execution."""

    if shots <= 0:
        raise ValueError("shots must be positive.")


def _combined_probability(first: float, second: float) -> float:
    """Combine independent error probabilities into one bounded probability."""

    combined = 1 - ((1 - first) * (1 - second))
    return max(0.0, min(1.0, combined))


def _has_zero_noise(noise_config: NoiseConfig) -> bool:
    """Return whether an enabled noisy run has no active error channels."""

    return (
        noise_config.depolar_prob == 0
        and noise_config.gate_error_prob == 0
        and noise_config.measurement_error_prob == 0
    )
