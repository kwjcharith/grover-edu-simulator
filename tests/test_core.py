from qiskit import QuantumCircuit
import pytest

from groverlab.grover_config import NoiseConfig
from groverlab.grover_core import (
    apply_diffuser,
    build_grover_circuit,
    decoherence_probabilities,
    decode_result,
    derive_pure_dephasing_time_us,
    get_gate_counts,
    get_statevector_probabilities,
    run_ideal_simulation,
    run_noisy_simulation,
)
from groverlab.grover_data import create_dataset_mapping


def test_apply_diffuser_has_expected_two_qubit_operations():
    qc = QuantumCircuit(2)

    apply_diffuser(qc, 2)

    assert qc.count_ops() == {"h": 6, "x": 4, "cx": 1}


def test_build_grover_circuit_adds_measurements_by_default():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")

    qc = build_grover_circuit(mapping, iterations=1)

    assert qc.num_qubits == 2
    assert qc.num_clbits == 2
    assert get_gate_counts(qc)["measure"] == 2


def test_ideal_simulation_finds_target_for_four_item_example():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")
    qc = build_grover_circuit(mapping, iterations=1)

    counts = run_ideal_simulation(qc, shots=128, seed=42)

    assert counts == {"10": 128}


def test_noisy_simulation_returns_none_when_noise_disabled():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")
    qc = build_grover_circuit(mapping, iterations=1)

    counts = run_noisy_simulation(qc, shots=128, noise_config=NoiseConfig(), seed=42)

    assert counts is None


def test_noisy_simulation_returns_counts_when_noise_enabled():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")
    qc = build_grover_circuit(mapping, iterations=1)

    counts = run_noisy_simulation(
        qc,
        shots=128,
        noise_config=NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.01,
            measurement_error_prob=0.01,
        ),
        seed=42,
    )

    assert counts is not None
    assert sum(counts.values()) == 128


def test_zero_noise_enabled_matches_ideal_simulation():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")
    qc = build_grover_circuit(mapping, iterations=1)

    ideal_counts = run_ideal_simulation(qc, shots=128, seed=42)
    noisy_counts = run_noisy_simulation(
        qc,
        shots=128,
        noise_config=NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.0,
            measurement_error_prob=0.0,
            gate_error_prob=0.0,
            t1_relaxation_us=500.0,
            t2_coherence_us=300.0,
        ),
        seed=42,
    )

    assert noisy_counts == ideal_counts


def test_decoherence_probabilities_derive_pure_dephasing_without_double_counting():
    config = NoiseConfig(
        noise_enabled=True,
        t1_relaxation_us=120.0,
        t2_coherence_us=80.0,
    )

    probabilities = decoherence_probabilities(config, gate_duration_us=0.05)

    assert derive_pure_dephasing_time_us(120.0, 80.0) == pytest.approx(120.0)
    assert probabilities["amplitude_damping_probability"] > 0
    assert probabilities["pure_dephasing_probability"] > 0


def test_noise_config_rejects_t2_above_physical_limit():
    with pytest.raises(ValueError, match="t2_coherence_us"):
        NoiseConfig(t1_relaxation_us=20.0, t2_coherence_us=50.0)


def test_decode_result_reports_target_success_probability():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")

    decoded = decode_result({"00": 60, "10": 40}, mapping)

    assert decoded["measured_bitstring"] == "00"
    assert decoded["decoded_item"] == "apple"
    assert decoded["found"] is False
    assert decoded["success_probability"] == 0.4


def test_statevector_probabilities_are_plain_dict_values():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")
    qc = build_grover_circuit(mapping, iterations=1, measure=False)

    probabilities = get_statevector_probabilities(qc)

    assert probabilities["10"] > 0.999
    assert all(isinstance(key, str) for key in probabilities)
    assert all(isinstance(value, float) for value in probabilities.values())


def test_build_grover_circuit_rejects_missing_target_by_default():
    mapping = create_dataset_mapping(["apple", "banana", "pineapple"], "abc")

    with pytest.raises(ValueError, match="no target state exists"):
        build_grover_circuit(mapping)


def test_build_grover_circuit_allows_no_target_demo():
    mapping = create_dataset_mapping(
        ["apple", "banana", "pineapple"],
        "abc",
        missing_target_mode="experimental",
    )

    qc = build_grover_circuit(mapping, allow_no_target=True)

    assert qc.count_ops() == {"h": 2, "measure": 2}
