import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from groverlab.grover_oracle import (
    apply_multi_controlled_z,
    apply_phase_oracle,
    apply_x_gates_for_zero_bits,
    build_phase_oracle,
    uncompute_x_gates,
    validate_target_bitstring,
)


def test_validate_target_bitstring_accepts_valid_state():
    assert validate_target_bitstring("101", 3) == "101"


def test_validate_target_bitstring_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_target_bitstring("10", 3)


def test_apply_x_gates_for_zero_bits_maps_zero_bits():
    qc = QuantumCircuit(3)

    apply_x_gates_for_zero_bits(qc, "101")

    assert qc.count_ops() == {"x": 1}


def test_uncompute_x_gates_reuses_zero_bit_mapping():
    qc = QuantumCircuit(3)

    apply_x_gates_for_zero_bits(qc, "101")
    uncompute_x_gates(qc, "101")

    assert qc.count_ops() == {"x": 2}


def test_oracle_builds_for_1_qubit():
    qc = QuantumCircuit(1)

    apply_phase_oracle(qc, "0")

    assert qc.num_qubits == 1
    assert qc.count_ops() == {"x": 2, "z": 1}


def test_oracle_builds_for_2_qubits():
    qc = QuantumCircuit(2)

    apply_phase_oracle(qc, "10")

    assert qc.num_qubits == 2
    assert qc.count_ops() == {"x": 2, "h": 2, "cx": 1}


def test_oracle_builds_for_3_qubits():
    qc = QuantumCircuit(3)

    apply_phase_oracle(qc, "101")

    assert qc.num_qubits == 3
    assert qc.count_ops() == {"x": 2, "h": 2, "ccx": 1}


def test_multi_controlled_z_has_expected_operations_for_one_qubit():
    qc = QuantumCircuit(1)

    apply_multi_controlled_z(qc, 1)

    assert qc.count_ops() == {"z": 1}


def test_multi_controlled_z_has_expected_operations_for_multiple_qubits():
    qc = QuantumCircuit(3)

    apply_multi_controlled_z(qc, 3)

    assert qc.count_ops() == {"h": 2, "ccx": 1}


def test_build_phase_oracle_returns_standalone_circuit():
    oracle = build_phase_oracle("10", 2)

    assert oracle.num_qubits == 2
    assert oracle.name == "phase_oracle"
    assert oracle.count_ops() == {"x": 2, "h": 2, "cx": 1}


def test_oracle_flips_phase_of_marked_state_only():
    oracle = build_phase_oracle("10", 2)

    for basis_state in ["00", "01", "10", "11"]:
        initial = Statevector.from_label(basis_state)
        evolved = initial.evolve(oracle)

        if basis_state == "10":
            assert evolved.equiv(-initial)
        else:
            assert evolved.equiv(initial)

