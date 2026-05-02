"""Oracle circuit logic for GroverLab.

The oracle is problem-specific: in this simulator it marks an encoded dataset
index, not raw CSV text.
"""

from __future__ import annotations

from qiskit import QuantumCircuit


def validate_target_bitstring(bitstring: str, num_qubits: int) -> str:
    """Validate a target state before circuit construction."""

    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")
    if len(bitstring) != num_qubits:
        raise ValueError("Target bitstring length must match num_qubits.")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("Target bitstring must contain only 0 and 1.")
    return bitstring


def apply_x_gates_for_zero_bits(qc: QuantumCircuit, target_binary: str) -> None:
    """Map zero bits in the target state to one bits before phase marking.

    Qiskit count strings are displayed with the highest-index qubit on the left,
    so the leftmost target bit maps to the highest-index qubit.
    """

    _validate_circuit_matches_target(qc, target_binary)
    n_qubits = len(target_binary)
    for bit_position, bit_value in enumerate(target_binary):
        if bit_value == "0":
            qc.x(n_qubits - bit_position - 1)


def apply_multi_controlled_z(qc: QuantumCircuit, n_qubits: int) -> None:
    """Apply a phase flip to the all-ones computational basis state."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    if qc.num_qubits < n_qubits:
        raise ValueError("Circuit does not contain enough qubits.")

    if n_qubits == 1:
        qc.z(0)
        return

    target_qubit = n_qubits - 1
    control_qubits = list(range(n_qubits - 1))
    qc.h(target_qubit)
    qc.mcx(control_qubits, target_qubit)
    qc.h(target_qubit)


def uncompute_x_gates(qc: QuantumCircuit, target_binary: str) -> None:
    """Undo zero-bit X mappings after phase marking."""

    apply_x_gates_for_zero_bits(qc, target_binary)


def apply_phase_oracle(qc: QuantumCircuit, target_binary: str) -> QuantumCircuit:
    """Flip the phase of the marked state only and return the circuit."""

    _validate_circuit_matches_target(qc, target_binary)
    n_qubits = len(target_binary)
    apply_x_gates_for_zero_bits(qc, target_binary)
    apply_multi_controlled_z(qc, n_qubits)
    uncompute_x_gates(qc, target_binary)
    return qc


def build_phase_oracle(target_bitstring: str, num_qubits: int) -> QuantumCircuit:
    """Build a standalone Qiskit phase-oracle circuit for the target state."""

    validate_target_bitstring(target_bitstring, num_qubits)
    qc = QuantumCircuit(num_qubits, name="phase_oracle")
    apply_phase_oracle(qc, target_bitstring)
    return qc


def _validate_circuit_matches_target(qc: QuantumCircuit, target_binary: str) -> None:
    """Validate a circuit and target bitstring before mutating the circuit."""

    validate_target_bitstring(target_binary, len(target_binary))
    if qc.num_qubits < len(target_binary):
        raise ValueError("Circuit does not contain enough qubits for target_binary.")

