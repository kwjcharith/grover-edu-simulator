"""Oracle construction contracts for GroverLab.

The oracle is problem-specific: in this simulator it marks an encoded dataset
index, not raw CSV text.
"""


def validate_target_bitstring(bitstring: str, num_qubits: int) -> str:
    """Validate a target state before circuit construction."""

    if len(bitstring) != num_qubits:
        raise ValueError("Target bitstring length must match num_qubits.")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("Target bitstring must contain only 0 and 1.")
    return bitstring


def build_phase_oracle(target_bitstring: str, num_qubits: int):
    """Build a Qiskit phase oracle for the target state.

    Full Qiskit circuit construction will be implemented in the next step.
    """

    validate_target_bitstring(target_bitstring, num_qubits)
    raise NotImplementedError("Qiskit oracle construction is planned next.")

