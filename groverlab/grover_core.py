"""Quantum circuit construction for ideal and noisy Grover simulation."""


def recommended_iterations(search_space_size: int, marked_states: int = 1) -> int:
    """Return the standard approximate number of Grover iterations."""

    if search_space_size < 1:
        raise ValueError("Search space size must be positive.")
    if marked_states < 1:
        raise ValueError("Marked states must be positive.")
    if marked_states > search_space_size:
        raise ValueError("Marked states cannot exceed the search space size.")

    from math import floor, pi, sqrt

    return max(1, floor((pi / 4) * sqrt(search_space_size / marked_states)))


def build_grover_circuit(target_bitstring: str, iterations: int):
    """Build the complete Grover circuit.

    This will apply Hadamards, the phase oracle, diffuser steps, and measurement.
    """

    if iterations < 1:
        raise ValueError("At least one Grover iteration is required.")
    raise NotImplementedError("Qiskit circuit construction is planned next.")

