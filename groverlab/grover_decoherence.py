"""Qiskit-free decoherence helper functions for GroverLab."""

from __future__ import annotations

from math import exp, inf, isinf

from groverlab.grover_config import NoiseConfig


SINGLE_QUBIT_GATE_DURATION_US = 0.05
MULTI_QUBIT_GATE_DURATION_US = 0.30
IDEAL_T1_RELAXATION_US = 500.0
IDEAL_T2_COHERENCE_US = 300.0


def derive_pure_dephasing_time_us(t1_relaxation_us: float, t2_coherence_us: float) -> float:
    """Derive pure dephasing time Tphi from T1 and T2 without double-counting T1."""

    denominator = (1 / t2_coherence_us) - (1 / (2 * t1_relaxation_us))
    if denominator <= 0:
        return inf
    return 1 / denominator


def decoherence_probabilities(
    noise_config: NoiseConfig,
    gate_duration_us: float = SINGLE_QUBIT_GATE_DURATION_US,
) -> dict[str, float]:
    """Convert physical coherence times into educational channel probabilities."""

    if gate_duration_us < 0:
        raise ValueError("gate_duration_us must be non-negative.")

    p_t1 = 1 - exp(-gate_duration_us / noise_config.t1_relaxation_us)
    t_phi = derive_pure_dephasing_time_us(
        noise_config.t1_relaxation_us,
        noise_config.t2_coherence_us,
    )
    p_phi = 0.0 if isinf(t_phi) else 1 - exp(-gate_duration_us / t_phi)
    return {
        "t_phi_us": t_phi,
        "amplitude_damping_probability": p_t1,
        "pure_dephasing_probability": p_phi,
    }
