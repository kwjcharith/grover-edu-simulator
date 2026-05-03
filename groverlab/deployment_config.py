"""Deployment safeguards for GroverLab public online demos."""

from __future__ import annotations

from typing import Any


PUBLIC_DEMO_MODE = True

MAX_PUBLIC_DATASET_SIZE = 1024
MAX_PUBLIC_QUBITS = 10
MAX_PUBLIC_SHOTS = 2048
MAX_PUBLIC_ITERATION_SWEEP_POINTS = 20
MAX_PUBLIC_NOISE_SWEEP_POINTS = 8

ALLOW_NOISY_SIMULATION_ONLINE = True
ALLOW_LARGE_DATASET_UPLOAD_ONLINE = False
ALLOW_FASTAPI_ON_STREAMLIT_CLOUD = False


def is_public_demo_mode() -> bool:
    """Return whether public demo safeguards are active."""

    return PUBLIC_DEMO_MODE


def validate_online_limits(
    dataset_size: int,
    n_qubits: int,
    shots: int,
    sweep_points: int | None = None,
) -> dict[str, Any]:
    """Validate or cap simulation settings for public Streamlit deployment."""

    warnings: list[str] = []
    adjusted_shots = shots
    adjusted_sweep_points = sweep_points

    if not PUBLIC_DEMO_MODE:
        return {
            "shots": adjusted_shots,
            "sweep_points": adjusted_sweep_points,
            "warnings": warnings,
        }

    if dataset_size > MAX_PUBLIC_DATASET_SIZE:
        raise ValueError(
            "This public online demo supports up to 1024 dataset items. Larger datasets should be tested locally."
        )

    if n_qubits > MAX_PUBLIC_QUBITS:
        raise ValueError(
            "This public online demo supports up to 10 qubits. Larger Grover simulations are computationally expensive on free cloud resources."
        )

    if shots > MAX_PUBLIC_SHOTS:
        adjusted_shots = MAX_PUBLIC_SHOTS
        warnings.append(
            f"Shots were capped at {MAX_PUBLIC_SHOTS} for the public online demo."
        )

    if sweep_points is not None and sweep_points > MAX_PUBLIC_ITERATION_SWEEP_POINTS:
        adjusted_sweep_points = MAX_PUBLIC_ITERATION_SWEEP_POINTS
        warnings.append(
            f"Sweep points were capped at {MAX_PUBLIC_ITERATION_SWEEP_POINTS} for the public online demo."
        )

    return {
        "shots": adjusted_shots,
        "sweep_points": adjusted_sweep_points,
        "warnings": warnings,
    }
