"""Shared configuration for GroverLab.

This module should remain free of UI concerns so it can be reused by the
Streamlit app, a future FastAPI backend, tests, and batch experiments.
"""

from dataclasses import dataclass


DEFAULT_SHOTS = 1024
DEFAULT_RANDOM_SEED = 42
MAX_ITEMS_FOR_LOCAL_DEMO = 1024


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for one Grover simulation run."""

    shots: int = DEFAULT_SHOTS
    seed: int = DEFAULT_RANDOM_SEED
    noisy: bool = False
    manual_iterations: int | None = None

