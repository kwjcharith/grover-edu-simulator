"""Shared configuration and result models for GroverLab.

This module must stay independent from Qiskit, Streamlit, and FastAPI so the
same dataclasses can be reused by the quantum engine, UI, API, tests, exports,
and research logging.
"""

from dataclasses import dataclass, field
from typing import Any


DEFAULT_SHOTS = 1024
DEFAULT_RANDOM_SEED = 42
MAX_ITEMS_FOR_LOCAL_DEMO = 1024


@dataclass(frozen=True)
class NoiseConfig:
    """Noise settings for optional noisy Grover simulation."""

    noise_enabled: bool = False
    depolar_prob: float = 0.0
    measurement_error_prob: float = 0.0
    gate_error_prob: float = 0.0

    def __post_init__(self) -> None:
        """Validate probability fields after initialization."""

        _validate_probability("depolar_prob", self.depolar_prob)
        _validate_probability("measurement_error_prob", self.measurement_error_prob)
        _validate_probability("gate_error_prob", self.gate_error_prob)


@dataclass(frozen=True)
class GroverConfig:
    """User-supplied configuration for one GroverLab run."""

    dataset_items: list[str]
    target_item: str
    shots: int = DEFAULT_SHOTS
    iterations: int | None = None
    seed: int | None = None
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    education_mode: bool = True
    allow_padding: bool = True
    missing_target_mode: str = "stop"

    def __post_init__(self) -> None:
        """Validate user-facing simulation configuration."""

        if not self.dataset_items:
            raise ValueError("dataset_items must not be empty.")
        if not self.target_item or not self.target_item.strip():
            raise ValueError("target_item must not be empty.")
        if self.shots <= 0:
            raise ValueError("shots must be positive.")
        if self.iterations is not None and self.iterations < 0:
            raise ValueError("iterations must be non-negative when provided.")
        if self.missing_target_mode not in {"stop", "experimental"}:
            raise ValueError("missing_target_mode must be 'stop' or 'experimental'.")


@dataclass(frozen=True)
class DatasetMapping:
    """Cleaned dataset mapped onto a quantum-searchable index space."""

    original_items: list[str]
    cleaned_items: list[str]
    n_items: int
    padded_size: int
    n_qubits: int
    target_item: str
    target_found: bool
    target_index: int
    target_binary: str
    unused_states: int
    warnings: list[str]
    missing_target_explanation: str | None


@dataclass(frozen=True)
class GroverResult:
    """Complete result object for simulation, analysis, education, and export."""

    config: GroverConfig
    mapping: DatasetMapping
    circuit_depth: int
    gate_counts: dict[str, Any]
    ideal_counts: dict[str, int]
    noisy_counts: dict[str, int] | None
    success_probability: float
    noisy_success_probability: float | None
    measured_bitstring: str
    decoded_item: str | None
    found: bool
    target_found: bool
    stopped_before_quantum_execution: bool
    missing_target_explanation: str | None
    runtime_seconds: float
    explanation_steps: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class ExperimentLog:
    """Anonymous research log record for educational evaluation."""

    session_id: str
    timestamp: str
    dataset_size: int
    target_index: int
    n_qubits: int
    padded_size: int
    unused_states: int
    iterations: int
    shots: int
    noise_enabled: bool
    depolar_prob: float
    measurement_error_prob: float
    success_probability: float
    decoded_correctly: bool
    runtime_seconds: float
    circuit_depth: int
    gate_counts: dict[str, Any]


@dataclass(frozen=True)
class SimulationConfig:
    """Compatibility config used by the initial skeleton runner.

    GroverConfig is the primary run configuration. This smaller dataclass remains
    temporarily so existing scaffold code and tests keep working while the
    runner is migrated.
    """

    shots: int = DEFAULT_SHOTS
    seed: int = DEFAULT_RANDOM_SEED
    noisy: bool = False
    manual_iterations: int | None = None

    def __post_init__(self) -> None:
        """Validate compatibility simulation settings."""

        if self.shots <= 0:
            raise ValueError("shots must be positive.")
        if self.manual_iterations is not None and self.manual_iterations < 0:
            raise ValueError("manual_iterations must be non-negative when provided.")


def _validate_probability(field_name: str, value: float) -> None:
    """Validate that a noise probability is in the inclusive range [0, 1]."""

    if value < 0 or value > 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")
