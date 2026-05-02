"""Simulation orchestration independent from Streamlit or FastAPI."""

from __future__ import annotations

from dataclasses import dataclass

from groverlab.grover_config import SimulationConfig
from groverlab.grover_core import recommended_iterations
from groverlab.grover_data import index_dataset, index_to_bitstring


@dataclass(frozen=True)
class GroverRunRequest:
    """Validated request for one Grover search run."""

    items: list[str]
    target_item: str
    config: SimulationConfig = SimulationConfig()


@dataclass(frozen=True)
class PreparedGroverRun:
    """Prepared data needed before circuit construction and simulation."""

    target_item: str
    target_index: int
    target_bitstring: str
    num_qubits: int
    search_space_size: int
    iterations: int
    padded_items: list[str | None]


def prepare_run(request: GroverRunRequest) -> PreparedGroverRun:
    """Clean data, validate the target, map it to a bitstring, and choose iterations."""

    dataset = index_dataset(request.items)
    target = request.target_item.strip()
    if target not in dataset.item_to_index:
        raise ValueError("Target item was not found in the cleaned dataset.")

    target_index = dataset.item_to_index[target]
    search_space_size = len(dataset.padded_items)
    iterations = request.config.manual_iterations or recommended_iterations(search_space_size)

    return PreparedGroverRun(
        target_item=target,
        target_index=target_index,
        target_bitstring=index_to_bitstring(target_index, dataset.num_qubits),
        num_qubits=dataset.num_qubits,
        search_space_size=search_space_size,
        iterations=iterations,
        padded_items=dataset.padded_items,
    )


def run_simulation(request: GroverRunRequest):
    """Run ideal or noisy Grover simulation after preparation.

    Qiskit Aer execution will be added after the project skeleton is in place.
    """

    prepared = prepare_run(request)
    raise NotImplementedError(f"Simulation engine pending for target {prepared.target_bitstring}.")

