"""Dataset cleaning, indexing, padding, and bitstring mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2


@dataclass(frozen=True)
class IndexedDataset:
    """A cleaned dataset mapped to quantum-searchable integer indices."""

    items: list[str]
    padded_items: list[str | None]
    item_to_index: dict[str, int]
    num_qubits: int


def clean_items(raw_items: list[str]) -> list[str]:
    """Strip whitespace and remove empty values while preserving order."""

    return [item.strip() for item in raw_items if item and item.strip()]


def required_qubits(item_count: int) -> int:
    """Return qubits required to encode all items, including one-item datasets."""

    if item_count < 1:
        raise ValueError("At least one item is required.")
    return max(1, ceil(log2(item_count)))


def pad_to_power_of_two(items: list[str]) -> list[str | None]:
    """Pad items with unused states so the search space has size 2^n."""

    qubits = required_qubits(len(items))
    target_size = 2**qubits
    return [*items, *([None] * (target_size - len(items)))]


def index_dataset(raw_items: list[str]) -> IndexedDataset:
    """Clean and index a dataset for quantum-searchable encoded indices."""

    items = clean_items(raw_items)
    if not items:
        raise ValueError("Dataset must contain at least one non-empty item.")

    item_to_index = {item: index for index, item in enumerate(items)}
    num_qubits = required_qubits(len(items))
    return IndexedDataset(
        items=items,
        padded_items=pad_to_power_of_two(items),
        item_to_index=item_to_index,
        num_qubits=num_qubits,
    )


def index_to_bitstring(index: int, num_qubits: int) -> str:
    """Encode a dataset index as a fixed-width computational-basis bitstring."""

    if index < 0:
        raise ValueError("Index must be non-negative.")
    if index >= 2**num_qubits:
        raise ValueError("Index is outside the representable search space.")
    return format(index, f"0{num_qubits}b")


def bitstring_to_index(bitstring: str) -> int:
    """Decode a computational-basis bitstring into a dataset index."""

    if not bitstring or any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("Bitstring must contain only 0 and 1 characters.")
    return int(bitstring, 2)

