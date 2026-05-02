"""Dataset parsing, cleaning, indexing, padding, and bitstring mapping."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path
from typing import Any

from groverlab.grover_config import DatasetMapping


@dataclass(frozen=True)
class IndexedDataset:
    """Compatibility model for the initial runner scaffold."""

    items: list[str]
    padded_items: list[str | None]
    item_to_index: dict[str, int]
    num_qubits: int


def parse_comma_text(text: str) -> list[str]:
    """Parse comma-separated text into cleaned items."""

    if text is None:
        raise ValueError("Input text must not be None.")
    return clean_items(text.split(","))


def load_csv_items(file_or_path: Any, column_name: str | None = None) -> list[str]:
    """Load cleaned items from a CSV path or file-like object.

    If column_name is omitted, the first CSV column is used. Header rows are
    detected with csv.Sniffer when possible.
    """

    csv_text = _read_csv_text(file_or_path)
    if not csv_text.strip():
        raise ValueError("CSV input is empty.")

    sample = csv_text[:2048]
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = column_name is not None

    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        raise ValueError("CSV input does not contain any rows.")

    if has_header:
        header = [cell.strip() for cell in rows[0]]
        data_rows = rows[1:]
        if not header:
            raise ValueError("CSV header is empty.")
        if column_name is None:
            column_index = 0
        elif column_name in header:
            column_index = header.index(column_name)
        else:
            raise ValueError(f"CSV column '{column_name}' was not found.")
    else:
        data_rows = rows
        if column_name is not None:
            raise ValueError("CSV has no detectable header, so column_name cannot be used.")
        column_index = 0

    values: list[str] = []
    for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if column_index >= len(row):
            raise ValueError(f"CSV row {row_number} does not contain column {column_index}.")
        values.append(row[column_index])

    return clean_items(values)


def clean_items(items: list[str]) -> list[str]:
    """Strip whitespace and remove empty items while preserving order."""

    if items is None:
        raise ValueError("Items must not be None.")

    cleaned: list[str] = []
    for item in items:
        if item is None:
            continue
        stripped = str(item).strip()
        if stripped:
            cleaned.append(stripped)
    return cleaned


def calculate_required_qubits(n_items: int) -> int:
    """Calculate qubits required to encode n_items basis states."""

    if n_items < 1:
        raise ValueError("At least one item is required to calculate qubits.")
    return max(1, ceil(log2(n_items)))


def calculate_padded_size(n_qubits: int) -> int:
    """Calculate the power-of-two padded search-space size."""

    if n_qubits < 0:
        raise ValueError("Number of qubits must be non-negative.")
    return 2**n_qubits


def encode_index_to_binary(index: int, n_qubits: int) -> str:
    """Encode a dataset index as a fixed-width computational-basis bitstring."""

    if n_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")
    if index < 0:
        raise ValueError("Index must be non-negative.")
    if index >= calculate_padded_size(n_qubits):
        raise ValueError("Index is outside the representable search space.")
    return format(index, f"0{n_qubits}b")


def create_dataset_mapping(items: list[str], target_item: str) -> DatasetMapping:
    """Create a complete dataset-to-quantum-index mapping."""

    cleaned_items = clean_items(items)
    target = target_item.strip() if target_item is not None else ""

    if not cleaned_items:
        raise ValueError("Dataset must contain at least one non-empty item.")
    if not target:
        raise ValueError("Target item must not be empty.")
    if target not in cleaned_items:
        raise ValueError(f"Target item '{target}' was not found in the dataset.")

    warnings = _mapping_warnings(cleaned_items)
    n_items = len(cleaned_items)
    n_qubits = calculate_required_qubits(n_items)
    padded_size = calculate_padded_size(n_qubits)
    target_index = cleaned_items.index(target)

    return DatasetMapping(
        original_items=list(items),
        cleaned_items=cleaned_items,
        n_items=n_items,
        padded_size=padded_size,
        n_qubits=n_qubits,
        target_item=target,
        target_index=target_index,
        target_binary=encode_index_to_binary(target_index, n_qubits),
        unused_states=padded_size - n_items,
        warnings=warnings,
    )


def decode_bitstring(bitstring: str, mapping: DatasetMapping) -> str | None:
    """Decode a measured bitstring back to a dataset item or padded unused state."""

    if not bitstring:
        raise ValueError("Bitstring must not be empty.")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("Bitstring must contain only 0 and 1 characters.")
    if len(bitstring) != mapping.n_qubits:
        raise ValueError("Bitstring length must match the mapping qubit count.")

    index = int(bitstring, 2)
    if index >= mapping.padded_size:
        raise ValueError("Bitstring is outside the padded search space.")
    if index >= mapping.n_items:
        return None
    return mapping.cleaned_items[index]


def required_qubits(item_count: int) -> int:
    """Compatibility alias for calculate_required_qubits."""

    return calculate_required_qubits(item_count)


def pad_to_power_of_two(items: list[str]) -> list[str | None]:
    """Pad items with None values so the search space has size 2^n."""

    cleaned = clean_items(items)
    padded_size = calculate_padded_size(calculate_required_qubits(len(cleaned)))
    return [*cleaned, *([None] * (padded_size - len(cleaned)))]


def index_dataset(raw_items: list[str]) -> IndexedDataset:
    """Compatibility helper for cleaning and indexing a dataset."""

    items = clean_items(raw_items)
    if not items:
        raise ValueError("Dataset must contain at least one non-empty item.")

    item_to_index: dict[str, int] = {}
    for index, item in enumerate(items):
        item_to_index.setdefault(item, index)

    num_qubits = calculate_required_qubits(len(items))
    return IndexedDataset(
        items=items,
        padded_items=pad_to_power_of_two(items),
        item_to_index=item_to_index,
        num_qubits=num_qubits,
    )


def index_to_bitstring(index: int, num_qubits: int) -> str:
    """Compatibility alias for encode_index_to_binary."""

    return encode_index_to_binary(index, num_qubits)


def bitstring_to_index(bitstring: str) -> int:
    """Decode a computational-basis bitstring into an integer index."""

    if not bitstring:
        raise ValueError("Bitstring must not be empty.")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("Bitstring must contain only 0 and 1 characters.")
    return int(bitstring, 2)


def _read_csv_text(file_or_path: Any) -> str:
    """Read text from common CSV inputs without importing UI frameworks."""

    if isinstance(file_or_path, (str, Path)):
        return Path(file_or_path).read_text(encoding="utf-8")

    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    content = file_or_path.read()

    if isinstance(content, bytes):
        return content.decode("utf-8-sig")
    if isinstance(content, str):
        return content
    raise TypeError("CSV input must be a path or file-like object returning str or bytes.")


def _mapping_warnings(cleaned_items: list[str]) -> list[str]:
    """Return non-fatal warnings discovered while mapping a dataset."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for item in cleaned_items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)

    warnings: list[str] = []
    if duplicates:
        duplicate_text = ", ".join(duplicates)
        warnings.append(
            "Duplicate items detected; the first matching index is used for "
            f"case-sensitive target matching: {duplicate_text}."
        )
    return warnings

