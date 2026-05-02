import io

import pytest

from groverlab.grover_data import (
    calculate_padded_size,
    calculate_required_qubits,
    clean_items,
    create_dataset_mapping,
    decode_bitstring,
    encode_index_to_binary,
    load_csv_items,
    parse_comma_text,
)


def test_parse_comma_text_strips_whitespace_and_preserves_order():
    assert parse_comma_text(" apple, mango , banana, orange ") == [
        "apple",
        "mango",
        "banana",
        "orange",
    ]


def test_clean_items_removes_empty_items():
    assert clean_items([" alpha ", "", " ", "beta", None]) == ["alpha", "beta"]


def test_load_csv_items_uses_first_column_by_default():
    csv_file = io.StringIO("item,category\napple,fruit\nbanana,fruit\n")

    assert load_csv_items(csv_file) == ["apple", "banana"]


def test_create_dataset_mapping_indexes_target_example():
    mapping = create_dataset_mapping(
        ["apple", "mango", "banana", "orange"],
        "banana",
    )

    assert mapping.target_index == 2
    assert mapping.n_qubits == 2
    assert mapping.padded_size == 4
    assert mapping.target_binary == "10"
    assert mapping.unused_states == 0


def test_duplicate_items_add_warning_and_preserve_first_target_index():
    mapping = create_dataset_mapping(["apple", "banana", "apple"], "apple")

    assert mapping.target_index == 0
    assert mapping.warnings
    assert "Duplicate items detected" in mapping.warnings[0]


def test_encode_index_to_binary_uses_fixed_width():
    assert encode_index_to_binary(2, 3) == "010"


def test_calculate_required_qubits():
    assert calculate_required_qubits(1) == 1
    assert calculate_required_qubits(2) == 1
    assert calculate_required_qubits(3) == 2
    assert calculate_required_qubits(8) == 3


def test_calculate_padded_size():
    assert calculate_padded_size(1) == 2
    assert calculate_padded_size(2) == 4
    assert calculate_padded_size(3) == 8


def test_padded_bitstring_decoding_returns_none():
    mapping = create_dataset_mapping(["apple", "banana", "orange"], "banana")

    assert mapping.padded_size == 4
    assert mapping.unused_states == 1
    assert decode_bitstring("11", mapping) is None


def test_decode_bitstring_returns_item_for_used_state():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")

    assert decode_bitstring("10", mapping) == "banana"


def test_missing_target_raises_value_error():
    with pytest.raises(ValueError, match="Target item 'pear' was not found"):
        create_dataset_mapping(["apple", "banana"], "pear")

