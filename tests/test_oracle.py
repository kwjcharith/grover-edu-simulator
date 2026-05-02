import pytest

from groverlab.grover_oracle import validate_target_bitstring


def test_validate_target_bitstring_accepts_valid_state():
    assert validate_target_bitstring("101", 3) == "101"


def test_validate_target_bitstring_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_target_bitstring("10", 3)

