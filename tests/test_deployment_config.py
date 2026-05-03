import pytest

from deployment_config import (
    MAX_PUBLIC_DATASET_SIZE,
    MAX_PUBLIC_QUBITS,
    MAX_PUBLIC_SHOTS,
    is_public_demo_mode,
    validate_online_limits,
)


def test_public_demo_mode_enabled_by_default():
    assert is_public_demo_mode() is True


def test_validate_online_limits_rejects_large_dataset():
    with pytest.raises(ValueError, match="supports up to 1024 dataset items"):
        validate_online_limits(
            dataset_size=MAX_PUBLIC_DATASET_SIZE + 1,
            n_qubits=MAX_PUBLIC_QUBITS,
            shots=1024,
        )


def test_validate_online_limits_rejects_large_qubit_count():
    with pytest.raises(ValueError, match="supports up to 10 qubits"):
        validate_online_limits(
            dataset_size=16,
            n_qubits=MAX_PUBLIC_QUBITS + 1,
            shots=1024,
        )


def test_validate_online_limits_caps_public_shots():
    result = validate_online_limits(
        dataset_size=16,
        n_qubits=4,
        shots=MAX_PUBLIC_SHOTS + 1024,
    )

    assert result["shots"] == MAX_PUBLIC_SHOTS
    assert result["warnings"]
