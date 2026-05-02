from groverlab.grover_config import GroverConfig, NoiseConfig, SimulationConfig
from groverlab.grover_runner import GroverRunRequest, prepare_run, run_grover_simulation


def test_prepare_run_maps_target_to_bitstring():
    request = GroverRunRequest(
        items=["red", "green", "blue", "yellow"],
        target_item="blue",
        config=SimulationConfig(manual_iterations=2),
    )

    prepared = prepare_run(request)

    assert prepared.target_index == 2
    assert prepared.target_bitstring == "10"
    assert prepared.num_qubits == 2
    assert prepared.iterations == 2


def test_ideal_grover_finds_target_for_four_items():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=128,
        iterations=1,
        seed=42,
    )

    result = run_grover_simulation(config)

    assert result.mapping.target_index == 2
    assert result.mapping.target_binary == "10"
    assert result.decoded_item == "banana"
    assert result.found is True
    assert result.success_probability > 0.95


def test_result_contains_mapping_counts_and_success_probability():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=64,
        iterations=1,
        seed=7,
    )

    result = run_grover_simulation(config)

    assert result.mapping.cleaned_items == ["apple", "mango", "banana", "orange"]
    assert result.ideal_counts
    assert result.ideal_counts == {"10": 64}
    assert result.success_probability == 1.0
    assert result.circuit_depth > 0
    assert result.gate_counts["measure"] == 2


def test_result_contains_educational_explanations_and_warnings():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana"],
        target_item="banana",
        shots=64,
        iterations=1,
        seed=7,
    )

    result = run_grover_simulation(config)

    assert result.explanation_steps
    assert result.warnings
    assert result.mapping.unused_states == 1


def test_noisy_simulation_runs_when_enabled():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=128,
        iterations=1,
        seed=42,
        noise_config=NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.01,
            measurement_error_prob=0.01,
        ),
    )

    result = run_grover_simulation(config)

    assert result.noisy_counts is not None
    assert sum(result.noisy_counts.values()) == 128
    assert result.noisy_success_probability is not None


def test_missing_target_stop_mode_does_not_build_quantum_circuit():
    config = GroverConfig(
        dataset_items=["apple", "banana", "pineapple"],
        target_item="abc",
        shots=128,
        missing_target_mode="stop",
    )

    result = run_grover_simulation(config)

    assert result.target_found is False
    assert result.stopped_before_quantum_execution is True
    assert result.circuit_depth == 0
    assert result.gate_counts == {}
    assert result.missing_target_explanation


def test_missing_target_stop_mode_returns_empty_counts():
    config = GroverConfig(
        dataset_items=["apple", "banana", "pineapple"],
        target_item="abc",
        shots=128,
        missing_target_mode="stop",
    )

    result = run_grover_simulation(config)

    assert result.ideal_counts == {}
    assert result.noisy_counts is None
    assert result.success_probability == 0.0
    assert result.found is False


def test_missing_target_experimental_mode_runs_uniform_demo():
    config = GroverConfig(
        dataset_items=["apple", "banana", "pineapple"],
        target_item="abc",
        shots=400,
        seed=42,
        missing_target_mode="experimental",
    )

    result = run_grover_simulation(config)

    assert result.target_found is False
    assert result.stopped_before_quantum_execution is False
    assert result.ideal_counts
    assert sum(result.ideal_counts.values()) == 400
    assert result.success_probability == 0.0
    assert result.gate_counts["h"] == 2
    assert result.gate_counts["measure"] == 2
    assert max(result.ideal_counts.values()) < 160


def test_missing_target_experimental_mode_found_false():
    config = GroverConfig(
        dataset_items=["apple", "banana", "pineapple"],
        target_item="abc",
        shots=128,
        seed=7,
        missing_target_mode="experimental",
    )

    result = run_grover_simulation(config)

    assert result.found is False
    assert result.decoded_item in {"apple", "banana", "pineapple", None}
    assert "experimental no-solution demonstration" in " ".join(result.warnings)
