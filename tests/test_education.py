from groverlab.grover_config import GroverConfig, NoiseConfig
from groverlab.grover_data import create_dataset_mapping
from groverlab.grover_education import (
    LEARNING_OUTCOMES,
    explain_dataset_mapping,
    explain_noise,
    generate_full_explanation,
    generate_quiz_questions,
    misconception_warnings,
)
from groverlab.grover_runner import run_grover_simulation


def test_explain_dataset_mapping_mentions_encoded_indices():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")

    explanation = explain_dataset_mapping(mapping)

    assert "index" in explanation
    assert "|10>" in explanation
    assert "not raw CSV text" in explanation


def test_misconception_warnings_include_required_warnings():
    config = GroverConfig(dataset_items=["apple", "banana", "orange"], target_item="banana")
    mapping = create_dataset_mapping(config.dataset_items, config.target_item)

    warnings = misconception_warnings(mapping, config)

    assert "Grover's algorithm does not directly search raw CSV text." in warnings
    assert "This simulator maps dataset entries to quantum-searchable indices." in warnings
    assert "The oracle is problem-specific and not free." in warnings
    assert "More Grover iterations are not always better." in warnings
    assert "Measurement is probabilistic." in warnings
    assert "Noise can reduce or destroy the quantum advantage." in warnings
    assert "Non-power-of-two datasets require padded unused states." in warnings
    assert any("padded unused state" in warning for warning in warnings)


def test_learning_outcomes_include_required_topics():
    assert "Explain classical-to-quantum index encoding." in LEARNING_OUTCOMES
    assert "Explain equal superposition." in LEARNING_OUTCOMES
    assert "Explain oracle phase marking." in LEARNING_OUTCOMES
    assert "Explain diffuser/amplitude amplification." in LEARNING_OUTCOMES
    assert "Explain probabilistic measurement." in LEARNING_OUTCOMES
    assert "Explain over-rotation." in LEARNING_OUTCOMES
    assert "Explain noise effects." in LEARNING_OUTCOMES


def test_generate_quiz_questions_returns_five_questions_with_answers():
    mapping = create_dataset_mapping(["apple", "mango", "banana", "orange"], "banana")

    quiz = generate_quiz_questions(mapping)

    assert len(quiz) == 5
    assert all("question" in item and "answer" in item for item in quiz)


def test_generate_full_explanation_contains_sections_and_activity():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=64,
        iterations=1,
        seed=42,
    )
    result = run_grover_simulation(config)

    explanation = generate_full_explanation(result)

    assert "dataset_mapping" in explanation["sections"]
    assert "measurement" in explanation["sections"]
    assert len(explanation["quiz_questions"]) == 5
    assert explanation["student_activity"]["steps"]


def test_explain_noise_describes_ideal_and_noisy_modes():
    ideal_config = GroverConfig(dataset_items=["apple", "banana"], target_item="banana")
    noisy_config = GroverConfig(
        dataset_items=["apple", "banana"],
        target_item="banana",
        noise_config=NoiseConfig(
            noise_enabled=True,
            depolar_prob=0.01,
            measurement_error_prob=0.02,
            gate_error_prob=0.03,
        ),
    )

    assert "idealized" in explain_noise(ideal_config.noise_config)
    assert "Noise is enabled" in explain_noise(noisy_config.noise_config)
