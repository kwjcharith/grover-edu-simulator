"""Educational explanations and misconception warnings for GroverLab."""

from __future__ import annotations

from typing import Any

from groverlab.grover_config import DatasetMapping, GroverConfig, GroverResult, NoiseConfig


CORE_WARNING = (
    "Grover's algorithm does not directly search raw CSV text. "
    "GroverLab maps dataset entries to quantum-searchable indices and searches "
    "those encoded indices."
)

MISCONCEPTION_WARNINGS = [
    "Grover's algorithm does not directly search raw CSV text.",
    "This simulator maps dataset entries to quantum-searchable indices.",
    "The oracle is problem-specific and not free.",
    "More Grover iterations are not always better.",
    "Measurement is probabilistic.",
    "Noise can reduce or destroy the quantum advantage.",
    "Non-power-of-two datasets require padded unused states.",
]

LEARNING_OUTCOMES = [
    "Explain classical-to-quantum index encoding.",
    "Explain equal superposition.",
    "Explain oracle phase marking.",
    "Explain diffuser/amplitude amplification.",
    "Explain probabilistic measurement.",
    "Explain over-rotation.",
    "Explain noise effects.",
]


def explain_dataset_mapping(mapping: DatasetMapping) -> str:
    """Explain how classical items are mapped to quantum-searchable indices."""

    return (
        f"The original dataset is cleaned into {mapping.n_items} item(s). "
        f"Each item is assigned a classical index, and that index is written as a "
        f"{mapping.n_qubits}-bit binary string so a quantum circuit can search it. "
        f"The target item '{mapping.target_item}' is at index {mapping.target_index}, "
        f"so GroverLab searches for the encoded state |{mapping.target_binary}>. "
        f"This is why GroverLab searches an index encoding, not raw CSV text."
    )


def explain_superposition(mapping: DatasetMapping) -> str:
    """Explain the equal superposition step."""

    return (
        f"Hadamard gates place the {mapping.n_qubits} qubits into an equal "
        f"superposition over {mapping.padded_size} possible basis states. "
        "Before the oracle is applied, each encoded index has the same amplitude, "
        "so the circuit represents all searchable indices at once."
    )


def explain_oracle(mapping: DatasetMapping) -> str:
    """Explain oracle phase marking for the target state."""

    return (
        f"The oracle recognizes the encoded target state |{mapping.target_binary}> "
        "and flips only that state's phase. The marked state is not measured yet; "
        "its sign is changed so the diffuser can amplify it. The oracle is "
        "problem-specific, which means it must be designed for the item or condition "
        "being searched and should not be treated as free."
    )


def explain_diffuser() -> str:
    """Explain Grover's diffuser in beginner-friendly language."""

    return (
        "The diffuser performs amplitude amplification, often described as "
        "inversion about the mean. After the oracle marks the target by phase, "
        "the diffuser increases the target state's amplitude and decreases many "
        "non-target amplitudes, making the target more likely to appear when measured."
    )


def explain_iterations(config: GroverConfig, mapping: DatasetMapping) -> str:
    """Explain why the number of Grover iterations matters."""

    if config.iterations is None:
        iteration_text = "GroverLab chooses a recommended iteration count automatically"
    else:
        iteration_text = f"This run uses {config.iterations} Grover iteration(s)"

    return (
        f"{iteration_text} for a padded search space of {mapping.padded_size} state(s). "
        "Grover iterations improve the target probability only up to a point. "
        "Too many iterations can rotate amplitude away from the target again, "
        "a behavior called over-rotation."
    )


def explain_measurement(result: GroverResult) -> str:
    """Explain how measurement counts are interpreted."""

    return (
        f"The simulator measured the bitstring {result.measured_bitstring} most often, "
        f"which decodes to {result.decoded_item!r}. The target success probability "
        f"was {result.success_probability:.3f}. Measurement is probabilistic, so a "
        "single shot is not a proof; the count distribution over many shots is the "
        "evidence used to interpret the result."
    )


def explain_noise(noise_config: NoiseConfig) -> str:
    """Explain noisy simulation settings and their educational meaning."""

    if not noise_config.noise_enabled:
        return (
            "Noise is disabled, so the simulation represents an idealized quantum "
            "computer. This is useful for learning the algorithm before studying "
            "hardware imperfections."
        )

    return (
        "Noise is enabled. Depolarizing noise can randomize quantum states, gate "
        "error can disturb circuit operations, and measurement error can flip the "
        "reported classical bit. These effects can reduce or destroy the quantum "
        f"advantage. Current settings: depolarizing={noise_config.depolar_prob:.3f}, "
        f"gate={noise_config.gate_error_prob:.3f}, measurement="
        f"{noise_config.measurement_error_prob:.3f}."
    )


def misconception_warnings(mapping: DatasetMapping, config: GroverConfig) -> list[str]:
    """Return warnings tailored to the current mapping and configuration."""

    warnings = [*MISCONCEPTION_WARNINGS, *mapping.warnings]
    if mapping.unused_states > 0:
        warnings.append(
            f"This dataset uses {mapping.unused_states} padded unused state(s), "
            "so some measured bitstrings may decode to no item."
        )
    if config.noise_config.noise_enabled:
        warnings.append("This run includes noise, so ideal and noisy results may differ.")
    if config.iterations is not None and config.iterations == 0:
        warnings.append("With zero iterations, the circuit samples the initial superposition.")
    return _deduplicate(warnings)


def generate_quiz_questions(mapping: DatasetMapping) -> list[dict[str, str]]:
    """Create five short quiz questions with answers."""

    return [
        {
            "question": "Does Grover's algorithm search raw CSV text directly?",
            "answer": "No. GroverLab maps dataset entries to encoded indices first.",
        },
        {
            "question": f"What binary state represents the target item '{mapping.target_item}'?",
            "answer": f"|{mapping.target_binary}>.",
        },
        {
            "question": "What does the oracle do to the marked state?",
            "answer": "It flips the phase of the marked state without measuring it.",
        },
        {
            "question": "Why can too many Grover iterations be a problem?",
            "answer": "They can cause over-rotation and reduce the target probability.",
        },
        {
            "question": "What can happen when noise is added to the simulation?",
            "answer": "Noise can lower success probability or destroy the advantage.",
        },
    ]


def generate_student_activity(result: GroverResult) -> dict[str, Any]:
    """Generate a concise classroom or lab activity from a simulation result."""

    return {
        "title": "Trace One Grover Search Run",
        "goal": "Connect a classical dataset item to its encoded quantum search state.",
        "steps": [
            "Write down the cleaned dataset and each item's index.",
            f"Find the target item '{result.mapping.target_item}' and verify its binary code.",
            "Sketch how Hadamard gates create equal superposition over all indices.",
            "Explain which state the oracle marks and why the oracle is problem-specific.",
            "Compare the ideal counts with the decoded result.",
            "Discuss how the result might change if noise or extra iterations are added.",
        ],
        "reflection_prompt": (
            "Where does the speedup idea appear, and what assumptions does the simulator "
            "make to show it?"
        ),
    }


def generate_full_explanation(result: GroverResult) -> dict[str, Any]:
    """Generate a full educational explanation bundle for a result."""

    return {
        "learning_outcomes": LEARNING_OUTCOMES,
        "warnings": misconception_warnings(result.mapping, result.config),
        "sections": {
            "dataset_mapping": explain_dataset_mapping(result.mapping),
            "superposition": explain_superposition(result.mapping),
            "oracle": explain_oracle(result.mapping),
            "diffuser": explain_diffuser(),
            "iterations": explain_iterations(result.config, result.mapping),
            "measurement": explain_measurement(result),
            "noise": explain_noise(result.config.noise_config),
        },
        "quiz_questions": generate_quiz_questions(result.mapping),
        "student_activity": generate_student_activity(result),
    }


def educational_overview() -> dict[str, list[str] | str]:
    """Return educational copy shared by UI, API, reports, and docs."""

    return {
        "core_warning": CORE_WARNING,
        "misconception_warnings": MISCONCEPTION_WARNINGS,
        "learning_outcomes": LEARNING_OUTCOMES,
    }


def _deduplicate(items: list[str]) -> list[str]:
    """Preserve item order while removing duplicate strings."""

    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items

