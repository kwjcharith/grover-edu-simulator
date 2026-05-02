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
    "Grover’s Algorithm does not automatically know whether a text item exists in a classical dataset. The target must correspond to a valid encoded state or to a condition that the oracle can evaluate.",
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

    if not mapping.target_found:
        return (
            f"The original dataset is cleaned into {mapping.n_items} item(s), but "
            f"the requested target item '{mapping.target_item}' is not present. "
            "GroverLab can still show the encoded index space, but there is no "
            "target index or target binary state for the oracle to mark."
        )

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

    if not mapping.target_found:
        return explain_missing_target(mapping.target_item, mapping)

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


def explain_missing_target(target_item: str, mapping: DatasetMapping) -> str:
    """Explain why a missing target prevents meaningful oracle construction."""

    return (
        "If the requested item is not in the dataset, Grover’s Algorithm cannot "
        "construct a meaningful oracle because there is no valid solution state "
        "to mark. In that case, no phase inversion occurs and the diffuser has no "
        "useful marked amplitude to amplify. Practical systems therefore handle "
        "this with a classical validation step before running the quantum circuit."
    )


def explain_no_solution_experiment(mapping: DatasetMapping) -> str:
    """Explain the experimental no-solution demonstration mode."""

    return (
        "This experimental run intentionally uses no marked state. The circuit "
        f"places {mapping.n_qubits} qubit(s) into superposition over "
        f"{mapping.padded_size} encoded states, but no oracle phase flip is "
        "applied. Because there is no solution state to amplify, measurement "
        "outcomes should remain approximately uniform or random. Any measured "
        "item is not a valid search success."
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

    if not result.target_found and result.stopped_before_quantum_execution:
        return (
            "No measurement was performed because GroverLab stopped before quantum "
            "execution. The requested target item was not in the dataset, so no "
            "valid marked state existed."
        )
    if not result.target_found:
        return (
            f"The simulator measured {result.measured_bitstring or 'no bitstring'} "
            "most often in the no-solution demonstration. Because no valid target "
            "state exists, the success probability is 0.000 and any decoded item "
            "should be treated as a random sample, not a successful search."
        )

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
    if not mapping.target_found and config.missing_target_mode == "experimental":
        warnings.append(
            "Experimental no-solution mode is for learning only; any measured item is random and should not be interpreted as a successful search."
        )
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
            "answer": f"|{mapping.target_binary}>." if mapping.target_found else "No binary target state exists because the item is missing.",
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

    if not result.target_found:
        return {
            "title": "Investigate a No-Solution Grover Case",
            "goal": "Explain why a missing target cannot be amplified by Grover's Algorithm.",
            "steps": [
                "List the dataset items and confirm the requested target is absent.",
                "Explain why no target index or binary state exists.",
                "Identify why the oracle cannot mark a solution state.",
                "If experimental mode was used, inspect the histogram and check whether it is roughly uniform.",
                "Explain why any measured item is not a successful search result.",
            ],
            "reflection_prompt": (
                "Why does a practical system validate the target classically before constructing the oracle?"
            ),
        }

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

    sections = {
        "dataset_mapping": explain_dataset_mapping(result.mapping),
        "superposition": explain_superposition(result.mapping),
        "oracle": explain_oracle(result.mapping),
        "diffuser": explain_diffuser(),
        "iterations": explain_iterations(result.config, result.mapping),
        "measurement": explain_measurement(result),
        "noise": explain_noise(result.config.noise_config),
    }
    if not result.target_found:
        sections["missing_target"] = explain_missing_target(result.mapping.target_item, result.mapping)
        if not result.stopped_before_quantum_execution:
            sections["no_solution_experiment"] = explain_no_solution_experiment(result.mapping)

    return {
        "learning_outcomes": LEARNING_OUTCOMES,
        "warnings": misconception_warnings(result.mapping, result.config),
        "sections": sections,
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
