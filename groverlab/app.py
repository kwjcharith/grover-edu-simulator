"""Streamlit entry point for standalone local and online GroverLab use."""

from __future__ import annotations

from dataclasses import replace
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from groverlab.deployment_config import (
    ALLOW_LARGE_DATASET_UPLOAD_ONLINE,
    ALLOW_NOISY_SIMULATION_ONLINE,
    MAX_PUBLIC_DATASET_SIZE,
    MAX_PUBLIC_ITERATION_SWEEP_POINTS,
    MAX_PUBLIC_NOISE_SWEEP_POINTS,
    MAX_PUBLIC_QUBITS,
    MAX_PUBLIC_SHOTS,
    is_public_demo_mode,
    validate_online_limits,
)
from groverlab.grover_analysis import (
    compare_classical_vs_grover,
    recommended_iterations as recommended_analysis_iterations,
    run_iteration_sweep,
    run_noise_sweep,
)
from groverlab.grover_config import GroverConfig, NoiseConfig
from groverlab.grover_data import (
    calculate_padded_size,
    calculate_required_qubits,
    clean_items,
    load_csv_items,
    parse_comma_text,
)
from groverlab.grover_education import CORE_WARNING, MISCONCEPTION_WARNINGS, generate_full_explanation
from groverlab.grover_export import (
    export_counts_csv,
    export_result_json,
    export_student_report_markdown,
)
from groverlab.grover_plots import (
    plot_classical_vs_grover,
    plot_counts_histogram,
    plot_counts_probability_comparison,
    plot_iteration_sweep,
    plot_noise_sweep,
)
from groverlab.grover_runner import run_grover_simulation


DEFAULT_DATASET_TEXT = "apple, mango, banana, orange"
PUBLIC_DEMO_LIMIT_MESSAGE = (
    "Grover’s Algorithm can be described theoretically for larger search spaces, "
    "but full classical simulation of quantum circuits becomes expensive as qubit "
    "count increases. This online demo intentionally limits simulation size."
)


def _depolarising_noise_label(value: float) -> str:
    """Return a plain-language interpretation of depolarising noise."""

    if value == 0:
        return "Ideal simulator"
    if value <= 0.001:
        return "High-fidelity quantum hardware"
    if value <= 0.005:
        return "Realistic NISQ system"
    if value <= 0.01:
        return "Noisy NISQ system"
    if value <= 0.03:
        return "Severe degradation"
    return "Experimental/extreme noise"


def _measurement_error_label(value: float) -> str:
    """Return a plain-language interpretation of measurement error."""

    if value == 0:
        return "Ideal measurement"
    if value <= 0.01:
        return "Realistic readout error"
    if value <= 0.03:
        return "Noticeable readout corruption"
    return "Severe measurement unreliability"


@st.cache_data(show_spinner=False)
def _cached_parse_comma_text(text: str) -> list[str]:
    """Parse small comma-separated datasets with Streamlit data caching."""

    return parse_comma_text(text)


@st.cache_data(show_spinner=False)
def _cached_clean_items(items: tuple[str, ...]) -> list[str]:
    """Clean small dataset values with Streamlit data caching."""

    return clean_items(list(items))


@st.cache_data(show_spinner=False)
def _cached_static_misconception_warnings() -> list[str]:
    """Cache static educational warning text."""

    return list(MISCONCEPTION_WARNINGS)


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(
        page_title="GroverLab",
        page_icon="G",
        layout="wide",
    )

    st.title("GroverLab: Interactive Grover’s Algorithm Simulator")
    public_demo_mode = is_public_demo_mode()
    if public_demo_mode:
        _render_public_demo_banner(st)

    st.header("Purpose and Educational Warning")
    st.write(
        "GroverLab is a research-grade educational simulator for exploring how "
        "Grover's Algorithm searches encoded indices."
    )
    st.warning(CORE_WARNING)

    with st.sidebar:
        st.header("Simulation Settings")
        max_shots = MAX_PUBLIC_SHOTS if public_demo_mode else 8192
        shots = st.slider("Shots", min_value=128, max_value=max_shots, value=1024, step=128)
        iteration_mode = st.radio("Iterations", ["Automatic", "Manual"], horizontal=True)
        manual_iterations = None
        if iteration_mode == "Manual":
            manual_iterations = st.number_input(
                "Manual Grover iterations",
                min_value=0,
                max_value=50,
                value=1,
                step=1,
            )
        missing_target_label = st.radio(
            "Missing target handling",
            ["Practical mode: stop before quantum execution", "Experimental mode: run no-solution demonstration"],
        )
        missing_target_mode = (
            "experimental"
            if missing_target_label.startswith("Experimental")
            else "stop"
        )

        noise_enabled = st.checkbox(
            "Enable noisy simulation",
            disabled=public_demo_mode and not ALLOW_NOISY_SIMULATION_ONLINE,
        )
        depolar_prob = 0.0
        measurement_error_prob = 0.0
        if noise_enabled:
            depolar_prob = st.slider(
                "Depolarising noise",
                min_value=0.0,
                max_value=0.07,
                value=0.001,
                step=0.0005,
                format="%.4f",
            )
            st.caption(_depolarising_noise_label(depolar_prob))
            st.metric("Approximate Gate Fidelity", f"{(1 - depolar_prob) * 100:.2f}%")
            st.caption(
                "This is an approximate educational interpretation based on a simplified depolarising noise model."
            )
            measurement_error_prob = st.slider(
                "Measurement error",
                min_value=0.0,
                max_value=0.07,
                value=0.01,
                step=0.001,
                format="%.3f",
            )
            st.caption(_measurement_error_label(measurement_error_prob))

        _render_noise_model_notes(st)

    st.header("Dataset Input")
    dataset_text = st.text_area(
        "Comma-separated dataset",
        value=DEFAULT_DATASET_TEXT,
        height=110,
    )

    st.header("CSV Upload Option")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    csv_column = st.text_input("CSV column name", value="", placeholder="Optional")

    items, dataset_source_error = _load_items(
        dataset_text,
        uploaded_file,
        csv_column or None,
        public_demo_mode,
    )
    if dataset_source_error:
        st.error(dataset_source_error)

    items = _cached_clean_items(tuple(items))
    if items:
        st.caption(f"Loaded {len(items)} cleaned item(s).")
        if len(items) <= MAX_PUBLIC_DATASET_SIZE or not public_demo_mode:
            st.dataframe(pd.DataFrame({"index": range(len(items)), "item": items}), width="stretch")
        else:
            st.info("The dataset is loaded for theoretical scaling analysis only.")
    _render_dataset_size_guidance(st, items)
    if public_demo_mode and items and _exceeds_public_limits(items):
        _render_theoretical_analysis_only(st, len(items))

    st.header("Target Item")
    target_from_select = None
    if items and (not public_demo_mode or len(items) <= MAX_PUBLIC_DATASET_SIZE):
        target_from_select = st.selectbox("Select target item", items)
    elif items:
        st.info("Target selection is disabled for oversized public-demo datasets. Use local mode for full simulation.")
    target_from_text = st.text_input("Or type target item", value="")
    target_item = target_from_text.strip() or target_from_select

    run_button = st.button(
        "Run Grover Simulation",
        type="primary",
        disabled=not items or not target_item,
    )

    st.header("Anonymous Research Logging Notice")
    st.info(
        "GroverLab research logs are anonymous. They store aggregate run metrics such as "
        "dataset size, qubit count, iterations, noise settings, success probability, "
        "runtime, circuit depth, and gate counts. They do not store names, emails, "
        "student IDs, IP addresses, or full dataset item names."
    )

    if not run_button:
        st.header("Misconception Warnings")
        for warning in _cached_static_misconception_warnings():
            st.info(warning)
        return

    try:
        if public_demo_mode:
            n_qubits = calculate_required_qubits(len(items))
            try:
                online_limits = validate_online_limits(len(items), n_qubits, shots)
            except ValueError as exc:
                st.warning(str(exc))
                st.info(PUBLIC_DEMO_LIMIT_MESSAGE)
                _render_theoretical_analysis_only(st, len(items))
                return
            shots = int(online_limits["shots"])
            for warning in online_limits["warnings"]:
                st.warning(warning)

        config = GroverConfig(
            dataset_items=items,
            target_item=target_item or "",
            shots=shots,
            iterations=int(manual_iterations) if manual_iterations is not None else None,
            noise_config=NoiseConfig(
                noise_enabled=noise_enabled,
                depolar_prob=depolar_prob,
                measurement_error_prob=measurement_error_prob,
            ),
            missing_target_mode=missing_target_mode,
        )
        with st.spinner("Running Grover simulation..."):
            result = run_grover_simulation(config)
    except (MemoryError, TimeoutError, RuntimeError) as exc:
        st.error(
            "The simulation could not complete in the public online environment. "
            "Try reducing dataset size, qubits, shots, noise sweeps, or run GroverLab locally."
        )
        st.caption(f"Anonymous error type: {type(exc).__name__}")
        return
    except Exception as exc:
        st.error(
            "The simulation could not complete in the public online environment. "
            "Try reducing dataset size, qubits, shots, noise sweeps, or run GroverLab locally."
        )
        st.caption(f"Anonymous error type: {type(exc).__name__}")
        return

    _render_mapping(st, result)
    _render_simulation_outputs(st, result)
    _render_explanations(st, result)
    _render_analysis(st, config, result)
    _render_exports(st, result)


def _load_items(
    dataset_text: str,
    uploaded_file,
    csv_column: str | None,
    public_demo_mode: bool,
) -> tuple[list[str], str | None]:
    """Load dataset items from CSV upload when present, otherwise comma text."""

    try:
        if uploaded_file is not None:
            return _load_uploaded_csv_items(uploaded_file, csv_column, public_demo_mode)
        return _cached_parse_comma_text(dataset_text), None
    except Exception as exc:
        return [], str(exc)


def _load_uploaded_csv_items(
    uploaded_file,
    csv_column: str | None,
    public_demo_mode: bool,
) -> tuple[list[str], str | None]:
    """Load CSV items while enforcing public-demo row limits early."""

    if not public_demo_mode or ALLOW_LARGE_DATASET_UPLOAD_ONLINE:
        uploaded_file.seek(0)
        return load_csv_items(uploaded_file, column_name=csv_column), None

    uploaded_file.seek(0)
    preview = pd.read_csv(uploaded_file, nrows=MAX_PUBLIC_DATASET_SIZE + 1)
    if len(preview) > MAX_PUBLIC_DATASET_SIZE:
        return (
            [f"uploaded_row_{index}" for index in range(MAX_PUBLIC_DATASET_SIZE + 1)],
            "Uploaded dataset is too large for the public online demo. Please upload a file with 1024 items or fewer, or run locally.",
        )
    if preview.empty:
        return [], "CSV input is empty."

    if csv_column:
        if csv_column not in preview.columns:
            return [], f"CSV column '{csv_column}' was not found."
        values = preview[csv_column].tolist()
    else:
        values = preview.iloc[:, 0].tolist()

    return clean_items(values), None


def _render_public_demo_banner(st) -> None:
    """Show online deployment constraints clearly near the top of the app."""

    st.info(
        "Public Online Demo Mode\n\n"
        "This free online version is designed for small educational demonstrations. "
        "Large Grover simulations can become computationally expensive because "
        "classical simulation grows exponentially with qubit count. For large "
        "datasets or deep noisy experiments, please run GroverLab locally.\n\n"
        f"Recommended dataset size: 4-256 items\n\n"
        f"Maximum public demo dataset size: {MAX_PUBLIC_DATASET_SIZE} items\n\n"
        f"Maximum public demo qubits: {MAX_PUBLIC_QUBITS}\n\n"
        f"Maximum shots: {MAX_PUBLIC_SHOTS}"
    )


def _render_dataset_size_guidance(st, items: list[str]) -> None:
    """Render educational guidance about dataset size and qubit scaling."""

    with st.expander("Dataset Size and Qubit Guidance"):
        st.write("Small datasets (4-64 items): Recommended for educational demonstrations.")
        st.write("Medium datasets (128-1024 items): Useful for advanced experiments and noise analysis.")
        st.write(
            "Large datasets (>4096 items): May become computationally expensive because classical simulation complexity grows exponentially with qubit count."
        )
        st.write(
            "Very large datasets (100000+ items): Currently not recommended for full simulation because the required number of qubits and circuit depth exceed practical classical simulation limits."
        )

        if not items:
            st.info("Add dataset items to see required qubits and padded search states.")
            return

        n_qubits = calculate_required_qubits(len(items))
        padded_size = calculate_padded_size(n_qubits)
        unused_states = padded_size - len(items)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset size", len(items))
        col2.metric("Required qubits", n_qubits)
        col3.metric("Padded searchable states", padded_size)
        col4.metric("Unused padded states", unused_states)


def _exceeds_public_limits(items: list[str]) -> bool:
    """Return whether the current dataset exceeds public simulation limits."""

    if not items:
        return False
    n_qubits = calculate_required_qubits(len(items))
    return len(items) > MAX_PUBLIC_DATASET_SIZE or n_qubits > MAX_PUBLIC_QUBITS


def _render_theoretical_analysis_only(st, dataset_size: int) -> None:
    """Render scaling information without building a quantum circuit."""

    if dataset_size < 1:
        return

    n_qubits = calculate_required_qubits(dataset_size)
    padded_size = calculate_padded_size(n_qubits)
    unused_states = padded_size - dataset_size
    recommended_iterations = recommended_analysis_iterations(dataset_size)

    st.header("Theoretical Analysis Only")
    st.warning(PUBLIC_DEMO_LIMIT_MESSAGE)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset size", dataset_size)
    col2.metric("Required qubits", n_qubits)
    col3.metric("Padded searchable states", padded_size)
    col4.metric("Recommended iterations", recommended_iterations)
    st.caption(f"Unused padded states: {unused_states}")
    st.write(
        "Full circuit simulation is disabled for this dataset in public online mode. "
        "This is a Streamlit Community Cloud resource safeguard, not a theoretical "
        "limit of Grover's Algorithm."
    )


def _render_noise_model_notes(st) -> None:
    """Render concise notes about the simplified noise model."""

    with st.expander("Noise Model Notes"):
        st.write(
            "Depolarising Noise: Approximates gate imperfections during quantum operations. "
            "Higher values reduce coherent interference and damage amplitude amplification."
        )
        st.write("Approximate relation: Gate Fidelity ≈ 1 − Depolarising Noise")
        st.write(
            "Measurement Error: Approximates incorrect readout of qubit states during the final measurement stage. "
            "The quantum computation may still be correct, but the observed classical bitstring may be wrong."
        )
        st.write("Important distinction:")
        st.write("- Gate errors affect the computation itself.")
        st.write("- Measurement errors affect the final observation.")
        st.write(
            "This simulator uses simplified noise models for educational purposes. Real quantum hardware exhibits additional effects such as decoherence, dephasing, crosstalk, leakage, and calibration drift."
        )


def _render_mapping(st, result) -> None:
    """Render dataset mapping and target encoding details."""

    st.header("Dataset Mapping")
    mapping = result.mapping
    st.dataframe(
        pd.DataFrame(
            {
                "index": range(mapping.n_items),
                "item": mapping.cleaned_items,
                "binary_state": [
                    format(index, f"0{mapping.n_qubits}b")
                    for index in range(mapping.n_items)
                ],
            }
        ),
        width="stretch",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target index", mapping.target_index)
    col2.metric("Target binary state", mapping.target_binary or "None")
    col3.metric("Qubits", mapping.n_qubits)
    col4.metric("Unused padded states", mapping.unused_states)


def _render_simulation_outputs(st, result) -> None:
    """Render ideal/noisy simulation summaries and histogram."""

    if result.stopped_before_quantum_execution:
        st.header("Missing Target")
        st.warning("Target item not found in dataset.")
        st.write(result.missing_target_explanation)
        st.info("No quantum circuit execution was performed. Missing-target validation was handled classically before constructing the oracle.")
        return

    st.header("Ideal Simulation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Most likely bitstring", result.measured_bitstring)
    col2.metric("Decoded item", result.decoded_item or "Unused state")
    col3.metric("Success probability", f"{result.success_probability:.3f}")

    st.header("Optional Noisy Simulation")
    if result.noisy_counts is None:
        st.info("Noisy simulation was not enabled for this run.")
    else:
        st.metric("Noisy success probability", f"{result.noisy_success_probability:.3f}")
        st.dataframe(
            pd.DataFrame(
                sorted(result.noisy_counts.items()),
                columns=["bitstring", "count"],
            ),
            width="stretch",
        )

    st.header("Circuit Summary")
    col1, col2 = st.columns(2)
    col1.metric("Circuit depth", result.circuit_depth)
    total_gates = sum(int(value) for value in result.gate_counts.values())
    col2.metric("Total gates", total_gates)
    st.dataframe(
        pd.DataFrame(
            sorted(result.gate_counts.items()),
            columns=["gate", "count"],
        ),
        width="stretch",
    )
    _render_large_circuit_warning(st, result.mapping.n_qubits, result.circuit_depth, total_gates)

    if result.noisy_counts is None:
        st.header("Measurement Histogram")
        histogram = plot_counts_histogram(result.ideal_counts, result.mapping.target_binary)
        st.pyplot(histogram)
    else:
        st.header("Final Measurement Probability Comparison")
        comparison = plot_counts_probability_comparison(
            result.ideal_counts,
            result.noisy_counts,
            result.mapping.target_binary,
        )
        st.pyplot(comparison)
        st.info(
            "The noisy simulation demonstrates how realistic hardware imperfections reduce Grover’s amplitude amplification advantage. As noise increases, probability spreads across multiple states, reducing the likelihood of measuring the correct target."
        )
        st.subheader("Measurement Probability Comparison")
        st.dataframe(
            _measurement_probability_table(
                result.ideal_counts,
                result.noisy_counts,
                result.mapping.target_binary,
            ),
            width="stretch",
        )

    if not result.target_found:
        st.warning(
            "This is an experimental no-solution demonstration. The distribution is expected to be approximately uniform or random, and any measured item is not a valid search success."
        )


def _render_large_circuit_warning(st, n_qubits: int, circuit_depth: int, total_gates: int) -> None:
    """Warn when circuit size makes noise especially consequential."""

    if n_qubits >= 10 or circuit_depth >= 200 or total_gates >= 1000:
        st.warning(
            "WARNING:\n"
            "Large Grover circuits are highly sensitive to noise.\n"
            "Even small gate errors may significantly reduce success probability in NISQ hardware."
        )


def _measurement_probability_table(
    ideal_counts: dict[str, int],
    noisy_counts: dict[str, int],
    target_binary: str,
) -> pd.DataFrame:
    """Build a compact probability comparison table for target and top states."""

    ideal_probabilities = _probabilities(ideal_counts)
    noisy_probabilities = _probabilities(noisy_counts)
    all_states = set(ideal_probabilities) | set(noisy_probabilities)
    ranked_states = sorted(
        all_states,
        key=lambda state: max(
            ideal_probabilities.get(state, 0.0),
            noisy_probabilities.get(state, 0.0),
        ),
        reverse=True,
    )

    selected_states: list[str] = []
    if target_binary:
        selected_states.append(target_binary)
    for state in ranked_states:
        if state not in selected_states:
            selected_states.append(state)
        if len(selected_states) >= 11:
            break

    return pd.DataFrame(
        {
            "State": selected_states,
            "Ideal Probability": [
                round(ideal_probabilities.get(state, 0.0), 4)
                for state in selected_states
            ],
            "Noisy Probability": [
                round(noisy_probabilities.get(state, 0.0), 4)
                for state in selected_states
            ],
        }
    )


def _probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Normalize measurement counts into probabilities."""

    total = sum(counts.values())
    if total <= 0:
        return {}
    return {state: count / total for state, count in counts.items()}


def _depolarising_sweep_values(selected_depolar_prob: float, max_points: int = 8) -> list[float]:
    """Return evenly spaced depolarising-noise values from 0 to selected value."""

    points = max(2, max_points)
    if selected_depolar_prob <= 0:
        return [0.0]
    return [
        round((selected_depolar_prob * index) / (points - 1), 6)
        for index in range(points)
    ]


def _measurement_error_scenarios(selected_measurement_error: float) -> list[tuple[str, float]]:
    """Return the three readout-error scenarios used in the noise sweep."""

    half_selected = selected_measurement_error * 0.5
    return [
        ("Measurement error = 0", 0.0),
        ("Measurement error = 0.5 x selected", half_selected),
        ("Measurement error = selected", selected_measurement_error),
    ]


def _config_with_measurement_error(config: GroverConfig, measurement_error: float) -> GroverConfig:
    """Return a sweep config with fixed measurement error and enabled noise."""

    noise_config = replace(
        config.noise_config,
        noise_enabled=True,
        measurement_error_prob=measurement_error,
    )
    return replace(config, noise_config=noise_config)


def _render_explanations(st, result) -> None:
    """Render educational explanations, warnings, and quiz prompts."""

    explanation = generate_full_explanation(result)

    st.header("Step-by-Step Explanation")
    for heading, body in explanation["sections"].items():
        with st.expander(heading.replace("_", " ").title(), expanded=heading == "dataset_mapping"):
            st.write(body)

    st.header("Misconception Warnings")
    for warning in explanation["warnings"]:
        st.warning(warning)

    st.subheader("Short Quiz Questions")
    for index, question in enumerate(explanation["quiz_questions"], start=1):
        st.write(f"{index}. {question['question']}")
        with st.expander("Answer"):
            st.write(question["answer"])


def _render_analysis(st, config: GroverConfig, result) -> None:
    """Render iteration sweep, noise sweep, and classical comparison."""

    if result.stopped_before_quantum_execution:
        return

    st.header("Ideal vs Noisy Iteration Sweep")
    max_iterations = max(2, min(10, (result.config.iterations or 1) * 2 + 2))
    if is_public_demo_mode():
        max_iterations = min(max_iterations, MAX_PUBLIC_ITERATION_SWEEP_POINTS - 1)
    with st.spinner("Running iteration sweep..."):
        iteration_results = run_iteration_sweep(config, max_iterations=max_iterations)
    st.dataframe(pd.DataFrame(iteration_results), width="stretch")
    st.pyplot(plot_iteration_sweep(iteration_results))

    st.header("Noise Sweep")
    sweep_points = MAX_PUBLIC_NOISE_SWEEP_POINTS if is_public_demo_mode() else 8
    noise_values = _depolarising_sweep_values(
        config.noise_config.depolar_prob,
        max_points=sweep_points,
    )
    st.caption(
        "Depolarising noise is swept evenly from 0 to the selected depolarising-noise value. "
        "Three readout-error scenarios are shown: 0, half the selected measurement error, and the selected measurement error."
    )

    scenarios = _measurement_error_scenarios(config.noise_config.measurement_error_prob)
    columns = st.columns(3)
    for column, (label, measurement_error) in zip(columns, scenarios):
        scenario_config = _config_with_measurement_error(config, measurement_error)
        with column:
            st.subheader(label)
            st.caption(f"Fixed measurement error: {measurement_error:.4f}")
            with st.spinner("Running noise sweep..."):
                noise_results = run_noise_sweep(scenario_config, noise_values=noise_values)
            st.pyplot(plot_noise_sweep(noise_results))
            with st.expander("Sweep data"):
                st.dataframe(pd.DataFrame(noise_results), width="stretch")

    st.header("Classical vs Grover Comparison")
    comparison = compare_classical_vs_grover(result.mapping.n_items)
    st.dataframe(pd.DataFrame([comparison]), width="stretch")
    st.pyplot(plot_classical_vs_grover(comparison))


def _render_exports(st, result) -> None:
    """Render JSON, CSV, and Markdown download buttons."""

    st.header("Export Results")
    json_data = _export_text(result, "result.json", export_result_json)
    csv_data = _export_text(result.ideal_counts, "counts.csv", export_counts_csv)
    markdown_data = _export_text(result, "student_report.md", export_student_report_markdown)

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download JSON",
        data=json_data,
        file_name="grover_result.json",
        mime="application/json",
    )
    col2.download_button(
        "Download CSV",
        data=csv_data,
        file_name="grover_counts.csv",
        mime="text/csv",
    )
    col3.download_button(
        "Download Markdown Report",
        data=markdown_data,
        file_name="grover_student_report.md",
        mime="text/markdown",
    )


def _export_text(payload, filename: str, exporter) -> str:
    """Run a file-based exporter and return the written text."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        exporter(payload, path)
        return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
