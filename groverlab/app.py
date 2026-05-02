"""Streamlit entry point for standalone local and online GroverLab use."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from groverlab.grover_analysis import (
    compare_classical_vs_grover,
    run_iteration_sweep,
    run_noise_sweep,
)
from groverlab.grover_config import GroverConfig, NoiseConfig
from groverlab.grover_data import clean_items, load_csv_items, parse_comma_text
from groverlab.grover_education import CORE_WARNING, MISCONCEPTION_WARNINGS, generate_full_explanation
from groverlab.grover_export import (
    export_counts_csv,
    export_result_json,
    export_student_report_markdown,
)
from groverlab.grover_plots import (
    plot_classical_vs_grover,
    plot_counts_histogram,
    plot_iteration_sweep,
    plot_noise_sweep,
)
from groverlab.grover_runner import run_grover_simulation


DEFAULT_DATASET_TEXT = "apple, mango, banana, orange"


def main() -> None:
    """Render the Streamlit application."""

    import streamlit as st

    st.set_page_config(
        page_title="GroverLab",
        page_icon="G",
        layout="wide",
    )

    st.title("GroverLab: Interactive Grover’s Algorithm Simulator")

    st.header("Purpose and Educational Warning")
    st.write(
        "GroverLab is a research-grade educational simulator for exploring how "
        "Grover's Algorithm searches encoded indices."
    )
    st.warning(CORE_WARNING)

    with st.sidebar:
        st.header("Simulation Settings")
        shots = st.slider("Shots", min_value=128, max_value=8192, value=1024, step=128)
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

        noise_enabled = st.checkbox("Enable noisy simulation")
        depolar_prob = 0.0
        measurement_error_prob = 0.0
        if noise_enabled:
            depolar_prob = st.slider(
                "Depolarising noise",
                min_value=0.0,
                max_value=0.5,
                value=0.01,
                step=0.01,
                format="%.2f",
            )
            measurement_error_prob = st.slider(
                "Measurement error",
                min_value=0.0,
                max_value=0.5,
                value=0.01,
                step=0.01,
                format="%.2f",
            )

    st.header("Dataset Input")
    dataset_text = st.text_area(
        "Comma-separated dataset",
        value=DEFAULT_DATASET_TEXT,
        height=110,
    )

    st.header("CSV Upload Option")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    csv_column = st.text_input("CSV column name", value="", placeholder="Optional")

    items, dataset_source_error = _load_items(dataset_text, uploaded_file, csv_column or None)
    if dataset_source_error:
        st.error(dataset_source_error)

    items = clean_items(items)
    if items:
        st.caption(f"Loaded {len(items)} cleaned item(s).")
        st.dataframe(pd.DataFrame({"index": range(len(items)), "item": items}), use_container_width=True)

    st.header("Target Item")
    target_from_select = None
    if items:
        target_from_select = st.selectbox("Select target item", items)
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
        for warning in MISCONCEPTION_WARNINGS:
            st.info(warning)
        return

    try:
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
        )
        with st.spinner("Running Grover simulation..."):
            result = run_grover_simulation(config)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return

    _render_mapping(st, result)
    _render_simulation_outputs(st, result)
    _render_explanations(st, result)
    _render_analysis(st, config, result)
    _render_exports(st, result)


def _load_items(dataset_text: str, uploaded_file, csv_column: str | None) -> tuple[list[str], str | None]:
    """Load dataset items from CSV upload when present, otherwise comma text."""

    try:
        if uploaded_file is not None:
            return load_csv_items(uploaded_file, column_name=csv_column), None
        return parse_comma_text(dataset_text), None
    except Exception as exc:
        return [], str(exc)


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
        use_container_width=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target index", mapping.target_index)
    col2.metric("Target binary state", mapping.target_binary)
    col3.metric("Qubits", mapping.n_qubits)
    col4.metric("Unused padded states", mapping.unused_states)


def _render_simulation_outputs(st, result) -> None:
    """Render ideal/noisy simulation summaries and histogram."""

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
            use_container_width=True,
        )

    st.header("Circuit Summary")
    col1, col2 = st.columns(2)
    col1.metric("Circuit depth", result.circuit_depth)
    col2.metric("Total gates", sum(int(value) for value in result.gate_counts.values()))
    st.dataframe(
        pd.DataFrame(
            sorted(result.gate_counts.items()),
            columns=["gate", "count"],
        ),
        use_container_width=True,
    )

    st.header("Measurement Histogram")
    histogram = plot_counts_histogram(result.ideal_counts, result.mapping.target_binary)
    st.pyplot(histogram)


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

    st.header("Iteration Sweep")
    max_iterations = max(2, min(10, (result.config.iterations or 1) * 2 + 2))
    with st.spinner("Running iteration sweep..."):
        iteration_results = run_iteration_sweep(config, max_iterations=max_iterations)
    st.dataframe(pd.DataFrame(iteration_results), use_container_width=True)
    st.pyplot(plot_iteration_sweep(iteration_results))

    st.header("Noise Sweep")
    noise_values = [0.0, 0.01, 0.03, 0.05]
    with st.spinner("Running noise sweep..."):
        noise_results = run_noise_sweep(config, noise_values=noise_values)
    st.dataframe(pd.DataFrame(noise_results), use_container_width=True)
    st.pyplot(plot_noise_sweep(noise_results))

    st.header("Classical vs Grover Comparison")
    comparison = compare_classical_vs_grover(result.mapping.n_items)
    st.dataframe(pd.DataFrame([comparison]), use_container_width=True)
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
