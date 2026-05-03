"""Streamlit entry point for standalone local and online GroverLab use."""

from __future__ import annotations

from dataclasses import replace
import json
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
    build_research_summary,
    compare_classical_vs_grover,
    ideal_grover_sweep_curves,
    probability_loss_by_iteration,
    recommended_iterations as recommended_analysis_iterations,
    run_decoherence_iteration_overlay,
    run_iteration_sweep,
    run_noisy_condition_iteration_overlay,
    run_noise_sweep,
    run_t1_iteration_overlay,
    run_t1_sweep,
    run_t2_iteration_overlay,
    run_t2_sweep,
    scalability_analysis,
)
from groverlab.grover_config import GroverConfig, NoiseConfig
from groverlab.grover_data import (
    calculate_padded_size,
    calculate_required_qubits,
    clean_items,
    decode_bitstring,
    load_csv_items,
    parse_comma_text,
)
from groverlab.grover_decoherence import decoherence_probabilities
from groverlab.grover_education import CORE_WARNING, MISCONCEPTION_WARNINGS, generate_full_explanation
from groverlab.grover_export import (
    export_counts_csv,
    export_result_json,
    export_student_report_markdown,
)
from groverlab.grover_plots import (
    plot_classical_vs_grover,
    plot_coherence_sweep,
    plot_counts_histogram,
    plot_counts_probability_comparison,
    plot_decoherence_iteration_overlay,
    plot_iteration_sweep,
    plot_iteration_overlay_curves,
    plot_noise_sweep,
    plot_probability_loss,
    plot_scalability_growth,
)


DEFAULT_DATASET_TEXT = "apple, mango, banana, orange"
PUBLIC_DEMO_LIMIT_MESSAGE = (
    "Grover’s Algorithm can be described theoretically for larger search spaces, "
    "but full classical simulation of quantum circuits becomes expensive as qubit "
    "count increases. This online demo intentionally limits simulation size."
)
HARDWARE_PRESETS = {
    "Ideal Simulator": {
        "depolar_prob": 0.0,
        "t1_relaxation_us": 500.0,
        "t2_coherence_us": 300.0,
        "measurement_error_prob": 0.0,
    },
    "High-Fidelity Hardware": {
        "depolar_prob": 0.0003,
        "t1_relaxation_us": 300.0,
        "t2_coherence_us": 200.0,
        "measurement_error_prob": 0.005,
    },
    "Typical Superconducting NISQ": {
        "depolar_prob": 0.001,
        "t1_relaxation_us": 120.0,
        "t2_coherence_us": 80.0,
        "measurement_error_prob": 0.01,
    },
    "Noisy NISQ": {
        "depolar_prob": 0.005,
        "t1_relaxation_us": 50.0,
        "t2_coherence_us": 30.0,
        "measurement_error_prob": 0.02,
    },
    "Extreme Experimental Noise": {
        "depolar_prob": 0.03,
        "t1_relaxation_us": 15.0,
        "t2_coherence_us": 10.0,
        "measurement_error_prob": 0.05,
    },
}


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


def _t1_label(value: float) -> str:
    """Return a plain-language interpretation of T1 relaxation time."""

    if value >= 300:
        return "Excellent coherence"
    if value >= 100:
        return "High-fidelity superconducting hardware"
    if value >= 50:
        return "Typical NISQ coherence"
    if value >= 20:
        return "Limited coherence"
    return "Severe relaxation"


def _t2_label(value: float) -> str:
    """Return a plain-language interpretation of T2 coherence time."""

    if value >= 200:
        return "Excellent phase coherence"
    if value >= 80:
        return "High-quality NISQ coherence"
    if value >= 40:
        return "Typical superconducting coherence"
    if value >= 10:
        return "Strong decoherence"
    return "Extreme phase instability"


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
        t1_relaxation_us = 500.0
        t2_coherence_us = 300.0
        if noise_enabled:
            preset_name = st.selectbox(
                "Hardware Preset",
                list(HARDWARE_PRESETS),
                index=2,
            )
            preset = HARDWARE_PRESETS[preset_name]
            depolar_prob = st.slider(
                "Depolarising noise",
                min_value=0.0,
                max_value=0.07,
                value=float(preset["depolar_prob"]),
                step=0.0005,
                format="%.4f",
            )
            st.caption(_depolarising_noise_label(depolar_prob))
            st.metric("Approximate Single-Gate Depolarising Fidelity", f"{(1 - depolar_prob) * 100:.2f}%")
            st.caption(
                "This is an approximate single-gate depolarising fidelity estimate. Full circuit performance also depends on circuit depth, T1/T2 coherence, measurement error, and accumulated decoherence."
            )
            t1_relaxation_us = st.slider(
                "T1 Relaxation Time (µs)",
                min_value=10.0,
                max_value=500.0,
                value=float(preset["t1_relaxation_us"]),
                step=5.0,
            )
            st.caption(f"{_t1_label(t1_relaxation_us)}. Models energy relaxation and amplitude decay toward |0>.")
            max_t2 = min(300.0, 2 * t1_relaxation_us)
            t2_default = min(float(preset["t2_coherence_us"]), max_t2)
            t2_coherence_us = st.slider(
                "T2 Coherence Time (µs)",
                min_value=5.0,
                max_value=max_t2,
                value=t2_default,
                step=5.0,
            )
            st.caption(f"{_t2_label(t2_coherence_us)}. Models phase coherence and interference stability.")
            measurement_error_prob = st.slider(
                "Measurement error",
                min_value=0.0,
                max_value=0.07,
                value=float(preset["measurement_error_prob"]),
                step=0.001,
                format="%.3f",
            )
            st.caption(_measurement_error_label(measurement_error_prob))
            _render_noise_summary_panel(
                st,
                NoiseConfig(
                    noise_enabled=True,
                    depolar_prob=depolar_prob,
                    measurement_error_prob=measurement_error_prob,
                    t1_relaxation_us=t1_relaxation_us,
                    t2_coherence_us=t2_coherence_us,
                ),
            )

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
                t1_relaxation_us=t1_relaxation_us,
                t2_coherence_us=t2_coherence_us,
            ),
            missing_target_mode=missing_target_mode,
        )
        with st.spinner("Running Grover simulation..."):
            from groverlab.grover_runner import run_grover_simulation

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
    _render_scalability_analysis(st, result.mapping.n_items)
    _render_simulation_outputs(st, result)
    _render_explanations(st, result)
    analysis_bundle = _render_analysis(st, config, result)
    _render_exports(st, result, analysis_bundle)


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
    """Render concise notes about the simplified decoherence-aware noise model."""

    with st.expander("Decoherence and Noise Notes"):
        st.write(
            "T1 Relaxation: T1 relaxation models energy loss from the environment. "
            "A qubit in |1> may decay toward |0> over time. This primarily affects "
            "amplitude preservation, deep circuits, and repeated Grover iterations."
        )
        st.write(
            "T2 Dephasing: T2 dephasing models phase instability. Grover's Algorithm "
            "relies heavily on constructive interference, so T2 dephasing can destroy "
            "interference and weaken amplitude amplification."
        )
        st.write(
            "Pure Dephasing: GroverLab derives pure dephasing time Tphi from T1 and T2 "
            "using 1/Tphi = 1/T2 - 1/(2T1), which avoids double-counting relaxation effects."
        )
        st.write(
            "Depolarising Noise: Approximates gate imperfections during quantum operations. "
            "Higher values reduce coherent interference and damage amplitude amplification."
        )
        st.write(
            "Measurement Error: Approximates incorrect readout of qubit states during the final measurement stage. "
            "The quantum computation may still be correct, but the observed classical bitstring may be wrong."
        )
        st.write("Important distinction:")
        st.write("- Gate errors affect the computation itself.")
        st.write("- T1/T2 decoherence accumulates over gate duration and circuit depth.")
        st.write("- Measurement errors affect the final observation.")
        st.write(
            "This simulator uses simplified approximate gate durations: 50 ns for single-qubit gates "
            "and 300 ns for multi-qubit gates. Real hardware gate durations vary across devices and calibration states."
        )


def _render_noise_summary_panel(st, noise_config: NoiseConfig) -> None:
    """Render the currently selected noise contributions."""

    probabilities = decoherence_probabilities(noise_config)
    t_phi = probabilities["t_phi_us"]
    t_phi_label = "infinite" if t_phi == float("inf") else f"{t_phi:.2f} µs"
    with st.expander("Current Noise Contributions", expanded=True):
        st.write("- Depolarising: gradual gate corruption")
        st.write("- T1 Relaxation: amplitude decay toward |0>")
        st.write("- T2 Dephasing: loss of phase coherence and interference")
        st.write("- Measurement Error: final readout corruption")
        st.caption(
            f"Derived Tphi: {t_phi_label}; "
            f"single-gate p_T1={probabilities['amplitude_damping_probability']:.6f}, "
            f"single-gate p_phi={probabilities['pure_dephasing_probability']:.6f}."
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
        st.header("Final Measurement Distribution: Ideal vs Noisy")
        comparison = plot_counts_probability_comparison(
            result.ideal_counts,
            result.noisy_counts,
            result.mapping.target_binary,
        )
        st.pyplot(comparison)
        st.info(
            "The noisy simulation demonstrates how realistic hardware imperfections reduce Grover’s amplitude amplification advantage. As noise increases, probability spreads across multiple states, reducing the likelihood of measuring the correct target."
        )
        st.subheader("Histogram Interpretation")
        st.dataframe(_histogram_interpretation_table(result), width="stretch")
        st.subheader("Measurement Probability Comparison")
        st.dataframe(
            _measurement_probability_table(
                result.ideal_counts,
                result.noisy_counts,
                result.mapping.target_binary,
                result.mapping,
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
            "Large Grover circuits are highly vulnerable to decoherence and accumulated gate noise. "
            "As circuit depth increases, T1 relaxation and T2 dephasing can significantly reduce amplification performance."
        )


def _measurement_probability_table(
    ideal_counts: dict[str, int],
    noisy_counts: dict[str, int],
    target_binary: str,
    mapping,
) -> pd.DataFrame:
    """Build a compact probability comparison table for target and top states."""

    ideal_probabilities = _probabilities(ideal_counts)
    noisy_probabilities = _probabilities(noisy_counts)

    selected_states: list[str] = []
    if target_binary:
        selected_states.append(target_binary)
    for state in _ranked_probability_states(noisy_probabilities):
        if state not in selected_states:
            selected_states.append(state)
        if len(selected_states) >= 11:
            break
    for state in _ranked_probability_states(ideal_probabilities):
        if state not in selected_states:
            selected_states.append(state)
        if len(selected_states) >= 21:
            break

    return pd.DataFrame(
        {
            "State": selected_states,
            "Decoded item": [_decode_state_for_table(state, mapping) for state in selected_states],
            "Ideal Probability": [
                round(ideal_probabilities.get(state, 0.0), 4)
                for state in selected_states
            ],
            "Noisy Probability": [
                round(noisy_probabilities.get(state, 0.0), 4)
                for state in selected_states
            ],
            "Difference from ideal": [
                round(
                    ideal_probabilities.get(state, 0.0)
                    - noisy_probabilities.get(state, 0.0),
                    4,
                )
                for state in selected_states
            ],
            "State type": [_state_type(state, target_binary, mapping) for state in selected_states],
        }
    )


def _histogram_interpretation_table(result) -> pd.DataFrame:
    """Summarize ideal-vs-noisy target and most-likely states."""

    ideal_probabilities = _probabilities(result.ideal_counts)
    noisy_probabilities = _probabilities(result.noisy_counts or {})
    ideal_most_likely = max(result.ideal_counts, key=result.ideal_counts.get)
    noisy_most_likely = max(result.noisy_counts, key=result.noisy_counts.get) if result.noisy_counts else ""
    target = result.mapping.target_binary
    return pd.DataFrame(
        [
            {
                "Metric": "Target probability",
                "Ideal": round(ideal_probabilities.get(target, 0.0), 4),
                "Noisy": round(noisy_probabilities.get(target, 0.0), 4),
            },
            {
                "Metric": "Most likely state",
                "Ideal": ideal_most_likely,
                "Noisy": noisy_most_likely,
            },
            {
                "Metric": "Decoded item",
                "Ideal": _decode_state_for_table(ideal_most_likely, result.mapping),
                "Noisy": _decode_state_for_table(noisy_most_likely, result.mapping),
            },
            {
                "Metric": "Probability degradation",
                "Ideal": "",
                "Noisy": round(
                    ideal_probabilities.get(target, 0.0)
                    - noisy_probabilities.get(target, 0.0),
                    4,
                ),
            },
        ]
    )


def _ranked_probability_states(probabilities: dict[str, float]) -> list[str]:
    """Return states ranked by probability descending."""

    return sorted(probabilities, key=probabilities.get, reverse=True)[:10]


def _decode_state_for_table(state: str, mapping) -> str:
    """Decode a state for display, handling padded states."""

    if not state:
        return ""
    try:
        return decode_bitstring(state, mapping) or "Padded unused state"
    except ValueError:
        return "Invalid state"


def _state_type(state: str, target_binary: str, mapping) -> str:
    """Classify a basis state for probability comparison."""

    if target_binary and state == target_binary:
        return "target"
    try:
        index = int(state, 2)
    except ValueError:
        return "invalid"
    if index >= mapping.n_items:
        return "padded unused state"
    return "valid non-target"


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


def _iteration_sweep_max_iterations(result) -> int:
    """Choose a sweep range that shows amplification and over-rotation."""

    recommended = recommended_analysis_iterations(result.mapping.n_items)
    upper_limit = max(6, (3 * recommended) + 3)
    upper_limit = min(upper_limit, result.mapping.padded_size + recommended)
    if is_public_demo_mode():
        upper_limit = min(upper_limit, MAX_PUBLIC_ITERATION_SWEEP_POINTS - 1)
    return max(2, upper_limit)


def _adaptive_noisy_ymax(rows: list[dict]) -> float:
    """Return the y-axis max used by adaptive noisy sweep plots."""

    probabilities = [float(row.get("target_probability", 0.0)) for row in rows]
    if not probabilities:
        return 0.05
    return max(0.05, max(probabilities) * 1.15)


def _classical_comparison_sizes(current_dataset_size: int) -> list[int]:
    """Return a broad range of dataset sizes for classical-vs-Grover comparison."""

    sizes = {
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
        10_000,
        100_000,
        1_000_000,
        current_dataset_size,
    }
    return sorted(size for size in sizes if size > 0)


def _research_summary_table(summary: dict) -> pd.DataFrame:
    """Convert a wide research summary into an easier-to-read metric table."""

    rows = [
        {
            "Metric": key.replace("_", " ").title(),
            "Value": value,
        }
        for key, value in summary.items()
    ]
    return pd.DataFrame(rows)


def _peak_summary_by_scenario(rows: list[dict]) -> list[dict]:
    """Summarise peak iteration and probability for every plotted line."""

    summaries: list[dict] = []
    scenarios = list(dict.fromkeys(row.get("scenario") for row in rows))
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row.get("scenario") == scenario]
        if not scenario_rows:
            continue
        peak = max(
            scenario_rows,
            key=lambda row: row.get("target_probability", row.get("success_probability", 0.0)),
        )
        summaries.append(
            {
                "scenario": scenario,
                "peak_iteration": peak.get("iterations"),
                "peak_probability": peak.get(
                    "target_probability",
                    peak.get("success_probability"),
                ),
                "recommended_iteration": peak.get("recommended_iteration"),
                "dataset_size": peak.get("dataset_size"),
                "qubits": peak.get("n_qubits"),
                "t1_relaxation_us": peak.get("t1_relaxation_us"),
                "t2_coherence_us": peak.get("t2_coherence_us"),
                "depolar_prob": peak.get("depolar_prob"),
                "measurement_error_prob": peak.get("measurement_error_prob"),
            }
        )
    return summaries


def _render_sweep_notation_notes(st) -> None:
    """Explain the notation used in sweep plots."""

    with st.expander("What do k and N mean?"):
        st.write(
            "k is the number of Grover iterations. One Grover iteration means one oracle phase-marking step plus one diffuser/amplitude-amplification step."
        )
        st.write(
            "N is the searchable basis-state space after padding to a power of two. For example, a dataset with 60 items needs 6 qubits, so N = 64 searchable states."
        )
        st.write(
            "The recommended k line is the ideal Grover iteration estimate. In noisy systems, the observed best iteration may shift or flatten."
        )


def _render_ideal_sweep_details(st, ideal_sizes: list[int]) -> None:
    """Display dataset-size details for ideal sweep lines."""

    rows = []
    for size in ideal_sizes:
        n_qubits = calculate_required_qubits(size)
        rows.append(
            {
                "Line label": f"N={calculate_padded_size(n_qubits)}",
                "Example dataset size": size,
                "Qubits": n_qubits,
                "Searchable states N": calculate_padded_size(n_qubits),
                "Recommended k": recommended_analysis_iterations(size),
            }
        )
    with st.expander("Ideal sweep line details"):
        st.dataframe(pd.DataFrame(rows), width="stretch")


def _render_noise_scenario_details(st) -> None:
    """Display the preset values used by the noisy sweep lines."""

    rows = [
        {
            "Scenario": "Low noise",
            "Depolarising": 0.0003,
            "Measurement error": 0.005,
            "T1 (µs)": 250,
            "T2 (µs)": 150,
            "Meaning": "High-quality NISQ conditions",
        },
        {
            "Scenario": "Medium noise",
            "Depolarising": 0.001,
            "Measurement error": 0.01,
            "T1 (µs)": 120,
            "T2 (µs)": 80,
            "Meaning": "Typical superconducting NISQ setting",
        },
        {
            "Scenario": "High noise",
            "Depolarising": 0.005,
            "Measurement error": 0.02,
            "T1 (µs)": 50,
            "T2 (µs)": 30,
            "Meaning": "Shorter coherence and stronger readout/gate noise",
        },
        {
            "Scenario": "Extreme noise",
            "Depolarising": 0.03,
            "Measurement error": 0.05,
            "T1 (µs)": 15,
            "T2 (µs)": 10,
            "Meaning": "Severe experimental degradation",
        },
        {
            "Scenario": "Selected setting",
            "Depolarising": "sidebar value",
            "Measurement error": "sidebar value",
            "T1 (µs)": "sidebar value",
            "T2 (µs)": "sidebar value",
            "Meaning": "Current user-selected hardware/noise setting",
        },
    ]
    with st.expander("Noise scenario settings"):
        st.dataframe(pd.DataFrame(rows), width="stretch")


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


def _render_analysis(st, config: GroverConfig, result) -> dict:
    """Render research-oriented iteration and noise robustness analysis."""

    if result.stopped_before_quantum_execution:
        return {}

    st.header("Advanced NISQ Robustness Analysis")
    st.write(
        "This section analyses how realistic NISQ noise and decoherence mechanisms affect Grover amplification stability. "
        "The focus is on how target probability changes with iteration count, dataset size, circuit depth, T1 relaxation, "
        "T2 dephasing, depolarising noise, and measurement error."
    )
    max_iterations = _iteration_sweep_max_iterations(result)

    st.subheader("Grover Robustness Under Noise and Decoherence")
    ideal_sizes = [16, 32, 64, 128, 256]
    ideal_sweep = ideal_grover_sweep_curves(ideal_sizes, max_iterations=max_iterations)
    st.subheader("Ideal Grover Sweep Analysis")
    st.pyplot(
        plot_iteration_overlay_curves(
            ideal_sweep,
            title="Ideal Grover Sweep Analysis",
            adaptive_y=False,
        )
    )
    st.caption(
        "The ideal sweep shows the expected oscillatory behaviour of Grover’s Algorithm. "
        "Target probability increases up to an optimal iteration count and then decreases due to over-rotation. "
        "This demonstrates why more Grover iterations are not always better."
    )
    _render_sweep_notation_notes(st)
    _render_ideal_sweep_details(st, ideal_sizes)

    sweep_config = replace(config, shots=min(config.shots, 512))
    st.subheader("Noisy Grover Sweep Analysis")
    with st.spinner("Running noisy robustness sweep..."):
        noisy_sweep = run_noisy_condition_iteration_overlay(
            sweep_config,
            max_iterations=max_iterations,
        )
    noisy_ymax = _adaptive_noisy_ymax(noisy_sweep)
    st.pyplot(
        plot_iteration_overlay_curves(
            noisy_sweep,
            title="Noisy Grover Sweep Analysis",
            adaptive_y=True,
        )
    )
    st.caption(
        "The noisy sweep shows how decoherence, gate noise, and measurement error suppress the Grover amplification peak. "
        "The adaptive y-axis highlights differences between noisy conditions that would otherwise be hidden when compared directly with the ideal curve."
    )
    _render_noise_scenario_details(st)

    st.subheader("T1 Sensitivity Analysis")
    with st.spinner("Running T1 iteration overlay..."):
        t1_overlay = run_t1_iteration_overlay(
            sweep_config,
            [500.0, 250.0, 120.0, 50.0, 15.0],
            max_iterations=max_iterations,
        )
    st.pyplot(
        plot_iteration_overlay_curves(
            t1_overlay,
            title="Grover Amplification Under T1 Relaxation",
            adaptive_y=True,
        )
    )
    st.caption(
        "This graph shows amplitude decay and target peak suppression as T1 becomes shorter while T2, depolarising noise, and measurement error are fixed."
    )

    st.subheader("T2 Sensitivity Analysis")
    t2_values = [
        value
        for value in [100.0, 80.0, 60.0, 40.0, 20.0]
        if value <= min(300.0, 2 * config.noise_config.t1_relaxation_us)
    ]
    if len(t2_values) < 5:
        t2_values = sorted(
            {
                max(5.0, round((min(100.0, 2 * config.noise_config.t1_relaxation_us) * index) / 5, 1))
                for index in range(1, 6)
            },
            reverse=True,
        )
    with st.spinner("Running T2 iteration overlay..."):
        t2_overlay = run_t2_iteration_overlay(
            sweep_config,
            t2_values,
            max_iterations=max_iterations,
        )
    st.pyplot(
        plot_iteration_overlay_curves(
            t2_overlay,
            title="Grover Amplification Under T2 Dephasing",
            adaptive_y=True,
        )
    )
    st.caption(
        "This graph shows interference collapse and flattening of the Grover amplification peak as T2 becomes shorter while T1, depolarising noise, and measurement error are fixed. "
        "T2 dephasing often suppresses and flattens the amplification peak more than it shifts the optimal iteration, so several T2 curves may peak at the same k while still showing different peak heights."
    )

    with st.spinner("Running current-setting iteration comparison..."):
        iteration_results = run_iteration_sweep(config, max_iterations=max_iterations)
    st.subheader("Difference From Ideal")
    probability_loss = probability_loss_by_iteration(iteration_results)
    if probability_loss:
        st.pyplot(plot_probability_loss(probability_loss))
        st.caption(
            "This graph visualises quantum advantage degradation: Delta P is the ideal target probability minus the noisy target probability. "
            "Near-zero Delta P at over-rotation troughs means both ideal and noisy target probabilities are low at the same iteration. It does not mean noise has disappeared or quantum advantage has recovered."
        )

    st.subheader("Research Summary")
    research_summary = build_research_summary(result, iteration_results)
    st.dataframe(_research_summary_table(research_summary), width="stretch")

    st.subheader("Ideal vs Noisy Iteration Comparison")
    st.caption(
        "This compact comparison keeps the current selected noisy setting next to the ideal curve for quick inspection."
    )
    st.dataframe(pd.DataFrame(iteration_results), width="stretch")
    st.pyplot(plot_iteration_sweep(iteration_results))

    st.subheader("Classical vs Grover Comparison")
    st.caption(
        "This is a theoretical query-scaling comparison on logarithmic axes. "
        "It can include very large datasets because it does not build or simulate those quantum circuits."
    )
    comparison = [
        compare_classical_vs_grover(size)
        for size in _classical_comparison_sizes(result.mapping.n_items)
    ]
    st.dataframe(pd.DataFrame(comparison), width="stretch")
    st.pyplot(plot_classical_vs_grover(comparison))

    return {
        "ideal_sweep_data": ideal_sweep,
        "noisy_sweep_data": noisy_sweep,
        "t1_sensitivity_data": t1_overlay,
        "t2_sensitivity_data": t2_overlay,
        "iteration_comparison_data": iteration_results,
        "probability_loss_data": probability_loss,
        "research_summary": research_summary,
        "ideal_peak_summary": _peak_summary_by_scenario(ideal_sweep),
        "noisy_peak_summary": _peak_summary_by_scenario(noisy_sweep),
        "t1_peak_summary": _peak_summary_by_scenario(t1_overlay),
        "t2_peak_summary": _peak_summary_by_scenario(t2_overlay),
        "noisy_adaptive_y_axis_max": noisy_ymax,
    }


def _render_advanced_robustness_analysis(
    st,
    config: GroverConfig,
    result,
    max_iterations: int,
) -> None:
    """Render advanced NISQ robustness analyses."""

    st.header("Multi-condition T1/T2 Iteration Overlay Analysis")
    with st.spinner("Running decoherence overlay analysis..."):
        overlay_results = run_decoherence_iteration_overlay(
            config,
            max_iterations=min(max_iterations, MAX_PUBLIC_ITERATION_SWEEP_POINTS - 1),
        )
    st.pyplot(plot_decoherence_iteration_overlay(overlay_results))
    st.caption(
        "This graph demonstrates peak suppression, interference collapse, over-rotation, "
        "decoherence sensitivity, optimal iteration shifts, and amplification instability "
        "under increasing T1/T2 decoherence."
    )

    st.header("T1/T2 Sensitivity Analysis")
    col1, col2 = st.columns(2)
    with col1:
        t1_values = [15.0, 30.0, 50.0, 120.0, 250.0, 500.0]
        with st.spinner("Running T1 sweep..."):
            t1_results = run_t1_sweep(config, t1_values)
        st.pyplot(
            plot_coherence_sweep(
                t1_results,
                x_key="t1_relaxation_us",
                title="T1 Relaxation Sweep",
                x_label="T1 relaxation time (µs)",
            )
        )
        st.caption(
            "This graph demonstrates how amplitude relaxation reduces state preservation. "
            "Shorter T1 values increase decay toward lower-energy states and weaken target-state amplification."
        )
    with col2:
        max_t2 = min(300.0, 2 * config.noise_config.t1_relaxation_us)
        candidate_t2_values = [10.0, 30.0, 50.0, 80.0, 150.0, 300.0]
        t2_values = [value for value in candidate_t2_values if value <= max_t2]
        if max_t2 not in t2_values:
            t2_values.append(max_t2)
        with st.spinner("Running T2 sweep..."):
            t2_results = run_t2_sweep(config, sorted(set(t2_values)))
        st.pyplot(
            plot_coherence_sweep(
                t2_results,
                x_key="t2_coherence_us",
                title="T2 Coherence Sweep",
                x_label="T2 coherence time (µs)",
            )
        )
        st.caption(
            "This graph demonstrates how phase decoherence weakens the interference pattern required for Grover amplification. "
            "As T2 dephasing increases, the target probability peak becomes lower and flatter."
        )


def _render_scalability_analysis(st, current_dataset_size: int) -> None:
    """Render simple NISQ scalability growth estimates."""

    st.header("Scalability and Circuit Growth")
    sizes = sorted(
        {
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            min(1024, max(1, current_dataset_size)),
        }
    )
    rows = scalability_analysis(sizes)
    st.dataframe(pd.DataFrame(rows), width="stretch")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(
            plot_scalability_growth(
                rows,
                y_key="n_qubits",
                title="Dataset Size vs Required Qubits",
                y_label="Required qubits",
            )
        )
    with col2:
        st.pyplot(
            plot_scalability_growth(
                rows,
                y_key="estimated_circuit_depth",
                title="Dataset Size vs Estimated Depth",
                y_label="Estimated circuit depth",
            )
        )
    with col3:
        st.pyplot(
            plot_scalability_growth(
                rows,
                y_key="estimated_total_gates",
                title="Dataset Size vs Estimated Gates",
                y_label="Estimated total gates",
            )
        )
    st.caption(
        "These estimates support NISQ scalability discussion. Classical simulation cost grows quickly as qubit count increases."
    )


def _render_exports(st, result, analysis_bundle: dict | None = None) -> None:
    """Render JSON, CSV, and Markdown download buttons."""

    st.header("Export Results")
    json_data = _export_text(result, "result.json", export_result_json)
    csv_data = _export_text(result.ideal_counts, "counts.csv", export_counts_csv)
    markdown_data = _export_text(result, "student_report.md", export_student_report_markdown)
    analysis_data = json.dumps(
        {
            "dataset_size": result.mapping.n_items,
            "qubits": result.mapping.n_qubits,
            "circuit_depth": result.circuit_depth,
            "total_gates": sum(int(value) for value in result.gate_counts.values()),
            "t1_relaxation_us": result.config.noise_config.t1_relaxation_us,
            "t2_coherence_us": result.config.noise_config.t2_coherence_us,
            "depolar_prob": result.config.noise_config.depolar_prob,
            "measurement_error_prob": result.config.noise_config.measurement_error_prob,
            **(analysis_bundle or {}),
        },
        indent=2,
        default=str,
    )

    col1, col2, col3, col4 = st.columns(4)
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
    col4.download_button(
        "Download Analysis JSON",
        data=analysis_data,
        file_name="grover_robustness_analysis.json",
        mime="application/json",
    )


def _export_text(payload, filename: str, exporter) -> str:
    """Run a file-based exporter and return the written text."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        exporter(payload, path)
        return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
