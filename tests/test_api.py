import pytest
from fastapi import HTTPException

from groverlab.api import (
    ExportReportRequest,
    IterationSweepRequest,
    NoiseSweepRequest,
    SimulationRequest,
    create_app,
)


def test_api_health():
    payload = _endpoint("/health")()

    assert payload["status"] == "ok"
    assert payload["service"] == "groverlab"


def test_api_simulate_success_response():
    payload = _endpoint("/simulate")(
        SimulationRequest(
            dataset_items=["apple", "mango", "banana", "orange"],
            target_item="banana",
            shots=64,
            iterations=1,
            seed=42,
        )
    )

    assert payload["target_found"] is True
    assert payload["found"] is True
    assert payload["mapping"]["target_binary"] == "10"
    assert payload["ideal_counts"] == {"10": 64}


def test_api_simulate_noisy_returns_noisy_counts():
    payload = _endpoint("/simulate-noisy")(
        SimulationRequest(
            dataset_items=["apple", "mango", "banana", "orange"],
            target_item="banana",
            shots=64,
            iterations=1,
            seed=42,
            depolar_prob=0.01,
            measurement_error_prob=0.01,
        )
    )

    assert payload["noisy_counts"] is not None
    assert sum(payload["noisy_counts"].values()) == 64


def test_api_iteration_sweep_returns_results():
    payload = _endpoint("/iteration-sweep")(
        IterationSweepRequest(
            dataset_items=["apple", "mango", "banana", "orange"],
            target_item="banana",
            shots=64,
            seed=42,
            max_iterations=2,
        )
    )

    assert [row["iterations"] for row in payload["results"]] == [0, 1, 2]


def test_api_noise_sweep_returns_results():
    payload = _endpoint("/noise-sweep")(
        NoiseSweepRequest(
            dataset_items=["apple", "mango", "banana", "orange"],
            target_item="banana",
            shots=64,
            iterations=1,
            seed=42,
            noise_values=[0.0, 0.01],
        )
    )

    assert [row["noise_value"] for row in payload["results"]] == [0.0, 0.01]


def test_api_export_report_returns_markdown_content():
    payload = _endpoint("/export-report")(
        ExportReportRequest(
            dataset_items=["apple", "mango", "banana", "orange"],
            target_item="banana",
            shots=64,
            iterations=1,
            seed=42,
        )
    )

    assert payload["mime_type"] == "text/markdown"
    assert "# GroverLab Student Report" in payload["content"]
    assert "Target item: banana" in payload["content"]


def test_api_missing_target_stop_response_fields():
    payload = _endpoint("/simulate")(
        SimulationRequest(
            dataset_items=["apple", "banana", "pineapple"],
            target_item="abc",
            missing_target_mode="stop",
        )
    )

    assert payload["target_found"] is False
    assert payload["stopped_before_quantum_execution"] is True
    assert payload["missing_target_explanation"]
    assert payload["ideal_counts"] == {}
    assert payload["found"] is False


def test_api_missing_target_experimental_response_fields():
    payload = _endpoint("/simulate")(
        SimulationRequest(
            dataset_items=["apple", "banana", "pineapple"],
            target_item="abc",
            missing_target_mode="experimental",
            shots=64,
            seed=42,
        )
    )

    assert payload["target_found"] is False
    assert payload["stopped_before_quantum_execution"] is False
    assert payload["ideal_counts"]
    assert payload["success_probability"] == 0.0
    assert payload["found"] is False


def test_api_validation_error_is_clear_http_400():
    with pytest.raises(HTTPException) as exc_info:
        _endpoint("/simulate")(
            SimulationRequest(
                dataset_items=[],
                target_item="banana",
            )
        )

    assert exc_info.value.status_code == 400
    assert "dataset_items must not be empty" in exc_info.value.detail


def _endpoint(path: str):
    app = create_app()
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"Missing {path} route")
