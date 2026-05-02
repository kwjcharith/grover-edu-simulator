"""FastAPI entry point for GroverLab online deployment."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from groverlab.grover_analysis import run_iteration_sweep, run_noise_sweep
from groverlab.grover_config import GroverConfig, NoiseConfig
from groverlab.grover_education import educational_overview
from groverlab.grover_export import export_student_report_markdown
from groverlab.grover_runner import run_grover_simulation


class SimulationRequest(BaseModel):
    """API request body for a GroverLab simulation."""

    dataset_items: list[str]
    target_item: str
    shots: int = 1024
    iterations: int | None = None
    seed: int | None = None
    noise_enabled: bool = False
    depolar_prob: float = 0.0
    measurement_error_prob: float = 0.0
    gate_error_prob: float = 0.0
    missing_target_mode: str = "stop"


class IterationSweepRequest(SimulationRequest):
    """Request body for an iteration sweep."""

    max_iterations: int | None = None


class NoiseSweepRequest(SimulationRequest):
    """Request body for a noise sweep."""

    noise_values: list[float]


class ExportReportRequest(SimulationRequest):
    """Request body for Markdown report export."""

    filename: str = "grover_student_report.md"


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""

    app = FastAPI(
        title="GroverLab API",
        version="0.1.0",
        description="API for GroverLab educational Grover's Algorithm simulations.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "groverlab"}

    @app.get("/education")
    def education() -> dict:
        return educational_overview()

    @app.post("/simulate")
    def simulate(request: SimulationRequest) -> dict[str, Any]:
        config = _config_from_request(request)
        result = _run_or_400(config)
        return _result_to_response(result)

    @app.post("/simulate-noisy")
    def simulate_noisy(request: SimulationRequest) -> dict[str, Any]:
        config = _config_from_request(request, force_noise=True)
        result = _run_or_400(config)
        return _result_to_response(result)

    @app.post("/iteration-sweep")
    def iteration_sweep(request: IterationSweepRequest) -> dict[str, Any]:
        config = _config_from_request(request)
        try:
            rows = run_iteration_sweep(config, max_iterations=request.max_iterations)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"results": _json_safe(rows)}

    @app.post("/noise-sweep")
    def noise_sweep(request: NoiseSweepRequest) -> dict[str, Any]:
        config = _config_from_request(request)
        try:
            rows = run_noise_sweep(config, noise_values=request.noise_values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"results": _json_safe(rows)}

    @app.post("/export-report")
    def export_report(request: ExportReportRequest) -> dict[str, Any]:
        config = _config_from_request(request)
        result = _run_or_400(config)
        return {
            "filename": request.filename,
            "mime_type": "text/markdown",
            "content": _render_markdown_report(result, request.filename),
            "target_found": result.target_found,
            "stopped_before_quantum_execution": result.stopped_before_quantum_execution,
            "warnings": result.warnings,
        }

    return app


def _config_from_request(
    request: SimulationRequest,
    force_noise: bool = False,
) -> GroverConfig:
    """Build a validated GroverConfig from an API request."""

    try:
        noise_enabled = request.noise_enabled or force_noise
        return GroverConfig(
            dataset_items=request.dataset_items,
            target_item=request.target_item,
            shots=request.shots,
            iterations=request.iterations,
            seed=request.seed,
            noise_config=NoiseConfig(
                noise_enabled=noise_enabled,
                depolar_prob=request.depolar_prob,
                measurement_error_prob=request.measurement_error_prob,
                gate_error_prob=request.gate_error_prob,
            ),
            missing_target_mode=request.missing_target_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_or_400(config: GroverConfig):
    """Run the shared Grover engine and convert validation errors to HTTP 400."""

    try:
        return run_grover_simulation(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _result_to_response(result) -> dict[str, Any]:
    """Convert a GroverResult to a JSON-compatible response dictionary."""

    return {
        "target_found": result.target_found,
        "stopped_before_quantum_execution": result.stopped_before_quantum_execution,
        "missing_target_explanation": result.missing_target_explanation,
        "warnings": result.warnings,
        "mapping": {
            "n_items": result.mapping.n_items,
            "padded_size": result.mapping.padded_size,
            "n_qubits": result.mapping.n_qubits,
            "target_found": result.mapping.target_found,
            "target_index": result.mapping.target_index,
            "target_binary": result.mapping.target_binary,
            "unused_states": result.mapping.unused_states,
            "missing_target_explanation": result.mapping.missing_target_explanation,
        },
        "ideal_counts": result.ideal_counts,
        "noisy_counts": result.noisy_counts,
        "success_probability": result.success_probability,
        "noisy_success_probability": result.noisy_success_probability,
        "measured_bitstring": result.measured_bitstring,
        "decoded_item": result.decoded_item,
        "found": result.found,
        "circuit_depth": result.circuit_depth,
        "gate_counts": result.gate_counts,
        "runtime_seconds": result.runtime_seconds,
        "explanation_steps": result.explanation_steps,
    }


def _render_markdown_report(result, filename: str) -> str:
    """Render a student report through the shared export system."""

    safe_name = Path(filename).name or "grover_student_report.md"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / safe_name
        export_student_report_markdown(result, path)
        return path.read_text(encoding="utf-8")


def _json_safe(value: Any) -> Any:
    """Recursively convert values to JSON-compatible Python objects."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


app = create_app()
