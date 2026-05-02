import json
from dataclasses import asdict

from groverlab.grover_config import ExperimentLog, GroverConfig
from groverlab.grover_export import (
    export_counts_csv,
    export_experiment_summary_csv,
    export_json,
    export_markdown_report,
    export_research_log,
    export_result_json,
    export_student_report_markdown,
)
from groverlab.grover_logging import (
    anonymize_log_payload,
    append_log_jsonl,
    append_research_log,
    create_experiment_log,
    create_session_id,
)
from groverlab.grover_runner import run_grover_simulation


def test_export_json_writes_payload(tmp_path):
    output = export_json({"target": "beta"}, tmp_path / "result.json")

    assert json.loads(output.read_text(encoding="utf-8")) == {"target": "beta"}


def test_result_json_export(tmp_path):
    result = _student_report_result()

    output = export_result_json(result, tmp_path / "result.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mapping"]["target_item"] == "banana"
    assert payload["mapping"]["target_binary"] == "10"
    assert payload["success_probability"] == 1.0


def test_counts_csv_export(tmp_path):
    output = export_counts_csv({"10": 12, "00": 4}, tmp_path / "counts.csv")

    text = output.read_text(encoding="utf-8")
    assert "bitstring,count" in text
    assert "10,12" in text


def test_experiment_summary_csv_export(tmp_path):
    result = _student_report_result()

    output = export_experiment_summary_csv([result], tmp_path / "summary.csv")

    text = output.read_text(encoding="utf-8")
    assert "dataset_size" in text
    assert "success_probability" in text
    assert "4" in text


def test_export_markdown_report_writes_sections(tmp_path):
    output = export_markdown_report("Run", {"Summary": "Done"}, tmp_path / "report.md")

    assert "# Run" in output.read_text(encoding="utf-8")


def test_student_report_markdown_export(tmp_path):
    result = _student_report_result()

    output = export_student_report_markdown(result, tmp_path / "student_report.md")

    text = output.read_text(encoding="utf-8")
    assert "# GroverLab Student Report" in text
    assert "Dataset size: 4" in text
    assert "Target item: banana" in text
    assert "Target binary state: |10>" in text
    assert "Step-by-Step Explanation" in text
    assert "Misconception Warnings" in text
    assert "Short Quiz" in text


def test_research_log_export_is_anonymous(tmp_path):
    result = _privacy_test_result()

    output = export_research_log(result, tmp_path / "research.jsonl")

    record = json.loads(output.read_text(encoding="utf-8").strip())
    serialized = json.dumps(record, sort_keys=True)
    assert record["dataset_size"] == 4
    assert record["target_index"] == 1
    assert "bob@example.com" not in serialized
    assert "Alice Smith" not in serialized
    assert "192.168.1.42" not in serialized


def test_create_session_id_returns_anonymous_string():
    session_id = create_session_id()

    assert isinstance(session_id, str)
    assert len(session_id) >= 16


def test_anonymize_log_payload_drops_unapproved_fields():
    payload = anonymize_log_payload(
        {
            "dataset_size": 4,
            "raw_items": ["secret"],
            "email": "student@example.com",
        }
    )

    assert payload == {"dataset_size": 4}


def test_anonymous_experiment_log_creation():
    result = _privacy_test_result()

    log = create_experiment_log(result, session_id="session-test")

    assert isinstance(log, ExperimentLog)
    assert log.session_id == "session-test"
    assert log.dataset_size == 4
    assert log.target_index == 1
    assert log.n_qubits == 2
    assert log.padded_size == 4
    assert log.shots == 64
    assert log.decoded_correctly is True


def test_log_does_not_include_names():
    serialized = _serialized_privacy_log()

    assert "Alice Smith" not in serialized
    assert "Bob Jones" not in serialized


def test_log_does_not_include_emails():
    serialized = _serialized_privacy_log()

    assert "alice@example.com" not in serialized
    assert "bob@example.com" not in serialized


def test_log_does_not_include_student_ids():
    serialized = _serialized_privacy_log()

    assert "S1234567" not in serialized
    assert "student_id" not in serialized


def test_log_does_not_include_ip_address():
    serialized = _serialized_privacy_log()

    assert "192.168.1.42" not in serialized
    assert "ip_address" not in serialized


def test_log_does_not_include_raw_dataset_values():
    serialized = _serialized_privacy_log()

    assert "Confidential Group A" not in serialized
    assert "cleaned_items" not in serialized
    assert "original_items" not in serialized
    assert "target_item" not in serialized


def test_append_log_jsonl_writes_anonymous_log(tmp_path):
    log = create_experiment_log(_privacy_test_result(), session_id="session-test")

    output = append_log_jsonl(log, tmp_path / "log.jsonl")

    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["session_id"] == "session-test"
    assert record["dataset_size"] == 4
    assert "target_item" not in record
    assert "cleaned_items" not in record


def test_append_research_log_writes_jsonl(tmp_path):
    output = append_research_log({"dataset_size": 4, "raw_items": ["secret"]}, tmp_path / "log.jsonl")

    line = output.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["dataset_size"] == 4
    assert "raw_items" not in record


def _privacy_test_result():
    config = GroverConfig(
        dataset_items=[
            "Alice Smith",
            "bob@example.com",
            "S1234567",
            "192.168.1.42 Confidential Group A",
        ],
        target_item="bob@example.com",
        shots=64,
        iterations=1,
        seed=42,
    )
    return run_grover_simulation(config)


def _student_report_result():
    config = GroverConfig(
        dataset_items=["apple", "mango", "banana", "orange"],
        target_item="banana",
        shots=64,
        iterations=1,
        seed=42,
    )
    return run_grover_simulation(config)


def _serialized_privacy_log() -> str:
    log = create_experiment_log(_privacy_test_result(), session_id="anonymous-session")
    return json.dumps(asdict(log), sort_keys=True)
