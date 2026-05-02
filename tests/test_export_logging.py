import json

from groverlab.grover_export import export_json, export_markdown_report
from groverlab.grover_logging import anonymize_log_payload, append_research_log


def test_export_json_writes_payload(tmp_path):
    output = export_json({"target": "beta"}, tmp_path / "result.json")

    assert json.loads(output.read_text(encoding="utf-8")) == {"target": "beta"}


def test_export_markdown_report_writes_sections(tmp_path):
    output = export_markdown_report("Run", {"Summary": "Done"}, tmp_path / "report.md")

    assert "# Run" in output.read_text(encoding="utf-8")


def test_anonymize_log_payload_drops_unapproved_fields():
    payload = anonymize_log_payload({"dataset_size": 4, "raw_items": ["secret"]})

    assert payload == {"dataset_size": 4}


def test_append_research_log_writes_jsonl(tmp_path):
    output = append_research_log({"dataset_size": 4}, tmp_path / "log.jsonl")

    line = output.read_text(encoding="utf-8").strip()
    assert json.loads(line)["dataset_size"] == 4

