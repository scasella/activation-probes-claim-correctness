from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import load_config
from ..evaluation.report_tables import markdown_table
from ..io import write_json
from ..utils import ensure_parent


def render_report(
    title: str,
    metrics_rows: list[dict[str, Any]],
    representative_examples: list[dict[str, Any]],
    threats_to_validity: list[str],
    output_path: Path,
) -> str:
    report_cfg = load_config("report.yaml")
    body = [
        f"# {title}",
        "",
        "## Main Results",
        "",
        markdown_table(metrics_rows),
        "",
        "## Representative Examples",
        "",
        markdown_table(representative_examples),
        "",
        "## Threats to Validity",
        "",
    ]
    body.extend(f"- {item}" for item in threats_to_validity)
    body.append("")
    body.append("## Required Sections Checklist")
    body.append("")
    body.extend(f"- {section}" for section in report_cfg["sections"])
    report_text = "\n".join(body)
    ensure_parent(output_path)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def write_decision_memo(payload: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, payload)
