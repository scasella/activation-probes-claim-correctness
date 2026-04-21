from __future__ import annotations

from typing import Any


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows_"
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = ["| " + " | ".join(str(row.get(header, "")) for header in headers) + " |" for row in rows]
    return "\n".join([header_row, sep_row, *body_rows])
