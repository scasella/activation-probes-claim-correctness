from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    char_start: int
    char_end: int


_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)


def segment_sentences(text: str) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing
        spans.append(TextSpan(text=text[start:end], char_start=start, char_end=end))
    if not spans and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        spans.append(TextSpan(text=text[start:end], char_start=start, char_end=end))
    return spans
