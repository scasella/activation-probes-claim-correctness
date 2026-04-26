from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import requests

from interp_experiment.io import read_jsonl, write_json, write_jsonl


USER_AGENT = "interp-felm-matched-generation-repair/0.1 (+https://openai.com/research)"


def _load_trafilatura() -> Any:
    try:
        import trafilatura
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "trafilatura is required for readability extraction. "
            "Run with: uv run --with trafilatura python scripts/extract_felm_matched_references_v2.py"
        ) from exc
    return trafilatura


def _fetch_html(url: str, *, timeout_sec: float) -> tuple[str | None, str | None]:
    try:
        response = requests.get(
            url,
            timeout=timeout_sec,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        )
        response.raise_for_status()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return response.text, None


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 FELM matched pilot reference cache with readability extraction.")
    parser.add_argument(
        "--input-cache-jsonl",
        type=Path,
        default=Path("data/felm/felm_wk_matched_pilot_reference_cache.jsonl"),
    )
    parser.add_argument(
        "--output-cache-jsonl",
        type=Path,
        default=Path("data/felm/felm_wk_matched_pilot_reference_cache_v2.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/runs/felm_wk_matched_pilot_reference_extraction_v2_summary.json"),
    )
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    args = parser.parse_args()

    trafilatura = _load_trafilatura()
    rows = list(read_jsonl(args.input_cache_jsonl))
    fetched_by_url: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    reference_stats: list[dict[str, Any]] = []

    for row in rows:
        new_refs: list[dict[str, Any]] = []
        for ref in row.get("references", []):
            url = str(ref.get("url") or "").strip()
            original_content = str(ref.get("content") or "").strip()
            original_len = len(original_content)
            if url not in fetched_by_url:
                html, fetch_error = _fetch_html(url, timeout_sec=args.timeout_sec) if url else (None, "missing_url")
                extracted = ""
                extraction_error = None
                if html:
                    try:
                        extracted = (
                            trafilatura.extract(
                                html,
                                url=url,
                                include_comments=False,
                                include_tables=False,
                                favor_recall=True,
                            )
                            or ""
                        ).strip()
                    except Exception as exc:
                        extraction_error = f"{type(exc).__name__}: {exc}"
                fetched_by_url[url] = {
                    "fetch_ok": html is not None,
                    "fetch_error": fetch_error,
                    "extracted_content": extracted,
                    "extracted_content_chars": len(extracted),
                    "extraction_ok": bool(extracted),
                    "extraction_error": extraction_error,
                }

            fetched = fetched_by_url[url]
            extracted_content = str(fetched["extracted_content"])
            extracted_len = int(fetched["extracted_content_chars"])
            use_extracted = extracted_len > original_len
            chosen_content = extracted_content if use_extracted else original_content
            chosen_source = "readability_extraction" if use_extracted else "original_ref_contents"
            if not chosen_content:
                chosen_source = "none"

            stat = {
                "example_id": row.get("example_id"),
                "url": url,
                "fetch_ok": bool(fetched["fetch_ok"]),
                "fetch_error": fetched["fetch_error"],
                "extraction_ok": bool(fetched["extraction_ok"]),
                "extraction_error": fetched["extraction_error"],
                "original_content_chars": original_len,
                "extracted_content_chars": extracted_len,
                "chosen_content_chars": len(chosen_content),
                "chosen_source": chosen_source,
                "used_longer_than_snippet_extraction": use_extracted,
            }
            reference_stats.append(stat)
            new_refs.append(
                {
                    **ref,
                    "content": chosen_content,
                    "content_chars": len(chosen_content),
                    "source": chosen_source,
                    "original_content": original_content,
                    "original_content_chars": original_len,
                    "extracted_content": extracted_content,
                    "extracted_content_chars": extracted_len,
                    "fetch_ok": bool(fetched["fetch_ok"]),
                    "fetch_error": fetched["fetch_error"],
                    "extraction_ok": bool(fetched["extraction_ok"]),
                    "extraction_error": fetched["extraction_error"],
                    "readability_extractor": "trafilatura",
                }
            )
        output_rows.append({**row, "references": new_refs, "reference_cache_version": "felm_matched_pilot_v2"})

    original_lengths = [int(item["original_content_chars"]) for item in reference_stats]
    extracted_lengths = [int(item["extracted_content_chars"]) for item in reference_stats]
    chosen_lengths = [int(item["chosen_content_chars"]) for item in reference_stats]
    n_references = len(reference_stats)
    n_fetch_ok = sum(1 for item in reference_stats if item["fetch_ok"])
    n_extraction_longer = sum(1 for item in reference_stats if item["used_longer_than_snippet_extraction"])
    n_fallback = sum(1 for item in reference_stats if item["chosen_source"] == "original_ref_contents")
    n_failed_fetch = n_references - n_fetch_ok
    n_failed_or_fallback = sum(
        1
        for item in reference_stats
        if (not item["fetch_ok"]) or item["chosen_source"] != "readability_extraction"
    )
    failed_or_fallback_rate = n_failed_or_fallback / n_references if n_references else 0.0
    unique_fetch_ok = sum(1 for item in fetched_by_url.values() if item["fetch_ok"])
    unique_fetch_failed = sum(1 for item in fetched_by_url.values() if not item["fetch_ok"])

    summary = {
        "n_examples": len(rows),
        "n_reference_entries": n_references,
        "n_unique_urls": len([url for url in fetched_by_url if url]),
        "n_unique_urls_successfully_fetched": unique_fetch_ok,
        "n_unique_urls_failed_fetch": unique_fetch_failed,
        "n_reference_entries_successfully_fetched": n_fetch_ok,
        "n_reference_entries_failed_fetch": n_failed_fetch,
        "n_urls_successfully_fetched": n_fetch_ok,
        "n_urls_failed_fetch": n_failed_fetch,
        "fetch_success_rate": n_fetch_ok / n_references if n_references else 0.0,
        "n_longer_than_snippet_extractions": n_extraction_longer,
        "n_fallback_to_original_ref_contents": n_fallback,
        "n_failed_or_fallback_to_snippet": n_failed_or_fallback,
        "failed_or_fallback_rate": failed_or_fallback_rate,
        "fallback_or_failed_rate": failed_or_fallback_rate,
        "repair_materially_improved_coverage": failed_or_fallback_rate <= 0.5,
        "original_content_chars_mean": _mean(original_lengths),
        "original_content_chars_median": _median(original_lengths),
        "extracted_content_chars_mean": _mean(extracted_lengths),
        "extracted_content_chars_median": _median(extracted_lengths),
        "chosen_content_chars_mean": _mean(chosen_lengths),
        "chosen_content_chars_median": _median(chosen_lengths),
        "reference_stats": reference_stats,
        "note": (
            "Each reference was fetched from its FELM URL and passed through trafilatura. "
            "When extracted text was not longer than FELM ref_contents, the original snippet was retained."
        ),
    }

    write_jsonl(args.output_cache_jsonl, output_rows)
    write_json(args.summary_json, summary)
    print(f"Wrote v2 reference cache to {args.output_cache_jsonl}", flush=True)
    print(f"Wrote v2 reference extraction summary to {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
