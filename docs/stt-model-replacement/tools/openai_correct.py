"""Correct phone transcripts with OpenAI while preserving each source transcript.

Set ``OPENAI_API_KEY`` and optionally ``OPENAI_MODEL`` in the current shell.
Only transcript text is sent to the API; audio and the API key stay off-device.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI

SYSTEM_PROMPT = (
    "Correct transcription errors only. Preserve the original meaning. "
    "Do not add information or infer words that are not supported by the input."
)
OUTPUT_FORMAT = {
    "type": "json_schema",
    "name": "corrected_transcript",
    "schema": {
        "type": "object",
        "properties": {"corrected_text": {"type": "string"}},
        "required": ["corrected_text"],
        "additionalProperties": False,
    },
    "strict": True,
}
OUTPUT_FIELDS = (
    "file",
    "stt",
    "hypothesis",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "status",
)


def correct(client: OpenAI, model: str, text: str) -> dict[str, Any]:
    """Send one transcript and return corrected text, latency, and token usage."""
    start = time.perf_counter()
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=text,
        reasoning={"effort": "low"},
        text={"format": OUTPUT_FORMAT},
        max_output_tokens=2048,
        store=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    if response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details else "unknown"
        raise ValueError(f"answer incomplete ({reason})")
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    raise ValueError(f"the LLM declined the request: {part.refusal}")
    corrected_text = json.loads(response.output_text)["corrected_text"].strip()
    if not corrected_text:
        raise ValueError("empty answer")
    return {
        "text": corrected_text,
        "latency_ms": latency_ms,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def failure_status(error: Exception, model: str) -> str:
    """Translate expected request failures into the workshop status column."""
    if isinstance(error, openai.AuthenticationError):
        return "error 401: wrong API key"
    if isinstance(error, openai.PermissionDeniedError):
        return "error 403: key has no access"
    if isinstance(error, openai.NotFoundError):
        return f"error: model '{model}' not found"
    if isinstance(error, openai.RateLimitError):
        if error.code == "insufficient_quota":
            return "error 429: no API credit (insufficient_quota)"
        return "error 429: rate limit (retries used up)"
    if isinstance(error, openai.APIStatusError):
        if error.code == "model_not_found":
            return f"error: model '{model}' not found"
        return f"error {error.status_code}"
    if isinstance(error, openai.APIConnectionError):
        return f"network error: {error}"
    if isinstance(error, (ValueError, KeyError, json.JSONDecodeError)):
        return f"bad answer: {error}"
    if isinstance(error, openai.OpenAIError):
        return f"API error: {error}"
    raise error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone_tsv", type=Path, help="Phone benchmark result TSV.")
    parser.add_argument("out_tsv", type=Path, help="New output TSV (never overwritten).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_tsv.exists():
        raise SystemExit(f"{args.out_tsv} already exists. Choose a new output name.")
    try:
        with args.phone_tsv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"file", "hypothesis"}.issubset(reader.fieldnames or []):
                raise SystemExit("Input TSV must have 'file' and 'hypothesis' columns.")
            phone_rows = list(reader)
    except OSError as error:
        raise SystemExit(f"Cannot read {args.phone_tsv}: {error}") from None
    if not phone_rows:
        raise SystemExit("Input TSV has no transcript rows.")

    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in this PowerShell session first.")
    client = OpenAI(timeout=30.0, max_retries=2)
    rows: list[dict[str, str]] = []
    for phone_row in phone_rows:
        original = phone_row["hypothesis"] or ""
        result: dict[str, Any] = {
            "text": original,
            "latency_ms": 0,
            "input_tokens": "",
            "output_tokens": "",
        }
        status = "ok"
        if not original.strip():
            status = "skipped: empty transcript"
        else:
            try:
                result = correct(client, model, original)
            except (
                openai.OpenAIError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as error:
                status = failure_status(error, model)
        rows.append(
            {
                "file": phone_row["file"],
                "stt": original,
                "hypothesis": result["text"],
                "latency_ms": f"{result['latency_ms']:.0f}",
                "input_tokens": str(result["input_tokens"]),
                "output_tokens": str(result["output_tokens"]),
                "status": status,
            }
        )
        print(f"{phone_row['file']}  {status}  {result['latency_ms']:.0f} ms")

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(row["status"] != "ok" for row in rows)
    print(f"Saved: {args.out_tsv} ({len(rows)} files, {failed} failed)")


if __name__ == "__main__":
    main()
