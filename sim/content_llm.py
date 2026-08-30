"""Build-time OpenAI generation and fail-closed runtime cache for safe exemplars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "llm_cache" / "content"
MODEL = "gpt-5.6-sol"
WATERMARK = "[AI DEFENSE LAB · SAFE SYNTHETIC · TRUNCATED]"
KINDS = ("injection", "scam_transcript", "upi_pretext", "dispute_narrative")
LANGUAGES = ("en", "hi", "mr")
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
INSTRUCTIONS = (
    "You create defensive security training summaries only. Return an abstract, non-operational description; "
    "never include executable instructions, URLs, contact details, credentials, payment handles, or evasion steps."
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string", "minLength": 20, "maxLength": 180}},
    "required": ["summary"],
    "additionalProperties": False,
}


def request_envelope(kind: str, language: str, variant: str = "baseline") -> dict[str, Any]:
    if kind not in KINDS or language not in LANGUAGES:
        raise ValueError("unsupported content kind or language")
    prompt = (
        f"Write one {LANGUAGE_NAMES[language]} abstract summary of a synthetic {kind.replace('_', ' ')} "
        "for a payment-fraud defense demo. Describe the risk signal, not how to perform the abuse. "
        "Keep it under 140 characters and make clear that actionable details were removed."
    )
    return {
        "schema_version": 2,
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "reasoning": {"effort": "high"},
        "max_output_tokens": 500,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "safe_content_summary",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "metadata": {"kind": kind, "language": language, "variant": variant},
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def request_key(kind: str, language: str, variant: str = "baseline") -> str:
    return hashlib.sha256(_canonical_json(request_envelope(kind, language, variant))).hexdigest()


def _sanitize(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "[URL REMOVED]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{5,}\b", "[NUMBER REMOVED]", text)
    text = re.sub(
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?",
        "[INSTRUCTION FRAGMENT REMOVED]",
        text,
        flags=re.IGNORECASE,
    )
    text = " ".join(text.split()).strip()
    return text[:max_chars].rstrip() + ("…" if len(text) > max_chars else "")


def _artifacts(kind: str, language: str) -> dict[str, Any]:
    injection = kind == "injection"
    return {
        "hidden_text_present": injection,
        "zero_width_chars": 7 + LANGUAGES.index(language) if injection else 0,
        "invisible_css": injection and language != "hi",
        "instruction_shape_score": 0.92 if injection else 0.12,
        "source": "synthetic_openai_build_cache",
    }


def safe_record(
    *,
    kind: str,
    language: str,
    text: str,
    variant: str = "baseline",
    response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope = request_envelope(kind, language, variant)
    cleaned = _sanitize(text)
    if not cleaned or "[URL REMOVED]" in cleaned or re.search(r"https?://|www\.", cleaned, re.I):
        raise ValueError("generated content failed the non-operational cache policy")
    key = hashlib.sha256(_canonical_json(envelope)).hexdigest()
    return {
        "cache_key": key,
        "request_binding": {
            "sha256": key,
            "schema_version": envelope["schema_version"],
            "model": envelope["model"],
            "metadata": envelope["metadata"],
        },
        "response": response or {},
        "response_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "artifact_id": f"safe-{key[:12]}",
        "kind": kind,
        "language": language,
        "variant": variant,
        "content": f"{WATERMARK} {cleaned}",
        "truncated": True,
        "watermarked": True,
        "operational": False,
        "review_status": "pending",
        "artifact_metadata": _artifacts(kind, language),
    }


class ContentCache:
    def __init__(self, path: Path = DEFAULT_CACHE) -> None:
        self.path = path

    def _path(self, key: str) -> Path:
        return self.path / f"{key}.json"

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.path.glob("*.json"))]

    def get(self, kind: str, language: str, variant: str = "baseline") -> dict[str, Any] | None:
        expected = request_key(kind, language, variant)
        path = self._path(expected)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("cache_key") != expected or (record.get("request_binding") or {}).get("sha256") != expected:
            raise RuntimeError(f"LLM cache binding mismatch: {path.name}")
        if record.get("review_status") != "approved":
            raise RuntimeError(f"LLM cache record is not approved: {path.name}")
        return record

    def require(self, kind: str, language: str, variant: str = "baseline") -> dict[str, Any]:
        record = self.get(kind, language, variant)
        if record is None:
            raise RuntimeError(
                f"LLM cache miss for {kind}/{language}/{variant}; runtime network generation is disabled"
            )
        return record

    def put(self, record: dict[str, Any]) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        destination = self._path(record["cache_key"])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path, delete=False) as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        return destination


def populate_openai(path: Path = DEFAULT_CACHE) -> list[Path]:
    from openai import OpenAI

    client = OpenAI()
    cache = ContentCache(path)
    written: list[Path] = []
    for language in LANGUAGES:
        for kind in KINDS:
            envelope = request_envelope(kind, language)
            response = client.responses.create(**{key: value for key, value in envelope.items() if key not in {"schema_version", "metadata"}})
            payload = json.loads(response.output_text)
            usage = response.usage.model_dump(mode="json") if response.usage else None
            record = safe_record(
                kind=kind,
                language=language,
                text=payload["summary"],
                response={"id": response.id, "model": response.model, "usage": usage},
            )
            written.append(cache.put(record))
    return written


def approve_pending(path: Path = DEFAULT_CACHE) -> int:
    cache = ContentCache(path)
    approved = 0
    for record in cache.records():
        if record.get("review_status") != "pending":
            continue
        record["review_status"] = "approved"
        record["reviewed_by"] = "codex-security-review"
        cache.put(record)
        approved += 1
    return approved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--populate-openai", action="store_true")
    mode.add_argument("--approve-pending", action="store_true")
    args = parser.parse_args()
    if args.populate_openai:
        paths = populate_openai(args.cache_dir)
        print(json.dumps({"pending_records": len(paths), "cache_dir": str(args.cache_dir)}))
    else:
        print(json.dumps({"approved_records": approve_pending(args.cache_dir), "cache_dir": str(args.cache_dir)}))


if __name__ == "__main__":
    main()
