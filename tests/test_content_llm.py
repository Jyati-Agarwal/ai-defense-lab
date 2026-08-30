import copy

import pytest

from sim.content_llm import (
    DEFAULT_CACHE,
    KINDS,
    LANGUAGES,
    MODEL,
    WATERMARK,
    ContentCache,
    request_key,
    safe_record,
)


def test_committed_multilingual_cache_is_full_hash_keyed_and_approved() -> None:
    records = ContentCache(DEFAULT_CACHE).records()

    assert len(records) == len(KINDS) * len(LANGUAGES) == 12
    assert {(record["kind"], record["language"]) for record in records} == {
        (kind, language) for kind in KINDS for language in LANGUAGES
    }
    for record in records:
        metadata = record["artifact_metadata"]
        assert record["cache_key"] == request_key(record["kind"], record["language"], record["variant"])
        assert record["request_binding"] == {
            "sha256": record["cache_key"],
            "schema_version": 2,
            "model": MODEL,
            "metadata": {"kind": record["kind"], "language": record["language"], "variant": record["variant"]},
        }
        assert "request_envelope" not in record
        assert record["content"].startswith(WATERMARK)
        assert record["truncated"] and record["watermarked"]
        assert record["operational"] is False
        assert record["review_status"] == "approved"
        assert record["reviewed_by"] == "codex-security-review"
        assert {"hidden_text_present", "zero_width_chars", "invisible_css", "instruction_shape_score", "source"} <= metadata.keys()
        assert "http://" not in record["content"].lower()
        assert "https://" not in record["content"].lower()
        assert "ignore previous instructions" not in record["content"].lower()


def test_runtime_cache_miss_and_pending_record_fail_closed(tmp_path) -> None:
    cache = ContentCache(tmp_path)
    with pytest.raises(RuntimeError, match="runtime network generation is disabled"):
        cache.require("injection", "en")

    pending = safe_record(kind="injection", language="en", text="Abstract defensive risk summary with actionable details removed.")
    cache.put(pending)
    with pytest.raises(RuntimeError, match="not approved"):
        cache.require("injection", "en")

    approved = copy.deepcopy(pending)
    approved.update(review_status="approved", reviewed_by="codex-security-review")
    cache.put(approved)
    assert cache.require("injection", "en")["cache_key"] == pending["cache_key"]
