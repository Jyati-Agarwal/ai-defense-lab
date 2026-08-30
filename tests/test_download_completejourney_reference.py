from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

import scripts.download_completejourney_reference as completejourney


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str = completejourney.SOURCE_URL) -> None:
        super().__init__(payload)
        self._url = url

    def geturl(self) -> str:
        return self._url


def _git_blob(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False).hexdigest()


def _pin(monkeypatch: pytest.MonkeyPatch, payload: bytes, *, frozen: bool = True) -> None:
    monkeypatch.setattr(completejourney, "RDS_SIZE_BYTES", len(payload))
    monkeypatch.setattr(completejourney, "GIT_BLOB_SHA1", _git_blob(payload))
    monkeypatch.setattr(
        completejourney,
        "RDS_SHA256",
        hashlib.sha256(payload).hexdigest() if frozen else None,
    )
    monkeypatch.setattr(completejourney, "urlopen", lambda *_args, **_kwargs: _Response(payload))


def _lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> tuple[Path, Path]:
    _pin(monkeypatch, payload, frozen=False)
    preregistration = tmp_path / "source-preregistration.json"
    receipt = tmp_path / "quarantine-receipt.json"
    completejourney.create_source_preregistration(preregistration)
    completejourney.derive_sha256(
        preregistration,
        receipt,
        completejourney.DERIVATION_CONFIRMATION,
    )
    monkeypatch.setattr(completejourney, "RDS_SHA256", hashlib.sha256(payload).hexdigest())
    return preregistration, receipt


def test_missing_sha_pin_blocks_before_network_or_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(completejourney, "RDS_SHA256", None)
    monkeypatch.setattr(
        completejourney,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not open before the SHA-256 pin is frozen"),
    )
    output = tmp_path / "missing" / "transactions.rds"

    with pytest.raises(RuntimeError, match="authorized derivation"):
        completejourney.download(output)

    assert not output.parent.exists()


def test_source_selection_preregistration_is_network_free_exact_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        completejourney,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("source preregistration must not open the network"),
    )
    path = tmp_path / "source-preregistration.json"

    payload = completejourney.create_source_preregistration(path)

    assert payload["sha256_state"]["value"] is None
    assert payload["real_data_opened"] is False
    assert payload["parser_invoked"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == completejourney.SOURCE_PREREGISTRATION_SHA256
    assert path.stat().st_mode & 0o222 == 0
    path.chmod(0o644)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        completejourney.create_source_preregistration(path)


def test_fake_download_is_commit_pinned_atomic_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic-rds-not-the-real-dataset"
    preregistration, receipt = _lineage(tmp_path, monkeypatch, payload)
    output = tmp_path / "transactions.rds"

    manifest = completejourney.download(
        output,
        preregistration_path=preregistration,
        receipt_path=receipt,
    )

    assert output.read_bytes() == payload
    assert manifest["source_url"].split("/")[5] == completejourney.GIT_COMMIT
    assert manifest["git_blob_sha1"] == _git_blob(payload)
    assert completejourney.verify_existing(
        output,
        preregistration_path=preregistration,
        receipt_path=receipt,
    ) == manifest
    output.chmod(0o644)
    output.write_bytes(b"X" + payload[1:])
    with pytest.raises(ValueError, match="differs"):
        completejourney.verify_existing(output)


def test_redirect_and_failed_replacement_are_rejected_without_destroying_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned"
    preregistration, receipt = _lineage(tmp_path, monkeypatch, payload)
    output = tmp_path / "transactions.rds"
    output.write_bytes(b"old-rds")
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"old": True}), encoding="utf-8")
    monkeypatch.setattr(
        completejourney,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload, "https://example.invalid/transactions.rds"),
    )

    with pytest.raises(ValueError, match="redirected"):
        completejourney.download(
            output,
            preregistration_path=preregistration,
            receipt_path=receipt,
        )

    assert output.read_bytes() == b"old-rds"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"old": True}


def test_sha_derivation_requires_exact_confirmation_and_retains_no_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"derive-only"
    _pin(monkeypatch, payload, frozen=False)
    monkeypatch.setattr(completejourney.tempfile, "tempdir", str(tmp_path))
    preregistration = tmp_path / "source-preregistration.json"
    receipt_path = tmp_path / "quarantine-receipt.json"
    completejourney.create_source_preregistration(preregistration)
    fake_opener = completejourney.urlopen
    calls = 0

    def forbidden_opener(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return fake_opener(*_args, **_kwargs)

    monkeypatch.setattr(completejourney, "urlopen", forbidden_opener)

    with pytest.raises(PermissionError, match="exact"):
        completejourney.derive_sha256(preregistration, receipt_path, "yes")
    assert calls == 0
    receipt = completejourney.derive_sha256(
        preregistration,
        receipt_path,
        completejourney.DERIVATION_CONFIRMATION,
    )
    assert calls == 1
    assert receipt["observed_rds_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["parser_invoked"] is False
    assert receipt["dataset_bytes_retained"] is False
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "quarantine-receipt.json",
        "source-preregistration.json",
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        completejourney.derive_sha256(
            preregistration,
            receipt_path,
            completejourney.DERIVATION_CONFIRMATION,
        )
