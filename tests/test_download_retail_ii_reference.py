from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

import scripts.download_retail_ii_reference as retail_ii


def _archive(member: str, workbook: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, workbook)
    return buffer.getvalue()


def _pin(monkeypatch: pytest.MonkeyPatch, archive: bytes, workbook: bytes) -> None:
    monkeypatch.setattr(retail_ii, "ARCHIVE_SIZE_BYTES", len(archive))
    monkeypatch.setattr(retail_ii, "WORKBOOK_SIZE_BYTES", len(workbook))
    monkeypatch.setattr(retail_ii, "ARCHIVE_SHA256", hashlib.sha256(archive).hexdigest())
    monkeypatch.setattr(retail_ii, "WORKBOOK_SHA256", hashlib.sha256(workbook).hexdigest())
    monkeypatch.setattr(retail_ii, "urlopen", lambda *_args, **_kwargs: io.BytesIO(archive))


def test_fake_download_then_cache_tamper_fails_closed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    workbook = b"pinned-workbook-bytes"
    archive = _archive(retail_ii.ARCHIVE_MEMBER, workbook)
    _pin(monkeypatch, archive, workbook)
    output = tmp_path / "online_retail_ii.xlsx"

    manifest = retail_ii.download(output)

    assert output.read_bytes() == workbook
    assert manifest["readiness_slice"] == retail_ii.READINESS_SLICE
    assert retail_ii.verify_existing(output) == manifest
    output.write_bytes(b"X" + workbook[1:])
    with pytest.raises(ValueError, match="workbook bytes"):
        retail_ii.verify_existing(output)


def test_download_rejects_unexpected_zip_member(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    workbook = b"pinned-workbook-bytes"
    archive = _archive("../online_retail_II.xlsx", workbook)
    _pin(monkeypatch, archive, workbook)

    with pytest.raises(ValueError, match="member contract"):
        retail_ii.download(tmp_path / "online_retail_ii.xlsx")
