"""Acquire the commit-pinned official Complete Journey transactions RDS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen


VERSION = "1.1.1"
GIT_TAG = "v1.1.1"
GIT_COMMIT = "5b5d06192b9856edd04e4d405787af2f2e4a1fef"
SOURCE_URL = (
    "https://raw.githubusercontent.com/bradleyboehmke/completejourney/"
    f"{GIT_COMMIT}/data/transactions.rds"
)
DATASET_PAGE = "https://cran.r-project.org/package=completejourney"
LICENSE = "CC0"
RDS_SIZE_BYTES = 12_775_661
GIT_BLOB_SHA1 = "f40e337238311130c1684cdf94d4128e4c6965f6"

# This must be populated from one explicitly authorized download after preregistration.
# A missing pin blocks before any network request.
RDS_SHA256: str | None = "1fa0700033f1e5d9bb6b09e2be063d8d68474d346e95c50f2833e09d083e0007"
DEFAULT_SOURCE_PREREGISTRATION = Path("results/completejourney_source_preregistration.json")
DEFAULT_QUARANTINE_RECEIPT = Path("results/completejourney_source_quarantine_receipt.json")

SOURCE_PREREGISTRATION_PAYLOAD = {
    "schema_version": 1,
    "artifact_type": "completejourney_source_selection_preregistration",
    "status": "SOURCE_SELECTED_SHA256_UNOBSERVED",
    "source": {
        "dataset": "completejourney",
        "version": VERSION,
        "git_tag": GIT_TAG,
        "git_commit": GIT_COMMIT,
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "source_url": SOURCE_URL,
        "rds_size_bytes": RDS_SIZE_BYTES,
        "git_blob_sha1": GIT_BLOB_SHA1,
    },
    "sha256_state": {
        "value": None,
        "reason": "unknown until the first explicitly authorized byte fetch",
    },
    "real_data_opened": False,
    "parser_invoked": False,
}


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


SOURCE_PREREGISTRATION_SHA256 = hashlib.sha256(_canonical_json(SOURCE_PREREGISTRATION_PAYLOAD)).hexdigest()
DERIVATION_CONFIRMATION = (
    f"AUTHORIZE_COMPLETEJOURNEY_SHA256_DERIVATION:{SOURCE_PREREGISTRATION_SHA256}:{GIT_COMMIT}:{GIT_BLOB_SHA1}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_sha256() -> str:
    if RDS_SHA256 is None:
        raise RuntimeError("Complete Journey SHA-256 is not frozen; authorized derivation is required")
    if len(RDS_SHA256) != 64 or any(character not in "0123456789abcdef" for character in RDS_SHA256):
        raise RuntimeError("Complete Journey SHA-256 pin is malformed")
    return RDS_SHA256


def _temporary_path(output: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=suffix, dir=output.parent)
    os.close(descriptor)
    return Path(name)


def _write_immutable_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = _temporary_path(path, ".part")
    try:
        with part.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        part.chmod(0o444)
        try:
            os.link(part, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace immutable artifact {path}") from None
    finally:
        part.unlink(missing_ok=True)


def create_source_preregistration(path: Path) -> dict[str, object]:
    """Freeze source selection without opening any network or dataset bytes."""
    payload = json.loads(json.dumps(SOURCE_PREREGISTRATION_PAYLOAD))
    _write_immutable_new(path, _canonical_json(payload))
    return payload


def _verify_source_preregistration(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("immutable Complete Journey source preregistration is missing or symbolic")
    payload = path.read_bytes()
    if payload != _canonical_json(SOURCE_PREREGISTRATION_PAYLOAD):
        raise ValueError("Complete Journey source preregistration differs from the frozen selection")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_PREREGISTRATION_SHA256:
        raise ValueError("Complete Journey source preregistration hash changed")
    return digest


def _copy_exact(source: BinaryIO, target: BinaryIO) -> None:
    copied = 0
    while block := source.read(min(1024 * 1024, RDS_SIZE_BYTES - copied + 1)):
        copied += len(block)
        if copied > RDS_SIZE_BYTES:
            raise ValueError("official Complete Journey RDS exceeds its pinned size")
        target.write(block)
    if copied != RDS_SIZE_BYTES:
        raise ValueError("official Complete Journey RDS size differs from its pinned size")


def _fetch_to(path: Path) -> None:
    request = Request(SOURCE_URL, headers={"User-Agent": "ai-defense-lab/1.0"})
    with urlopen(request, timeout=120) as source, path.open("wb") as target:
        final_url = source.geturl()
        if final_url != SOURCE_URL:
            raise ValueError("official Complete Journey source redirected away from the pinned URL")
        _copy_exact(source, target)


def _validate_bytes(path: Path, *, expected_sha256: str) -> None:
    if path.stat().st_size != RDS_SIZE_BYTES:
        raise ValueError("official Complete Journey RDS size differs from its pinned size")
    if git_blob_sha1(path) != GIT_BLOB_SHA1:
        raise ValueError("official Complete Journey RDS Git blob differs from the pinned artifact")
    if sha256_file(path) != expected_sha256:
        raise ValueError("official Complete Journey RDS SHA-256 differs from the pinned artifact")


def derive_sha256(
    preregistration_path: Path,
    receipt_path: Path,
    confirmation: str,
) -> dict[str, object]:
    """Perform one authorized hash-only read and persist no dataset bytes."""
    preregistration_sha256 = _verify_source_preregistration(preregistration_path)
    if confirmation != DERIVATION_CONFIRMATION:
        raise PermissionError("exact Complete Journey SHA-256 derivation confirmation is required")
    if RDS_SHA256 is not None:
        raise RuntimeError("Complete Journey SHA-256 is already frozen; derivation cannot be repeated")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("Complete Journey quarantine receipt is immutable and already exists")
    with tempfile.TemporaryDirectory(prefix="completejourney-sha256-") as directory:
        candidate = Path(directory) / "transactions.rds"
        _fetch_to(candidate)
        if git_blob_sha1(candidate) != GIT_BLOB_SHA1:
            raise ValueError("official Complete Journey RDS Git blob differs from the pinned artifact")
        receipt: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "completejourney_source_quarantine_receipt",
            "status": "SHA256_OBSERVED_UNPARSED",
            "source_preregistration_sha256": preregistration_sha256,
            "source_url": SOURCE_URL,
            "observed_size_bytes": candidate.stat().st_size,
            "observed_git_blob_sha1": GIT_BLOB_SHA1,
            "observed_rds_sha256": sha256_file(candidate),
            "parser_invoked": False,
            "dataset_bytes_retained": False,
        }
        _write_immutable_new(receipt_path, _canonical_json(receipt))
        return receipt


def _verify_quarantine_receipt(path: Path, preregistration_sha256: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("immutable Complete Journey quarantine receipt is missing or symbolic")
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Complete Journey quarantine receipt is not valid JSON") from error
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "source_preregistration_sha256",
        "source_url",
        "observed_size_bytes",
        "observed_git_blob_sha1",
        "observed_rds_sha256",
        "parser_invoked",
        "dataset_bytes_retained",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys or raw != _canonical_json(receipt):
        raise ValueError("Complete Journey quarantine receipt contract changed")
    if (
        receipt["schema_version"] != 1
        or receipt["artifact_type"] != "completejourney_source_quarantine_receipt"
        or receipt["status"] != "SHA256_OBSERVED_UNPARSED"
        or receipt["source_preregistration_sha256"] != preregistration_sha256
        or receipt["source_url"] != SOURCE_URL
        or receipt["observed_size_bytes"] != RDS_SIZE_BYTES
        or receipt["observed_git_blob_sha1"] != GIT_BLOB_SHA1
        or receipt["parser_invoked"] is not False
        or receipt["dataset_bytes_retained"] is not False
        or not isinstance(receipt["observed_rds_sha256"], str)
        or len(receipt["observed_rds_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in receipt["observed_rds_sha256"])
    ):
        raise ValueError("Complete Journey quarantine receipt provenance changed")
    return receipt, hashlib.sha256(raw).hexdigest()


def manifest_for(
    path: Path,
    *,
    source_preregistration_sha256: str,
    quarantine_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "completejourney",
        "version": VERSION,
        "git_tag": GIT_TAG,
        "git_commit": GIT_COMMIT,
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "source_url": SOURCE_URL,
        "rds_size_bytes": path.stat().st_size,
        "git_blob_sha1": GIT_BLOB_SHA1,
        "rds_sha256": sha256_file(path),
        "source_preregistration_sha256": source_preregistration_sha256,
        "quarantine_receipt_sha256": quarantine_receipt_sha256,
    }


def download(
    output: Path,
    *,
    preregistration_path: Path = DEFAULT_SOURCE_PREREGISTRATION,
    receipt_path: Path = DEFAULT_QUARANTINE_RECEIPT,
) -> dict[str, object]:
    expected_sha256 = _frozen_sha256()  # Fail before mkdir, temp files, or network access.
    preregistration_sha256 = _verify_source_preregistration(preregistration_path)
    receipt, receipt_sha256 = _verify_quarantine_receipt(receipt_path, preregistration_sha256)
    if receipt["observed_rds_sha256"] != expected_sha256:
        raise ValueError("frozen Complete Journey SHA-256 differs from its quarantine receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    rds_part = _temporary_path(output, ".rds.part")
    manifest_part = _temporary_path(output, ".json.part")
    manifest_path = output.with_suffix(".manifest.json")
    try:
        _fetch_to(rds_part)
        _validate_bytes(rds_part, expected_sha256=expected_sha256)
        manifest = manifest_for(
            rds_part,
            source_preregistration_sha256=preregistration_sha256,
            quarantine_receipt_sha256=receipt_sha256,
        )
        manifest_part.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rds_part.chmod(0o444)
        manifest_part.chmod(0o444)
        os.replace(rds_part, output)
        os.replace(manifest_part, manifest_path)
        return manifest
    finally:
        rds_part.unlink(missing_ok=True)
        manifest_part.unlink(missing_ok=True)


def verify_existing(
    output: Path,
    *,
    preregistration_path: Path | None = None,
    receipt_path: Path | None = None,
) -> dict[str, object]:
    expected_sha256 = _frozen_sha256()
    manifest_path = output.with_suffix(".manifest.json")
    if output.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Complete Journey cache must not use symbolic links")
    if not output.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Complete Journey RDS and manifest must both exist")
    _validate_bytes(output, expected_sha256=expected_sha256)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Complete Journey cache manifest is not an object")
    if manifest_bytes != _canonical_json(manifest):
        raise ValueError("Complete Journey cache manifest is not canonical")
    preregistration_sha256 = manifest.get("source_preregistration_sha256")
    receipt_sha256 = manifest.get("quarantine_receipt_sha256")
    if (
        not isinstance(preregistration_sha256, str)
        or len(preregistration_sha256) != 64
        or any(character not in "0123456789abcdef" for character in preregistration_sha256)
        or not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise ValueError("Complete Journey cache lineage is missing")
    if preregistration_path is not None or receipt_path is not None:
        if preregistration_path is None or receipt_path is None:
            raise ValueError("both Complete Journey lineage artifact paths are required")
        actual_preregistration_sha256 = _verify_source_preregistration(preregistration_path)
        receipt, actual_receipt_sha256 = _verify_quarantine_receipt(receipt_path, actual_preregistration_sha256)
        if (
            preregistration_sha256 != actual_preregistration_sha256
            or receipt_sha256 != actual_receipt_sha256
            or receipt["observed_rds_sha256"] != expected_sha256
        ):
            raise ValueError("Complete Journey cache lineage differs from immutable artifacts")
    if manifest != manifest_for(
        output,
        source_preregistration_sha256=preregistration_sha256,
        quarantine_receipt_sha256=receipt_sha256,
    ):
        raise ValueError("Complete Journey cache provenance does not match the pinned manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/reference/completejourney/transactions.rds"))
    parser.add_argument("--force", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preregister-source", action="store_true")
    mode.add_argument("--derive-sha256", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--source-preregistration", type=Path, default=DEFAULT_SOURCE_PREREGISTRATION)
    parser.add_argument("--quarantine-receipt", type=Path, default=DEFAULT_QUARANTINE_RECEIPT)
    args = parser.parse_args()
    if args.preregister_source:
        print(json.dumps(create_source_preregistration(args.source_preregistration), sort_keys=True))
        return
    if args.derive_sha256:
        print(
            json.dumps(
                derive_sha256(args.source_preregistration, args.quarantine_receipt, args.confirmation),
                sort_keys=True,
            )
        )
        return
    manifest = (
        download(
            args.output,
            preregistration_path=args.source_preregistration,
            receipt_path=args.quarantine_receipt,
        )
        if args.force or not args.output.exists()
        else verify_existing(
            args.output,
            preregistration_path=args.source_preregistration,
            receipt_path=args.quarantine_receipt,
        )
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
