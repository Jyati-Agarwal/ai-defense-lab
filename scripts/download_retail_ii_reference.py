"""Download the pinned official UCI Online Retail II workbook (dataset 502)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_ID = 502
SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
DOI = "10.24432/C5CG6D"
LICENSE = "CC BY 4.0"
ARCHIVE_MEMBER = "online_retail_II.xlsx"
ARCHIVE_SIZE_BYTES = 45_622_418
WORKBOOK_SIZE_BYTES = 45_622_278
ARCHIVE_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
WORKBOOK_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
READINESS_SLICE = {
    "sheet": "Year 2009-2010",
    "invoice_date_before": "2010-12-01",
    "transaction_disjoint_from_uci352": True,
    "customer_disjoint": "unknown",
    "source_independent": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_exact(source: object, target: object, expected_size: int) -> None:
    copied = 0
    while block := source.read(min(1024 * 1024, expected_size - copied + 1)):  # type: ignore[attr-defined]
        copied += len(block)
        if copied > expected_size:
            raise ValueError("official UCI artifact exceeds its pinned size")
        target.write(block)  # type: ignore[attr-defined]
    if copied != expected_size:
        raise ValueError("official UCI artifact size differs from its pinned size")


def _temporary_path(output: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=suffix, dir=output.parent)
    os.close(descriptor)
    return Path(name)


def manifest_for(path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "name": "Online Retail II",
        "dataset_page": DATASET_PAGE,
        "doi": DOI,
        "license": LICENSE,
        "source_url": SOURCE_URL,
        "archive_member": ARCHIVE_MEMBER,
        "archive_size_bytes": ARCHIVE_SIZE_BYTES,
        "workbook_size_bytes": path.stat().st_size,
        "archive_sha256": ARCHIVE_SHA256,
        "workbook_sha256": sha256_file(path),
        "readiness_slice": READINESS_SLICE,
    }


def _validate_workbook_bytes(path: Path) -> None:
    if path.stat().st_size != WORKBOOK_SIZE_BYTES or sha256_file(path) != WORKBOOK_SHA256:
        raise ValueError("official UCI workbook bytes differ from the pinned artifact")


def download(output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_part = _temporary_path(output, ".zip.part")
    workbook_part = _temporary_path(output, ".xlsx.part")
    manifest_part = _temporary_path(output, ".json.part")
    manifest_path = output.with_suffix(".manifest.json")
    try:
        request = Request(SOURCE_URL, headers={"User-Agent": "ai-defense-lab/1.0"})
        with urlopen(request, timeout=120) as source, archive_part.open("wb") as target:
            _copy_exact(source, target, ARCHIVE_SIZE_BYTES)
        if sha256_file(archive_part) != ARCHIVE_SHA256:
            raise ValueError("official UCI archive SHA-256 differs from the pinned artifact")

        with zipfile.ZipFile(archive_part) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != ARCHIVE_MEMBER or members[0].is_dir():
                raise ValueError("official UCI archive member contract changed")
            if members[0].file_size != WORKBOOK_SIZE_BYTES:
                raise ValueError("official UCI workbook size differs from the pinned artifact")
            with archive.open(members[0]) as source, workbook_part.open("wb") as target:
                _copy_exact(source, target, WORKBOOK_SIZE_BYTES)

        _validate_workbook_bytes(workbook_part)
        manifest = manifest_for(workbook_part)
        manifest_part.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        workbook_part.chmod(0o644)
        manifest_part.chmod(0o644)
        os.replace(workbook_part, output)
        os.replace(manifest_part, manifest_path)
        return manifest
    finally:
        archive_part.unlink(missing_ok=True)
        workbook_part.unlink(missing_ok=True)
        manifest_part.unlink(missing_ok=True)


def verify_existing(output: Path) -> dict[str, object]:
    manifest_path = output.with_suffix(".manifest.json")
    if not output.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("workbook and manifest must both exist")
    _validate_workbook_bytes(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != manifest_for(output):
        raise ValueError("existing UCI workbook provenance does not match the pinned manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/reference/online_retail_ii.xlsx"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = download(args.output) if args.force or not args.output.exists() else verify_existing(args.output)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
