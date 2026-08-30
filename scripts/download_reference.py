"""Download and reduce the public OpenML ULB credit-card dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATASET_ID = 1597
METADATA_URL = f"https://www.openml.org/api/v1/json/data/{DATASET_ID}"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
EXPECTED_PARQUET_SHA256 = "b7efcb35a428bbe22347a05d2437d9177bab07ce61e51214a17bec584ad9496d"
EXPECTED_MARGINALS_SHA256 = "a481c635286387380d3eb6413cd4533fe477fdd9699dda13e2ce1c575347fe11"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frame(frame: pd.DataFrame, *, expected_rows: int, expected_fraud: int) -> None:
    if list(frame.columns) != ["Time", "Amount", "Class"]:
        raise ValueError("OpenML dataset 1597 no longer has the expected Time/Amount/Class contract")
    if len(frame) != expected_rows or int(frame["Class"].sum()) != expected_fraud:
        raise ValueError("OpenML dataset 1597 row or fraud counts changed")
    if frame.isna().any().any() or (frame["Amount"] < 0).any():
        raise ValueError("OpenML dataset 1597 contains invalid marginal values")


def download(output: Path) -> dict[str, object]:
    request = Request(METADATA_URL, headers={"User-Agent": "ai-defense-lab/1.0"})
    with urlopen(request, timeout=30) as response:
        metadata = json.load(response)["data_set_description"]
    if metadata.get("id") != str(DATASET_ID) or metadata.get("status") != "active":
        raise RuntimeError("OpenML dataset 1597 is missing or inactive")

    parquet_url = str(metadata["parquet_url"])
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet = output.with_suffix(".pq.part")
    try:
        with urlopen(Request(parquet_url, headers={"User-Agent": "ai-defense-lab/1.0"}), timeout=120) as source:
            with parquet.open("wb") as target:
                shutil.copyfileobj(source, target)
        frame = pd.read_parquet(parquet, columns=["Time", "Amount", "Class"])
        frame["Class"] = frame["Class"].astype(int)
        validate_frame(frame, expected_rows=EXPECTED_ROWS, expected_fraud=EXPECTED_FRAUD)
        frame.to_csv(output, index=False)
        parquet_sha256 = sha256_file(parquet)
        marginals_sha256 = sha256_file(output)
        if parquet_sha256 != EXPECTED_PARQUET_SHA256 or marginals_sha256 != EXPECTED_MARGINALS_SHA256:
            raise ValueError("OpenML dataset 1597 bytes differ from the pinned reference")
        result = {
            "dataset_id": DATASET_ID,
            "name": metadata["name"],
            "version": metadata["version"],
            "source": METADATA_URL,
            "parquet_url": parquet_url,
            "rows": len(frame),
            "fraud_rows": int(frame["Class"].sum()),
            "legitimate_rows": int((frame["Class"] == 0).sum()),
            "parquet_sha256": parquet_sha256,
            "marginals_sha256": marginals_sha256,
        }
        output.with_suffix(".manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    finally:
        parquet.unlink(missing_ok=True)


def verify_existing(output: Path) -> dict[str, object]:
    manifest_path = output.with_suffix(".manifest.json")
    if not output.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("marginals CSV and manifest must both exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("rows") != EXPECTED_ROWS
        or manifest.get("fraud_rows") != EXPECTED_FRAUD
        or manifest.get("parquet_sha256") != EXPECTED_PARQUET_SHA256
        or manifest.get("marginals_sha256") != EXPECTED_MARGINALS_SHA256
        or sha256_file(output) != EXPECTED_MARGINALS_SHA256
    ):
        raise ValueError("cached OpenML dataset 1597 provenance does not match the pinned reference")
    frame = pd.read_csv(output)
    validate_frame(frame, expected_rows=EXPECTED_ROWS, expected_fraud=EXPECTED_FRAUD)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/reference/creditcard_marginals.csv"))
    args = parser.parse_args()
    result = verify_existing(args.output) if args.output.exists() else download(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
