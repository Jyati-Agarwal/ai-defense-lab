from pathlib import Path

import pytest

from scripts.download_reference import verify_existing


ROOT = Path(__file__).resolve().parents[1]


def test_cached_openml_reference_is_byte_verified(tmp_path: Path) -> None:
    source = ROOT / "data" / "reference" / "creditcard_marginals.csv"
    manifest = source.with_suffix(".manifest.json")
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    copied.with_suffix(".manifest.json").write_bytes(manifest.read_bytes())

    assert verify_existing(copied)["marginals_sha256"] == verify_existing(source)["marginals_sha256"]
    copied.write_bytes(b"tampered," + copied.read_bytes())
    with pytest.raises(ValueError, match="pinned reference"):
        verify_existing(copied)
