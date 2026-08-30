import pandas as pd
import pytest

from scripts.download_reference import validate_frame


def test_reference_contract_rejects_changed_counts() -> None:
    frame = pd.DataFrame({"Time": [0, 1], "Amount": [10.0, 20.0], "Class": [0, 1]})
    validate_frame(frame, expected_rows=2, expected_fraud=1)
    with pytest.raises(ValueError):
        validate_frame(frame, expected_rows=3, expected_fraud=1)

