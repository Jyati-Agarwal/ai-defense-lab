import json
from collections import Counter

import yaml
from jsonschema import Draft202012Validator

from atlas.render import ROOT, STAGES, build_matrix, load_and_validate, render, spec_paths


def test_all_48_specs_validate_and_are_split_exactly() -> None:
    schema = yaml.safe_load((ROOT / "schema.yaml").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    paths = spec_paths()
    attacks = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]

    assert len(paths) == 48
    assert all(not list(validator.iter_errors(attack)) for attack in attacks)
    assert Counter(attack["tier"] for attack in attacks) == {"classic": 20, "agentic": 28}
    assert len({attack["id"] for attack in attacks}) == 48
    assert len({attack["name"] for attack in attacks}) == 48


def test_render_has_no_empty_kill_chain_stage(tmp_path) -> None:
    output = tmp_path / "atlas_matrix.json"
    matrix = render(output)

    assert matrix == json.loads(output.read_text(encoding="utf-8"))
    assert matrix["total"] == 48
    assert [column["stage"] for column in matrix["kill_chain"]] == list(STAGES)
    assert all(column["attack_ids"] for column in matrix["kill_chain"])
    assert build_matrix(load_and_validate()) == matrix
