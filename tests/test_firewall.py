import ast
from pathlib import Path

from defend.harness import generator_input_tree_hash, import_firewall_violations


ROOT = Path(__file__).resolve().parents[1]


def test_static_heldout_import_firewall() -> None:
    assert import_firewall_violations(ROOT / "defend") == []


def test_firewall_rejects_attack_loader_and_dynamic_heldout_import(tmp_path: Path) -> None:
    (tmp_path / "loader.py").write_text(
        "from sim.attacks import all_attacks\nall_attacks(include_heldout=True)\n",
        encoding="utf-8",
    )
    (tmp_path / "dynamic.py").write_text(
        "import importlib\nimportlib.import_module('sim.attacks.heldout')\n",
        encoding="utf-8",
    )
    (tmp_path / "aliased.py").write_text(
        "from importlib import import_module as load\nload('sim.' + 'attacks.heldout')\n",
        encoding="utf-8",
    )
    (tmp_path / "assigned.py").write_text(
        "load = __import__\nload('sim.' + 'attacks.heldout')\n",
        encoding="utf-8",
    )
    (tmp_path / "computed.py").write_text(
        "import importlib as loader\nloader.import_module('.'.join(('sim', 'attacks', 'heldout')))\n",
        encoding="utf-8",
    )

    assert len(import_firewall_violations(tmp_path)) == 5


def test_feature_source_cannot_read_protected_metadata() -> None:
    protected = {"label", "atlas_id", "attack_family", "attack_primitives", "split"}
    violations = []
    for path in (ROOT / "defend" / "features").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in protected:
                violations.append(f"{path}:{node.lineno}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in protected
            ):
                violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_generator_hash_includes_package_initializer(tmp_path: Path) -> None:
    for relative in (
        "sim/__init__.py",
        "sim/population.py",
        "sim/mandate_twin.py",
        "sim/primitives.py",
        "sim/content_llm.py",
        "sim/attacks/catalog.py",
        "data/market_prices.csv",
        "data/llm_cache/content/approved.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    before = generator_input_tree_hash(tmp_path)
    (tmp_path / "sim/__init__.py").write_text("changed", encoding="utf-8")

    assert generator_input_tree_hash(tmp_path) != before
