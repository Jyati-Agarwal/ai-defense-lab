from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


class ScorerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_scorer(project_root: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Load the single frozen-model inference entry point."""

    for module_name, function_name in (
        ("defend.inference", "score_composer"),
        ("defend.runtime", "score_composer"),
    ):
        try:
            function = getattr(import_module(module_name), function_name)
        except (ImportError, AttributeError):
            continue
        return lambda payload, fn=function: fn(payload, project_root=project_root)
    raise ScorerUnavailable(
        "Frozen scorer is missing. Run `make all` to generate model.pkl and its runtime."
    )

