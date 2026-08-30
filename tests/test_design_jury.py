import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.design_jury as design_jury
from scripts.design_jury import (
    AXES,
    CODEX_PROVIDER,
    OPENAI_PROVIDER,
    SCREENS,
    TRANSIENT_LIFECYCLE_EXCLUSION,
    VIEWPORTS,
    cache_key,
    main,
    request_binding,
    request_spec,
    run,
)


def _axe_evidence(screen: str) -> dict[str, Any]:
    return {
        "scope": f"#{screen}",
        "violations": [],
        "incomplete": [
            {
                "id": f"manual-review-{screen}",
                "impact": "serious",
                "help": "Measured manual-review evidence",
                "nodeCount": 1,
                "nodesTruncated": False,
                "nodes": [
                    {
                        "target": [f"#{screen}"],
                        "html": f'<section id="{screen}">',
                        "failureSummary": "Review the measured selector.",
                    }
                ],
            }
        ],
    }


def _motion_sample(viewport: str, screen: str, *, reduced: bool) -> dict[str, Any]:
    target = f"#{screen} .measured-motion"
    start = {
        "opacity": "0" if not reduced else "1",
        "transform": "none",
        "box": {"x": 0, "y": 0, "width": 100, "height": 40},
    }
    end = {
        "opacity": "1",
        "transform": "none",
        "box": {"x": 0, "y": 0, "width": 100, "height": 40},
    }
    changes = [] if reduced else [{"target": target, "before": start, "after": end}]
    trajectories = [] if reduced else [
        {
            "target": target,
            "frames": [
                {"frame": 0, "elapsedMs": 0, "state": start},
                {"frame": 12, "elapsedMs": 200, "state": end},
            ],
        }
    ]
    return {
        "selector": f"#{screen}",
        "trigger": "reduced-scroll-entry" if reduced else "scroll-entry",
        "mediaMatches": reduced,
        "sampledElements": 4,
        "sampleFrames": 12,
        "sampleDurationMs": 200,
        "averageFrameMs": 16.67,
        "worstFrameMs": 17.1,
        "transitionElementCount": 1,
        "transitionCapableElementCount": 1,
        "activeAnimationCount": 0,
        "registeredAnimationCount": 0,
        "changedElementCount": len(changes),
        "transientLifecycleExclusions": {
            **TRANSIENT_LIFECYCLE_EXCLUSION,
            "excludedElementCount": 0,
        },
        "changesTruncated": False,
        "changes": changes,
        "trajectoriesTruncated": False,
        "trajectories": trajectories,
        "screenMarker": f"{viewport}:{screen}:{'reduced' if reduced else 'standard'}",
    }


def _keyboard_evidence(viewport: str, scope: str) -> dict[str, Any]:
    marker = f"{viewport}:{scope}:keyboard"
    return {
        "scope": scope,
        "expectedFocusableCount": 1,
        "visitedFocusableCount": 1,
        "allExpectedVisited": True,
        "allFocusVisible": True,
        "allHaveFocusIndicator": True,
        "steps": [
            {
                "index": 0,
                "target": f"{scope} [data-keyboard-marker='{marker}']",
                "name": marker,
                "role": "button",
                "focusVisible": True,
                "hasVisibleFocusIndicator": True,
            }
        ],
    }


def _evidence() -> dict[str, Any]:
    row = {
        "viewport": {"width": 1440, "height": 900},
        "screenCount": len(SCREENS),
        "capturedScreens": list(SCREENS),
        "apiResponses": {"/metrics": {"status": 200, "fixtureHeader": None, "method": "GET"}},
        "fixtureRailCount": 0,
        "browserFailures": {"console": [], "page": [], "request": []},
        "touchTargetViolations": [],
        "accessibility": {"scope": "document", "violations": [], "incomplete": []},
        "animationFrames": {"start": {"opacity": "1"}, "mid": {"opacity": "0.5"}, "end": {"opacity": "0"}},
        "reducedMotion": {"mediaMatches": True},
        "frameTiming": {"longFrameRatio": 0},
        "pagePerformance": {
            "fcpMs": 600,
            "lcpMs": 800,
            "cls": 0.02,
            "clsAttribution": {
                "algorithm": "maximum-session-window",
                "sessionGapLimitMs": 1_000,
                "sessionDurationLimitMs": 5_000,
                "windows": [
                    {
                        "startTime": 400,
                        "endTime": 400,
                        "value": 0.02,
                        "shifts": [
                            {
                                "startTime": 400,
                                "value": 0.02,
                                "sources": [
                                    {
                                        "target": "#thesis",
                                        "previousRect": {
                                            "x": 0,
                                            "y": 0,
                                            "width": 100,
                                            "height": 40,
                                        },
                                        "currentRect": {
                                            "x": 0,
                                            "y": 8,
                                            "width": 100,
                                            "height": 40,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "recentInputShifts": [],
            },
        },
        "documentWidths": {"viewport": 1440, "document": 1440},
        "navigation": {"lineCount": 1},
        "skipLinkTarget": {"width": 44, "height": 44},
    }
    return {
        "viewports": {
            viewport: {
                **row,
                "viewport": {"name": viewport},
                "headerEvidence": {
                    "scope": "header",
                    "captureMode": "actual-global-header",
                    "screenshot": f"{viewport}-header.png",
                    "accessibility": {
                        "scope": "header",
                        "violations": [],
                        "incomplete": [],
                    },
                    "keyboard": _keyboard_evidence(viewport, "header"),
                    "widths": {"viewport": 1440, "content": 1200},
                    "box": {"left": 0, "right": 1440, "width": 1440},
                },
                "screenEvidence": {
                    screen: {
                        "capture": {
                            "sectionMode": "isolated-section",
                            "fixedHeaderVisibility": "hidden-during-section-capture",
                            "screenshot": f"{viewport}-{screen}.png",
                            "headerScreenshot": f"{viewport}-header.png",
                        },
                        "accessibility": _axe_evidence(screen),
                        "keyboard": _keyboard_evidence(viewport, f"#{screen}"),
                        "touchTargetViolations": [],
                        "documentWidths": {
                            "viewport": 1440,
                            "sectionClientWidth": 1200,
                            "sectionScrollWidth": 1200,
                            "contentLeft": 0,
                            "contentRight": 1200,
                            "overflowsViewport": False,
                            "box": {"left": 0, "right": 1200, "width": 1200},
                        },
                        "motion": {
                            "standard": _motion_sample(viewport, screen, reduced=False),
                            "reduced": _motion_sample(viewport, screen, reduced=True),
                        },
                    }
                    for screen in SCREENS
                },
            }
            for viewport in VIEWPORTS
        }
    }


def _screenshots(root: Path, evidence: dict[str, Any] | None = None) -> Path:
    root.mkdir()
    (root / "evidence.json").write_text(json.dumps(evidence or _evidence()), encoding="utf-8")
    for screen in SCREENS:
        for viewport in VIEWPORTS:
            (root / f"{viewport}-{screen}.png").write_bytes(f"{viewport}:{screen}".encode())
    for viewport in VIEWPORTS:
        (root / f"{viewport}-header.png").write_bytes(f"{viewport}:header".encode())
    return root


class FakeResponses:
    def __init__(self, *, delay: float = 0, fail: tuple[str, str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.delay = delay
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def create(self, **kwargs: Any) -> SimpleNamespace:
        schema = kwargs["text"]["format"]["schema"]
        screen = schema["properties"]["screen"]["enum"][0]
        axis = schema["properties"]["axis"]["enum"][0]
        with self.lock:
            self.calls.append(kwargs)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if (screen, axis) == self.fail:
                raise RuntimeError("OPENAI_API_KEY=must-not-leak")
            critique = {"screen": screen, "axis": axis, "score": 8, "evidence": "Measured evidence supports this score.", "fix": "Keep measured proof current."}
            return SimpleNamespace(id=f"resp_{screen}_{axis}", model=kwargs["model"], usage=None, output_text=json.dumps(critique))
        finally:
            with self.lock:
                self.active -= 1


class FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        self.responses = FakeResponses(**kwargs)


class FakeCodex:
    def __init__(self, *, delay: float = 0, fail: tuple[str, str] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.schemas: list[dict[str, Any]] = []
        self.delay = delay
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        with self.lock:
            self.calls.append((command, kwargs))
        if command == [design_jury.CODEX_BINARY, "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.147.0\n", stderr="secret")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            schema_path = Path(command[command.index("--output-schema") + 1])
            output_path = Path(command[command.index("--output-last-message") + 1])
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            with self.lock:
                self.schemas.append(schema)
            screen = schema["properties"]["screen"]["enum"][0]
            axis = schema["properties"]["axis"]["enum"][0]
            if (screen, axis) == self.fail and self.fail is not None:
                if self.fail_mode == "malformed":
                    output_path.write_text("{not-json", encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="secret")
                return subprocess.CompletedProcess(command, 17, stdout="", stderr="secret")
            output_path.write_text(
                json.dumps(
                    {
                        "screen": screen,
                        "axis": axis,
                        "score": 8,
                        "evidence": "Measured evidence supports this score.",
                        "fix": "Keep measured proof current.",
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="secret")
        finally:
            with self.lock:
                self.active -= 1

    fail_mode = "nonzero"


def _codex_calls(fake: FakeCodex) -> list[tuple[list[str], dict[str, Any]]]:
    return [call for call in fake.calls if call[0] != [design_jury.CODEX_BINARY, "--version"]]


@pytest.mark.parametrize(
    ("argv", "expected_provider", "expected_populate"),
    [
        (["design_jury.py", "--from-cache"], CODEX_PROVIDER, False),
        (["design_jury.py", "--populate-openai"], OPENAI_PROVIDER, True),
    ],
)
def test_cli_selects_codex_for_cache_and_preserves_openai_population(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_provider: str,
    expected_populate: bool,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"gate_passed": True, "minimum_score": 8}

    monkeypatch.setattr(design_jury, "run", fake_run)
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert calls[0]["provider"] == expected_provider
    assert calls[0]["populate"] is expected_populate


def test_design_jury_cache_is_bound_to_section_and_header_screenshots(tmp_path: Path) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    spec, _ = request_spec("thesis", "aesthetics", screenshots, _evidence())
    assert spec["schema_version"] == 4
    assert [image["capture"] for image in spec["images"]] == [
        "section",
        "section",
        "section",
        "header",
        "header",
        "header",
    ]
    original = cache_key(spec)
    (screenshots / "tablet-thesis.png").write_bytes(b"changed")
    changed, _ = request_spec("thesis", "aesthetics", screenshots, _evidence())
    assert cache_key(changed) != original
    section_changed = cache_key(changed)
    (screenshots / "mobile-header.png").write_bytes(b"changed-header")
    header_changed, _ = request_spec("thesis", "aesthetics", screenshots, _evidence())
    assert cache_key(header_changed) != section_changed
    assert "prompt" not in request_binding(changed)
    with pytest.raises(RuntimeError, match="cache miss for thesis/aesthetics"):
        run(screenshots, tmp_path / "cache", tmp_path / "jury.json", populate=False)


def test_codex_provider_builds_strict_six_image_command_and_sanitizes_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    fake = FakeCodex()
    monkeypatch.setattr(design_jury.subprocess, "run", fake)

    result = run(
        screenshots,
        tmp_path / "cache",
        tmp_path / "jury.json",
        populate=True,
        provider=CODEX_PROVIDER,
        max_workers=4,
    )

    assert result["provider"] == CODEX_PROVIDER
    assert sum(command == [design_jury.CODEX_BINARY, "--version"] for command, _ in fake.calls) == 1
    calls = _codex_calls(fake)
    assert len(calls) == len(SCREENS) * len(AXES)
    for command, kwargs in calls:
        assert command[:2] == [design_jury.CODEX_BINARY, "exec"]
        assert command[command.index("--model") + 1] == "gpt-5.6-sol"
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert all(flag in command for flag in ("--ephemeral", "--ignore-user-config", "--skip-git-repo-check"))
        assert command.count("--image") == 6
        assert kwargs["input"].startswith(
            "Be strict and evidence-bound. Never reward hidden functionality"
        )
        assert Path(kwargs["cwd"]).name.startswith("design-jury-codex-")
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert all(name not in kwargs["env"] for name in design_jury.CODEX_ENV_BLOCKLIST)
    assert len(fake.schemas) == len(calls)
    assert all(schema["additionalProperties"] is False for schema in fake.schemas)
    assert all(schema["required"] == ["screen", "axis", "score", "evidence", "fix"] for schema in fake.schemas)

    spec, _ = request_spec("thesis", "aesthetics", screenshots, _evidence(), provider=CODEX_PROVIDER)
    record = json.loads((tmp_path / "cache" / f"{cache_key(spec)}.json").read_text(encoding="utf-8"))
    assert record["provider"] == CODEX_PROVIDER
    assert set(record["provenance"]) == {"provider", "cli_version", "model", "output_sha256"}
    assert record["provenance"]["cli_version"] == "codex-cli 0.147.0"
    assert len(record["provenance"]["output_sha256"]) == 64
    assert "response" not in record


def test_provider_is_bound_into_cache_key_and_tampering_cannot_cross_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    openai_spec, _ = request_spec("thesis", "aesthetics", screenshots, _evidence(), provider=OPENAI_PROVIDER)
    codex_spec, _ = request_spec("thesis", "aesthetics", screenshots, _evidence(), provider=CODEX_PROVIDER)
    assert cache_key(openai_spec) != cache_key(codex_spec)
    assert request_binding(openai_spec)["provider"] == OPENAI_PROVIDER
    assert request_binding(codex_spec)["provider"] == CODEX_PROVIDER

    fake = FakeCodex()
    monkeypatch.setattr(design_jury.subprocess, "run", fake)
    cache = tmp_path / "cache"
    run(screenshots, cache, tmp_path / "first.json", populate=True, provider=CODEX_PROVIDER)
    cache_path = cache / f"{cache_key(codex_spec)}.json"
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    record["provider"] = OPENAI_PROVIDER
    cache_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cache binding or payload mismatch for thesis/aesthetics"):
        run(screenshots, cache, tmp_path / "tampered.json", populate=False, provider=CODEX_PROVIDER)


@pytest.mark.parametrize("fail_mode", ["malformed", "nonzero"])
def test_codex_malformed_or_nonzero_output_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_mode: str
) -> None:
    screenshots = _screenshots(tmp_path / fail_mode)
    fake = FakeCodex(fail=("thesis", "aesthetics"))
    fake.fail_mode = fail_mode
    monkeypatch.setattr(design_jury.subprocess, "run", fake)
    cache = tmp_path / f"cache-{fail_mode}"
    output = tmp_path / f"{fail_mode}.json"

    with pytest.raises(RuntimeError, match="jury population failed for thesis/aesthetics") as raised:
        run(screenshots, cache, output, populate=True, provider=CODEX_PROVIDER, max_workers=4)
    assert "secret" not in str(raised.value)
    assert not output.exists()
    assert len(list(cache.glob("*.json"))) == len(SCREENS) * len(AXES) - 1


def test_codex_concurrency_and_resume_reuse_only_valid_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    cache = tmp_path / "cache"
    first_fake = FakeCodex(delay=0.002)
    monkeypatch.setattr(design_jury.subprocess, "run", first_fake)
    first = run(
        screenshots,
        cache,
        tmp_path / "first.json",
        populate=True,
        provider=CODEX_PROVIDER,
        max_workers=3,
    )
    assert first["gate_passed"] is True
    assert 1 < first_fake.max_active <= 3

    resumed_fake = FakeCodex()
    monkeypatch.setattr(design_jury.subprocess, "run", resumed_fake)
    resumed = run(
        screenshots,
        cache,
        tmp_path / "resumed.json",
        populate=True,
        provider=CODEX_PROVIDER,
        max_workers=3,
    )
    assert resumed["screens"] == first["screens"]
    assert resumed_fake.calls == []


def test_motion_and_accessibility_prompts_use_only_the_requested_screen_evidence(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    screenshots = _screenshots(tmp_path / "screens", evidence)

    thesis_motion, _ = request_spec("thesis", "motion_quality", screenshots, evidence)
    thesis_accessibility, _ = request_spec("thesis", "accessibility", screenshots, evidence)

    for viewport in VIEWPORTS:
        assert f'"screenshot": "{viewport}-thesis.png"' in thesis_motion["prompt"]
        assert f'"screenshot": "{viewport}-header.png"' in thesis_motion["prompt"]
        assert f"{viewport}:thesis:standard" in thesis_motion["prompt"]
        assert f"{viewport}:thesis:reduced" in thesis_motion["prompt"]
        assert f"{viewport}:#thesis:keyboard" in thesis_accessibility["prompt"]
        assert f"{viewport}:thesis:reduced" in thesis_accessibility["prompt"]
        assert "manual-review-thesis" in thesis_accessibility["prompt"]
        assert f"{viewport}:atlas:standard" not in thesis_motion["prompt"]
        assert f'"screenshot": "{viewport}-atlas.png"' not in thesis_motion["prompt"]
        assert f"{viewport}:#atlas:keyboard" not in thesis_accessibility["prompt"]
        assert f"{viewport}:atlas:reduced" not in thesis_accessibility["prompt"]
        assert "manual-review-atlas" not in thesis_accessibility["prompt"]


def test_design_jury_rejects_missing_or_violating_screen_evidence(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    del evidence["viewports"]["desktop"]["screenEvidence"]["thesis"]["motion"]["reduced"]
    screenshots = _screenshots(tmp_path / "missing", evidence)
    with pytest.raises(ValueError, match="reduced motion for desktop/thesis is missing"):
        run(screenshots, tmp_path / "cache-missing", tmp_path / "missing.json", populate=False)

    evidence = _evidence()
    evidence["viewports"]["desktop"]["screenEvidence"]["thesis"]["motion"]["reduced"]["mediaMatches"] = False
    screenshots = _screenshots(tmp_path / "invalid-reduced", evidence)
    with pytest.raises(ValueError, match="wrong reduced-motion media state"):
        run(screenshots, tmp_path / "cache-invalid-reduced", tmp_path / "invalid-reduced.json", populate=False)

    evidence = _evidence()
    evidence["viewports"]["desktop"]["screenEvidence"]["thesis"]["motion"]["standard"]["trajectories"][0]["frames"][-1]["frame"] = 11
    screenshots = _screenshots(tmp_path / "invalid-trajectory", evidence)
    with pytest.raises(ValueError, match="incomplete trajectory"):
        run(screenshots, tmp_path / "cache-invalid-trajectory", tmp_path / "invalid-trajectory.json", populate=False)

    evidence = _evidence()
    del evidence["viewports"]["desktop"]["screenEvidence"]["mandate"]["motion"]["reduced"]["transientLifecycleExclusions"]
    screenshots = _screenshots(tmp_path / "missing-lifecycle-exclusions", evidence)
    with pytest.raises(ValueError, match="invalid transient lifecycle exclusions"):
        run(screenshots, tmp_path / "cache-missing-lifecycle-exclusions", tmp_path / "missing-lifecycle-exclusions.json", populate=False)

    evidence = _evidence()
    evidence["viewports"]["desktop"]["screenEvidence"]["mandate"]["motion"]["reduced"]["transientLifecycleExclusions"]["reason"] = "silently ignored"
    screenshots = _screenshots(tmp_path / "invalid-lifecycle-reason", evidence)
    with pytest.raises(ValueError, match="invalid transient lifecycle exclusions"):
        run(screenshots, tmp_path / "cache-invalid-lifecycle-reason", tmp_path / "invalid-lifecycle-reason.json", populate=False)

    evidence = _evidence()
    evidence["viewports"]["desktop"]["screenEvidence"]["thesis"]["accessibility"]["violations"] = [
        {
            "id": "button-name",
            "impact": "critical",
            "help": "Buttons must have discernible text",
            "nodeCount": 1,
            "nodesTruncated": False,
            "nodes": [
                {
                    "target": ["#thesis button"],
                    "html": "<button></button>",
                    "failureSummary": "Fix the missing name.",
                }
            ],
        }
    ]
    screenshots = _screenshots(tmp_path / "violating", evidence)
    with pytest.raises(ValueError, match="accessibility violations for desktop/thesis"):
        run(screenshots, tmp_path / "cache-violating", tmp_path / "violating.json", populate=False)


def test_design_jury_rejects_invalid_cls_windows_and_attribution(tmp_path: Path) -> None:
    evidence = _evidence()
    del evidence["viewports"]["desktop"]["pagePerformance"]["clsAttribution"]
    screenshots = _screenshots(tmp_path / "missing-cls-attribution", evidence)
    with pytest.raises(ValueError, match="invalid CLS attribution"):
        run(screenshots, tmp_path / "cache-missing-cls", tmp_path / "missing-cls.json", populate=False)

    evidence = _evidence()
    performance = evidence["viewports"]["mobile"]["pagePerformance"]
    performance["clsAttribution"]["windows"][0]["shifts"].append(
        {"startTime": 1_600, "value": 0.01, "sources": []}
    )
    performance["clsAttribution"]["windows"][0]["endTime"] = 1_600
    performance["clsAttribution"]["windows"][0]["value"] = 0.03
    performance["cls"] = 0.03
    screenshots = _screenshots(tmp_path / "invalid-cls-gap", evidence)
    with pytest.raises(ValueError, match="invalid CLS session window"):
        run(screenshots, tmp_path / "cache-invalid-cls-gap", tmp_path / "invalid-cls-gap.json", populate=False)

    evidence = _evidence()
    performance = evidence["viewports"]["mobile"]["pagePerformance"]
    performance["clsAttribution"]["windows"][0]["shifts"].append(
        {"startTime": 300, "value": 0.01, "sources": []}
    )
    performance["clsAttribution"]["windows"][0]["endTime"] = 300
    performance["clsAttribution"]["windows"][0]["value"] = 0.03
    performance["cls"] = 0.03
    screenshots = _screenshots(tmp_path / "reversed-cls-time", evidence)
    with pytest.raises(ValueError, match="invalid CLS session window"):
        run(screenshots, tmp_path / "cache-reversed-cls", tmp_path / "reversed-cls.json", populate=False)

    evidence = _evidence()
    performance = evidence["viewports"]["tablet"]["pagePerformance"]
    performance["cls"] = 0.04
    screenshots = _screenshots(tmp_path / "invalid-cls-maximum", evidence)
    with pytest.raises(ValueError, match="inconsistent CLS maximum"):
        run(screenshots, tmp_path / "cache-invalid-cls-max", tmp_path / "invalid-cls-max.json", populate=False)

    evidence = _evidence()
    performance = evidence["viewports"]["desktop"]["pagePerformance"]
    performance["clsAttribution"]["recentInputShifts"] = [
        {"startTime": 900, "value": 0.01, "sources": [{"target": "", "previousRect": {}, "currentRect": {}}]}
    ]
    screenshots = _screenshots(tmp_path / "invalid-cls-source", evidence)
    with pytest.raises(ValueError, match="invalid layout-shift source"):
        run(screenshots, tmp_path / "cache-invalid-cls-source", tmp_path / "invalid-cls-source.json", populate=False)


def test_design_jury_rejects_reused_cross_screen_measurements(tmp_path: Path) -> None:
    evidence = _evidence()
    evidence["viewports"]["desktop"]["screenEvidence"]["atlas"]["accessibility"]["scope"] = "#thesis"
    screenshots = _screenshots(tmp_path / "wrong-axe-scope", evidence)
    with pytest.raises(ValueError, match="wrong measurement scope"):
        run(screenshots, tmp_path / "cache-wrong-axe", tmp_path / "wrong-axe.json", populate=False)

    evidence = _evidence()
    evidence["viewports"]["mobile"]["screenEvidence"]["composer"]["motion"]["standard"]["selector"] = "#unknowns"
    screenshots = _screenshots(tmp_path / "wrong-motion-selector", evidence)
    with pytest.raises(ValueError, match="wrong measurement selector"):
        run(screenshots, tmp_path / "cache-wrong-motion", tmp_path / "wrong-motion.json", populate=False)

    evidence = _evidence()
    evidence["viewports"]["tablet"]["screenEvidence"]["atlas"]["keyboard"]["scope"] = "#thesis"
    screenshots = _screenshots(tmp_path / "wrong-keyboard-scope", evidence)
    with pytest.raises(ValueError, match="screen keyboard evidence for tablet/atlas has the wrong measurement scope"):
        run(screenshots, tmp_path / "cache-wrong-keyboard", tmp_path / "wrong-keyboard.json", populate=False)


def test_design_jury_rejects_missing_capture_and_header_proof(tmp_path: Path) -> None:
    evidence = _evidence()
    del evidence["viewports"]["desktop"]["screenEvidence"]["thesis"]["capture"]
    screenshots = _screenshots(tmp_path / "missing-capture", evidence)
    with pytest.raises(ValueError, match="capture proof for desktop/thesis is missing"):
        run(screenshots, tmp_path / "cache-missing-capture", tmp_path / "missing-capture.json", populate=False)

    evidence = _evidence()
    del evidence["viewports"]["mobile"]["headerEvidence"]
    screenshots = _screenshots(tmp_path / "missing-header-evidence", evidence)
    with pytest.raises(ValueError, match="header evidence for mobile is missing"):
        run(screenshots, tmp_path / "cache-missing-header", tmp_path / "missing-header.json", populate=False)

    screenshots = _screenshots(tmp_path / "missing-header-file")
    (screenshots / "tablet-header.png").unlink()
    with pytest.raises(FileNotFoundError, match="tablet-header.png"):
        request_spec("thesis", "aesthetics", screenshots, _evidence())


def test_design_jury_rejects_stale_header_capture(tmp_path: Path) -> None:
    screenshots = _screenshots(tmp_path / "stale-header")
    os.utime(screenshots / "desktop-header.png", ns=(1, 1))
    with pytest.raises(RuntimeError, match="jury evidence is stale"):
        run(screenshots, tmp_path / "cache", tmp_path / "jury.json", populate=False)


def test_design_jury_makes_one_openai_call_per_screen_and_critic(tmp_path: Path) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    client = FakeClient()
    cache = tmp_path / "cache"
    result = run(screenshots, cache, tmp_path / "jury.json", populate=True, client=client)

    assert result["gate_passed"] is True
    assert result["independent_critic_calls"] == len(SCREENS) * len(AXES) == 84
    assert len(client.responses.calls) == 84
    assert len(list(cache.glob("*.json"))) == 84
    assert all(len(call["input"][0]["content"]) == 1 + 2 * len(VIEWPORTS) for call in client.responses.calls)
    assert all(call["store"] is False for call in client.responses.calls)
    assert {
        (
            call["text"]["format"]["schema"]["properties"]["screen"]["enum"][0],
            call["text"]["format"]["schema"]["properties"]["axis"]["enum"][0],
        )
        for call in client.responses.calls
    } == {(screen, axis) for screen in SCREENS for axis in AXES}

    cached = run(screenshots, cache, tmp_path / "jury-from-cache.json", populate=False)
    assert cached["minimum_score"] == 8
    assert cached["screens"] == result["screens"]


def test_populate_reuses_only_valid_bound_cache_records(tmp_path: Path) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    cache = tmp_path / "cache"
    run(screenshots, cache, tmp_path / "first.json", populate=True, client=FakeClient())

    resumed = FakeClient()
    run(screenshots, cache, tmp_path / "resumed.json", populate=True, client=resumed)
    assert resumed.responses.calls == []

    spec, _ = request_spec("thesis", "aesthetics", screenshots, _evidence())
    cache_path = cache / f"{cache_key(spec)}.json"
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    record["critique"]["axis"] = "clarity"
    cache_path.write_text(json.dumps(record), encoding="utf-8")
    repaired = FakeClient()
    run(screenshots, cache, tmp_path / "repaired.json", populate=True, client=repaired)
    assert len(repaired.responses.calls) == 1

    record = json.loads(cache_path.read_text(encoding="utf-8"))
    record["request_binding"]["model"] = "wrong-model"
    cache_path.write_text(json.dumps(record), encoding="utf-8")
    rebound = FakeClient()
    run(screenshots, cache, tmp_path / "rebound.json", populate=True, client=rebound)
    assert len(rebound.responses.calls) == 1


def test_population_is_bounded_and_output_order_is_deterministic(tmp_path: Path) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    client = FakeClient(delay=0.002)
    result = run(screenshots, tmp_path / "cache", tmp_path / "jury.json", populate=True, client=client, max_workers=3)

    assert 1 < client.responses.max_active <= 3
    assert [row["screen"] for row in result["screens"]] == list(SCREENS)
    assert all(list(row["axes"]) == list(AXES) for row in result["screens"])


def test_population_failure_keeps_valid_caches_for_resume_without_final_output(tmp_path: Path) -> None:
    screenshots = _screenshots(tmp_path / "screens")
    cache = tmp_path / "cache"
    output = tmp_path / "jury.json"
    failing = FakeClient(delay=0.001, fail=("live-wire", "clarity"))

    with pytest.raises(RuntimeError, match="jury population failed for live-wire/clarity") as raised:
        run(screenshots, cache, output, populate=True, client=failing, max_workers=4)
    assert "must-not-leak" not in str(raised.value)
    assert not output.exists()
    assert len(list(cache.glob("*.json"))) == len(SCREENS) * len(AXES) - 1
    assert failing.responses.max_active <= 4

    resumed = FakeClient()
    result = run(screenshots, cache, output, populate=True, client=resumed, max_workers=4)
    assert result["gate_passed"] is True
    assert output.is_file()
    assert len(resumed.responses.calls) == 1
