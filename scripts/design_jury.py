"""Run seven independent vision critics over real-server screen evidence."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.6-sol"
OPENAI_PROVIDER = "openai"
CODEX_PROVIDER = "codex"
PROVIDERS = (OPENAI_PROVIDER, CODEX_PROVIDER)
CODEX_BINARY = "codex"
CODEX_VERSION_MAX_LENGTH = 80
CODEX_ENV_BLOCKLIST = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "BASE_URL",
    "ORG_ID",
)
VIEWPORTS = ("desktop", "tablet", "mobile")
SCREENS = (
    "thesis",
    "atlas",
    "live-wire",
    "mandate",
    "injection",
    "precedent",
    "arena",
    "vault",
    "metrics",
    "economics",
    "composer",
    "unknowns",
)
CRITICS = {
    "aesthetics": "Judge composition, typography, color, polish, and visual hierarchy across all three viewports.",
    "motion_quality": "Judge motion only from the measured animation frames, reduced-motion proof, and frame timing.",
    "clarity": "Judge whether the screen communicates its claim and next action without explanation from a presenter.",
    "relevance": "Judge whether the screen advances the intent-integrity defense story and Mastercard hackathon brief.",
    "memorability": "Judge whether the screen has a distinctive, demo-worthy visual idea rather than generic dashboard styling.",
    "performance": "Judge performance only from measured browser timing, layout stability, and transfer evidence.",
    "accessibility": "Judge accessibility only from measured WCAG, keyboard, touch-target, overflow, and reduced-motion evidence.",
}
AXES = tuple(CRITICS)
TRANSIENT_LIFECYCLE_EXCLUSION = {
    "selector": ".react-flow__edges > *",
    "reason": (
        "React Flow replaces its rendered edge subtree for one frame during layout; "
        "the stable edge container, viewport, and app nodes remain sampled."
    ),
}
CLS_SESSION_GAP_LIMIT_MS = 1_000
CLS_SESSION_DURATION_LIMIT_MS = 5_000


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _critic_schema(screen: str, axis: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "screen": {"type": "string", "enum": [screen]},
            "axis": {"type": "string", "enum": [axis]},
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
            "evidence": {"type": "string", "minLength": 12},
            "fix": {"type": "string", "minLength": 4},
        },
        "required": ["screen", "axis", "score", "evidence", "fix"],
        "additionalProperties": False,
    }


def _axis_evidence(
    evidence: dict[str, Any], screen: str, axis: str
) -> dict[str, Any]:
    shared = (
        "viewport",
        "screenCount",
        "capturedScreens",
        "apiResponses",
        "fixtureRailCount",
        "browserFailures",
    )
    viewports: dict[str, Any] = {}
    for viewport, row in evidence["viewports"].items():
        selected = {key: row[key] for key in shared if key in row}
        screen_evidence = row["screenEvidence"][screen]
        selected.update(
            {
                "screenCapture": screen_evidence["capture"],
                "headerEvidence": row["headerEvidence"],
            }
        )
        if axis == "motion_quality":
            selected.update(
                {
                    "screenMotion": screen_evidence["motion"],
                    "frameTiming": row["frameTiming"],
                }
            )
        elif axis == "performance":
            selected.update(
                {
                    key: row[key]
                    for key in ("pagePerformance", "frameTiming", "documentWidths")
                }
            )
        elif axis == "accessibility":
            selected.update(
                {
                    "screenAccessibility": screen_evidence["accessibility"],
                    "screenTouchTargetViolations": screen_evidence[
                        "touchTargetViolations"
                    ],
                    "screenDocumentWidths": screen_evidence["documentWidths"],
                    "screenKeyboard": screen_evidence["keyboard"],
                    "screenReducedMotion": screen_evidence["motion"]["reduced"],
                    "documentAccessibility": row["accessibility"],
                }
            )
            if screen == "thesis":
                selected.update(
                    {
                        key: row[key]
                        for key in ("navigation", "skipLinkTarget")
                    }
                )
        viewports[viewport] = selected
    return {"screen": screen, "viewports": viewports}


def request_spec(
    screen: str,
    axis: str,
    screenshots: Path,
    evidence: dict[str, Any],
    *,
    provider: str = OPENAI_PROVIDER,
) -> tuple[dict[str, Any], list[Path]]:
    if screen not in SCREENS:
        raise ValueError(f"unknown screen: {screen}")
    if axis not in AXES:
        raise ValueError(f"unknown critic axis: {axis}")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown jury provider: {provider}")
    image_inputs = [
        (viewport, "section", screenshots / f"{viewport}-{screen}.png")
        for viewport in VIEWPORTS
    ] + [
        (viewport, "header", screenshots / f"{viewport}-header.png")
        for viewport in VIEWPORTS
    ]
    images = [path for _, _, path in image_inputs]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing real-server jury screenshots: {missing}")
    image_evidence = [
        {
            "viewport": viewport,
            "capture": capture,
            "screenshot": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for viewport, capture, path in image_inputs
    ]
    prompt = (
        f"You are the independent `{axis}` critic for the AI Defense Lab `{screen}` screen. {CRITICS[axis]} "
        "Return only this axis; do not infer evidence or coordinate with another critic. A score of 8 means genuinely "
        "hackathon-demo ready, not merely functional. Score below 8 when required evidence is absent. "
        "The screenshots are from the real server: the first three are the isolated section at desktop, tablet, and "
        "mobile; the final three are the actual global header at those viewports. Measured Playwright evidence: "
        + json.dumps(
            _axis_evidence(evidence, screen, axis),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    schema = _critic_schema(screen, axis)
    spec = {
        "schema_version": 4,
        "provider": provider,
        "model": MODEL,
        "critic": axis,
        "instructions": "Be strict and evidence-bound. Never reward hidden functionality or treat missing proof as a pass.",
        "prompt": prompt,
        "images": image_evidence,
        "reasoning": {"effort": "high"},
        "max_output_tokens": 1_000,
        "store": False,
        "text": {"format": {"type": "json_schema", "name": f"design_jury_{axis}", "strict": True, "schema": schema}},
    }
    return spec, images


def cache_key(spec: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(spec)).hexdigest()


def request_binding(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha256": cache_key(spec),
        "schema_version": spec["schema_version"],
        "provider": spec["provider"],
        "model": spec["model"],
        "critic": spec["critic"],
        "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest(),
        "instructions_sha256": hashlib.sha256(spec["instructions"].encode()).hexdigest(),
        "images": spec["images"],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate_axe_evidence(
    value: object, label: str, *, expected_scope: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")
    if value.get("scope") != expected_scope:
        raise ValueError(f"{label} has the wrong measurement scope")
    for kind in ("violations", "incomplete"):
        findings = value.get(kind)
        if not isinstance(findings, list):
            raise ValueError(f"{label} has no {kind} list")
        for finding in findings:
            if not isinstance(finding, dict):
                raise ValueError(f"{label} has an invalid {kind} finding")
            node_count = finding.get("nodeCount")
            nodes = finding.get("nodes")
            truncated = finding.get("nodesTruncated")
            if (
                not isinstance(finding.get("id"), str)
                or not finding["id"]
                or isinstance(node_count, bool)
                or not isinstance(node_count, int)
                or node_count < 1
                or not isinstance(nodes, list)
                or not isinstance(truncated, bool)
                or truncated != (len(nodes) < node_count)
                or len(nodes) > node_count
                or (kind == "violations" and truncated)
            ):
                raise ValueError(f"{label} has an invalid {kind} summary")
            for node in nodes:
                if (
                    not isinstance(node, dict)
                    or not isinstance(node.get("target"), list)
                    or not node["target"]
                    or not all(isinstance(target, str) and target for target in node["target"])
                    or not isinstance(node.get("html"), str)
                    or not node["html"]
                    or not isinstance(node.get("failureSummary"), (str, type(None)))
                ):
                    raise ValueError(f"{label} has an invalid {kind} node")
    return value


def _validate_keyboard_evidence(
    value: object, label: str, *, expected_scope: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")
    if value.get("scope") != expected_scope:
        raise ValueError(f"{label} has the wrong measurement scope")
    steps = value.get("steps")
    expected = value.get("expectedFocusableCount")
    visited = value.get("visitedFocusableCount")
    if (
        not isinstance(steps, list)
        or isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < 0
        or isinstance(visited, bool)
        or not isinstance(visited, int)
        or visited < 0
        or expected != visited
        or visited != len(steps)
    ):
        raise ValueError(f"{label} has an invalid traversal summary")
    for field in (
        "allExpectedVisited",
        "allFocusVisible",
        "allHaveFocusIndicator",
    ):
        if value.get(field) is not True:
            raise ValueError(f"{label} failed {field}")
    indexes: set[int] = set()
    for step in steps:
        index = step.get("index") if isinstance(step, dict) else None
        if (
            not isinstance(step, dict)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in indexes
            or not all(
                isinstance(step.get(field), str) and step[field]
                for field in ("target", "name", "role")
            )
            or step.get("focusVisible") is not True
            or step.get("hasVisibleFocusIndicator") is not True
        ):
            raise ValueError(f"{label} has an invalid traversal step")
        indexes.add(index)
    return value


def _validate_capture_evidence(
    value: object, viewport: str, screen: str
) -> dict[str, Any]:
    label = f"capture proof for {viewport}/{screen}"
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")
    expected = {
        "sectionMode": "isolated-section",
        "fixedHeaderVisibility": "hidden-during-section-capture",
        "screenshot": f"{viewport}-{screen}.png",
        "headerScreenshot": f"{viewport}-header.png",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError(f"{label} is invalid")
    return value


def _validate_header_evidence(value: object, viewport: str) -> dict[str, Any]:
    label = f"header evidence for {viewport}"
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")
    if (
        value.get("scope") != "header"
        or value.get("captureMode") != "actual-global-header"
        or value.get("screenshot") != f"{viewport}-header.png"
    ):
        raise ValueError(f"{label} has invalid capture proof")
    accessibility = _validate_axe_evidence(
        value.get("accessibility"),
        f"header accessibility for {viewport}",
        expected_scope="header",
    )
    if accessibility["violations"] != []:
        raise ValueError(f"real Playwright evidence contains header accessibility violations for {viewport}")
    _validate_keyboard_evidence(
        value.get("keyboard"),
        f"header keyboard evidence for {viewport}",
        expected_scope="header",
    )
    return value


def _validate_motion_sample(
    value: object, label: str, *, reduced: bool, expected_selector: str
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")
    if value.get("selector") != expected_selector:
        raise ValueError(f"{label} has the wrong measurement selector")
    numeric_fields = (
        "sampledElements",
        "sampleFrames",
        "sampleDurationMs",
        "averageFrameMs",
        "worstFrameMs",
        "transitionElementCount",
        "transitionCapableElementCount",
        "activeAnimationCount",
        "registeredAnimationCount",
        "changedElementCount",
    )
    for field in numeric_fields:
        measurement = value.get(field)
        if (
            isinstance(measurement, bool)
            or not isinstance(measurement, (int, float))
            or measurement < 0
        ):
            raise ValueError(f"{label} has an invalid {field}")
    if value["sampledElements"] < 1 or value["sampleFrames"] < 1:
        raise ValueError(f"{label} contains no measured samples")
    if not isinstance(value.get("trigger"), str) or not value["trigger"].strip():
        raise ValueError(f"{label} has no measurement trigger")
    if value["transitionCapableElementCount"] != value["transitionElementCount"]:
        raise ValueError(f"{label} has inconsistent transition counts")
    if value["activeAnimationCount"] > value["registeredAnimationCount"]:
        raise ValueError(f"{label} has inconsistent animation counts")
    if value.get("mediaMatches") is not reduced:
        raise ValueError(f"{label} has the wrong reduced-motion media state")
    lifecycle_exclusions = value.get("transientLifecycleExclusions")
    if (
        not isinstance(lifecycle_exclusions, dict)
        or set(lifecycle_exclusions)
        != {"selector", "reason", "excludedElementCount"}
        or lifecycle_exclusions.get("selector")
        != TRANSIENT_LIFECYCLE_EXCLUSION["selector"]
        or lifecycle_exclusions.get("reason")
        != TRANSIENT_LIFECYCLE_EXCLUSION["reason"]
        or isinstance(lifecycle_exclusions.get("excludedElementCount"), bool)
        or not isinstance(lifecycle_exclusions.get("excludedElementCount"), int)
        or lifecycle_exclusions["excludedElementCount"] < 0
    ):
        raise ValueError(f"{label} has invalid transient lifecycle exclusions")
    changes = value.get("changes")
    changed_count = value["changedElementCount"]
    if (
        not isinstance(changes, list)
        or not isinstance(value.get("changesTruncated"), bool)
        or value["changesTruncated"] != (len(changes) < changed_count)
        or len(changes) > changed_count
        or changed_count > value["sampledElements"]
    ):
        raise ValueError(f"{label} has an invalid change summary")
    change_targets: set[str] = set()
    for change in changes:
        target = change.get("target") if isinstance(change, dict) else None
        if not isinstance(target, str) or not target or target in change_targets:
            raise ValueError(f"{label} has an invalid change target")
        change_targets.add(target)
    trajectories = value.get("trajectories")
    trajectories_truncated = value.get("trajectoriesTruncated")
    if (
        not isinstance(trajectories, list)
        or not isinstance(trajectories_truncated, bool)
        or len(trajectories) > changed_count
        or trajectories_truncated != (len(trajectories) < changed_count)
        or (changed_count == 0 and trajectories != [])
        or (changed_count > 0 and not trajectories)
    ):
        raise ValueError(f"{label} has an invalid trajectory summary")
    trajectory_targets: set[str] = set()
    for trajectory in trajectories:
        target = trajectory.get("target") if isinstance(trajectory, dict) else None
        frames = trajectory.get("frames") if isinstance(trajectory, dict) else None
        if (
            not isinstance(target, str)
            or not target
            or target in trajectory_targets
            or target not in change_targets
            or not isinstance(frames, list)
            or len(frames) < 2
        ):
            raise ValueError(f"{label} has an invalid trajectory")
        trajectory_targets.add(target)
        previous_frame = -1
        previous_elapsed = -1.0
        for frame in frames:
            frame_index = frame.get("frame") if isinstance(frame, dict) else None
            elapsed = frame.get("elapsedMs") if isinstance(frame, dict) else None
            state = frame.get("state") if isinstance(frame, dict) else None
            box = state.get("box") if isinstance(state, dict) else None
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index <= previous_frame
                or frame_index > value["sampleFrames"]
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or elapsed < previous_elapsed
                or not isinstance(state, dict)
                or not isinstance(state.get("opacity"), str)
                or not isinstance(state.get("transform"), str)
                or not isinstance(box, dict)
                or any(
                    isinstance(box.get(field), bool)
                    or not isinstance(box.get(field), (int, float))
                    for field in ("x", "y", "width", "height")
                )
            ):
                raise ValueError(f"{label} has an invalid trajectory frame")
            previous_frame = frame_index
            previous_elapsed = elapsed
        if frames[0]["frame"] != 0 or frames[-1]["frame"] != value["sampleFrames"]:
            raise ValueError(f"{label} has an incomplete trajectory")
    if reduced and changed_count != 0:
        raise ValueError(f"{label} recorded motion despite reduced-motion mode")


def _validate_page_performance(value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing")

    def number(measurement: object, *, minimum: float = 0) -> bool:
        return (
            not isinstance(measurement, bool)
            and isinstance(measurement, (int, float))
            and math.isfinite(measurement)
            and measurement >= minimum
        )

    cls = value.get("cls")
    if not number(cls) or cls >= 0.1:
        raise ValueError(f"{label} has invalid CLS")
    attribution = value.get("clsAttribution")
    if (
        not isinstance(attribution, dict)
        or set(attribution)
        != {
            "algorithm",
            "sessionGapLimitMs",
            "sessionDurationLimitMs",
            "windows",
            "recentInputShifts",
        }
        or attribution.get("algorithm") != "maximum-session-window"
        or attribution.get("sessionGapLimitMs") != CLS_SESSION_GAP_LIMIT_MS
        or attribution.get("sessionDurationLimitMs")
        != CLS_SESSION_DURATION_LIMIT_MS
        or not isinstance(attribution.get("windows"), list)
        or not isinstance(attribution.get("recentInputShifts"), list)
    ):
        raise ValueError(f"{label} has invalid CLS attribution")

    def validate_shift(shift: object) -> dict[str, Any]:
        if (
            not isinstance(shift, dict)
            or set(shift) != {"startTime", "value", "sources"}
            or not number(shift.get("startTime"))
            or not number(shift.get("value"))
            or not isinstance(shift.get("sources"), list)
        ):
            raise ValueError(f"{label} has invalid layout-shift attribution")
        for source in shift["sources"]:
            if (
                not isinstance(source, dict)
                or set(source) != {"target", "previousRect", "currentRect"}
                or not isinstance(source.get("target"), str)
                or not source["target"].strip()
            ):
                raise ValueError(f"{label} has invalid layout-shift source")
            for rect_name in ("previousRect", "currentRect"):
                rect = source.get(rect_name)
                if (
                    not isinstance(rect, dict)
                    or set(rect) != {"x", "y", "width", "height"}
                    or any(not number(rect.get(field), minimum=-math.inf) for field in ("x", "y"))
                    or any(not number(rect.get(field)) for field in ("width", "height"))
                ):
                    raise ValueError(f"{label} has invalid layout-shift source rectangle")
        return shift

    for shift in attribution["recentInputShifts"]:
        validate_shift(shift)

    previous_window: dict[str, Any] | None = None
    window_scores: list[float] = []
    for window in attribution["windows"]:
        if (
            not isinstance(window, dict)
            or set(window) != {"startTime", "endTime", "value", "shifts"}
            or not number(window.get("startTime"))
            or not number(window.get("endTime"))
            or not number(window.get("value"))
            or not isinstance(window.get("shifts"), list)
            or not window["shifts"]
        ):
            raise ValueError(f"{label} has invalid CLS session window")
        shifts = [validate_shift(shift) for shift in window["shifts"]]
        if (
            not math.isclose(window["startTime"], shifts[0]["startTime"], abs_tol=1e-9)
            or not math.isclose(window["endTime"], shifts[-1]["startTime"], abs_tol=1e-9)
            or window["endTime"] < window["startTime"]
            or window["endTime"] - window["startTime"]
            > CLS_SESSION_DURATION_LIMIT_MS
            or any(
                current["startTime"] < prior["startTime"]
                or current["startTime"] - prior["startTime"]
                > CLS_SESSION_GAP_LIMIT_MS
                for prior, current in zip(shifts, shifts[1:])
            )
            or any(
                shift["startTime"] - window["startTime"]
                > CLS_SESSION_DURATION_LIMIT_MS
                for shift in shifts
            )
            or not math.isclose(
                window["value"],
                sum(shift["value"] for shift in shifts),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{label} has invalid CLS session window")
        if previous_window is not None and not (
            window["startTime"] - previous_window["endTime"]
            > CLS_SESSION_GAP_LIMIT_MS
            or window["startTime"] - previous_window["startTime"]
            > CLS_SESSION_DURATION_LIMIT_MS
        ):
            raise ValueError(f"{label} has incorrectly split CLS session windows")
        previous_window = window
        window_scores.append(window["value"])

    expected_cls = max(window_scores, default=0)
    if not math.isclose(cls, expected_cls, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} has inconsistent CLS maximum")


def _validate_evidence(evidence: object) -> dict[str, Any]:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("viewports"), dict):
        raise ValueError("real Playwright evidence must contain viewport measurements")
    viewports = evidence["viewports"]
    if set(viewports) != set(VIEWPORTS):
        raise ValueError("real Playwright evidence must contain desktop, tablet, and mobile")
    for viewport in VIEWPORTS:
        row = viewports[viewport]
        if not isinstance(row, dict) or row.get("screenCount") != len(SCREENS) or row.get("capturedScreens") != list(SCREENS):
            raise ValueError(f"real Playwright evidence is incomplete for {viewport}")
        if row.get("fixtureRailCount") != 0 or row.get("touchTargetViolations") != []:
            raise ValueError(f"real Playwright evidence contains fixture or touch-target failures for {viewport}")
        failures = row.get("browserFailures")
        if not isinstance(failures, dict) or any(failures.get(kind) != [] for kind in ("console", "page", "request")):
            raise ValueError(f"real Playwright evidence contains browser failures for {viewport}")
        accessibility = _validate_axe_evidence(
            row.get("accessibility"),
            f"document accessibility for {viewport}",
            expected_scope="document",
        )
        if accessibility["violations"] != []:
            raise ValueError(f"real Playwright evidence contains accessibility violations for {viewport}")
        _validate_page_performance(
            row.get("pagePerformance"), f"page performance for {viewport}"
        )
        _validate_header_evidence(row.get("headerEvidence"), viewport)
        screen_evidence = row.get("screenEvidence")
        if not isinstance(screen_evidence, dict) or set(screen_evidence) != set(
            SCREENS
        ):
            raise ValueError(
                f"real Playwright evidence has incomplete screen evidence for {viewport}"
            )
        for screen in SCREENS:
            measured = screen_evidence[screen]
            if not isinstance(measured, dict):
                raise ValueError(
                    f"real Playwright evidence is invalid for {viewport}/{screen}"
                )
            _validate_capture_evidence(measured.get("capture"), viewport, screen)
            screen_accessibility = _validate_axe_evidence(
                measured.get("accessibility"),
                f"screen accessibility for {viewport}/{screen}",
                expected_scope=f"#{screen}",
            )
            if screen_accessibility["violations"] != []:
                raise ValueError(
                    f"real Playwright evidence contains accessibility violations for {viewport}/{screen}"
                )
            if measured.get("touchTargetViolations") != []:
                raise ValueError(
                    f"real Playwright evidence contains touch-target failures for {viewport}/{screen}"
                )
            _validate_keyboard_evidence(
                measured.get("keyboard"),
                f"screen keyboard evidence for {viewport}/{screen}",
                expected_scope=f"#{screen}",
            )
            widths = measured.get("documentWidths")
            if not isinstance(widths, dict) or widths.get("overflowsViewport") is not False:
                raise ValueError(
                    f"real Playwright evidence contains overflow for {viewport}/{screen}"
                )
            motion = measured.get("motion")
            if not isinstance(motion, dict):
                raise ValueError(
                    f"real Playwright evidence has no motion proof for {viewport}/{screen}"
                )
            _validate_motion_sample(
                motion.get("standard"),
                f"standard motion for {viewport}/{screen}",
                reduced=False,
                expected_selector=f"#{screen}",
            )
            _validate_motion_sample(
                motion.get("reduced"),
                f"reduced motion for {viewport}/{screen}",
                reduced=True,
                expected_selector=f"#{screen}",
            )
        responses = row.get("apiResponses")
        if not isinstance(responses, dict) or not responses:
            raise ValueError(f"real Playwright evidence contains no API responses for {viewport}")
        if any(not isinstance(response, dict) or response.get("status") != 200 or response.get("fixtureHeader") is not None for response in responses.values()):
            raise ValueError(f"real Playwright evidence contains non-production API responses for {viewport}")
    return evidence


def _require_fresh_capture(screenshots: Path) -> None:
    frontend = ROOT / "app" / "frontend"
    sources = [path for path in (frontend / "src").rglob("*") if path.is_file()]
    sources.extend((frontend / "tests-real" / "real-server.spec.ts", frontend / "playwright.real.config.ts"))
    newest_source = max(path.stat().st_mtime_ns for path in sources)
    evidence_files = [
        screenshots / "evidence.json",
        *(screenshots / f"{viewport}-{screen}.png" for viewport in VIEWPORTS for screen in SCREENS),
        *(screenshots / f"{viewport}-header.png" for viewport in VIEWPORTS),
    ]
    stale = [path.name for path in evidence_files if path.is_file() and path.stat().st_mtime_ns < newest_source]
    if stale:
        raise RuntimeError(f"real-server jury evidence is stale; rerun the real Playwright suite ({len(stale)} files)")


def _validate_critique(screen: str, axis: str, critique: object) -> dict[str, Any]:
    if not isinstance(critique, dict) or set(critique) != {"screen", "axis", "score", "evidence", "fix"}:
        raise ValueError(f"jury returned an invalid payload for {screen}/{axis}")
    if critique["screen"] != screen or critique["axis"] != axis:
        raise ValueError(f"jury returned the wrong screen or axis for {screen}/{axis}")
    score = critique["score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
        raise ValueError(f"jury returned an invalid score for {screen}/{axis}")
    if not isinstance(critique["evidence"], str) or len(critique["evidence"].strip()) < 12:
        raise ValueError(f"jury returned insufficient evidence for {screen}/{axis}")
    if not isinstance(critique["fix"], str) or len(critique["fix"].strip()) < 4:
        raise ValueError(f"jury returned an invalid fix for {screen}/{axis}")
    return critique


def _call_critic(client: Any, spec: dict[str, Any], images: list[Path]) -> tuple[dict[str, Any], Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": spec["prompt"]}]
    for image in images:
        encoded = base64.b64encode(image.read_bytes()).decode()
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "high"})
    response = client.responses.create(
        model=spec["model"],
        instructions=spec["instructions"],
        input=[{"role": "user", "content": content}],
        reasoning=spec["reasoning"],
        max_output_tokens=spec["max_output_tokens"],
        store=False,
        text=spec["text"],
    )
    return json.loads(response.output_text), response


def _codex_child_env() -> dict[str, str]:
    environment = os.environ.copy()
    for name in CODEX_ENV_BLOCKLIST:
        environment.pop(name, None)
    return environment


def _bounded_cli_version(value: str) -> str:
    first_line = value.strip().splitlines()[0].strip() if value.strip() else ""
    if not first_line:
        raise RuntimeError("codex CLI version was unavailable")
    return first_line[:CODEX_VERSION_MAX_LENGTH]


def _codex_cli_version() -> str:
    try:
        result = subprocess.run(
            [CODEX_BINARY, "--version"],
            cwd=ROOT,
            env=_codex_child_env(),
            capture_output=True,
            check=False,
            shell=False,
            text=True,
        )
    except Exception:
        raise RuntimeError("codex CLI version check failed") from None
    if result.returncode != 0:
        raise RuntimeError("codex CLI version check failed")
    try:
        return _bounded_cli_version(result.stdout)
    except (AttributeError, TypeError):
        raise RuntimeError("codex CLI version was unavailable") from None


def _call_codex(
    spec: dict[str, Any],
    images: list[Path],
    *,
    cli_version: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if len(images) != 2 * len(VIEWPORTS):
        raise ValueError("codex jury requires six images")
    schema = spec["text"]["format"]["schema"]
    with tempfile.TemporaryDirectory(prefix="design-jury-codex-") as directory:
        directory_path = Path(directory)
        schema_path = directory_path / "output-schema.json"
        output_path = directory_path / "last-message.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        command = [
            CODEX_BINARY,
            "exec",
            "--model",
            spec["model"],
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
        ]
        for image in images:
            command.extend(("--image", str(image)))
        try:
            result = subprocess.run(
                command,
                cwd=directory_path,
                env=_codex_child_env(),
                input=f"{spec['instructions']}\n\n{spec['prompt']}",
                capture_output=True,
                check=False,
                shell=False,
                text=True,
            )
        except Exception:
            raise RuntimeError("codex CLI invocation failed") from None
        if result.returncode != 0:
            raise RuntimeError("codex CLI returned a nonzero exit status")
        try:
            raw_output = output_path.read_bytes()
            output_text = raw_output.decode("utf-8")
        except (OSError, UnicodeError):
            raise RuntimeError("codex CLI produced no final JSON output") from None
        output_sha256 = hashlib.sha256(raw_output).hexdigest()
        try:
            parsed = json.loads(output_text)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("codex CLI returned malformed JSON") from None

    screen = spec["text"]["format"]["schema"]["properties"]["screen"]["enum"][0]
    axis = spec["text"]["format"]["schema"]["properties"]["axis"]["enum"][0]
    critique = _validate_critique(screen, axis, parsed)
    return critique, {
        "provider": CODEX_PROVIDER,
        "cli_version": _bounded_cli_version(cli_version),
        "model": spec["model"],
        "output_sha256": output_sha256,
    }


def _validate_codex_provenance(record: dict[str, Any], spec: dict[str, Any]) -> dict[str, str]:
    provenance = record.get("provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {"provider", "cli_version", "model", "output_sha256"}
        or provenance.get("provider") != CODEX_PROVIDER
        or provenance.get("model") != spec["model"]
        or not isinstance(provenance.get("cli_version"), str)
        or not 1 <= len(provenance["cli_version"]) <= CODEX_VERSION_MAX_LENGTH
        or "\n" in provenance["cli_version"]
        or not isinstance(provenance.get("output_sha256"), str)
        or len(provenance["output_sha256"]) != hashlib.sha256().digest_size * 2
        or any(character not in "0123456789abcdef" for character in provenance["output_sha256"])
    ):
        raise ValueError("cache Codex provenance is invalid")
    return provenance


def _cached_record(cache_path: Path, spec: dict[str, Any], screen: str, axis: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    key = cache_key(spec)
    if (
        not isinstance(record, dict)
        or record.get("cache_key") != key
        or record.get("provider") != spec["provider"]
        or record.get("request_binding") != request_binding(spec)
    ):
        raise ValueError("cache binding mismatch")
    critique = _validate_critique(screen, axis, record.get("critique"))
    if spec["provider"] == CODEX_PROVIDER:
        _validate_codex_provenance(record, spec)
    else:
        response = record.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("id"), str) or not response["id"]:
            raise ValueError("cache response metadata is invalid")
    return record, critique


def run(
    screenshots: Path,
    cache_dir: Path,
    output: Path,
    *,
    populate: bool,
    provider: str = OPENAI_PROVIDER,
    client: Any = None,
    max_workers: int = 8,
) -> dict[str, Any]:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown jury provider: {provider}")
    evidence_path = screenshots / "evidence.json"
    if not evidence_path.is_file():
        raise FileNotFoundError("real Playwright evidence.json is required")
    _require_fresh_capture(screenshots)
    evidence = _validate_evidence(json.loads(evidence_path.read_text(encoding="utf-8")))
    if populate and provider == OPENAI_PROVIDER and client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for build-time jury population")
        from openai import OpenAI

        client = OpenAI()

    jobs = [
        (
            screen,
            axis,
            *request_spec(screen, axis, screenshots, evidence, provider=provider),
        )
        for screen in SCREENS
        for axis in AXES
    ]
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    codex_version: str | None = None
    codex_version_lock = threading.Lock()

    def get_codex_version() -> str:
        nonlocal codex_version
        with codex_version_lock:
            if codex_version is None:
                codex_version = _codex_cli_version()
            return codex_version

    def load_or_populate(job: tuple[str, str, dict[str, Any], list[Path]]) -> tuple[dict[str, Any], dict[str, Any]]:
        screen, axis, spec, images = job
        key = cache_key(spec)
        cache_path = cache_dir / f"{key}.json"
        if cache_path.is_file():
            try:
                return _cached_record(cache_path, spec, screen, axis)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                if not populate:
                    raise RuntimeError(f"jury cache binding or payload mismatch for {screen}/{axis}") from None
        elif not populate:
            raise RuntimeError(f"jury cache miss for {screen}/{axis}; run the explicit build-time populate mode")

        if provider == CODEX_PROVIDER:
            critique, provenance = _call_codex(
                spec,
                images,
                cli_version=get_codex_version(),
            )
            response = None
        else:
            critique, response = _call_critic(client, spec, images)
            critique = _validate_critique(screen, axis, critique)
        record = {
            "cache_key": key,
            "provider": provider,
            "request_binding": request_binding(spec),
            "critique": critique,
        }
        if provider == CODEX_PROVIDER:
            record["provenance"] = provenance
        else:
            record["response"] = {
                "id": response.id,
                "model": response.model,
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
            }
        _write_json(cache_path, record)
        return record, critique

    if populate:
        completed: list[tuple[dict[str, Any], dict[str, Any]] | None] = [None] * len(jobs)
        failures: list[tuple[int, Exception]] = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
            futures = {executor.submit(load_or_populate, job): index for index, job in enumerate(jobs)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    completed[index] = future.result()
                except Exception as error:
                    failures.append((index, error))
        if failures:
            index, error = min(failures, key=lambda failure: failure[0])
            screen, axis, *_ = jobs[index]
            raise RuntimeError(f"jury population failed for {screen}/{axis} ({type(error).__name__})") from None
        job_results = [result for result in completed if result is not None]
    else:
        job_results = [load_or_populate(job) for job in jobs]

    rows = []
    for screen_index, screen in enumerate(SCREENS):
        axes: dict[str, Any] = {}
        critic_calls: dict[str, Any] = {}
        for axis_index, axis in enumerate(AXES):
            spec = jobs[screen_index * len(AXES) + axis_index][2]
            record, critique = job_results[screen_index * len(AXES) + axis_index]
            key = cache_key(spec)
            axes[axis] = {field: critique[field] for field in ("score", "evidence", "fix")}
            if provider == CODEX_PROVIDER:
                critic_calls[axis] = {"cache_key": key, **record["provenance"]}
            else:
                critic_calls[axis] = {"cache_key": key, "provider": provider, "response_id": record["response"]["id"]}
        failing = sorted((axis for axis in AXES if axes[axis]["score"] < 8), key=lambda axis: (axes[axis]["score"], axis))
        rows.append({
            "screen": screen,
            "verdict": "pass" if not failing else "fix",
            "axes": axes,
            "priority_fixes": [f"{axis}: {axes[axis]['fix']}" for axis in failing[:3]],
            "critic_calls": critic_calls,
            "images": jobs[screen_index * len(AXES)][2]["images"],
        })
    result = {
        "provider": provider,
        "model": MODEL,
        "independent_critic_calls": len(SCREENS) * len(AXES),
        "screens": rows,
        "minimum_score": min(row["axes"][axis]["score"] for row in rows for axis in AXES),
        "gate_passed": all(row["verdict"] == "pass" for row in rows),
        "source": "real-server Playwright screenshots plus measured browser evidence",
    }
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path, default=ROOT / "app" / "frontend" / "jury-screens")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "llm_cache" / "design_jury")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "design_jury.json")
    parser.add_argument("--max-workers", type=int, default=8)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--populate-openai", action="store_true")
    mode.add_argument("--populate-codex", action="store_true")
    mode.add_argument("--from-cache", action="store_true")
    args = parser.parse_args()
    if args.populate_openai:
        provider = OPENAI_PROVIDER
        populate = True
    elif args.populate_codex:
        provider = CODEX_PROVIDER
        populate = True
    else:
        provider = CODEX_PROVIDER
        populate = False
    result = run(
        args.screenshots,
        args.cache_dir,
        args.output,
        populate=populate,
        provider=provider,
        max_workers=args.max_workers,
    )
    print(json.dumps({"gate_passed": result["gate_passed"], "minimum_score": result["minimum_score"]}))
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
