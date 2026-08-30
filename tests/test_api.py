from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

import app.backend.main as backend_main
from app.backend.artifacts import PUBLIC_KEY_ENV, generate_signing_key, sign_release
from app.backend.main import create_app
from app.backend.store import ArtifactStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(root: Path, name: str, payload: object) -> None:
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def scored_event() -> dict[str, object]:
    return {
        "event_id": "evt_test",
        "scores": {"fused": 0.91},
        "decision": {"action": "BLOCK", "reason_codes": ["Intent and cart diverge"]},
    }


def write_scored_event(root: Path, event: dict[str, object] | None = None) -> None:
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "scored_events.jsonl").write_text(
        json.dumps(event or scored_event()) + "\n", encoding="utf-8"
    )


def injected_scorer(payload: dict[str, object]) -> dict[str, object]:
    return {
        "input": payload,
        "decision": {"action": "BLOCK", "reason_codes": ["Intent and cart diverge"]},
    }


def fixture_private_key(root: Path) -> Path:
    return root.parent / f"{root.name}-api-keys" / "release.pem"


def fixture_public_key(root: Path) -> Path:
    return root.parent / f"{root.name}-api-keys" / "release.pub.pem"


def ensure_fixture_keys(root: Path) -> None:
    private_key = fixture_private_key(root)
    public_key = fixture_public_key(root)
    if not private_key.exists() and not public_key.exists():
        generate_signing_key(root, private_key, public_key)


def fixture_release(root: Path) -> dict[str, object]:
    ensure_fixture_keys(root)
    results = root / "results"
    dist = root / "app" / "frontend" / "dist"
    payload = {
        "schema_version": 4,
        "status": "passed",
        "required_artifact_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in results.glob("*")
            if path.is_file()
        },
        "runtime_code": {
            "file_sha256": {
                path.relative_to(root).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in dist.rglob("*")
                if path.is_file()
            }
        },
    }
    return sign_release(payload, root, fixture_private_key(root))


def fixture_app(
    root: Path,
    scorer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
):
    ensure_fixture_keys(root)
    return create_app(
        root,
        scorer=scorer,
        release_verifier=fixture_release,
        release_public_key=fixture_public_key(root),
    )


def p99(samples: list[float]) -> float:
    return sorted(samples)[math.ceil(len(samples) * 0.99) - 1]


def test_cached_endpoints_and_composer(tmp_path: Path) -> None:
    for name in ("atlas_matrix", "metrics", "generalization", "arena_log", "phylogeny", "vaccination"):
        write_json(tmp_path, name, {"artifact": name})
    write_json(tmp_path, "precedents", {"evt_test": {"support": 0.91}})
    event = scored_event() | {"label": {"is_fraud": True}}
    (tmp_path / "results" / "scored_events.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        assert client.get("/health").json() == {"status": "ready"}
        assert client.get("/metrics").json() == {"artifact": "metrics"}
        streamed = client.get("/stream?n=1").json()["items"][0]
        case = client.get("/case/evt_test").json()
        assert streamed["event_id"] == "evt_test"
        assert "label" not in streamed
        assert "label" not in case
        assert client.get("/precedent/evt_test").json()["support"] == 0.91
        composed = client.post("/compose", json={"amount_minor": 7_800_000}).json()
    assert composed["decision"]["action"] == "BLOCK"
    assert composed["decision"]["reason_codes"]
    assert composed["decision"]["latency_ms"] >= 0


def test_startup_prewarms_scored_events_once_before_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = {
        **scored_event(),
        "event_id": "evt_source",
        "label": {
            "attack_family": "injection_to_payment",
            "is_fraud": False,
        },
        "decision": {
            "action": "BLOCK",
            "legacy_rules_action": "ALLOW",
            "reason_codes": ["Intent and cart diverge"],
        },
    }
    results = tmp_path / "results"
    results.mkdir()
    (results / "scored_events.jsonl").write_text(
        json.dumps(event) + "\n", encoding="utf-8"
    )
    write_json(
        tmp_path,
        "precedents",
        {
            "evt_source": {
                "event_id": "evt_source",
                "verdict": "suppressed",
                "precedent_support": 0.91,
            }
        },
    )

    scored_reads = 0
    original_verified_bytes = ArtifactStore._verified_bytes

    def tracked_verified_bytes(store: ArtifactStore, filename: str) -> bytes:
        nonlocal scored_reads
        if filename == "scored_events.jsonl":
            scored_reads += 1
        return original_verified_bytes(store, filename)

    monkeypatch.setattr(ArtifactStore, "_verified_bytes", tracked_verified_bytes)

    def scorer(payload: dict[str, object]) -> dict[str, object]:
        assert scored_reads == 1
        return injected_scorer(payload)

    with TestClient(fixture_app(tmp_path, scorer=scorer)) as client:
        assert client.get("/health").json() == {"status": "ready"}
        streamed = client.get("/stream?n=1")
        assert streamed.status_code == 200
        assert streamed.json()["items"][0]["event_id"] == "evt_flagship"
        assert streamed.json()["items"][0]["source_event_id"] == "evt_source"
        assert client.get("/case/evt_flagship").status_code == 200
        assert client.get("/precedent/evt_borderline").status_code == 200

    assert scored_reads == 1


def test_missing_artifact_is_not_fabricated(tmp_path: Path) -> None:
    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        response = client.get("/metrics")
    assert response.status_code == 503


def test_raw_events_are_never_served_as_scored_events(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "events.jsonl"
    raw_path.parent.mkdir()
    raw_path.write_text(json.dumps(scored_event()) + "\n", encoding="utf-8")

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        response = client.get("/stream")
    assert response.status_code == 503


@pytest.mark.parametrize(
    "event",
    [
        {
            "event_id": "evt_invalid",
            "scores": {"fused": None},
            "decision": {"action": "BLOCK", "reason_codes": ["x"]},
        },
        {
            "event_id": "evt_invalid",
            "scores": {"fused": 0.9},
            "decision": {"action": None, "reason_codes": ["x"]},
        },
        {
            "event_id": "evt_invalid",
            "scores": {"fused": 0.9},
            "decision": {"action": "BLOCK", "reason_codes": None},
        },
        {
            "event_id": "evt_invalid",
            "scores": {"fused": 0.9},
            "decision": {"action": "BLOCK", "reason_codes": []},
        },
    ],
)
def test_incomplete_scored_event_fails_closed(tmp_path: Path, event: dict[str, object]) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "scored_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        health = client.get("/health")
        response = client.get("/stream")
    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"
    assert response.status_code == 503


def test_borderline_alias_does_not_relabel_a_confirmed_precedent(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    event = scored_event() | {"label": {"is_fraud": False}}
    (results / "scored_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    write_json(
        tmp_path,
        "precedents",
        {
            "evt_test": {
                "event_id": "evt_test",
                "precedent_support": 0.99,
                "verdict": "confirmed",
            }
        },
    )

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        response = client.get("/precedent/evt_borderline")
    assert response.status_code == 404


def test_real_artifacts_and_real_scorer_end_to_end() -> None:
    scored_path = PROJECT_ROOT / "results" / "scored_events.jsonl"
    precedents_path = PROJECT_ROOT / "results" / "precedents.json"
    scored_bytes = scored_path.read_bytes()
    precedents_bytes = precedents_path.read_bytes()
    cold_started = perf_counter()
    with TestClient(create_app(PROJECT_ROOT)) as client:
        cold_startup_ms = (perf_counter() - cold_started) * 1_000

        assert client.get("/health").json() == {"status": "ready"}
        for route, artifact in {
            "/atlas": "atlas_matrix",
            "/metrics": "metrics",
            "/generalization": "generalization",
            "/arena": "arena_log",
            "/phylogeny": "phylogeny",
            "/vaccination": "vaccination",
        }.items():
            response = client.get(route)
            assert response.status_code == 200
            assert response.json() == json.loads(
                (PROJECT_ROOT / "results" / f"{artifact}.json").read_text(encoding="utf-8")
            )

        scored_rows = [
            json.loads(line)
            for line in scored_bytes.decode().splitlines()
            if line.strip()
        ]
        public_rows = [
            {key: value for key, value in row.items() if key != "label"}
            for row in scored_rows
        ]
        stream = client.get("/stream?n=8")
        assert stream.status_code == 200
        flagship_source = next(
            row
            for row in scored_rows
            if row["event_id"] != "evt_flagship"
            and row["label"]["attack_family"] == "injection_to_payment"
            and row["decision"]["legacy_rules_action"] == "ALLOW"
            and row["decision"]["action"] == "BLOCK"
        )
        public_flagship = {
            **{key: value for key, value in flagship_source.items() if key != "label"},
            "source_event_id": flagship_source["event_id"],
            "event_id": "evt_flagship",
        }
        assert stream.json() == {"items": [public_flagship, *public_rows[:7]], "count": 8}

        case = client.get("/case/evt_flagship")
        assert case.status_code == 200
        assert case.json() == public_flagship

        precedents = json.loads(precedents_bytes)
        scored_by_id = {row["event_id"]: row for row in scored_rows}
        precedent_source = max(
            (
                value
                for value in precedents.values()
                if value.get("event_id") != "evt_borderline"
                and value.get("source_event_id") is None
                and value.get("verdict") == "suppressed"
                and scored_by_id[value["event_id"]]["label"]["is_fraud"] is False
            ),
            key=lambda value: value["precedent_support"],
        )
        precedent = client.get("/precedent/evt_borderline")
        assert precedent.status_code == 200
        assert precedent.json() == {
            **precedent_source,
            "source_event_id": precedent_source["event_id"],
            "event_id": "evt_borderline",
        }

        preflight = client.options(
            "/compose",
            headers={
                "Origin": "http://127.0.0.1:4173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://127.0.0.1:4173"

        payloads = [
            {"channel": "agentic_checkout", "injection_style": "hidden_text", "amount_minor": 7_800_000},
            {"channel": "agentic_checkout", "injection_style": "zero_width", "amount_minor": 500_000},
            {"channel": "agentic_checkout", "injection_style": "none", "amount_minor": 250_000},
            {"channel": "card_ecom", "injection_style": "none", "amount_minor": 1_200_000},
            {"channel": "upi_p2p", "injection_style": "none", "amount_minor": 75_000},
            {"channel": "upi_collect", "injection_style": "none", "amount_minor": 90_000},
            {"channel": "wire", "injection_style": "none", "amount_minor": 5_000_000},
        ]
        wall_samples: list[float] = []
        scorer_samples: list[float] = []
        for index in range(100):
            started = perf_counter()
            response = client.post("/compose", json=payloads[index % len(payloads)])
            wall_samples.append((perf_counter() - started) * 1_000)
            assert response.status_code == 200, response.text
            result = response.json()
            assert isinstance(result["scores"]["fused"], (int, float))
            assert result["decision"]["action"]
            assert result["decision"]["reason_codes"]
            scorer_samples.append(result["decision"]["latency_ms"])

    assert scored_path.read_bytes() == scored_bytes
    assert precedents_path.read_bytes() == precedents_bytes
    wall_p99_ms = p99(wall_samples)
    scorer_p99_ms = p99(scorer_samples)
    print(
        f"cold_startup_ms={cold_startup_ms:.3f} "
        f"warm_wall_p99_ms={wall_p99_ms:.3f} "
        f"warm_scorer_p99_ms={scorer_p99_ms:.3f} calls={len(wall_samples)}"
    )
    assert wall_p99_ms < 100
    assert scorer_p99_ms < 100


def test_missing_model_returns_503(tmp_path: Path) -> None:
    write_scored_event(tmp_path)
    with TestClient(fixture_app(tmp_path)) as client:
        health = client.get("/health")
        response = client.post("/compose", json={})
    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"
    assert response.status_code == 503
    assert "model.pkl" in response.json()["detail"]


def test_missing_release_blocks_artifacts_and_does_not_call_injected_scorer(tmp_path: Path) -> None:
    calls = 0

    def scorer(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return injected_scorer(payload)

    write_json(tmp_path, "metrics", {"must_not": "be served"})
    dist = tmp_path / "app" / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><title>unsigned</title>", encoding="utf-8"
    )
    with TestClient(create_app(tmp_path, scorer=scorer)) as client:
        health = client.get("/health")
        metrics = client.get("/metrics")
        compose = client.post("/compose", json={})
        frontend = client.get("/")
    assert {
        health.status_code,
        metrics.status_code,
        compose.status_code,
        frontend.status_code,
    } == {503}
    assert b"unsigned" not in frontend.content
    assert health.json()["status"] == "not_ready"
    assert "release_manifest.json" in health.json()["detail"]
    assert calls == 0


def test_unsigned_injected_release_cannot_make_runtime_ready(tmp_path: Path) -> None:
    calls = 0
    ensure_fixture_keys(tmp_path)

    def scorer(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return injected_scorer(payload)

    def unsigned_release(_root: Path) -> dict[str, object]:
        return {"status": "passed", "required_artifact_sha256": {}}

    dist = tmp_path / "app" / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><title>unsigned</title>", encoding="utf-8"
    )
    with TestClient(
        create_app(
            tmp_path,
            scorer=scorer,
            release_verifier=unsigned_release,
            release_public_key=fixture_public_key(tmp_path),
        )
    ) as client:
        health = client.get("/health")
        metrics = client.get("/metrics")
        frontend = client.get("/")
    assert health.status_code == metrics.status_code == frontend.status_code == 503
    assert b"unsigned" not in frontend.content
    assert "signed custody record" in health.json()["detail"]
    assert calls == 0


def test_missing_configured_public_key_cannot_make_runtime_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PUBLIC_KEY_ENV, raising=False)
    ensure_fixture_keys(tmp_path)

    with TestClient(
        create_app(
            tmp_path,
            scorer=injected_scorer,
            release_verifier=fixture_release,
        )
    ) as client:
        health = client.get("/health")
    assert health.status_code == 503
    assert "public key is not configured" in health.json()["detail"]


def test_artifact_changed_after_startup_is_not_served(tmp_path: Path) -> None:
    write_json(tmp_path, "metrics", {"version": 1})
    write_scored_event(tmp_path)
    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        write_json(tmp_path, "metrics", {"version": 2})
        response = client.get("/metrics")
        health = client.get("/health")
    assert response.status_code == 503
    assert "no longer matches" in response.json()["detail"]
    assert health.status_code == 503
    assert health.json()["status"] == "not_ready"


@pytest.mark.parametrize(
    ("request_path", "relative_path", "original", "tampered"),
    [
        ("/", "index.html", b"<!doctype html><title>verified</title>", b"tampered index"),
        ("/assets/app.js", "assets/app.js", b"console.log('verified')", b"tampered asset"),
    ],
)
def test_static_tamper_after_startup_fails_closed_without_health_poll(
    tmp_path: Path,
    request_path: str,
    relative_path: str,
    original: bytes,
    tampered: bytes,
) -> None:
    path = tmp_path / "app" / "frontend" / "dist" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    write_scored_event(tmp_path)

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        unchanged = client.get(request_path)
        path.write_bytes(tampered)
        response = client.get(request_path)

        assert unchanged.status_code == 200
        assert unchanged.content == original
        assert response.status_code == 503
        assert tampered not in response.content
        assert client.app.state.ready is False


def test_static_fifo_fails_without_a_blocking_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "app" / "frontend" / "dist" / "index.html"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"signed")
    write_scored_event(tmp_path)

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        path.unlink()
        os.mkfifo(path)
        real_open = os.open
        final_flags: list[int] = []

        def guarded_open(
            name: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if name == "index.html" and dir_fd is not None:
                final_flags.append(flags)
                if not flags & os.O_NONBLOCK:
                    raise OSError("test refused a blocking FIFO open")
            return real_open(name, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(backend_main.os, "open", guarded_open)
        response = client.get("/")

        assert final_flags[-1] & os.O_NONBLOCK
        assert response.status_code == 503
        assert client.app.state.ready is False


def test_oversized_static_file_is_rejected_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "app" / "frontend" / "dist" / "index.html"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"signed")
    write_scored_event(tmp_path)

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        path.write_bytes(b"signed but larger")

        def refuse_read(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("oversized static file was opened for reading")

        monkeypatch.setattr(backend_main.os, "fdopen", refuse_read)
        response = client.get("/")

        assert response.status_code == 503
        assert client.app.state.ready is False


def test_intermediate_static_symlink_fails_closed(tmp_path: Path) -> None:
    dist = tmp_path / "app" / "frontend" / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_bytes(b"index")
    (assets / "app.js").write_bytes(b"signed")
    write_scored_event(tmp_path)

    with TestClient(fixture_app(tmp_path, scorer=injected_scorer)) as client:
        assets.rename(dist / "moved-assets")
        assets.symlink_to(dist / "moved-assets", target_is_directory=True)
        response = client.get("/assets/app.js")

        assert response.status_code == 503
        assert response.content != b"signed"
        assert client.app.state.ready is False
