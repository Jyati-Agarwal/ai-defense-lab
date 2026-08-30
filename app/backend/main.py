from __future__ import annotations

import hashlib
import hmac
import mimetypes
import os
import stat
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, Mapping

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from scripts.release import ReleaseInvalid, verify_release

from .artifacts import (
    ArtifactInvalid,
    ArtifactMissing,
    artifact_store_from_signed_release,
)
from .scoring import ScorerUnavailable, load_scorer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ComposeRequest(BaseModel):
    channel: Literal["agentic_checkout", "card_ecom", "upi_p2p", "upi_collect", "wire"] = "agentic_checkout"
    injection_style: Literal["hidden_text", "zero_width", "instruction_shift", "none"] = "hidden_text"
    amount_minor: int = Field(default=7_800_000, ge=1, le=100_000_000)
    hops: int = Field(default=2, ge=0, le=8)
    delegation_depth: int = Field(default=2, ge=0, le=8)
    ttl_delay_seconds: int = Field(default=0, ge=0, le=604_800)
    fanout: int = Field(default=1, ge=1, le=20)
    evasion_aggressiveness: float = Field(default=0.5, ge=0, le=1)
    lie_about_presence: bool = False


LOCAL_ORIGINS = (
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def allowed_origins() -> list[str]:
    """Local dev origins plus any comma-separated extras from the environment.

    Deployed frontends (e.g. the Vercel static build) live on a different origin
    than the API, so the deployment supplies them via AI_DEFENSE_ALLOWED_ORIGINS
    rather than baking hostnames into the image.
    """
    extra = os.environ.get("AI_DEFENSE_ALLOWED_ORIGINS", "")
    origins = list(LOCAL_ORIGINS)
    for candidate in extra.split(","):
        candidate = candidate.strip().rstrip("/")
        if candidate and candidate not in origins:
            origins.append(candidate)
    return origins


def create_app(
    project_root: Path = PROJECT_ROOT,
    scorer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    release_verifier: Callable[[Path], dict[str, Any]] | None = None,
    release_public_key: Path | None = None,
) -> FastAPI:
    verifier = release_verifier or (
        lambda root: verify_release(root, public_key_path=release_public_key)
    )
    dist = project_root / "app" / "frontend" / "dist"

    def open_static_file(relative: str) -> int:
        parts = relative.split("/")
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise OSError("unsafe static path")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory = os.open(dist, directory_flags)
        try:
            for part in parts[:-1]:
                child = os.open(part, directory_flags, dir_fd=directory)
                os.close(directory)
                directory = child
            return os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory,
            )
        finally:
            os.close(directory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            release = verifier(app.state.project_root)
            store = artifact_store_from_signed_release(
                app.state.project_root, release, release_public_key
            )
            store.scored_events()
            runtime_code = release.get("runtime_code")
            runtime_hashes = (
                runtime_code.get("file_sha256")
                if isinstance(runtime_code, Mapping)
                else None
            )
            if dist.is_dir() and not isinstance(runtime_hashes, Mapping):
                raise ReleaseInvalid("signed release has no runtime code file map")
            prefix = "app/frontend/dist/"
            static_hashes = {
                name.removeprefix(prefix): digest
                for name, digest in (runtime_hashes or {}).items()
                if isinstance(name, str) and name.startswith(prefix)
            }
            static_sizes: dict[str, int] = {}
            for relative in static_hashes:
                descriptor = open_static_file(relative)
                try:
                    metadata = os.fstat(descriptor)
                    if not stat.S_ISREG(metadata.st_mode):
                        raise OSError("not a regular file")
                    static_sizes[relative] = metadata.st_size
                finally:
                    os.close(descriptor)
        except Exception as error:
            app.state.readiness_detail = f"release verification failed: {error}"
            yield
            return
        try:
            active_scorer = scorer or load_scorer(app.state.project_root)
            probe = active_scorer(ComposeRequest().model_dump())
            if not isinstance(probe, dict) or not (probe.get("decision") or {}).get("reason_codes"):
                raise ScorerUnavailable("frozen scorer startup probe returned no reason codes")
            if verifier(app.state.project_root) != release:
                raise ScorerUnavailable("release evidence changed during scorer startup")
        except Exception as error:
            app.state.readiness_detail = f"scorer startup failed: {error}"
            yield
            return
        app.state.release = release
        app.state.static_hashes = static_hashes
        app.state.static_sizes = static_sizes
        app.state.store = store
        app.state.scorer = active_scorer
        app.state.ready = True
        app.state.readiness_detail = "verified release and frozen scorer are ready"
        yield

    app = FastAPI(
        title="AI Defense Lab",
        version="1.0.0",
        description="Cached red-team/blue-team evidence for agentic-payment intent integrity.",
        lifespan=lifespan,
    )
    app.state.project_root = project_root
    app.state.store = None
    app.state.scorer = None
    app.state.release = None
    app.state.static_hashes = {}
    app.state.static_sizes = {}
    app.state.ready = False
    app.state.readiness_detail = "application lifespan has not started"
    app.state.release_verifier = verifier

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_origin_regex=os.environ.get("AI_DEFENSE_ALLOWED_ORIGIN_REGEX") or None,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.middleware("http")
    async def require_verified_release(request: Request, call_next):
        if request.url.path != "/health" and not app.state.ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "detail": app.state.readiness_detail},
            )
        return await call_next(request)

    def require_ready() -> None:
        if not app.state.ready:
            raise HTTPException(503, app.state.readiness_detail)

    def read_artifact(name: str) -> Any:
        require_ready()
        try:
            return app.state.store.json(name)
        except (ArtifactInvalid, ArtifactMissing) as error:
            raise HTTPException(503, f"Verified artifact unavailable: {error}") from error

    def read_scored_events() -> tuple[dict[str, Any], ...]:
        require_ready()
        try:
            return app.state.store.scored_events()
        except (ArtifactInvalid, ArtifactMissing) as error:
            raise HTTPException(503, f"Scored event cache unavailable: {error}") from error

    def public_event(event: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in event.items() if key != "label"}

    @app.get("/health", responses={503: {"description": "Release or scorer is not ready"}})
    def health() -> Any:
        if app.state.ready:
            try:
                current_release = app.state.release_verifier(app.state.project_root)
                if current_release != app.state.release:
                    raise ReleaseInvalid("release evidence changed after startup")
            except Exception as error:
                app.state.ready = False
                app.state.readiness_detail = f"release re-verification failed: {error}"
        if not app.state.ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "detail": app.state.readiness_detail},
            )
        return {"status": "ready"}

    @app.get("/atlas")
    def atlas() -> Any:
        return read_artifact("atlas_matrix")

    @app.get("/metrics")
    def metrics() -> Any:
        return read_artifact("metrics")

    @app.get("/generalization")
    def generalization() -> Any:
        return read_artifact("generalization")

    @app.get("/arena")
    def arena() -> Any:
        return read_artifact("arena_log")

    @app.get("/phylogeny")
    def phylogeny() -> Any:
        return read_artifact("phylogeny")

    @app.get("/vaccination")
    def vaccination() -> Any:
        return read_artifact("vaccination")

    @app.get("/stream")
    def stream(n: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        events = read_scored_events()
        flagship = app.state.store.event("evt_flagship")
        if flagship is None:
            selected = events[:n]
        else:
            source_id = flagship.get("source_event_id")
            selected = (
                flagship,
                *(event for event in events if event.get("event_id") != source_id),
            )[:n]
        return {"items": [public_event(event) for event in selected], "count": len(selected)}

    @app.get("/case/{event_id}")
    def case(event_id: str) -> dict[str, Any]:
        require_ready()
        try:
            event = app.state.store.event(event_id)
        except (ArtifactInvalid, ArtifactMissing) as error:
            raise HTTPException(503, f"Scored event cache unavailable: {error}") from error
        if event is None:
            raise HTTPException(404, "Case not found")
        return public_event(event)

    @app.get("/precedent/{event_id}")
    def precedent(event_id: str) -> Any:
        require_ready()
        try:
            match = app.state.store.precedent(event_id)
        except ArtifactMissing as error:
            raise HTTPException(503, f"Precomputed artifact missing: {error}") from error
        except ArtifactInvalid as error:
            raise HTTPException(503, f"Scored event cache unavailable: {error}") from error
        if match is not None:
            return match
        raise HTTPException(404, "Precedent result not found")

    @app.post("/compose")
    def compose(request: ComposeRequest) -> dict[str, Any]:
        require_ready()
        try:
            active_scorer = app.state.scorer
            if active_scorer is None:
                raise ScorerUnavailable("frozen scorer was not initialized")
            started = perf_counter()
            result = active_scorer(request.model_dump())
        except (FileNotFoundError, ScorerUnavailable) as error:
            raise HTTPException(503, str(error)) from error
        if not isinstance(result, dict):
            raise HTTPException(500, "Scorer returned an invalid result")
        decision = result.setdefault("decision", {})
        decision["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        if not decision.get("reason_codes"):
            raise HTTPException(500, "Scorer returned no reason codes")
        return result

    if dist.is_dir():
        @app.api_route("/{frontend_path:path}", methods=["GET", "HEAD"])
        def frontend(frontend_path: str) -> Response:
            relative = frontend_path or "index.html"
            expected = app.state.static_hashes.get(relative)
            expected_size = app.state.static_sizes.get(relative)
            if expected is None or expected_size is None:
                raise HTTPException(404, "Static file not found")

            descriptor: int | None = None
            try:
                descriptor = open_static_file(relative)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
                    raise OSError("not the verified regular file size")
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = None
                    payload = handle.read(expected_size + 1)
                if len(payload) != expected_size:
                    raise OSError("size changed while reading")
            except OSError as error:
                app.state.ready = False
                app.state.readiness_detail = (
                    f"static verification failed for {relative}: {error}"
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "detail": app.state.readiness_detail,
                    },
                )
            finally:
                if descriptor is not None:
                    os.close(descriptor)

            if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected):
                app.state.ready = False
                app.state.readiness_detail = (
                    f"static verification failed for {relative}: hash mismatch"
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "detail": app.state.readiness_detail,
                    },
                )
            return Response(
                content=payload,
                media_type=mimetypes.guess_type(relative)[0]
                or "application/octet-stream",
            )

    return app


app = create_app()
