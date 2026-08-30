#!/bin/sh
set -eu

project_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
image=${IMAGE_NAME:-ai-defense-lab:container-smoke}
container="ai-defense-lab-smoke-$$"
python=${PYTHON:-"$project_root/.venv/bin/python"}
public_key=${AI_DEFENSE_RELEASE_PUBLIC_KEY:-}
container_public_key=/run/ai-defense/release-ed25519.pub.pem
trust_dir=
staged_public_key=
started=0

cleanup() {
    if [ "$started" -eq 1 ]; then
        docker rm -f "$container" >/dev/null 2>&1 || true
    fi
    if [ -n "$staged_public_key" ]; then
        rm -f -- "$staged_public_key" 2>/dev/null || true
    fi
    if [ -n "$trust_dir" ]; then
        rmdir -- "$trust_dir" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

if [ -z "$public_key" ]; then
    echo "AI_DEFENSE_RELEASE_PUBLIC_KEY must name an external absolute public-key path" >&2
    exit 2
fi
case "$public_key" in
    /*) ;;
    *)
        echo "AI_DEFENSE_RELEASE_PUBLIC_KEY must be an absolute path" >&2
        exit 2
        ;;
esac
case "$public_key" in
    "$project_root"|"$project_root"/*)
        echo "AI_DEFENSE_RELEASE_PUBLIC_KEY must be outside the repository" >&2
        exit 2
        ;;
esac
if [ ! -x "$python" ]; then
    echo "Python runtime is missing: $python" >&2
    exit 2
fi
"$python" - "$project_root" "$public_key" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
key = Path(sys.argv[2]).resolve(strict=True)
try:
    key.relative_to(root)
except ValueError:
    raise SystemExit(0)
raise SystemExit("AI_DEFENSE_RELEASE_PUBLIC_KEY must be outside the repository")
PY
"$python" "$project_root/scripts/release.py" --root "$project_root" --verify \
    --public-key "$public_key" >/dev/null

if docker container inspect "$container" >/dev/null 2>&1; then
    echo "refusing to replace existing container: $container" >&2
    exit 1
fi

trust_dir=$(mktemp -d)
chmod 0711 "$trust_dir"
staged_public_key="$trust_dir/release-ed25519.pub.pem"
cp -- "$public_key" "$staged_public_key"
chmod 0444 "$staged_public_key"
cmp -s -- "$public_key" "$staged_public_key"
"$python" "$project_root/scripts/release.py" --root "$project_root" --verify \
    --public-key "$staged_public_key" >/dev/null

docker build --tag "$image" "$project_root"
docker run --rm --network none --read-only --user 0 --cap-drop ALL --cap-add CHOWN \
    --mount "type=bind,src=$trust_dir,dst=/trust" \
    "$image" chown 10001:10001 /trust/release-ed25519.pub.pem
cmp -s -- "$public_key" "$staged_public_key"
docker run --detach --name "$container" --network none --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=64m \
    --cap-drop ALL --security-opt no-new-privileges \
    --env "AI_DEFENSE_RELEASE_PUBLIC_KEY=$container_public_key" \
    --mount "type=bind,src=$staged_public_key,dst=$container_public_key,readonly" \
    "$image" >/dev/null
started=1

attempt=0
while [ "$attempt" -lt 180 ]; do
    if docker exec "$container" python -c \
        'import json, urllib.request; assert json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2))["status"] == "ready"' \
        >/dev/null 2>&1; then
        break
    fi
    if [ "$(docker inspect --format '{{.State.Running}}' "$container")" != "true" ]; then
        docker logs "$container" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ "$attempt" -eq 180 ]; then
    docker logs "$container" >&2
    echo "container did not become ready" >&2
    exit 1
fi

[ "$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$container")" = "none" ]
[ "$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "$container")" = "true" ]
[ "$(docker exec "$container" id -u)" = "10001" ]
[ "$(docker exec "$container" printenv HF_HUB_OFFLINE)" = "1" ]
[ "$(docker exec "$container" printenv AI_DEFENSE_RELEASE_PUBLIC_KEY)" = "$container_public_key" ]
[ "$(docker exec "$container" stat -c '%u:%a' "$container_public_key")" = "10001:444" ]
[ "$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/run/ai-defense/release-ed25519.pub.pem"}}{{.RW}}{{end}}{{end}}' "$container")" = "false" ]

docker exec -i "$container" python - <<'PY'
import json
from urllib.request import Request, urlopen


def request(path, *, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    method = "GET" if body is None else "POST"
    with urlopen(
        Request(f"http://127.0.0.1:8000{path}", data=body, headers=headers, method=method),
        timeout=30,
    ) as response:
        assert response.status == 200, (path, response.status)
        return response, response.read()


response, body = request("/")
assert "text/html" in response.headers.get("Content-Type", "")
assert b"<!doctype html" in body.lower()

for path in (
    "/atlas",
    "/metrics",
    "/generalization",
    "/arena",
    "/phylogeny",
    "/vaccination",
    "/stream?n=20",
    "/case/evt_flagship",
    "/precedent/evt_borderline",
):
    _, body = request(path)
    assert json.loads(body), path

response, body = request(
    "/compose",
    payload={
        "channel": "agentic_checkout",
        "injection_style": "instruction_shift",
        "amount_minor": 7_800_000,
        "hops": 2,
        "delegation_depth": 2,
        "ttl_delay_seconds": 86_400,
        "fanout": 4,
        "evasion_aggressiveness": 0.8,
        "lie_about_presence": True,
    },
)
composed = json.loads(body)
assert response.headers.get("X-AI-Defense-Fixture") is None
assert composed["event_id"].startswith("evt_composer_")
assert composed["decision"]["reason_codes"]
assert composed["decision"]["latency_ms"] >= 0
_, body = request("/health")
assert json.loads(body)["status"] == "ready"
print("offline container smoke passed")
PY
