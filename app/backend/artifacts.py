from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .store import ArtifactInvalid, ArtifactMissing, ArtifactStore


PRIVATE_KEY_ENV = "AI_DEFENSE_RELEASE_PRIVATE_KEY"
PUBLIC_KEY_ENV = "AI_DEFENSE_RELEASE_PUBLIC_KEY"
CUSTODY_FIELD = "custody"
CUSTODY_ALGORITHM = "Ed25519"
CUSTODY_PAYLOAD_FORMAT = "utf8-json-sort-keys-compact-v1"
DEFAULT_PRIVATE_KEY = Path.home() / ".config" / "ai-defense-lab" / "release-ed25519.pem"
_CUSTODY_KEYS = frozenset(
    {"algorithm", "payload_format", "public_key_sha256", "signature"}
)
_MAX_KEY_BYTES = 64 * 1024


class CustodyInvalid(ArtifactInvalid):
    pass


def canonical_release_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CustodyInvalid("release payload cannot be canonically encoded") from error


def _absolute_path(path: str | Path, label: str) -> Path:
    try:
        candidate = Path(path).expanduser()
    except (TypeError, RuntimeError) as error:
        raise CustodyInvalid(f"{label} path is invalid") from error
    if not candidate.is_absolute():
        raise CustodyInvalid(f"{label} path must be absolute")
    return Path(os.path.abspath(candidate))


def _reject_symlinks(path: Path, label: str) -> None:
    current = path
    while True:
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise CustodyInvalid(f"{label} path must not contain symlinks")
        except FileNotFoundError:
            pass
        except OSError as error:
            raise CustodyInvalid(f"{label} path is unsafe") from error
        if current.parent == current:
            return
        current = current.parent


def _outside_project(path: Path, project_root: Path, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(project_root.resolve(strict=True))
    except ValueError:
        return
    except OSError as error:
        raise CustodyInvalid(f"{label} path is unsafe") from error
    raise CustodyInvalid(f"{label} must be outside the repository")


def _configured_path(
    explicit: str | Path | None,
    environment_name: str,
    label: str,
) -> Path:
    value = explicit if explicit is not None else os.environ.get(environment_name)
    if not value:
        raise CustodyInvalid(
            f"{label} is not configured; set {environment_name} to an absolute path"
        )
    return _absolute_path(value, label)


def _open_key_directory(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise CustodyInvalid(f"{label} directory is missing or unsafe") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or metadata.st_uid not in {0, os.geteuid()}
    ):
        os.close(descriptor)
        raise CustodyInvalid(f"{label} directory is not a trusted location")
    return descriptor


def _read_key(
    path: Path,
    label: str,
    *,
    project_root: Path | None = None,
    private: bool = False,
) -> bytes:
    _reject_symlinks(path, label)
    if project_root is None:
        if private:
            raise CustodyInvalid("project root is required for private-key safety")
    else:
        _outside_project(path, project_root, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = _open_key_directory(path.parent, label)
    try:
        descriptor = os.open(path.name, flags, dir_fd=directory)
    except OSError as error:
        os.close(directory)
        raise CustodyInvalid(f"{label} is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or (
            metadata.st_dev,
            metadata.st_ino,
        ) != (current.st_dev, current.st_ino):
            raise CustodyInvalid(f"{label} is not a stable regular file")
        if metadata.st_uid not in {0, os.geteuid()}:
            raise CustodyInvalid(f"{label} has an untrusted owner")
        if private and metadata.st_nlink != 1:
            raise CustodyInvalid(f"{label} must have exactly one hard link")
        if private and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise CustodyInvalid(f"{label} permissions must not allow group or other access")
        if not private and stat.S_IMODE(metadata.st_mode) & 0o022:
            raise CustodyInvalid(f"{label} must not be group or other writable")
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        if private and descriptor_path.exists():
            _outside_project(descriptor_path.resolve(strict=True), project_root, label)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(_MAX_KEY_BYTES + 1)
        if not payload or len(payload) > _MAX_KEY_BYTES:
            raise CustodyInvalid(f"{label} has an invalid size")
        return payload
    except OSError as error:
        raise CustodyInvalid(f"{label} is unreadable") from error
    finally:
        os.close(descriptor)
        os.close(directory)


def _private_key(path: Path, project_root: Path) -> Ed25519PrivateKey:
    payload = _read_key(
        path, "release private key", project_root=project_root, private=True
    )
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as error:
        raise CustodyInvalid("release private key is not a valid unencrypted PEM key") from error
    if not isinstance(key, Ed25519PrivateKey):
        raise CustodyInvalid("release private key is not Ed25519")
    return key


def _public_key(path: Path, project_root: Path) -> Ed25519PublicKey:
    payload = _read_key(path, "release public key", project_root=project_root)
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as error:
        raise CustodyInvalid("release public key is not a valid PEM key") from error
    if not isinstance(key, Ed25519PublicKey):
        raise CustodyInvalid("release public key is not Ed25519")
    return key


def _public_key_sha256(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sign_release(
    payload: Mapping[str, Any],
    project_root: Path,
    private_key_path: str | Path | None = None,
) -> dict[str, Any]:
    if CUSTODY_FIELD in payload:
        raise CustodyInvalid("unsigned release payload must not contain custody metadata")
    key = _private_key(
        _configured_path(private_key_path, PRIVATE_KEY_ENV, "release private key"),
        project_root,
    )
    signature = key.sign(canonical_release_bytes(payload))
    return {
        **payload,
        CUSTODY_FIELD: {
            "algorithm": CUSTODY_ALGORITHM,
            "payload_format": CUSTODY_PAYLOAD_FORMAT,
            "public_key_sha256": _public_key_sha256(key.public_key()),
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    }


def verify_release_custody(
    manifest: Mapping[str, Any],
    public_key_path: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    if project_root is None:
        raise CustodyInvalid("project root is required for public-key safety")
    custody = manifest.get(CUSTODY_FIELD)
    if not isinstance(custody, Mapping) or set(custody) != _CUSTODY_KEYS:
        raise CustodyInvalid("release manifest has no valid signed custody record")
    if (
        custody.get("algorithm") != CUSTODY_ALGORITHM
        or custody.get("payload_format") != CUSTODY_PAYLOAD_FORMAT
    ):
        raise CustodyInvalid("release manifest uses an unsupported custody contract")
    encoded_signature = custody.get("signature")
    if not isinstance(encoded_signature, str):
        raise CustodyInvalid("release manifest signature is invalid")
    try:
        signature = base64.b64decode(encoded_signature, validate=True)
    except (binascii.Error, ValueError) as error:
        raise CustodyInvalid("release manifest signature is invalid") from error

    key = _public_key(
        _configured_path(public_key_path, PUBLIC_KEY_ENV, "release public key"),
        project_root,
    )
    fingerprint = custody.get("public_key_sha256")
    if not _is_sha256(fingerprint) or not hmac.compare_digest(
        fingerprint, _public_key_sha256(key)
    ):
        raise CustodyInvalid("configured public key does not match the signed release")

    payload = {key: value for key, value in manifest.items() if key != CUSTODY_FIELD}
    try:
        key.verify(signature, canonical_release_bytes(payload))
    except InvalidSignature as error:
        raise CustodyInvalid("release manifest signature verification failed") from error
    return payload


def artifact_store_from_signed_release(
    project_root: Path,
    manifest: Mapping[str, Any],
    public_key_path: str | Path | None = None,
) -> ArtifactStore:
    release = verify_release_custody(
        manifest, public_key_path, project_root=project_root
    )
    if release.get("status") != "passed":
        raise CustodyInvalid("release manifest status is not passed")
    hashes = release.get("required_artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise CustodyInvalid("release manifest has no required artifact hash map")
    return ArtifactStore(project_root, hashes)


def _prepare_new_path(path: str | Path, label: str, project_root: Path) -> Path:
    candidate = _absolute_path(path, label)
    _outside_project(candidate, project_root, label)
    candidate.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlinks(candidate, label)
    _outside_project(candidate, project_root, label)
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"{label} already exists")
    return candidate


def _write_new(path: Path, payload: bytes, mode: int) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory = _open_key_directory(path.parent, "release key")
    try:
        descriptor = os.open(path.name, flags, mode, dir_fd=directory)
    except Exception:
        os.close(directory)
        raise
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
        os.close(directory)


def generate_signing_key(
    project_root: Path,
    private_key_path: str | Path = DEFAULT_PRIVATE_KEY,
    public_key_path: str | Path | None = None,
) -> tuple[Path, Path]:
    private_path = _prepare_new_path(
        private_key_path, "release private key", project_root
    )
    default_public = private_path.with_name(
        f"{private_path.stem}.pub{private_path.suffix}"
    )
    public_path = _prepare_new_path(
        public_key_path or default_public, "release public key", project_root
    )
    if private_path == public_path:
        raise CustodyInvalid("release private and public key paths must differ")

    key = Ed25519PrivateKey.generate()
    private_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_bytes = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_new(private_path, private_bytes, 0o600)
    try:
        _write_new(public_path, public_bytes, 0o644)
    except Exception:
        private_path.unlink(missing_ok=True)
        raise
    return private_path, public_path


__all__ = (
    "ArtifactInvalid",
    "ArtifactMissing",
    "ArtifactStore",
    "CustodyInvalid",
    "DEFAULT_PRIVATE_KEY",
    "PRIVATE_KEY_ENV",
    "PUBLIC_KEY_ENV",
    "artifact_store_from_signed_release",
    "canonical_release_bytes",
    "generate_signing_key",
    "sign_release",
    "verify_release_custody",
)
