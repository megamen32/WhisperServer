#!/usr/bin/env python3
"""Seal and launch one reproducible Whisper Python/native runtime.

Build with uv sync --locked --extra diarization --extra dev --no-install-project.
Seal only after committing the source. The launch check uses metadata, not heavy
model imports or GPU allocation. Full native file hashing runs once at seal time.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

MANIFEST = ".whisper-runtime.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def installed(distributions=None):
    """Reject conflicting ownership before trusting the dependency metadata."""
    packages, owners, native = {}, {}, {}
    for dist in distributions if distributions is not None else metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        packages[name] = dist.version
        for item in dist.files or ():
            relative = str(item)
            if not relative.startswith("nvidia/") or ".so" not in relative:
                continue
            path = Path(dist.locate_file(item)).resolve()
            previous = owners.get(path)
            if previous and previous != name:
                raise RuntimeError(f"Native library owned by both {previous} and {name}: {relative}")
            owners[path] = name
            native[relative] = (path, getattr(item, "hash", None))
    return dict(sorted(packages.items())), native


def library_dirs(native, prefix=None):
    prefix = Path(prefix or sys.prefix).resolve()
    directories = set()
    for path, _ in native.values():
        if not path.is_relative_to(prefix):
            raise RuntimeError(f"Native library is outside this runtime: {path}")
        if not path.is_file():
            raise RuntimeError(f"Native library is missing: {path}")
        directories.add(str(path.parent))
    return sorted(directories)


def source_revision(root):
    changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root)
    if changed.returncode:
        raise RuntimeError("Source contains uncommitted changes; commit the candidate before sealing or launching")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def inventory(root):
    root = Path(root).resolve()
    packages, native = installed()
    value = {
        "schema": 1,
        "python": sys.version.split()[0],
        "source_revision": source_revision(root),
        "lock_sha256": digest(root / "uv.lock"),
        "packages": packages,
    }
    return value, native


def seal(root):
    value, native = inventory(root)
    library_dirs(native)
    # Check the bytes against the wheel RECORD, including every cuDNN sublibrary.
    # An apparently compatible pip/uv metadata set can still overwrite files.
    checked = {}
    for relative, (path, recorded) in sorted(native.items()):
        if recorded is None or recorded.mode != "sha256":
            raise RuntimeError(f"Native wheel has no SHA256 record: {relative}")
        actual = digest(path)
        encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b"=").decode()
        if encoded != recorded.value:
            raise RuntimeError(f"Native library differs from its wheel: {relative}")
        checked[relative] = {"sha256": actual, "size": path.stat().st_size}
    value["native"] = checked
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["runtime_id"] = hashlib.sha256(payload).hexdigest()
    path = Path(sys.prefix) / MANIFEST
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    return value


def checked_environment(root):
    path = Path(sys.prefix) / MANIFEST
    if not path.is_file():
        raise RuntimeError("Runtime is not sealed; run scripts/runtime.py seal after building and testing it")
    expected = json.loads(path.read_text())
    actual, native = inventory(root)
    for key in ("schema", "python", "source_revision", "lock_sha256", "packages"):
        if expected.get(key) != actual[key]:
            raise RuntimeError(f"Runtime changed after validation: {key}")
    # Startup stays light. The full hashes were verified before activation.
    if set(expected.get("native", {})) != set(native):
        raise RuntimeError("Native library inventory changed after validation")
    for relative, (file, _) in native.items():
        if not file.is_file() or file.stat().st_size != expected["native"][relative]["size"]:
            raise RuntimeError(f"Native library changed after validation: {relative}")
    env = dict(os.environ)
    # Do not inherit an unrelated environment's CUDA library search paths.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs(native))
    env["WHISPER_BUILD_REVISION"] = actual["source_revision"]
    env["WHISPER_LOCK_SHA256"] = actual["lock_sha256"]
    env["WHISPER_RUNTIME_ID"] = expected["runtime_id"]
    return env, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "seal", "check", "run"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7653)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            value, native = inventory(args.root)
            value["library_dirs"] = library_dirs(native)
        elif args.command == "seal":
            value = seal(args.root)
        else:
            env, value = checked_environment(args.root)
            if args.command == "run":
                os.chdir(args.root)
                os.execve(sys.executable, [sys.executable, "-m", "uvicorn", "main:app",
                                          "--host", args.host, "--port", str(args.port)], env)
        print(json.dumps({key: value[key] for key in ("source_revision", "lock_sha256", "runtime_id", "library_dirs") if key in value}, indent=2))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Whisper runtime: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
