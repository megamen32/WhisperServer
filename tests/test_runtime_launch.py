"""Runtime gates catch the native-library collision that broke real inference."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("runtime_launcher", Path(__file__).parents[1] / "scripts/runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def fake_distribution(root, name, files):
    return SimpleNamespace(metadata={"Name": name}, version="9.20.0", files=files,
                           locate_file=lambda item: root / str(item))


def test_conflicting_cuda_wheels_are_rejected_even_when_versions_resolve(tmp_path):
    shared = "nvidia/cudnn/lib/libcudnn.so.9"
    distributions = [fake_distribution(tmp_path, name, [shared]) for name in
                     ("nvidia-cudnn-cu12", "nvidia-cudnn-cu13")]
    with pytest.raises(RuntimeError, match="owned by both"):
        runtime.installed(distributions)


def test_external_library_symlink_cannot_reintroduce_second_runtime(tmp_path):
    env = tmp_path / "active"
    env.mkdir()
    old = tmp_path / "old.so"
    old.write_bytes(b"old runtime")
    with pytest.raises(RuntimeError, match="outside this runtime"):
        runtime.library_dirs({"nvidia/cudnn/lib/libcudnn.so.9": (old, None)}, env)


def test_changed_lock_cannot_launch_sealed_environment(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))
    expected = {"schema": 1, "python": "3.10.12", "source_revision": "abc",
                "lock_sha256": "old", "packages": {}, "native": {}, "runtime_id": "checked"}
    (tmp_path / runtime.MANIFEST).write_text(json.dumps(expected))
    actual = {key: expected[key] for key in ("schema", "python", "source_revision", "lock_sha256", "packages")}
    actual["lock_sha256"] = "new"
    monkeypatch.setattr(runtime, "inventory", lambda root: (actual, {}))
    with pytest.raises(RuntimeError, match="lock_sha256"):
        runtime.checked_environment(tmp_path)


def test_launcher_discards_foreign_cuda_paths(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))
    expected = {"schema": 1, "python": "3.10.12", "source_revision": "abc",
                "lock_sha256": "lock", "packages": {}, "native": {}, "runtime_id": "checked"}
    (tmp_path / runtime.MANIFEST).write_text(json.dumps(expected))
    actual = {key: expected[key] for key in ("schema", "python", "source_revision", "lock_sha256", "packages")}
    monkeypatch.setattr(runtime, "inventory", lambda root: (actual, {}))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/foreign/old-environment/lib")
    env, _ = runtime.checked_environment(tmp_path)
    assert env["LD_LIBRARY_PATH"] == ""
    assert env["WHISPER_RUNTIME_ID"] == "checked"
    assert env["WHISPER_BUILD_REVISION"] == "abc"


def test_uncommitted_source_cannot_claim_a_sealed_revision(tmp_path):
    import subprocess
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    source = tmp_path / 'main.py'
    source.write_text('version = 1\n')
    subprocess.run(['git', 'add', 'main.py'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Runtime test', '-c', 'user.email=runtime-test@example.invalid',
                    'commit', '-qm', 'fixture'], cwd=tmp_path, check=True)
    assert runtime.source_revision(tmp_path)
    source.write_text('version = 2\n')
    with pytest.raises(RuntimeError, match='uncommitted'):
        runtime.source_revision(tmp_path)
