"""Real-browser smoke test for the Web UI upload and transcription path."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _enabled(value: str | None) -> bool:
    """Return whether an environment flag is enabled."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _server_is_available(base_url: str) -> bool:
    """Check that a live WhisperServer can serve the Web UI before launching Chromium."""
    try:
        with urllib.request.urlopen(f"{base_url}/", timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _run_playwright_cli(session: str, *arguments: str, timeout: int = 30, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run one command in a named real-browser Playwright CLI session."""
    codex_home = Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))
    wrapper = codex_home / "skills" / "playwright" / "scripts" / "playwright_cli.sh"
    if not wrapper.exists():
        pytest.skip(f"Playwright CLI wrapper is missing: {wrapper}")

    environment = os.environ.copy()
    environment["PLAYWRIGHT_CLI_SESSION"] = session
    result = subprocess.run(
        [str(wrapper), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
        check=False,
    )
    if check and result.returncode != 0:
        pytest.fail(
            f"playwright-cli {' '.join(arguments)} failed with {result.returncode}:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def test_webui_uploads_and_transcribes_in_real_browser(tmp_path: Path) -> None:
    """Exercise model selection, file upload, streaming response, and final text."""
    if not _enabled(os.getenv("BROWSER_TESTS")):
        pytest.skip("set BROWSER_TESTS=1 to run the real-browser Web UI test")
    if shutil.which("npx") is None:
        pytest.skip("npx is required by the Playwright CLI wrapper")
    codex_home = Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))
    wrapper = codex_home / "skills" / "playwright" / "scripts" / "playwright_cli.sh"
    if not wrapper.exists():
        pytest.skip(f"Playwright CLI wrapper is missing: {wrapper}")

    base_url = os.getenv("BROWSER_BASE_URL", "http://127.0.0.1:7653").rstrip("/")
    if not _server_is_available(base_url):
        pytest.skip(f"WhisperServer is unavailable at {base_url}")

    espeak = shutil.which("espeak") or shutil.which("espeak-ng")
    if espeak is None:
        pytest.skip("espeak or espeak-ng is required to create deterministic test audio")

    audio_path = tmp_path / "browser-upload.wav"
    subprocess.run(
        [espeak, "-v", "en", "-w", str(audio_path), "This is a browser upload test."],
        check=True,
        capture_output=True,
        text=True,
    )

    session = f"whisper-webui-{os.getpid()}"
    try:
        open_args = ["open", base_url]
        if _enabled(os.getenv("BROWSER_HEADED")):
            open_args.append("--headed")
        _run_playwright_cli(session, *open_args, timeout=60)
        _run_playwright_cli(
            session,
            "run-code",
            "await page.waitForFunction(() => document.querySelector('#model-select')?.value === 'large-v3', null, {timeout: 15000})",
            timeout=30,
        )
        snapshot = _run_playwright_cli(session, "snapshot", timeout=30)
        drop_zone_match = re.search(r"(?:paragraph|generic).*?\[ref=([^\]]+)\].*?Перетащите файл сюда", snapshot.stdout, re.DOTALL)
        assert drop_zone_match, f"file drop zone is missing from browser snapshot:\n{snapshot.stdout}"

        selected_model = _run_playwright_cli(
            session,
            "eval",
            "document.querySelector('#model-select').value",
            timeout=30,
        )
        assert "large-v3" in selected_model.stdout, selected_model.stdout

        _run_playwright_cli(
            session,
            "eval",
            "(() => { window.__whisperTranscribeRequested = false; const originalOpen = XMLHttpRequest.prototype.open; XMLHttpRequest.prototype.open = function(method, url, ...rest) { if (String(url).includes('/web/transcribe')) window.__whisperTranscribeRequested = true; return originalOpen.call(this, method, url, ...rest); }; })()",
            timeout=30,
        )
        _run_playwright_cli(session, "click", drop_zone_match.group(1), timeout=30)
        _run_playwright_cli(session, "upload", str(audio_path), timeout=30)
        _run_playwright_cli(session, "snapshot", timeout=30)
        wait_result = _run_playwright_cli(
            session,
            "run-code",
            "await page.waitForFunction(() => document.querySelector('#status-bar')?.textContent === 'Готово' || Boolean(document.querySelector('#error-output')?.textContent), null, {timeout: 180000})",
            timeout=190,
            check=False,
        )
        if wait_result.returncode != 0:
            pytest.fail(f"browser transcription did not finish:\n{wait_result.stdout}\n{wait_result.stderr}")

        status = _run_playwright_cli(
            session,
            "eval",
            "document.querySelector('#status-bar').textContent",
            timeout=30,
        )
        error = _run_playwright_cli(
            session,
            "eval",
            "document.querySelector('#error-output').textContent",
            timeout=30,
        )
        transcript = _run_playwright_cli(
            session,
            "eval",
            "document.querySelector('#final-output').textContent",
            timeout=30,
        )
        request_seen = _run_playwright_cli(
            session,
            "eval",
            "window.__whisperTranscribeRequested === true",
            timeout=30,
        )

        assert "true" in request_seen.stdout.lower(), "browser did not issue /web/transcribe request"
        assert "Готово" in status.stdout, f"Web UI error: {error.stdout}"
        normalized = transcript.stdout.casefold()
        assert "browser" in normalized and "upload" in normalized, transcript.stdout
    finally:
        _run_playwright_cli(session, "close", timeout=30, check=False)
