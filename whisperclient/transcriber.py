"""
Utility functions for speech-to-text.
Local transcription is delegated to whisper_cli.py via subprocess
to avoid loading heavy dependencies (torch, ctranslate2) at import time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional

# No heavy imports here — only standard library + lightweight deps
import aiohttp
import requests


def _get_cli_path() -> str:
    """Return absolute path to whisper_cli.py."""
    # Option 1: same directory as this module
    cli_path = Path(__file__).parent / "whisper_cli.py"
    if cli_path.exists():
        return str(cli_path.resolve())
    # Option 2: installed as console script
    return "whisper_cli"


def _has_cuda() -> bool:
    """Check if NVIDIA GPU is available without importing torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_cli_sync(
    file_path: str,
    model: str,
    device: Optional[str],
    include_words: bool,
    stream: bool,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Execute whisper_cli.py synchronously."""
    cmd = [
        sys.executable,
        _get_cli_path(),
        file_path,
        "--model", model,
    ]
    if device:
        cmd.extend(["--device", device])
    if include_words:
        cmd.append("--words")
    if stream:
        cmd.append("--stream")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_cli_output(output: str) -> Optional[dict]:
    """Parse JSON from CLI stdout. Handle multi-line NDJSON for streaming."""
    lines = output.strip().split("\n")
    last_result = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("type") == "result":
                last_result = data
        except json.JSONDecodeError:
            continue
    return last_result


def transcribe_sync(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Transcribe audio file.
    Tries remote server first, falls back to local CLI subprocess.
    """
    # Lazy import to avoid circular deps / early load
    import whisperclient

    url = whisperclient.server_url
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model

    # === Remote attempt ===
    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "application/octet-stream")
            }
            params = {"model": model_name, "api_key": key}
            if language:
                params["language"] = language
            response = requests.post(url, files=files, params=params, timeout=600)
            if response.ok:
                return response.json()["text"]
            else:
                logging.warning(
                    f"[Whisper remote {response.status_code}] {response.text[:200]}"
                )
    except Exception:
        logging.warning("Remote whisper request failed", exc_info=True)

    # === Local CLI fallback ===
    logging.info("Falling back to local CLI transcription")
    try:
        device = "cuda" if _has_cuda() else "cpu"
        proc = _run_cli_sync(
            file_path=file_path,
            model=model_name or "base",
            device=device,
            include_words=False,
            stream=False,
            timeout=600,
        )
        if proc.returncode == 0:
            result = _parse_cli_output(proc.stdout)
            if result and "text" in result:
                return result["text"]
        else:
            logging.error(f"CLI failed: {proc.stderr[:300]}")
    except subprocess.TimeoutExpired:
        logging.error("Local CLI transcription timed out")
    except Exception:
        logging.exception("Local CLI transcription crashed")

    return "[TRANSCRIPTION ERROR]"


def transcribe_stream_sync(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Stream transcription results.
    Remote streaming first, then local CLI streaming fallback.
    """
    import whisperclient

    url = whisperclient.server_url
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model

    # === Remote streaming attempt ===
    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "application/octet-stream")
            }
            params = {"model": model_name, "api_key": key, "stream": "true"}
            if language:
                params["language"] = language
            with requests.post(
                url, files=files, params=params, timeout=600, stream=True
            ) as resp:
                if resp.ok:
                    for line in resp.iter_lines():
                        if line:
                            yield json.loads(line.decode("utf-8"))
                    return
                else:
                    logging.warning(
                        f"[Whisper remote {resp.status_code}] {resp.text[:200]}"
                    )
    except Exception:
        logging.warning("Remote streaming failed", exc_info=True)

    # === Local CLI streaming fallback ===
    logging.info("Falling back to local CLI streaming")
    try:
        device = "cuda" if _has_cuda() else "cpu"
        proc = _run_cli_sync(
            file_path=file_path,
            model=model_name or "base",
            device=device,
            include_words=False,
            stream=True,
            timeout=600,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                if line.strip():
                    yield json.loads(line)
        else:
            logging.error(f"CLI streaming failed: {proc.stderr[:300]}")
            yield {"result": {"text": "[TRANSCRIPTION ERROR]"}}
    except Exception:
        logging.exception("Local CLI streaming crashed")
        yield {"result": {"text": "[TRANSCRIPTION ERROR]"}}


async def transcribe_with_fallback(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Async transcription with remote fallback + local CLI."""
    import whisperclient

    url = whisperclient.server_url
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model

    # === Remote async attempt ===
    try:
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )
                params = {"model": model_name, "api_key": key}
                if language:
                    params["language"] = language
                async with session.post(
                    url, data=form, params=params, timeout=600
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["text"]
                    else:
                        error_text = await resp.text()
                        logging.warning(
                            f"[Whisper remote {resp.status}] {error_text[:200]}"
                        )
    except Exception:
        logging.warning("Remote async whisper failed", exc_info=True)

    # === Local CLI fallback (run in executor) ===
    logging.info("Falling back to local CLI (async)")
    loop = asyncio.get_event_loop()
    try:
        device = "cuda" if _has_cuda() else "cpu"
        proc = await loop.run_in_executor(
            None,
            _run_cli_sync,
            file_path,
            model_name or "base",
            device,
            False,  # include_words
            False,  # stream
            600,    # timeout
        )
        if proc.returncode == 0:
            result = _parse_cli_output(proc.stdout)
            if result and "text" in result:
                return result["text"]
    except Exception:
        logging.exception("Local CLI async fallback failed")

    return "[TRANSCRIPTION ERROR]"


async def transcribe_stream_with_fallback(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Async streaming with remote fallback + local CLI streaming."""
    import whisperclient

    url = whisperclient.server_url
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model

    # === Remote async streaming ===
    try:
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )
                params = {"model": model_name, "api_key": key, "stream": "true"}
                if language:
                    params["language"] = language
                async with session.post(
                    url, data=form, params=params, timeout=600
                ) as resp:
                    if resp.status == 200:
                        async for line in resp.content:
                            line = line.decode("utf-8").strip()
                            if line:
                                yield json.loads(line)
                        return
                    else:
                        error_text = await resp.text()
                        logging.warning(
                            f"[Whisper remote {resp.status}] {error_text[:200]}"
                        )
    except Exception:
        logging.warning("Remote async streaming failed", exc_info=True)

    # === Local CLI streaming fallback ===
    logging.info("Falling back to local CLI streaming (async)")
    try:
        device = "cuda" if _has_cuda() else "cpu"
        cmd = [
            sys.executable,
            _get_cli_path(),
            file_path,
            "--model", model_name or "base",
            "--device", device,
            "--stream",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)
            await proc.wait()
        finally:
            if proc.returncode != 0:
                stderr = await proc.stderr.read()
                logging.error(f"CLI streaming failed: {stderr.decode()[:300]}")
    except Exception:
        logging.exception("Local CLI async streaming crashed")
        yield {"result": {"text": "[TRANSCRIPTION ERROR]"}}


async def voice_to_text(message) -> str:
    """
    Helper for aiogram: download voice message and transcribe.
    """
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        file_path = tmp.name

    try:
        await message.download_media(file_path)
        text = await transcribe_with_fallback(file_path)
        return text
    finally:
        try:
            os.remove(file_path)
        except OSError:
            logging.warning("Failed to remove temp file", exc_info=True)


if __name__ == "__main__":
    # Quick self-test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>")
        sys.exit(1)
    result = transcribe_sync(sys.argv[1])
    print(result)