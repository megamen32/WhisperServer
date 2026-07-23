#!/usr/bin/env python3
"""
CLI wrapper for local Faster Whisper transcription.
Designed to be called as a subprocess: heavy imports stay isolated here.
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# Heavy imports ONLY here — loaded only when CLI is invoked
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from parakeet_backend import ParakeetModel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


def _determine_compute_type(device: str) -> str:
    """Return optimal compute_type for given device."""
    if device == "cuda":
        return "float16"
    return "int8"  # CPU: faster + less memory


def _load_model(model_name: str, device: str):
    """Load a local model through its native backend."""
    if model_name == "parakeet-v3":
        return ParakeetModel.from_pretrained(device=device)
    return WhisperModel(model_name, device=device, compute_type=_determine_compute_type(device))


def transcribe_oneshot(
    audio_path: str,
    model_name: str,
    device: str,
    include_words: bool,
) -> dict:
    """Run full transcription and return complete result."""
    model = _load_model(model_name, device)

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=include_words,
        condition_on_previous_text=False,
    )

    all_segments = []
    all_words = []

    for seg in segments:
        seg_data = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        all_segments.append(seg_data)
        if include_words and seg.words:
            for w in seg.words:
                all_words.append({
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word.strip(),
                })

    result = {
        "type": "result",
        "text": " ".join(s["text"] for s in all_segments).strip(),
        "segments": all_segments,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
    }
    if include_words:
        result["words"] = all_words

    return result


def transcribe_streaming(
    audio_path: str,
    model_name: str,
    device: str,
    include_words: bool,
):
    """Yield transcription segments as NDJSON lines."""
    model = _load_model(model_name, device)

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=include_words,
        condition_on_previous_text=False,
    )

    # Send metadata first
    meta = {
        "type": "info",
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
    }
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    all_segments = []
    for seg in segments:
        seg_data = {
            "type": "segment",
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        if include_words and seg.words:
            seg_data["words"] = [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word.strip(),
                }
                for w in seg.words
            ]
        print(json.dumps(seg_data, ensure_ascii=False), flush=True)
        all_segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    # Send final aggregated result
    final = {
        "type": "result",
        "text": " ".join(s["text"] for s in all_segments).strip(),
        "segments": all_segments,
    }
    print(json.dumps(final, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Local Whisper CLI")
    parser.add_argument("file", help="Path to audio file (ogg, wav, mp3, etc.)")
    parser.add_argument(
        "--model",
        default="parakeet-v3",
        choices=["tiny", "base", "small", "medium", "large-v3", "parakeet-v3"],
        help="Model size",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--words",
        action="store_true",
        help="Include word-level timestamps",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output as NDJSON (one JSON per line)",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        error = {"type": "error", "error": f"File not found: {args.file}"}
        print(json.dumps(error), file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        if args.stream:
            transcribe_streaming(
                args.file,
                args.model,
                args.device,
                args.words,
            )
        else:
            result = transcribe_oneshot(
                args.file,
                args.model,
                args.device,
                args.words,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as e:
        error = {"type": "error", "error": str(e)}
        print(json.dumps(error), file=sys.stderr, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
