"""Example of using the Whisper Server with OpenAI Python SDK.

This example shows how to use the OpenAI-compatible /v1/audio/transcriptions
endpoint with the official OpenAI Python SDK.

Requirements:
    pip install openai

Usage:
    python example_openai_usage.py <audio_file>
"""

import os
import sys
from pathlib import Path
from openai import OpenAI


def transcribe_with_openai_sdk(audio_file_path: str, model: str = "whisper-1") -> str:
    """
    Transcribe audio using OpenAI SDK pointed at local Whisper Server.

    Args:
        audio_file_path: Path to audio file (mp3, wav, m4a, ogg, flac, mp4, webm)
        model: Model to use (default: "whisper-1")

    Returns:
        Transcribed text
    """
    # Initialize OpenAI client pointing to local server
    # Make sure the server is running on http://localhost:7653
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
        base_url="http://localhost:7653/v1"
    )

    # Open audio file
    with open(audio_file_path, "rb") as audio_file:
        # Create transcription using OpenAI SDK
        # This will POST to /v1/audio/transcriptions on our server
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language="en",  # Optional: specify language
            temperature=0.0,  # Optional: set temperature (0.0-2.0)
        )

    return transcript.text


async def transcribe_async(audio_file_path: str) -> str:
    """
    Async example of transcription.

    Note: OpenAI SDK is synchronous, so we use run_in_executor.
    """
    import asyncio

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"),
        base_url="http://localhost:7653/v1"
    )

    loop = asyncio.get_event_loop()

    def transcribe():
        with open(audio_file_path, "rb") as f:
            return client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            ).text

    # Run the synchronous SDK call in a thread pool
    text = await loop.run_in_executor(None, transcribe)
    return text


def main():
    """Main example."""
    if len(sys.argv) < 2:
        print("Usage: python example_openai_usage.py <audio_file>")
        print("\nExample:")
        print("  python example_openai_usage.py audio.mp3")
        print("\nSupported formats: mp3, wav, m4a, ogg, flac, mp4, webm")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Verify file exists
    if not Path(audio_file).exists():
        print(f"Error: File '{audio_file}' not found")
        sys.exit(1)

    print(f"Transcribing {audio_file}...")
    print("Make sure the Whisper Server is running on http://localhost:7653")
    print()

    try:
        text = transcribe_with_openai_sdk(audio_file)
        print("✓ Transcription successful!")
        print(f"\nTranscribed text:\n{text}")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. The server is running: python main.py")
        print("  2. The audio file format is supported")
        sys.exit(1)


if __name__ == "__main__":
    main()
