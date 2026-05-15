# Development Guide

This guide covers setting up the development environment and running tests.

## Project Setup with `uv`

This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager for faster and more reliable dependency management.

### Installation

If you don't have `uv` installed:

```bash
pip install uv
```

Or follow [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Install Dependencies

```bash
# Install main dependencies
uv pip install -r requirements.txt

# Install development dependencies (for testing)
uv pip install -r requirements-dev.txt
```

Or use `uv sync` once we have proper dependency groups set up:

```bash
uv sync
uv sync --extra dev
```

## Running the Server

```bash
python main.py
```

The server will start on `http://localhost:7653`

## Running Tests

### Run all tests

```bash
pytest -v
```

### Run specific test file

```bash
pytest tests/test_openai_transcription.py -v
```

### Run specific test

```bash
pytest tests/test_openai_transcription.py::TestOpenAITranscriptionEndpoint::test_endpoint_exists -v
```

### Run with coverage

```bash
pytest --cov=. --cov-report=html
```

## Using the OpenAI SDK

### Installation

```bash
uv pip install openai
```

### Example Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",  # Can be anything for local server
    base_url="http://localhost:7653/v1"
)

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="en",
        temperature=0.0,
    )
    print(transcript.text)
```

### Using the Example Script

```bash
python tests/example_openai_usage.py audio.mp3
```

## API Endpoint

### OpenAI-Compatible Endpoint

**Endpoint:** `POST /v1/audio/transcriptions`

**Request (multipart/form-data):**
- `file` (required): Audio file (mp3, wav, m4a, ogg, flac, mp4, webm)
- `model` (required): Model name (e.g., "whisper-1", "base", "small", "medium", "large-v3")
- `language` (optional): Language code (e.g., "en", "es", "fr")
- `temperature` (optional): Temperature for decoding (0.0-2.0, default: 0.0)

**Response:**
```json
{
  "text": "The transcribed text"
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:7653/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

**Example with Python SDK:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:7653/v1")
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("audio.mp3", "rb")
)
print(result.text)
```

## Project Structure

```
.
├── main.py                    # Main FastAPI application
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── pyproject.toml            # Project metadata
├── tests/
│   ├── __init__.py
│   ├── test_openai_transcription.py   # Unit and integration tests
│   └── example_openai_usage.py        # Example usage with OpenAI SDK
└── DEVELOPMENT.md            # This file
```

## Supported Models

The following faster-whisper models are supported:

- `tiny` - Smallest, fastest (~39M parameters)
- `base` - Small model (~74M parameters)
- `small` - Medium model (~244M parameters)
- `medium` - Large model (~769M parameters)
- `distil-large-v3` - Distilled large model (~756M parameters)
- `large-v3` - Largest model (~1.5B parameters) - **Default for "whisper-1"**
- `large-v2` - Previous large version
- `large` - Alias for large-v3

## OpenAI API Compatibility

Our implementation is compatible with the official [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

Specification: `/openapi.yaml`

**Supported OpenAI Models:**
- `whisper-1` → Maps to `large-v3` (configurable via `OPENAI_DEFAULT_MODEL` env var)

**Response Format:**
- Follows OpenAI spec with `text` field for transcribed content
- Includes `language` and `language_probability` for additional info

## Troubleshooting

### ImportError: No module named 'openai'

Install the OpenAI SDK:
```bash
uv pip install openai>=1.0.0
```

### CUDA/GPU Issues

The server automatically detects and uses CUDA if available. For CPU-only:
```bash
# Set environment variable before running
export CUDA_VISIBLE_DEVICES=""
python main.py
```

### Model Loading Takes Too Long

Models are loaded on first use. The largest model (large-v3) requires ~4-5GB VRAM and takes 10-30 seconds to load initially.

### Connection Refused

Make sure the server is running:
```bash
python main.py
```

Server should be accessible at `http://localhost:7653`

## Environment Variables

- `API_KEY`: API key for authentication (default: "bad-key")
- `MODEL`: Default model to load (default: "base")
- `TG_BOT_ENABLED`: Enable Telegram bot (default: false)
- `TG_BOT_TOKEN`: Telegram bot token
- `OPENAI_DEFAULT_MODEL`: Model to use for "whisper-1" (default: "large-v3")

## Performance Tips

1. **Use larger models on GPU**: They're significantly faster than CPU
2. **Reuse client connections**: Don't create new OpenAI() instances for each request
3. **Use beam_size wisely**: Default 5 is good balance, increase for accuracy, decrease for speed
4. **Batch similar requests**: The server uses a queue that prioritizes by model size

## Contributing

When adding tests:
1. Use the test templates in `tests/test_openai_transcription.py`
2. Mark async tests with `@pytest.mark.asyncio`
3. Use fixtures for common setup (see `sample_audio` fixture)
4. Follow OpenAI spec for response format
