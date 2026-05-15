# Quick Start Guide

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or using uv (faster)
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Run Server

```bash
python main.py
```

Server starts on `http://localhost:7653`

## Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",
    base_url="http://localhost:7653/v1"
)

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f
    )
    print(transcript.text)
```

## Run Tests

```bash
# Validation tests
pytest tests/test_openai_endpoint_simple.py -v

# Run specific tests
pytest tests/test_openai_endpoint_simple.py::TestOpenAIEndpointValidation -v
```

## Example Usage

```bash
python tests/example_openai_usage.py audio.mp3
```

## API Endpoint

**POST** `/v1/audio/transcriptions`

**Request (multipart/form-data):**
- `file` - Audio file (mp3, wav, m4a, ogg, flac, mp4, webm)
- `model` - Model name ("whisper-1", "base", "small", "medium", "large-v3")
- `language` (optional) - Language code ("en", "es", "fr", etc.)
- `temperature` (optional) - 0.0-2.0 (default: 0.0)

**Response:**
```json
{
  "text": "Transcribed audio text here"
}
```

## Curl Example

```bash
curl -X POST "http://localhost:7653/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

## More Information

- [OpenAI Integration Guide](OPENAI_INTEGRATION.md)
- [Development Guide](DEVELOPMENT.md)
- [API Specification](openapi.yaml)

## Supported Models

- `tiny` - 39M params, fastest
- `base` - 74M params
- `small` - 244M params
- `medium` - 769M params
- `distil-large-v3` - Distilled large model
- `large-v3` - 1.5B params (largest, most accurate)
- `large-v2` - Previous large version
- `whisper-1` - Maps to large-v3

## Environment Variables

```bash
export API_KEY="your-secret-key"
export OPENAI_DEFAULT_MODEL="large-v3"
export TG_BOT_ENABLED=false
```

## Troubleshooting

**Port already in use:**
```bash
lsof -i :7653
```

**Check GPU:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**View logs:**
```bash
# Server runs in foreground, logs appear in terminal
```

**Tests timing out:**
Some integration tests timeout. Use simple validation tests:
```bash
pytest tests/test_openai_endpoint_simple.py -v -k "validation"
```

---

🚀 **Ready to go!** The server is now OpenAI-compatible. See [OPENAI_INTEGRATION.md](OPENAI_INTEGRATION.md) for more examples and advanced usage.
