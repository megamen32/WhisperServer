# OpenAI API Integration

Whisper Server now includes a full OpenAI-compatible `/v1/audio/transcriptions` endpoint following the official OpenAI API specification.

## Quick Start

### 1. Install Dependencies

```bash
# Main dependencies
pip install -r requirements.txt

# Development dependencies (for testing and examples)
pip install -r requirements-dev.txt
```

Using `uv`:
```bash
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### 2. Start the Server

```bash
python main.py
```

Server runs on `http://localhost:7653`

### 3. Use with OpenAI SDK

```python
from openai import OpenAI

# Point to local Whisper Server
client = OpenAI(
    api_key="any-key",  # Can be anything for local server
    base_url="http://localhost:7653/v1"
)

# Transcribe audio
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="en",
        temperature=0.0,
    )
    print(transcript.text)
```

## Endpoint Specification

### POST /v1/audio/transcriptions

OpenAI-compatible audio transcription endpoint.

**Request Format:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | ✓ | - | Audio file (mp3, wav, m4a, ogg, flac, mp4, webm) |
| `model` | string | ✓ | - | Model ID (e.g., "whisper-1", "base", "medium", "large-v3") |
| `language` | string | ✗ | - | Language code (e.g., "en", "es", "fr") for faster/more accurate transcription |
| `temperature` | float | ✗ | 0.0 | Temperature for decoding (0.0-2.0, higher = more creative) |

**Supported Audio Formats:**
- MP3 (`.mp3`)
- WAV (`.wav`)
- MP4 Audio (`.m4a`, `.mp4`)
- OGG (`.ogg`)
- FLAC (`.flac`)
- WebM (`.webm`)

**Response Format (JSON):**

```json
{
  "text": "The transcribed text here"
}
```

Additional optional fields in response:
```json
{
  "text": "...",
  "language": "en",
  "language_probability": 0.95
}
```

### Supported Models

The endpoint supports faster-whisper model names that map to OpenAI models:

```
whisper-1       → large-v3 (2.7GB, ~5-30 seconds per minute of audio)
base            → base (74M parameters)
small           → small (244M parameters)
medium          → medium (769M parameters)
large-v3        → large-v3 (1.5B parameters, default)
large-v2        → large-v2 (previous large version)
distil-large-v3 → distil-large-v3 (distilled version)
tiny            → tiny (39M parameters, fastest)
```

## Example Usage

### With cURL

```bash
curl -X POST "http://localhost:7653/v1/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

### With Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7653/v1",
    api_key="dummy-key"  # Not used for local server
)

# Simple transcription
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("meeting.mp3", "rb")
)
print(f"Transcription: {transcript.text}")

# With optional parameters
transcript = client.audio.transcriptions.create(
    model="small",  # Use smaller model for faster processing
    file=open("audio.wav", "rb"),
    language="es",  # Spanish audio
    temperature=0.5
)
print(transcript.text)

# Batch processing
files = ["audio1.mp3", "audio2.wav", "audio3.flac"]
for filename in files:
    with open(filename, "rb") as f:
        result = client.audio.transcriptions.create(
            model="base",  # Use smaller model for speed
            file=f
        )
        print(f"{filename}: {result.text}")
```

### Using the Example Script

Run the provided example:

```bash
python tests/example_openai_usage.py audio.mp3
```

### With Other Languages

```python
# Works with any language supported by Whisper
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}

client = OpenAI(base_url="http://localhost:7653/v1")

for lang_name, lang_code in languages.items():
    result = client.audio.transcriptions.create(
        model="large-v3",
        file=open(f"sample_{lang_code}.mp3", "rb"),
        language=lang_code,
    )
    print(f"{lang_name}: {result.text}")
```

## Testing

### Run Unit Tests

```bash
# Validation tests (fast)
pytest tests/test_openai_endpoint_simple.py -v

# Full test suite
pytest tests/ -v
```

### Test Coverage

✓ Endpoint validation
✓ File format checking
✓ Model name mapping
✓ Parameter acceptance
✓ Error handling
✓ OpenAI SDK compatibility

## Configuration

### Environment Variables

```bash
# Default API key (for authentication)
export API_KEY="your-secret-key"

# Default model to load
export MODEL="base"

# Map "whisper-1" to a different model
export OPENAI_DEFAULT_MODEL="large-v3"

# Telegram bot (optional)
export TG_BOT_ENABLED=false
export TG_BOT_TOKEN="your-token"
```

### Custom Configuration

To change the model mapping, edit `OPENAI_MODEL_MAP` in `main.py`:

```python
OPENAI_MODEL_MAP = {
    "whisper-1": "large-v3",  # Default behavior
    "gpt-4o-transcribe": "large-v3",  # Custom mapping
}
```

## Performance

### Model Performance Comparison

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | ⚡⚡⚡ | ⭐ | 1GB |
| base | 74M | ⚡⚡ | ⭐⭐ | 1GB |
| small | 244M | ⚡ | ⭐⭐⭐ | 2GB |
| medium | 769M | 🔋 | ⭐⭐⭐⭐ | 5GB |
| large-v3 | 1.5B | 🔋🔋 | ⭐⭐⭐⭐⭐ | 6GB |

### Optimization Tips

1. **Start with smaller models** for faster feedback
2. **Use GPU** if available (auto-detected)
3. **Specify language** when known for better accuracy
4. **Adjust temperature** for different use cases:
   - `0.0` - Deterministic, best for important content
   - `0.5` - Balanced, default
   - `1.0+` - More creative interpretations

## Error Handling

### Common Errors

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:7653/v1")

try:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=open("audio.mp3", "rb")
    )
except FileNotFoundError:
    print("Audio file not found")
except Exception as e:
    print(f"Error: {e}")
```

### Expected Error Codes

| Code | Reason | Solution |
|------|--------|----------|
| 400 | Bad request (unsupported format, empty file, invalid model) | Check file format and model name |
| 422 | Validation error (missing required field) | Ensure `file` and `model` parameters are provided |
| 500 | Server error during transcription | Check server logs, try smaller model |
| 504 | Timeout (transcription took too long) | Use smaller model or try again |

## Compatibility

✅ **Fully compatible with:**
- OpenAI Python SDK v1.0.0+
- OpenAI API specification (as of May 2026)
- Standard OpenAI client configuration

**Not compatible with:**
- Streaming transcriptions (todo)
- Advanced parameters like `timestamp_granularities`
- Response formats like `srt`, `vtt`

## Project Structure

```
.
├── main.py                              # FastAPI app with OpenAI endpoint
├── requirements.txt                     # Production dependencies
├── requirements-dev.txt                 # Development dependencies
├── pyproject.toml                       # Project config
├── OPENAI_INTEGRATION.md               # This file
├── DEVELOPMENT.md                       # Development guide
├── tests/
│   ├── test_openai_endpoint_simple.py   # Unit tests
│   ├── test_openai_transcription.py     # Integration tests
│   └── example_openai_usage.py          # Example code
└── openapi.yaml                         # OpenAI spec reference
```

## References

- [Official OpenAI Audio API Documentation](https://platform.openai.com/docs/api-reference/audio/createTranscription)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Faster Whisper Documentation](https://github.com/SYSTRAN/faster-whisper)
- [Whisper Model Cards](https://huggingface.co/spaces/hf-audio/whisper-large-v3-demo)

## Troubleshooting

### Server doesn't start

```bash
# Check if port 7653 is already in use
lsof -i :7653

# Use different port (modify main.py)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Models not loading

```bash
# Ensure you have CUDA available (optional, CPU works too)
python -c "import torch; print(torch.cuda.is_available())"

# Download models manually
from faster_whisper import WhisperModel
model = WhisperModel("large-v3")
```

### Slow transcription

1. Use smaller model (`base` or `small`)
2. Check GPU usage: `nvidia-smi`
3. Monitor system resources: `htop`

### CORS Issues

If accessing from browser, add CORS middleware in `main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Support

For issues, check:
1. [DEVELOPMENT.md](DEVELOPMENT.md) - Development guide
2. [FastAPI docs](https://fastapi.tiangolo.com/)
3. [OpenAI Python SDK docs](https://github.com/openai/openai-python)
4. Server logs for detailed error messages
