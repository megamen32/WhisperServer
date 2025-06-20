# WhisperServer

A high-performance transcription server using [Faster Whisper](https://github.com/guillaumekln/faster-whisper), built with FastAPI, supporting:

- ✅ On-the-fly model selection (`tiny`, `base`, `small`, etc.)
- ⚡ Priority-based queue (lightweight models are processed first)
- 💾 Disk-based caching (via `diskcache`) to avoid reprocessing same audio
- 🧠 Multiprocessing architecture for parallel decoding
- 🖥️ GPU and CPU modes supported (automatically selected)

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

**Or manually:**

```bash
pip install fastapi uvicorn faster-whisper diskcache python-multipart
```

If using GPU:

* Make sure you have installed CUDA and cuDNN (e.g. CUDA 12.4)
* `torch.cuda.is_available()` should return `True`

---

### 2. Start the server

```bash
python main.py
```

By default, the server runs at `http://localhost:7653`

---

## 📡 API Usage

### `POST /transcribe`

Transcribe any audio file using a chosen model.

#### Request

* `file`: audio file (`.mp3`, `.wav`, etc.)
* `model`: model name (default: `base`)
* `stream`: return results progressively when `true`
* `words`: include word-level timestamps when `true`

#### Example using curl

```bash
# regular response
curl.exe -F "file=@C:/path/to/audio.mp3" "http://localhost:7653/transcribe?model=base"
# with word timestamps
curl.exe -F "file=@C:/path/to/audio.mp3" "http://localhost:7653/transcribe?model=base&words=true"
# streaming response
curl.exe -N -F "file=@C:/path/to/audio.mp3" "http://localhost:7653/transcribe?model=base&stream=true"
```

#### Example Response

```json
{
  "text": "This is the transcribed result."
}
```

### `GET /status`

Returns current queue size and information about loaded models.

```bash
curl http://localhost:7653/status
```

---

## 🧠 Caching

The server uses `diskcache` to avoid redundant transcription for the same file. Cached results are keyed by the SHA256 hash of the audio + model name.

Cached files are stored in `./whisper_cache` with a 30 day TTL and a 10 GB size limit. Old entries are pruned automatically.

---

## 🔧 Model Priority

Requests using smaller models like `tiny` and `base` are processed before heavier models like `large-v2`.

You can customize priorities in the `MODEL_PRIORITY` dictionary.

---

## 🛑 Shutdown Handling

The server cleanly shuts down:

* Cancels the response listener
* Terminates the model process
* Cleans up temporary files

---

## 💡 TODO Ideas

* ~~Add `/status` endpoint to monitor queue size and model usage~~ (done)
* ~~Add TTL-based expiration to cache~~ (done)
* ~~Support streaming or chunked transcription~~ (done)

---

## License

MIT — use it, modify it, profit from it. No warranty.


