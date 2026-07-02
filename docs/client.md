# Python client

Пакет `whisperclient` нужен проектам, которым нужна транскрибация без жёсткой зависимости от доступности сервера.

## Синхронное использование

```python
import whisperclient
from whisperclient import transcribe_sync

whisperclient.api_key = "dev-local-key"
whisperclient.model = "large-v3"
whisperclient.whisper_url = "http://127.0.0.1:7653/transcribe"

text = transcribe_sync("voice.ogg")
print(text)
```

## Streaming

```python
from whisperclient import transcribe_stream_sync

for event in transcribe_stream_sync("voice.ogg"):
    print(event)
```

## Fallback

Логика клиента:

1. попробовать удалённый WhisperServer;
2. если сервер недоступен или вернул ошибку — запустить локальный `whisperclient/whisper_cli.py`;
3. выбрать `cuda`, если доступна CUDA, иначе `cpu`;
4. вернуть результат в том же формате, насколько это возможно.

## CLI helper

```bash
python whisperclient/whisper_cli.py audio.mp3 --model base --device cpu
python whisperclient/whisper_cli.py audio.mp3 --model large-v3 --device cuda --stream
python whisperclient/whisper_cli.py audio.mp3 --model base --words
```
