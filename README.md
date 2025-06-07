# whisperclient

Python-клиент для сервиса голосовой расшифровки Whisper.

## Установка
```bash
pip install whisperclient  # после публикации
```

## Использование
```python
from whisperclient import transcribe_sync
whisperclient.api_key='secret-key'
text = transcribe_sync("audio.ogg")
print(text)
```
