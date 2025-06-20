# bezrabotnyiwhisper

Python-клиент для сервиса голосовой расшифровки Whisper.

## Установка
```bash
pip install bezrabotnyiwhisper  # после публикации
```

## Использование
```python
from whisperclient import transcribe_sync, transcribe_stream_sync
import whisperclient

# Настраиваем ключ и модель
whisperclient.api_key = 'secret-key'
whisperclient.model = 'large-v3'

# Обычная расшифровка
text = transcribe_sync("audio.ogg")
print(text)

# Стриминг-режим
for chunk in transcribe_stream_sync("audio.ogg"):
    print(chunk)
```
