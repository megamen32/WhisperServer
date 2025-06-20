from .transcriber import (
    transcribe_sync,
    transcribe_with_fallback,
    voice_to_text,
    transcribe_stream_sync,
    transcribe_stream_with_fallback,
)

# Глобальные переменные для настройки клиента (API‑ключ и модель)
import os

api_key = os.getenv("API_KEY", "bad-key")
# Модель по умолчанию используется при обращении к серверу
model = os.getenv("MODEL", "large-v3")
