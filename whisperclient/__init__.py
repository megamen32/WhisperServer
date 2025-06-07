from .transcriber import transcribe_sync, transcribe_with_fallback, voice_to_text

# Глобальная переменная для API-ключа (можно менять снаружи)
import os
api_key = os.getenv("API_KEY", "bad-key")
