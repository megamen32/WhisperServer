# Configuration

Конфигурация читается из `.env` через `python-dotenv` и из переменных окружения процесса.

## Основные переменные

| Переменная | Значение по умолчанию | Описание |
|---|---|---|
| `API_KEY` | пусто / disabled | ключ для OpenAI-compatible endpoint и внутренних клиентов |
| `MODEL` | `parakeet-v3` | модель по умолчанию для `/transcribe` |
| `OPENAI_DEFAULT_MODEL` | `parakeet-v3` | fallback-модель для умного alias `whisper-1`, если ничего не загружено |
| `WHISPER_METRICS_JSONL` | `whisper_metrics.jsonl` | путь к JSONL-телеметрии каждой транскрибации; пустое значение отключает запись |
| `WHISPER_VAD_ENABLED` | `1` | включить VAD по умолчанию |
| `WHISPER_VAD_THRESHOLD` | `0.50` | порог Silero VAD |
| `WHISPER_VAD_MIN_SPEECH_MS` | `250` | минимальная длительность речи |
| `WHISPER_VAD_MIN_SILENCE_MS` | `700` | пауза, после которой закрывается speech chunk |
| `WHISPER_VAD_SPEECH_PAD_MS` | `200` | padding вокруг speech chunk |
| `WEBUI_SESSION_TTL_SECONDS` | `7200` | срок жизни web UI session/csrf token |

## Telegram bot

| Переменная | Описание |
|---|---|
| `TG_BOT_ENABLED` | `1/true/yes/on` включает запуск `telegram_bot.py` из `main.py` |
| `TG_BOT_TOKEN` | token Telegram bot API |
| `TG_BOT_SERVER_URL` | endpoint сервера, обычно `http://127.0.0.1:7653/transcribe` |
| `TG_BOT_API_KEY` | ключ, который бот передаёт серверу; если пусто, используется `API_KEY` |
| `TG_BOT_MODEL` | модель для бота |
| `TG_BOT_LANG` | язык для бота, например `ru` |
| `TG_BOT_MAX_SIZE_MB` | максимальный размер файла |
| `TG_BOT_CONCURRENCY` | параллельные обработки |
| `TG_BOT_HTTP_PROXY` | proxy для Telegram API |

## CUDA broker

| Переменная | Значение по умолчанию | Описание |
|---|---|---|
| `CUDABROKER_CLIENT_ID` | `whisper` | id клиента в broker |
| `CUDABROKER_WHISPER_TTL_SECONDS` | `900` | сколько держать модель после последнего использования |
| `CUDABROKER_VRAM_MB_<MODEL>` | встроенная таблица | override оценки VRAM для модели |
| `CUDABROKER_CPU_CAPABLE_<MODEL>` | встроенная таблица | можно ли безопасно выполнить модель на CPU |

`.env.example` безопасен для git. Реальный `.env` должен оставаться локальным и уже исключён через `.gitignore`.
