# Configuration

Конфигурация читается из `.env` через `python-dotenv` и из переменных окружения процесса.

## Основные переменные

| Переменная | Значение по умолчанию | Описание |
|---|---|---|
| `API_KEY` | пусто / disabled | ключ для OpenAI-compatible endpoint и внутренних клиентов |
| `MODEL` | `base` | модель по умолчанию для `/transcribe` |
| `OPENAI_DEFAULT_MODEL` | `large-v3` | модель для alias `whisper-1`, если ничего подходящего не загружено |
| `OPENAI_WHISPER_MIN_MODEL` | `medium` | минимальный уровень модели для переиспользования под `whisper-1` |
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
