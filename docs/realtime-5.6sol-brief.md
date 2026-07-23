# Brief for 5.6sol: собственный realtime или внешний ASR-движок

## Роль критика

Ты — senior architect/reviewer. Нужно принять решение для существующего
WhisperServer, а не предложить абстрактный «идеальный» ASR-сервис. Проверяй
утверждения по фактам ниже и явно отделяй:

1. OpenAI-compatible batch REST (`POST /v1/audio/transcriptions`);
2. Deepgram-compatible realtime WebSocket;
3. кастомный realtime WebSocket;
4. настоящий OpenAI Realtime protocol (`/v1/realtime`, `session.update`,
   `input_audio_buffer.append`, `response.audio_transcript.delta` и т.п.).

Не считай два проекта совместимыми с OpenAI Realtime только потому, что у них
есть OpenAI-compatible REST endpoint.

## Цель пользователя

Нужно добавить настоящий realtime ASR для микрофона с partial/final текстом,
VAD и нормальным завершением utterance. Текущий batch-сервис работает и уже
имеет полезную инфраструктуру. Нужно понять, что рациональнее:

- написать только compatibility/gateway слой и использовать внешний ASR
  worker;
- написать весь realtime pipeline внутри текущего проекта;
- перейти на готовый проект целиком.

Главное ограничение: нельзя потерять текущий контроль над загрузкой моделей,
приоритетами и VRAM. В проекте уже есть CUDA broker, который решает, можно ли
загружать модель, задаёт TTL и CPU fallback.

## Что уже есть в WhisperServer

Репозиторий: `/home/roomhacker/PycharmProjects/WhisperServer`.

Ключевые факты:

- `main.py` — FastAPI-приложение, multiprocessing queue и отдельный model
  worker.
- Есть `POST /v1/audio/transcriptions`, `GET /v1/models`, `POST /transcribe`,
  `POST /web/transcribe`, Web UI и Telegram client.
- Batch worker получает целый `audio_bytes`, пишет временный файл, выбирает
  модель и вызывает backend `.transcribe(...)`.
- Для Whisper используется faster-whisper.
- Для Parakeet v3 добавлен отдельный NeMo adapter в
  `parakeet_backend.py`. Он нормализует NeMo hypothesis в общий контракт
  `segments + info`.
- `MODEL_PRIORITY`, `model_cache`, `ManagedModel`, `BROKER_VRAM_MB`,
  `BROKER_CPU_CAPABLE` и `BROKER_TTL_SECONDS` дают модельный routing и
  координацию VRAM.
- `ManagedModel` оборачивает загрузчик модели, поэтому тяжёлые модели
  загружаются лениво и могут быть вытеснены/переведены на CPU.
- VAD уже есть: Silero VAD плюс decoder thresholds. Для batch это хороший
  safety net, но это не realtime endpointing.
- Текущий streaming — это выдача сегментов после загрузки готового файла.
  Для Parakeet decode фактически batch, поэтому SSE/NDJSON не означает
  потоковую обработку микрофона.
- В текущем API `model` выбирает конкретную локальную модель. Это важно не
  сломать ради внешнего realtime worker.
- Уже есть black-box audio tests для TTS → transcription, silence → empty и
  low-level noise → empty, но они проверяют batch endpoint и требуют live
  server при `BLACKBOX_TESTS=1`.

## Исследованные проекты

### WhisperLiveKit

- Репозиторий: `https://github.com/QuentinFuxa/WhisperLiveKit`
- Локальная копия: `/tmp/research-whisperlivekit`
- Исследованный commit: `362d709`
- Лицензия кода: Apache-2.0. Внутри есть зависимости с отдельными
  лицензиями, например SimulStreaming/whisper_streaming MIT и NeMo Apache-2.0.
- Размер checkout: около 201 tracked files.

#### Что это реально делает

Это не просто FastAPI-обёртка вокруг `model.transcribe`. Внутри есть:

- per-session `AudioProcessor`;
- конвертация входного аудио через FFmpeg в PCM;
- Silero VAD iterator;
- буфер аудио и watchdog/cleanup для задач;
- `TranscriptionEngine` singleton с выбранным backend;
- SimulStreaming/AlignAtt policy или LocalAgreement policy;
- partial transcript и commit/locking стабильных фрагментов;
- diff protocol для клиентов, которым не нужно пересылать весь snapshot;
- optional diarization;
- optional translation;
- native WebSocket `/asr`;
- Deepgram-compatible WebSocket `/v1/listen`;
- OpenAI-compatible batch REST.

Это полезная часть, которую сложно аккуратно повторить самостоятельно: online
decoding policy, пересмотр последних слов, local agreement, endpointing,
обрезание буфера и согласование partial/final.

#### Протоколы

Batch REST:

- `POST /v1/audio/transcriptions`;
- `GET /v1/models`;
- `GET /health`;
- OpenAI SDK может подключаться к `base_url=http://host:8000/v1`;
- `model` в REST принимается, но игнорируется: используется backend всего
  запущенного WLK процесса.

Deepgram-compatible WebSocket `/v1/listen`:

- клиент отправляет binary audio frames;
- `KeepAlive`, `Finalize`, `CloseStream` — JSON-команды;
- сервер отправляет `Metadata`, `Results`, `UtteranceEnd`, опционально
  `SpeechStarted`;
- `Results` имеют `is_final` и `speech_final`;
- query parameters похожи на Deepgram: language, punctuate,
  interim_results, vad_events, endpointing и т.д.;
- WLK прямо документирует ограничения: authentication для этого compatibility
  слоя отсутствует/не равна полноценной auth-модели Deepgram, word timestamps
  интерполируются от сегментов, confidence scores равны 0.

Native `/asr`:

- пер-session query parameters `language`, `target_language`, `mode=full|diff`,
  `token`;
- bundled Web UI использует full snapshots;
- diff protocol предназначен для интеграторов, которым нужен append/update
  поток.

#### Модели и backend

Базовые зависимости включают FastAPI, uvicorn, websockets, faster-whisper,
torch, torchaudio, librosa, soundfile и huggingface-hub. Optional extras:

- Qwen3 vLLM;
- Qwen3 causal streaming;
- Voxtral HF/MLX;
- MLX Whisper для Apple Silicon;
- diarization;
- Nemo для Sortformer diarization.

CLI по умолчанию использует Whisper `base`. В коде есть faster-whisper,
SimulStreaming и LocalAgreement backend. Qwen3 causal backend интересен для
мультиязычного realtime, но это отдельный тяжёлый optional stack.

#### Главный конфликт с нашим проектом

WLK — server-wide model/backend. Его REST `model` не делает routing. В core
есть singleton `TranscriptionEngine`, который рассчитан на общий engine для
нескольких сессий. Из исследованного кода не видно механизма, эквивалентного
нашему `ManagedModel` с per-model VRAM arbitration и TTL eviction.

Поэтому WLK можно использовать как отдельный realtime worker, но не стоит
считать его drop-in заменой текущему gateway, если нужны:

- несколько моделей в одном процессе;
- lazy load по запросу;
- модельный приоритет;
- решение CUDA broker «грузить или не грузить»;
- единые API keys и status/usage в текущем домене.

### FunASR

- Репозиторий: `https://github.com/modelscope/FunASR`
- Локальная копия: `/tmp/research-funasr`
- Исследованный commit: `6c3b47c`
- Код репозитория: MIT.
- Для моделей есть отдельный `MODEL_LICENSE`; лицензии конкретных checkpoint
  нельзя автоматически приравнивать к MIT-лицензии Python toolkit.
- В `MODEL_LICENSE` есть формулировки про reference/learning purpose,
  принятие рисков и автоматическое прекращение лицензии при нарушении условий.
  Для коммерческого/публичного сервиса это нужно отдельно проверить юристом.

#### Что это реально делает

FunASR — большой toolkit, а не один сервер и не одна модель. В модельном zoo
есть:

- Fun-ASR-Nano, около 800M;
- Fun-ASR-MLT-Nano, 31 language;
- SenseVoiceSmall, ASR + emotion/events;
- Paraformer-zh и Paraformer-zh-streaming;
- Qwen3-ASR 1.7B, 52 languages;
- GLM-ASR-Nano 1.5B;
- Whisper large-v3/turbo;
- FSMN VAD, CAM++ speaker model, emotion2vec.

Важная деталь: таблица FunASR явно показывает Paraformer streaming для zh/en,
а Fun-ASR-Nano — zh/en/ja и китайские dialects. Нельзя автоматически
предположить, что любой FunASR realtime checkpoint хорош для русского.
Qwen3/MLT требуют отдельной проверки на русском. Для русского решения нужны
наши собственные black-box/WER/latency measurements.

#### OpenAI API

`funasr-server --device cuda` поднимает FastAPI OpenAI-compatible batch API.
Исследованный `_server_app.py` содержит:

- `POST /v1/audio/transcriptions`;
- `GET /v1/models`;
- `GET /health`;
- file upload и response formats;
- VAD/punctuation/speaker pipeline через `AutoModel`;
- отдельный vLLM path для Fun-ASR-Nano;
- fallback path через обычный FunASR AutoModel;
- custom model path и hub selection.

Это полезный batch replacement, но не OpenAI Realtime WebSocket. Не следует
говорить «FunASR уже OpenAI-compatible realtime», пока не показан отдельный
adapter/protocol.

#### Настоящий realtime entrypoint

В репозитории есть отдельный `funasr/bin/realtime_ws.py`. Это standalone
WebSocket-сервер на `websockets`, не тот же самый `/v1/audio/transcriptions`.

Wire protocol:

- клиент сначала отправляет текстовую команду `START`;
- затем отправляет binary audio chunks;
- `STOP` завершает сессию;
- `COMMIT` принудительно заканчивает текущую реплику в `--endpoint-mode client`;
- есть команды для `HOTWORDS`, postprocess hotwords и `LANGUAGE`;
- ответы — кастомный JSON, а не OpenAI Realtime events;
- response содержит `sentences`, `partial`, `partial_start_ms`, `duration_ms`,
  `is_final`;
- server endpoint mode использует VAD и сам завершает utterance;
- client endpoint mode не грузит VAD и ждёт COMMIT клиента.

Настройки, которые важны для эксплуатации:

- `--decode-interval`, default 0.48s;
- `--partial-window-sec`, default 15s, чтобы длинная речь не вызывала
  O(L^2) re-encode на каждом partial decode;
- `--enable-spk` для streaming speaker diarization;
- hotword decoding и отдельные deterministic postprocess corrections;
- `--gpu-memory-utilization`, tensor parallel, dtype, ws ping/timeout/size;
- server/client endpoint mode;
- global model loading перед обслуживанием sessions.

Это технически полезный realtime worker, но protocol translation всё равно
понадобится для OpenAI Realtime или нашего собственного стабильного API.

## Сравнение

| Свойство | Текущий WhisperServer | WhisperLiveKit | FunASR batch | FunASR realtime_ws |
|---|---|---|---|---|
| Batch OpenAI REST | Да | Да | Да | Нет |
| OpenAI Realtime | Нет | Нет | Нет | Нет |
| Deepgram-like WS | Нет | Да | Нет | Нет |
| Native realtime | Нет | Да | Нет | Да |
| VAD | Silero + decoder guards | Silero/session VAD | pipeline VAD | FSMN/dynamic endpointing |
| Partial/final | batch file segments | yes, stable/diff | batch | yes, custom JSON |
| Dynamic model per request | Да | Нет, model ignored | Обычно fixed process | Fixed process/global models |
| CUDA broker/TTL | Да | Нет | Нет | Нет |
| Speaker diarization | Нет/не основной путь | optional | toolkit support | optional streaming |
| Russian guarantee | проверяем своим тестом | зависит от backend | зависит от checkpoint | зависит от checkpoint |
| OpenAI Realtime adapter | нужно написать | нужно написать | нужно написать | нужно написать |

## Что повторять, а что не повторять

### Не повторять с нуля

Не стоит самостоятельно реализовывать заново:

- streaming audio buffer;
- VAD state machine;
- endpointing;
- partial decode policy;
- stable-prefix/local-agreement;
- word rollback/commit;
- backpressure и disconnect cleanup;
- timestamp reconciliation;
- long-session memory bounds.

Именно здесь находится большая часть скрытой сложности. SSE для результата
готового файла не является доказательством, что эти задачи решены.

### Повторить в нашем gateway

Разумно написать самостоятельно только protocol/control-plane слой:

- auth/API keys;
- `/v1/realtime` или другой целевой compatibility endpoint;
- session registry и limits;
- преобразование OpenAI events в worker events;
- routing к WLK/FunASR worker;
- worker health и reconnect;
- сохранение текущего `/v1/audio/transcriptions`;
- status/metrics;
- интеграция с CUDA broker на уровне worker lifecycle.

Это сохраняет наш главный актив — управление ресурсами — и не требует
изобретать ASR decoding.

## Варианты решения

### Вариант A: полностью написать realtime внутри WhisperServer

Плюсы: единый процесс, полный контроль, можно напрямую подключить broker.

Минусы: нужно реализовать streaming backend для каждой модели. Parakeet
batch adapter не превращается в realtime автоматически. Потребуются session
state, chunked model APIs, VAD endpointing, partial/final semantics,
disconnect/backpressure и протокол.

Оценка: это не «несколько endpoint-ов», а отдельный серьёзный подсервис.
Риск регрессии текущего batch API высокий.

### Вариант B: текущий WhisperServer как gateway + WhisperLiveKit worker

Плюсы:

- самый зрелый найденный streaming pipeline;
- Deepgram-compatible `/v1/listen` можно использовать уже сейчас;
- native `/asr` и diff protocol доступны для собственного adapter;
- WLK уже решает VAD, online decoding, partial/final и cleanup;
- код Apache-2.0.

Минусы:

- WLK model field не маршрутизирует модель;
- модель грузится на уровне процесса/singleton;
- нет готового OpenAI Realtime protocol;
- нужна auth/proxy/worker lifecycle обвязка;
- русское качество зависит от реально выбранного backend, а не от названия
  WLK.

Это лучший кандидат для POC.

### Вариант C: gateway + FunASR realtime worker

Плюсы:

- есть server-side/client-side endpoint mode;
- decode interval, partial window, hotwords и speaker support уже сделаны;
- можно получить native realtime result schema;
- toolkit даёт много ASR/VAD/punctuation моделей.

Минусы:

- custom protocol;
- OpenAI-compatible REST и realtime — разные entrypoints;
- default realtime model ориентирован не на русский;
- модельные лицензии требуют отдельной проверки;
- worker имеет глобальные модели и не интегрирован с нашим broker.

Это хороший экспериментальный worker, особенно если тесты покажут преимущество
конкретной модели на русских данных. Это не очевидная замена текущему проекту.

### Вариант D: полностью перейти на один из проектов

Переход оправдан только если одновременно выполняются условия:

1. нужен именно тот protocol, который проект уже поддерживает;
2. устраивает его модельная политика;
3. устраивает его auth и lifecycle;
4. устраивает качество на наших русских black-box тестах;
5. приемлема его лицензия checkpoint-ов;
6. готовы отказаться от/вынести CUDA broker и текущего model routing.

Сейчас доказательств для полного перехода нет.

## Предварительная рекомендация

Не переносить текущий сервис целиком и не писать ASR streaming с нуля.

1. Оставить текущий WhisperServer control plane и batch API.
2. Поднять WhisperLiveKit как отдельный realtime worker для первого POC.
3. Сделать небольшой gateway adapter с auth и одним стабильным публичным
   протоколом.
4. Сначала выбрать target protocol: OpenAI Realtime или Deepgram-compatible.
5. Если нужен именно OpenAI Realtime, реализовать только mapping событий, а
   decoding оставить WLK.
6. После POC сравнить WLK с FunASR `realtime_ws.py` на одинаковых аудио:
   русская чистая речь, шум, тишина, телефон, длинная речь, RU/EN code-switch,
   имена и термины.
7. Только после измерений решать, нужна ли поддержка нескольких workers и
   broker-aware process lifecycle.

Первоначальная ставка: **WhisperServer gateway + WhisperLiveKit worker**.
FunASR держать как второй worker-кандидат, а не как автоматическую замену.

## Вопросы, на которые должен ответить 5.6sol

1. Какой protocol имеет наибольшую практическую ценность для нашего клиента:
   OpenAI Realtime или Deepgram-compatible WS?
2. Стоит ли делать `/v1/realtime` compatibility layer поверх WLK, или лучше
   публично выбрать Deepgram protocol и не притворяться полной OpenAI Realtime
   совместимостью?
3. Может ли WLK с Qwen3 causal/Whisper дать приемлемый русский realtime, или
   FunASR MLT/Qwen/другая модель имеет доказанное преимущество?
4. Как сохранить model routing и CUDA broker, если WLK/FunASR требуют
   server-wide model и грузят её до session handling?
5. Нужно ли запускать отдельный worker на модель и выключать его по TTL, или
   для realtime warm model должна постоянно жить на GPU?
6. Какие события обязательны: partial delta, stable text, speech started,
   speech stopped, utterance final, commit ack, errors, reconnect?
7. Где должна быть VAD truth: в worker или в gateway? Как избежать двойного
   endpointing и потерянного начала слова?
8. Какой минимальный POC докажет решение за один короткий цикл и не затронет
   текущий batch API?
9. Какие license/model-card проверки обязательны перед использованием FunASR
   checkpoint в публичном сервисе?
10. Какой вариант ты рекомендуешь с confidence и почему: WLK worker,
    FunASR worker, собственный realtime или полный migration?

## Требуемый формат ответа критика

Дай:

1. verdict: build / WLK / FunASR / hybrid / migrate;
2. confidence 0-100;
3. три главных аргумента;
4. три главных риска;
5. архитектуру первого POC;
6. точный protocol и event mapping;
7. план измерений и критерии pass/fail;
8. что оставить из текущего WhisperServer;
9. что точно не делать;
10. условия, при которых verdict меняется.

Не предлагай миграцию только на основании наличия OpenAI REST endpoint.
