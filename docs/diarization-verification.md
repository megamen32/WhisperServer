# Community-1: PASS

Проверено 2026-10-13 на работающем WhisperServer server-88. Код модели:
`ebdab08`; финальный Web UI с сохранением переносов строк: `7804b73`.
Условия Hugging Face приняты, все веса доступны, HF_TOKEN хранится только
в серверном `.env`. Дополнительных действий владельца не требуется.

## Реальные consumer checks

- Реальные голоса openslr/librispeech_asr, clean validation, спикеры 2277 и 777:
  запись 25,04 секунды с последовательностью первый → второй → первый.
  Число говорящих не задавалось. Community-1 определила два голоса и сохранила
  метку вернувшегося первого. Inference CPU 28,13 секунды; memory.peak
  отдельного проверочного cgroup — 948547584 байт (~905 MiB).
- Свежий `/v1/audio/transcriptions` с Whisper tiny возвращает `speaker_text`,
  `segments[].speaker` и `speaker_turns` с этой же последовательностью.
- Свежий запрос без выбора ASR-модели использует `gigaam-v3` и возвращает
  SPEAKER_00 → SPEAKER_01 → SPEAKER_01 → SPEAKER_00 за 43,35 секунды.
  На английском материале проверялись метки, а не качество русского GigaAM ASR.
- Реальный headed AgentBrowser на server-88: выбрать tiny, включить
  «Различать говорящих», загрузить контрольный файл, дождаться «Готово».
  Подписанные реплики отображаются отдельными строками. Первая свежая загрузка
  и повторная загрузка из кэша проверены; кэш сохраняет метки.
- Кнопка «Скопировать»: реальный системный буфер собственного Xvfb-профиля
  прочитан через xclip; текст полностью равен отображённому speaker_text,
  включая обе метки и переносы строк. Screenshot показывает
  «Результат скопирован». Финальная браузерная проверка: PASS, 2,845 секунды,
  пик RAM 251875328 байт, peak tasks 128; собственная сессия закрыта.

## Артефакты в каноническом checkout

- `.tmp/community-1/fixture-longturn-manifest.json`: исходные голоса и текст.
- `.tmp/community-1/real-longturn-result.json`: интервалы, время и память модели.
- `.tmp/community-1/real-api-result.json`: свежий Whisper API response.
- `.tmp/community-1/default-gigaam-result.json`: свежий GigaAM API response.
- `.tmp/community-1/ui-accepted.json`: отображённый текст и настоящий clipboard.
- `.tmp/community-1/community-1-success.png`: финальное состояние интерфейса.
- `.tmp/community-1/ui_accept.py`: воспроизводимый браузерный сценарий;
  запускать в описанном конечном cgroup и с affinity на два доступных CPU.

## Проверки кода и бюджет

До включения модели: 44 focused tests и 3 alias tests прошли. После снижения
пакетов: затронутые 21 тест прошли. AST-граф обновлён. Для простого CSS-fix
отдельный unit test не добавлялся: результат проверен на реальном интерфейсе.
Исходные неудачные попытки не считались GREEN: мягкий RAM-лимит проверочного
процесса оказался ниже рабочего набора; размер пакетов снижен до 1. Browser
runner дополнительно ограничен двумя доступными CPU через affinity, потоками
Tokio/Rayon 2, одним renderer и одним raster thread. RAM/CPU/tasks лимиты
рабочего Whisper не увеличивались. Системный clipboard проверен напрямую
в собственном тестовом дисплее, без изменения разрешений пользовательских
браузеров. Подробный конечный бюджет: [diarization.md](diarization.md).

## Практические ограничения

На варианте записи, где второй голос звучал только 2,2 секунды, автоматическое
определение объединило голоса. Это реальное ограничение; качество не обещается
для любых коротких реплик. Плотные длинные разговоры и сложные перекрытия остаются slow nightly coverage.
Порог 20 минут снят; новую focused-проверку длинного файла см. ниже.


## Длинный файл и стандартные параметры OpenAI: PASS

Код `d7cf585`, развёрнут на server-88 без повышения лимитов сервиса.
50 fast unit / focused integration проверок прошли за 9,21 секунды.
После перезапуска выполнен реальный публичный OpenAI SDK запрос:
`model="whisper-1", response_format="diarized_json", chunking_strategy="auto", stream=True`.
Файл `.tmp/community-1/over20min.flac` имеет полную длину 1285,04 секунды
(21:25): исходные два голоса и возврат первого плюс 1260 секунд тишины.
Это проверка снятия порога и восстановления времени, не нагрузочное измерение
21 минуты непрерывной речи и не оценка качества русского ASR на английском аудио.

Через SSE SDK получил A → B → B → B → A, последняя реплика началась в
1278,5185625 секунды. Первый голос после паузы сохранил метку A. Свежий
запрос занял 49,04 секунды. Повторный обычный SDK запрос из кэша вернул
типизированные duration=1285,04, task=transcribe и те же сегменты/спикеры.
Общая обычная release-проверка с tests, deployment, restart и API canary
заняла 82,33 секунды, меньше предела 180 секунд.

Воспроизводимый сценарий: `.venv/bin/python .tmp/community-1/long_release.py`;
все тяжёлые проверки имеют конечные cgroup-бюджеты. Результат:
`.tmp/community-1/long-accepted.json`. Плотные часовые записи требуют
slow nightly замера в штатном runner после его отдельного admission;
в этом цикле такой запуск не выполнялся. Прежняя приёмка коротких голосов,
UI и настоящего clipboard выше сохраняется.


Дополнительная реальная UI-приёмка длинного файла также PASS: headed
AgentBrowser на server-88, выбрать gigaam-v3, включить говорящих, загрузить
21:25 FLAC, дождаться Готово, нажать Скопировать. Отображаются Спикер 1 →
Спикер 2 → Спикер 1, системный clipboard собственного Xvfb полностью равен
тексту и сохраняет переносы. Сценарий занял 40 секунд, memory.peak 243339264
байт, tasks.peak 128. Артефакты `.tmp/community-1/long-ui-accepted.json`,
`.tmp/community-1/long-ui-success.png`, `.tmp/community-1/long_ui_accept.py`.

Первоначальные две браузерные попытки зависли на пустой странице при TasksMax
128 (pids.events max подтверждает ограничение); они не считались GREEN.
После снижения affinity браузера с двух доступных CPU до одного полный
consumer canary прошёл в прежних RAM/tasks-пределах. Эта более низкая
параллельность рекомендована для повторения; серверный ASR-бюджет не менялся.
Собственная браузерная сессия и unit завершены, чужие процессы не затрагивались.

## Managed GPU and real long recordings: PASS

2026-10-13: deployed `2e64328` on the existing server-88 service. Community-1
uses CUDA through ManagedModel/cudabroker,1536MiB reservation and priority11;
its active lease protects the whole model pass. Weights and WeSpeaker frame
caches are released before the ASR stage. Cudabroker client0.2.1 is pinned;
sync/async heartbeat no longer renews its own activity grace, and model
finalizers run before acknowledging free memory. SDK source reconciled with
previously deployed client code; the broker server itself was not restarted.

Real GPU canary: two consecutive25.04s recordings retained two voices and the
returning first voice. Two load/inference/unload cycles took9.48s excluding
import; peak CUDA allocated119.62MiB, reserved154MiB, RAM1566273536bytes.
Model CUDA bytes after each unload:0 above the original8.125MiB cuBLAS
workspace. Initial39MiB WeSpeaker LRU retention was reproduced and repaired;
failed initial attempts were not GREEN.

After the final intentional restart, a fresh public OpenAI SDK request used
model=whisper-1, response_format=diarized_json, chunking_strategy=auto,
cache=false. The observer saw an active whisper-community-1 lease, priority11,
and no such service lease after the result. Two speaker labels and the
returning first remained correct.63 focused checks plus deployment/restart/API
canary completed in57.89s, below180s. Evidence:
`.tmp/diarization-practice-20261013/gpu-api-accepted.json` and `gpu-canary.json`.
Old queued requests were allowed to finish before the deployment.

Full user-provided video47:17: CUDA diarization116.39s, whole worker319.48s.
Automatic clustering proposed four voices when the user expected three;
the additional short cluster had about22s in40turns. Preserve this baseline.
The separate run with official pyannote num_speakers=3 took331.11s and
returned three labelled voices,653 nonempty ASR segments. This number is
conditioned on the user's hint, not an independent count-accuracy claim.

Full user audio1:26:06: CUDA diarization236.72s, automatically three voices;
1132 source turns and1040 nonempty ASR segments. A combined two-file
verification process hit systemd-oomd pressure after saving the audio turns.
The second file was resumed alone using the verified saved CUDA turns;
recognition took343.03s. Productive phase total579.75s (~9:40), excluding
failed work and implementation time. RAM/GPU limits were not raised; files,
checkpoints and completed results were retained. This is authorized user-file
processing, not nightly benchmark coverage or a dense-voice release test.

The real video also exposed a270-sample turn shorter than GigaAM's320-sample
FFT window. Decoder input is padded to200ms, with original segment boundaries
unchanged. A real CPU FFT regression check and the full CUDA worker rerun pass.
No user recordings/transcript content are committed. Local artifacts:
`video-2026-10-13*.json/txt/srt`, `shkolny-bulvar-30.json/txt/srt`, `summary.json`
and `Проверка.md` inside `.tmp/diarization-practice-20261013/`.
