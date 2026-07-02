# Security

## Secrets

- Реальные ключи держите только в `.env`, systemd EnvironmentFile или секрет-хранилище.
- `.env` исключён из git через `.gitignore`.
- `.env.example` содержит только безопасные placeholders.
- В тестах используйте фиктивные значения, например `WHISPER_TEST_API_KEY`.

## API key

`/v1/audio/transcriptions` проверяет `Authorization: Bearer ...` или `X-API-Key`, если `API_KEY` задан.

Для локальных экспериментов можно использовать короткий ключ. Для сервера в сети нужен длинный случайный ключ:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Web UI

Web UI не получает API key. Вместо этого используется server-side session cookie и одноразовый CSRF token. Дополнительно проверяется same-origin request.

## Git history cleanup

Если секрет попал в историю:

1. замените значение в рабочем дереве;
2. перепишите историю `git filter-repo` или `git filter-branch`;
3. удалите `refs/original`, reflog и выполните `git gc --prune=now`;
4. force-push в remote;
5. считайте старый секрет скомпрометированным и замените его в реальной системе.

Даже после чистки git старый ключ мог попасть в CI logs, IDE index, backup или fork. Ротация обязательна.
