# Development

Актуальный guide перенесён в [docs/development.md](docs/development.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
python -m py_compile main.py telegram_bot.py whisperclient/*.py tests/*.py
```
