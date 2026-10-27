FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Сначала только зависимости — этот слой кэшируется, пока не меняется pyproject.toml
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY alembic.ini .
COPY migrations ./migrations
COPY app ./app

# кэш моделей Hugging Face (Whisper) — том в compose; каталог создаём заранее,
# иначе Docker создаст его от root и пользователь приложения не сможет писать
RUN useradd --create-home --uid 1000 yourist \
    && mkdir -p /home/yourist/.cache/huggingface \
    && chown -R yourist:yourist /home/yourist/.cache
USER yourist

EXPOSE 8000

# порт берётся из PORT, чтобы один образ обслуживал и api, и corpus
ENV PORT=8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/v1/health', timeout=3)"

# миграции применяет отдельный одноразовый контейнер (см. docker-compose.yml)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\" --proxy-headers"]
