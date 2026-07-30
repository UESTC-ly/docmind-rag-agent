FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# Web API and Celery worker share this immutable production image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UPLOAD_DIR=/var/lib/docmind/uploads \
    SKILL_WORKSPACE_DIR=/var/lib/docmind/skill-workspaces \
    AGENT_CHECKPOINT_PATH=/var/lib/docmind/agent-checkpoints.sqlite3

WORKDIR /opt/docmind

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin docmind \
    && mkdir -p /var/lib/docmind/uploads /var/lib/docmind/skill-workspaces \
    && chown -R docmind:docmind /var/lib/docmind

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY frontend ./frontend

USER docmind

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

ENTRYPOINT ["tini", "--"]

# The worker service overrides this command in docker-compose.production.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
