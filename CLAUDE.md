# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

DocMind is an agentic document intelligence system. Users upload documents; the Agent autonomously selects and chains Skills (RAG search, mindmap, relation graph, report, weekly report, presentation, web search, generic Codex-style packages) via OpenAI Function Calling to answer questions and produce structured outputs.

## Development Commands

### Start infrastructure (PostgreSQL, Redis, Qdrant)
```bash
./start.sh          # macOS/Linux one-command startup
# Windows: powershell -ExecutionPolicy Bypass -File .\start.ps1
```

### Activate virtualenv and install dependencies
```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run API server
```bash
uv run uvicorn app.main:app --reload
```

### Run Celery worker (required for document parsing — must run alongside the API)
```bash
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

API docs: http://127.0.0.1:8000/docs

## Architecture

```
FastAPI (async) ──► Agent Orchestrator ──► Skills (pluggable)
     │                     │                  ├─ Python-backed skills
  routers/           Function Calling         │  kb_search / mindmap / graph
  services/           ReAct loop              │  report / weekly / ppt / web
  models/                                     └─ Codex-style generic packages
     ▼
PostgreSQL (metadata)  Qdrant (vectors)  Redis (Celery queue)
```

### Dual database engines

`app/database.py` exposes two engines because FastAPI uses async I/O and Celery tasks run synchronously:
- `AsyncSessionLocal` (asyncpg) — used in all FastAPI route handlers via the `get_db` dependency in `app/database.py`
- `SyncSessionLocal` (psycopg2) — used inside `app/tasks/document_tasks.py` (Celery)

Never use the sync session in async route handlers or vice versa.

### Document processing pipeline

Upload → `document_service.create_document` saves to `./uploads/` with a UUID filename → dispatches `process_document.delay(doc.id)` to Celery → worker parses file (`app/utils/file_parser.py`), chunks text, embeds via `embedding_service`, stores vectors in Qdrant with payload `{user_id, document_id, chunk_index}`, updates `Document.status` to `completed`.

Table auto-creation (`Base.metadata.create_all`) runs on startup (dev only). For production use Alembic migrations.

### Agent orchestration (`app/agent/orchestrator.py`)

`run_agent()` is synchronous — it (and all LLM/skill calls) is invoked via `asyncio.to_thread` from the async service layer (`agent_service.chat_with_agent`), never directly in routes. The loop runs at most `settings.agent_max_steps` (default 6) iterations. Each iteration sends messages + all tool definitions to the LLM; if it returns `tool_calls`, skills are executed and their results are appended as `role: tool` messages before the next iteration.

Structured outputs (`type` in `{"mindmap", "relation_graph", "report"}`) are collected in `artifacts`; the execution path is recorded in `trace` for debugging and frontend display.

### Skills system (`app/skills/`)

DocMind v1.0.0 supports two paths:

1. Python-backed skills: subclass `BaseSkill`, register with `@register_skill`, and import the module in `app/skills/__init__.py`.
2. Codex-style generic packages: copy a folder with `SKILL.md` into `app/skills/packages/<slug>/`; if no Python skill already owns its name, it is auto-registered as `GenericPackageSkill`.

Current skills include `search_knowledge_base`, `generate_mindmap`, `generate_relation_graph`, `generate_report`, `generate_weekly_report`, `generate_presentation`, `web_search`, and sample generic package `codex_note`.

Generic skills write only inside `skill_workspaces/user_<id>/<skill_slug>/` by default. Shell is disabled unless `SKILL_SHELL_ENABLED=true` and the command is in `SKILL_SHELL_ALLOWED_COMMANDS`.

## Configuration (`.env`)

All settings are in `app/config.py` (`Settings` class via pydantic-settings). Required keys:

| Key | Notes |
|---|---|
| `DATABASE_URL` | Must use `postgresql+asyncpg://` scheme |
| `SECRET_KEY` | JWT signing key |
| `OPENAI_API_KEY` | Also used for embeddings unless overridden |

Optional overrides: `OPENAI_BASE_URL` (proxy), `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM` (must match model), `CHAT_MODEL`.

`settings.sync_database_url` is derived automatically by replacing `+asyncpg` with nothing.

## Key Design Decisions

- **Celery tasks are sync**: avoids running an event loop inside a worker. The `SyncSessionLocal` + psycopg2 path exists specifically for this.
- **Qdrant payloads always include `user_id`**: all vector searches filter by `user_id` to enforce data isolation between users.
- **`bcrypt==4.0.1` is pinned**: passlib 1.7.4 is incompatible with bcrypt 5.x — do not upgrade bcrypt without also upgrading passlib.
- **`embedding_batch_size`**: some providers (e.g. DashScope) cap batch size; the default of 10 is conservative. Adjust in `.env` if your provider allows larger batches.
