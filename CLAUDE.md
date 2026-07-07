# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

DocMind is an agentic document intelligence system. Users upload documents; the Agent autonomously selects and chains Skills (RAG search, mindmap, relation graph, report, web search) via OpenAI Function Calling to answer questions and produce structured outputs.

## Development Commands

### Start infrastructure (PostgreSQL, Redis, Qdrant)
```bash
colima start          # if using colima on macOS
docker-compose up -d
docker-compose ps     # verify all three containers are Up
```

### Activate virtualenv and install dependencies
```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run API server
```bash
uvicorn app.main:app --reload
```

### Run Celery worker (required for document parsing — must run alongside the API)
```bash
celery -A app.celery_app worker --loglevel=info
```

API docs: http://127.0.0.1:8000/docs

## Architecture

```
FastAPI (async) ──► Agent Orchestrator ──► Skills (pluggable)
     │                     │                  ├─ kb_search    (RAG)
  routers/           Function Calling         ├─ mindmap
  services/           ReAct loop              ├─ graph
  models/                                     ├─ report
                                              └─ web_search
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

| File | Skill name | Description |
|---|---|---|
| `kb_search.py` | `search_knowledge_base` | Qdrant vector search → RAG answer |
| `mindmap.py` | `generate_mindmap` | LLM generates mindmap JSON |
| `graph.py` | `generate_relation_graph` | LLM generates relation graph JSON |
| `report.py` | `generate_report` | LLM writes a structured report |
| `web_search.py` | `web_search` | DuckDuckGo search via `ddgs` |

**Adding a skill:** subclass `BaseSkill` in a new file under `app/skills/`, decorate with `@register_skill`, add one import line to `app/skills/__init__.py`. The agent picks it up automatically — no changes to the orchestrator.

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
