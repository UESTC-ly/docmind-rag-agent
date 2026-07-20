# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

DocMind v3.1.0 is an agentic document intelligence system. Its outer Agent orchestration runs as a durable LangGraph with human approval before high-risk Skills, renewable cross-worker run leases, SQLite checkpoint recovery, and retention-based checkpoint cleanup. Users upload documents; the Agent autonomously selects and chains Python-backed or Codex-style Skills through OpenAI Function Calling. Online chat, document-grounded Skills, and evaluation share the same hybrid retrieval pipeline: dense candidates, database-side keyword candidates, scored RRF fusion, and bounded reranking.

The repository supports two deployment profiles:

- **Web development/production**: PostgreSQL, Redis/Celery, Qdrant Server, Python 3.12, and `uv`.
- **Installed desktop application**: a Tauri-bundled PyInstaller sidecar with SQLite, Qdrant local mode, and the local task executor. End users do not install Docker, PostgreSQL, Redis, Qdrant Server, `uv`, or Python.

## Development Commands

### Start the Web stack

```bash
./start.sh          # macOS/Linux one-command startup
# Windows: powershell -ExecutionPolicy Bypass -File .\start.ps1
```

The launchers start PostgreSQL/Redis/Qdrant, apply `alembic upgrade head`, start the Celery worker, and then start FastAPI.

### Install dependencies and run components manually

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
uv run alembic upgrade head
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
uv run uvicorn app.main:app --reload
```

Celery is required in the Web profile for both document parsing and evaluation runs. API docs: <http://127.0.0.1:8000/docs>.

### Run automated checks

```bash
uv run pytest --cov=app --cov-report=term-missing

cd frontend
npm ci
npm run test:e2e

cd ../desktop
npm ci
npm run test:sidecar
npm run test:rust
```

## Architecture

```text
FastAPI (async) ──► LangGraph Outer Agent ──► Skills (Python + generic packages)
     │                       │                    │
     ├──────────────► shared hybrid retrieval ◄──┘
     │                 dense + DB keyword
     │                    RRF + reranker
     ▼
Web: PostgreSQL + Qdrant Server + Redis/Celery
Desktop: SQLite + Qdrant local + local task executor
```

### Database engines and migrations

`app/database.py` exposes async and sync engines:

- `AsyncSessionLocal` is used by FastAPI handlers.
- `SyncSessionLocal` is used by Celery tasks and the desktop local task executor.

The URL determines the profile: Web normally uses `postgresql+asyncpg://`; desktop injects `sqlite+aiosqlite:///...`. Never mix sync and async sessions. Alembic is the schema source of truth. Application startup applies migrations and must not call `Base.metadata.create_all`; tests may still create disposable schemas directly.

For an existing v2.1 database that was created with `create_all`, verify it matches the v2.1 schema, then run:

```bash
uv run alembic stamp 0001_v210_baseline
uv run alembic upgrade head
```

Do not stamp a fresh database; run `uv run alembic upgrade head` instead.

### Background pipelines

Document upload calls `dispatch_document()`. Web mode sends `process_document` to Celery; desktop mode submits the same work to the bounded local executor. The worker parses, chunks, embeds, writes Qdrant, and updates document status.

`POST /eval/runs` creates a pending run and returns `202 Accepted` immediately. `dispatch_evaluation()` routes execution to Celery or the desktop local executor. The frontend polls `GET /eval/runs/{id}` through pending/running/completed/failed. The runner is idempotent and stores retrieval diagnostics in `EvalResult`.

### Retrieval

`app/services/retrieval.py` owns the shared retrieval path used by online RAG, Skills, and evaluation:

- Dense candidates come from Qdrant.
- PostgreSQL keyword recall uses `to_tsvector`, `websearch_to_tsquery`, `ts_rank_cd`, and a migration-managed GIN index.
- SQLite uses a SQL-side bounded `LIKE` compatibility score.
- Ownership/document filters, ranking, and `LIMIT` stay in the database query.
- Scored RRF preserves source ranks/scores, then `app/services/reranker.py` applies a deterministic local reranker or an optional HTTP reranker with local fallback.

### Agent orchestration and Skills

`run_agent()` is synchronous and is called from the async service layer via `asyncio.to_thread`. The outer orchestration is a LangGraph with supervisor, tool-selection, approval, and single-tool execution nodes. It is capped by `settings.agent_max_steps`. HTTP runs use a persistent SQLite saver and a random run/thread ID; `POST /agent/resume` resumes an interrupt after ownership validation.

High-risk Python Skills may set `requires_approval = True`. Generic packages that declare configured high-risk capabilities are approved before the whole outer Skill invocation. The internal Generic Skill ReAct runner is not a LangGraph subgraph in v3.1, and full Multi-Agent decomposition is explicitly deferred.

Every mutating Agent entrypoint acquires an exclusive run lease before inspecting or advancing its checkpoint. Web/Celery profiles resolve `AGENT_RUN_LOCK_BACKEND=auto` to Redis; desktop/local profiles resolve it to SQLite. The lease has an owner token, TTL, heartbeat renewal, and conditional release. Do not remove the lease from initial/idempotent execution, resume, or recover paths, and do not make status GETs take the mutating lease.

`app/agent/checkpoint_cleanup.py` runs a bounded maintenance cycle from the FastAPI lifespan. It keeps completed checkpoints for 30 days and incomplete/waiting checkpoints for 90 days by default, coordinates workers with a maintenance lease, and acquires the normal run lease before deletion. Cleanup owns only LangGraph checkpoints/writes, tool receipts, and lifecycle metadata; it must not delete conversations or other business records.

Skills have two execution paths:

1. Python-backed Skills subclass `BaseSkill` and register with `@register_skill`.
2. Codex-style packages live in `app/skills/packages/<slug>/` and declare runtime requirements in `docmind.json`.

`status: ready` means a package has been audited; it does not grant host authority. `requires` is checked against host-controlled capabilities at runtime. Repository access, package scripts, MCP, browser, and App control are disabled by default in the Web profile and require explicit enablement/configuration.

Production adapters are in `app/skills/adapters/`:

- Streamable HTTP MCP with server/tool allowlists and optional environment-sourced bearer tokens.
- Playwright browser sessions with host allowlists and workspace-confined screenshots.
- Authenticated loopback App bridge with application/action allowlists.

All adapters return the normalized `status`, `summary`, `next_actions`, and `artifacts` observation contract. Generic Skill workspace, package resource, and repository roots are validated separately.

## Configuration

All settings are in `app/config.py` via pydantic-settings.

Required application values are `DATABASE_URL`, `SECRET_KEY`, and `OPENAI_API_KEY`. Important v3.1 groups are:

- Retrieval: `DENSE_CANDIDATES`, `KEYWORD_CANDIDATES`, `RRF_K`, `RERANKER_MODE`, and optional `RERANKER_HTTP_*`.
- Tasks/storage: `TASK_EXECUTION_MODE` and optional `QDRANT_PATH` for desktop local persistence.
- Agent durability: `AGENT_CHECKPOINT_PATH`, `AGENT_RUN_LOCK_*`, `AGENT_CHECKPOINT_*_RETENTION_DAYS`, `AGENT_CHECKPOINT_CLEANUP_*`, `AGENT_HIGH_RISK_SKILLS`, and `AGENT_HIGH_RISK_CAPABILITIES`.
- Capabilities: `SKILL_PACKAGE_SCRIPTS_ENABLED`, `SKILL_REPOSITORY_*`, `SKILL_MCP_*`, `SKILL_BROWSER_*`, and `SKILL_APP_*`.
- Desktop: `DOCMIND_ENV_FILE`, `DOCMIND_FRONTEND_DIR`, and `ALEMBIC_CONFIG` are injected by the sidecar host.

## Key Design Decisions

- Qdrant payloads always include `user_id`; vector searches always filter by it.
- SQL keyword queries join documents and apply `user_id`/`document_id` filters before a database-level limit.
- HTTP reranker failure falls back to deterministic local ranking instead of failing document Q&A.
- Host capability checks are authoritative; a copied manifest cannot enable repository, script, browser, MCP, or App access.
- `interrupt()` must stay before high-risk side effects. Nodes that can be replayed after resume must not perform untracked pre-interrupt writes.
- Desktop runtime-owned database/vector/task settings override dotenv values, preventing an old config from reconnecting the installed app to external infrastructure.
- `bcrypt==4.0.1` remains pinned because passlib 1.7.4 is incompatible with bcrypt 5.x.
- `EMBEDDING_DIM` must match the configured embedding model; changing dimensions requires rebuilding the vector collection.
