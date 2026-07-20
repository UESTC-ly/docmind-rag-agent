# Changelog

## v3.1.0 - 2026-07-20

### Cross-worker Agent run coordination

- Added an exclusive, renewable lease around every mutating Agent run operation. Web/Celery deployments use Redis so separate processes or hosts cannot execute the same `run_id` concurrently; the self-contained desktop/local profile uses an owner-token SQLite lease beside the checkpoint file.
- Leases are non-blocking, expire after a bounded TTL, renew on a heartbeat, and release only when the owner token still matches. A stale worker therefore cannot delete a newer worker's lease.
- `POST /agent/chat`, `POST /agent/resume`, and `POST /agent/runs/{run_id}/recover` now return `423 Locked` with `Retry-After` when another worker owns the run, and fail closed with `503 Service Unavailable` when the configured lock backend is unavailable or ownership is lost.
- Kept `GET /agent/runs/{run_id}` read-only and available during execution. The frontend retains its client-generated run ID after transient `423`, `429`, network, or server failures so it can inspect or recover the original checkpoint instead of creating duplicate work.

### Checkpoint lifecycle maintenance

- Added a DocMind-owned run lifecycle index next to the LangGraph SQLite tables without duplicating graph messages or private state.
- Added bounded periodic cleanup with separate defaults for completed runs (30 days) and incomplete/waiting/recoverable runs (90 days), an hourly interval, and a 100-run batch limit. Cleanup removes LangGraph checkpoints/writes, tool receipts, run metadata, and expired local lease rows; it does not delete conversations, messages, documents, or generated business data.
- Added a maintenance lease so multiple API workers do not repeat the same cleanup cycle, and reacquires each run's normal execution lease before deletion. Actively executing runs are skipped rather than interrupted.
- Added safe v3.0 upgrade backfill: pre-v3.1 checkpoints receive a lifecycle entry whose retention clock starts at first v3.1 maintenance, so an upgrade cannot immediately purge older recoverable state.

### Runtime, desktop, and verification

- Updated FastAPI, frontend, desktop, Tauri, Rust, MCP client, and sidecar version declarations to `3.1.0`.
- Desktop hosts explicitly force the SQLite lease backend and keep the lease/checkpoint database in the per-user data directory; installed desktop users still do not need Redis.
- Added unit/API coverage for lease exclusion, expiry takeover, heartbeat ownership, stale-owner release protection, Redis failures, HTTP conflict mapping, retention, maintenance coordination, active-run skipping, and v3.0 backfill. Added a Playwright regression proving a transient cross-worker conflict does not discard the pending run ID.

Redis coordinates mutation but does not turn the SQLite LangGraph saver into a cross-host state store. Multi-host recovery still requires sticky routing/shared storage or a future production shared checkpointer. Generic Skill subgraphs and full Multi-Agent migration remain intentionally deferred. See `doc/10-v3.1.0-分布式运行锁与生命周期维护.md`.

## v3.0.0 - 2026-07-20

### Durable LangGraph Agent

- Replaced the handwritten outer ReAct loop with an explicit LangGraph state graph containing `supervisor`, `select_tool`, `approval_gate`, and `execute_tool` nodes while preserving selected-skill and document-first-tool enforcement.
- Added SQLite-backed LangGraph checkpoints keyed by a random run/thread ID. Web and desktop requests can inspect a run and resume it after application or process restart with the same checkpoint.
- Added `POST /agent/resume` and `GET /agent/runs/{run_id}`. Checkpoint ownership is verified against the authenticated user before status or resume data is returned.
- Added client-generated idempotent run IDs and `POST /agent/runs/{run_id}/recover`. Per-tool execution receipts reuse completed results after checkpoint-write crashes; an in-flight result with unknown side effects interrupts for an explicit retry decision instead of running twice silently. Completed Assistant history is also repaired idempotently when a graph commit survives but its business transaction does not.
- Added human approval before any side effect from an explicitly high-risk outer Skill. Approvers may approve, reject, comment, or replace the proposed arguments; rejection is returned to the Supervisor as a tool observation.
- Generic Skills that declare package-script, MCP, browser, App, or repository capabilities are gated before entering the package. Their existing internal ReAct runner is intentionally not represented as a subgraph yet, so this release does not claim per-inner-action checkpointing.
- Added frontend approval/rejection controls and deterministic Playwright coverage for the interrupt/resume flow.

### Runtime and compatibility

- Pinned `langgraph==1.2.9` and `langgraph-checkpoint-sqlite==3.1.0` in both Web and frozen desktop runtimes, and included LangGraph modules/metadata in the PyInstaller sidecar.
- Desktop checkpoints now live beside other user data rather than inside the application bundle. No application-database schema migration is required for v3.0.0.
- Canonicalized the macOS script-sandbox probe paths so `/var` and `/tmp` symlink resolution cannot incorrectly disable the frozen sidecar's supported sandbox backend.
- Kept existing v2.2 desktop SQLite files writable by registering the PostgreSQL-compatible `now()` timestamp function on both request and background-task connections.
- Preserved the existing `/agent/chat` fields and added `run_id`, `thread_id`, `status`, and optional `approval` metadata.
- Deferred Generic Skill subgraph migration and full Multi-Agent decomposition to a later release so those changes can receive their own state, idempotency, session-lifecycle, and evaluation design.

See `doc/09-v3.0.0-LangGraph实施与发布.md` for architecture, API, recovery, and rollout guidance.

## v2.2.0 - 2026-07-13

### Agent capabilities

- Added production Streamable HTTP MCP, Playwright browser, and authenticated loopback App bridge adapters with explicit enable switches, allowlists, timeouts, and structured observations.
- Closed Generic Skill authorization at both discovery and execution time: every tool now requires the intersection of host-granted and package-declared capabilities, so a forged tool call or edited manifest cannot self-authorize.
- Audited all copied packages. Nine are ready (`codex-note`, `jupyter-notebook`, `openai-docs`, `playwright`, `presentation`, `screenshot`, `security-best-practices`, `security-threat-model`, and `weekly-report`); four remain honestly blocked because their required action surface is not implemented (`gh-fix-ci`, `pdf`, `playwright-interactive`, and `security-ownership-map`).
- Added confined package asset/script tools and read-only repository/Git tools without widening the default Web deployment's permissions. Arbitrary Generic Skill shell access is unavailable; audited package scripts run only when the macOS `sandbox-exec` write-confinement probe succeeds, reject escaping arguments/symlinks, and register only workspace-contained artifacts.

### Retrieval and evaluation

- Moved evaluation runs off the request path: `POST /eval/runs` now returns `202 Accepted`, Web deployments execute through Celery, desktop deployments use the local task executor, and the frontend polls run state.
- Added CAS lease claims, heartbeats, token-guarded terminal writes, a per-run/sample result uniqueness constraint, retry-to-pending semantics, Celery late acknowledgement/worker-loss redelivery, and desktop startup recovery for interrupted local evaluations.
- Unified online chat, Skills, and evaluation on the same dense + database keyword candidate pipeline, scored RRF fusion, and bounded second-stage reranking.
- Added a deterministic local reranker and an optional HTTP cross-encoder mode that falls back to local ranking when the provider is unavailable.
- Moved keyword recall into the database with PostgreSQL full-text ranking and a bounded SQLite SQL compatibility query; both paths apply ownership/document filters and `LIMIT` before returning rows.
- Persisted retrieval mode, reranker mode, candidate provenance, scores, and final ordering in evaluation details.

### Schema and quality gates

- Added Alembic v2.1 baseline and v2.2 migrations, including evaluation retrieval trace fields, the independently recoverable PostgreSQL full-text GIN index, and evaluation leases/result uniqueness. Runtime startup no longer creates tables with SQLAlchemy metadata.
- Hardened concurrent-index recovery to accept only the exact non-partial, single-expression GIN definition; invalid or drifted indexes are rebuilt or rejected instead of being stamped silently.
- Added Playwright Page Object-based E2E coverage for registration/automatic login, independent login, upload, streaming chat, Skills/artifacts, asynchronous evaluation polling, and Chromium visual baselines, including a negative self-test that proves visual drift is rejected. A separate full-stack route launches the real FastAPI application on temporary SQLite/Qdrant-local storage and verifies real JWT-protected APIs and login persistence without mocks.
- Split macOS CI into Python lint/typecheck/tests, JavaScript syntax checks, Rust fmt/clippy/tests, and a dedicated Chromium E2E/visual workflow with trace, screenshot, video, and HTML-report artifacts on failure. Linux/Windows baselines are outside v2.2 acceptance.

### Self-contained desktop runtime

- Replaced the external Docker/uv/Python desktop bootstrap with a PyInstaller-frozen FastAPI sidecar bundled by Tauri.
- Desktop production mode now uses SQLite, Qdrant local persistence, the in-process task executor, and Alembic migrations in the per-user application-data directory.
- End users no longer need Docker, PostgreSQL, Redis, Qdrant Server, `uv`, or a system Python installation. AI/embedding provider credentials and network access remain external by design.
- Restored complete macOS ad-hoc bundle signing, kept hardened runtime enabled, and granted only the library-validation entitlement required by the managed PyInstaller sidecar; CI verifies both the bundle signature and DMG checksum.
- Tauri now requests graceful sidecar termination before a bounded forced shutdown, including startup failures; the release workflow smoke-tests the post-sign bundled sidecar before accepting the app artifact.

See `doc/08-v2.2.0发布说明.md` for upgrade and deployment guidance.

## v2.1.0 - 2026-07-11

- Prevent document-scoped Agent requests and explicitly selected skill cards from bypassing tool execution.
- Reuse hybrid vector + keyword RRF retrieval in knowledge search, reports, weekly reports, presentations, and document-grounded generic skills.
- Guarantee downloadable artifacts for reports, mindmaps, relation graphs, and generic-skill text fallbacks.
- Quarantine unreviewed or unsupported third-party skill packages with `docmind.json` runtime compatibility metadata.

## v2.0.1 - 2026-07-10

- Configured Tauri ad-hoc signing for macOS bundles so Apple Silicon machines do not classify downloaded GitHub Release builds as damaged.
- Added the Homebrew and system executable paths to Finder-launched desktop processes so Docker, Colima, and uv are discovered outside a terminal session.
- A Developer ID certificate and Apple notarization remain necessary to remove macOS's unidentified-developer warning completely.

## v2.0.0 - 2026-07-10

### Desktop application

- Added a Tauri 2 desktop shell that reuses the existing FastAPI API and single-page frontend.
- Added desktop-owned lifecycle management for Docker Compose, an isolated Python environment, Celery, and FastAPI.
- Added native file pickers for uploads, native save dialogs for generated artifacts, native delete confirmation, and desktop notifications for completed parsing.
- Kept browser mode and root cross-platform startup scripts intact.
- Kept user configuration, uploads, Skill workspaces, and logs outside the application bundle.

### Compatibility

- Pinned the Rust Tauri dependency graph to versions compatible with Rust 1.87.
- Preserved immutable `v1.0.0` and `v1.1.0` Git tags as rollback points.

## v1.0.0 - 2026-07-09

DocMind first public release.

### Highlights

- Agentic RAG document Q&A with FastAPI, PostgreSQL, Qdrant, Redis, and Celery.
- ReAct-style Agent orchestration with OpenAI Function Calling.
- Python-backed Skills: knowledge-base search, mindmap, relation graph, report, weekly report, presentation, and web search.
- Codex-style generic Skills packages via `SKILL.md`, `templates/`, `references/`, `scripts/`, and `assets/`.
- Downloadable artifacts for Markdown weekly reports, PPTX presentations, and generic skill outputs.
- Native single-page frontend served by FastAPI.
- RAG evaluation loop with retrieval metrics and LLM-as-judge generation metrics.
- Cross-platform local launchers for macOS, Linux, and Windows.
- Automated pytest coverage with mocked external services and a CI coverage gate.

### Safety notes

- `.env` is ignored and must not be committed.
- Generic skill file I/O is scoped to `skill_workspaces/user_<id>/<skill_slug>/`.
- Generic skill shell execution is disabled by default and allowlist-gated when enabled.
