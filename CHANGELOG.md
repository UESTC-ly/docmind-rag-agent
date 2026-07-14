# Changelog

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
