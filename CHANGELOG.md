# Changelog

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
- 189 pytest tests with mocked external services and CI coverage gate.

### Safety notes

- `.env` is ignored and must not be committed.
- Generic skill file I/O is scoped to `skill_workspaces/user_<id>/<skill_slug>/`.
- Generic skill shell execution is disabled by default and allowlist-gated when enabled.
