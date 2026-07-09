# Changelog

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
