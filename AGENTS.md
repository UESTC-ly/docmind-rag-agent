# Repository Guidelines

## Project Structure & Module Organization

DocMind combines a Python 3.12 FastAPI/LangGraph backend, a browser frontend, and a Tauri desktop application. Backend code lives in `app/`: keep HTTP handlers in `routers/`, Pydantic contracts in `schemas/`, ORM types in `models/`, business logic in `services/`, durable Agent orchestration in `agent/`, and background work in `tasks/`. Database revisions belong in `alembic/`; backend tests live in `tests/`.

The dependency-free web UI is under `frontend/` (`js/`, `styles/`, and Playwright specs in `e2e/`). Native packaging, Rust host code, and the frozen Python sidecar live in `desktop/`. Treat `frontend/playwright-report/`, `frontend/test-results/`, `desktop/dist/`, and `.omx/` as generated state. `learn/` is private local material and must not be staged or released.

## Build, Test, and Development Commands

- `uv venv --python 3.12 && uv pip install -r requirements.txt` creates the backend environment.
- `./start.sh` starts infrastructure, applies Alembic migrations, launches Celery, and serves FastAPI.
- `uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=90` runs the required Python suite and coverage gate.
- `uv run ruff check app tests --select E9,F63,F7,F82` catches runtime-breaking Python errors.
- `cd frontend && npm ci && npm run test:e2e:ci` runs Chromium mock and visual tests; use `npm run test:e2e:fullstack` for the real FastAPI path.
- `cd desktop && npm ci && npm run test:sidecar && npm run test:rust` validates the packaged sidecar and Rust host. `npm run build` creates native artifacts.

## Coding Style & Naming Conventions

Use four-space Python indentation, type hints at API/model boundaries, `snake_case` functions/modules, and `PascalCase` classes. Keep async database sessions in request paths and sync sessions in worker paths. JavaScript uses ES modules, two-space indentation, and descriptive `camelCase` names. Format Rust with `cargo fmt`; Clippy warnings are errors. Reuse existing service and capability-gate patterns rather than adding parallel abstractions.

## Testing Guidelines

Name pytest files and functions `test_*.py`; async tests use pytest-asyncio automatically. Mock external LLM, vector, browser, MCP, and App boundaries. Name Playwright cases `*.spec.js`, update visual baselines only for intentional UI changes, and add regression coverage for bug fixes.

## Commit & Pull Request Guidelines

History uses concise, intent-first imperative subjects, followed by rationale and Lore trailers such as `Constraint:`, `Rejected:`, `Tested:`, and `Not-tested:`. PRs should describe behavior and risk, link issues, call out migrations or configuration changes, list exact verification commands, and include screenshots for UI changes. Never commit `.env`, credentials, generated reports, `.omx/`, or `learn/`.
