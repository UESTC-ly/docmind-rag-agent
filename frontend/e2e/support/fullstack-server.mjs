import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const supportDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(supportDir, "../../..");
const runtimeRoot = mkdtempSync(join(tmpdir(), "docmind-fullstack-e2e-"));
const configuredPython = process.env.DOCMIND_E2E_PYTHON || ".venv/bin/python";
const python = isAbsolute(configuredPython)
  ? configuredPython
  : resolve(repoRoot, configuredPython);
const port = Number(process.env.DOCMIND_E2E_FULLSTACK_PORT || 4180);

if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  rmSync(runtimeRoot, { recursive: true, force: true });
  throw new Error(`Invalid DOCMIND_E2E_FULLSTACK_PORT: ${port}`);
}
if (!existsSync(python)) {
  rmSync(runtimeRoot, { recursive: true, force: true });
  throw new Error(
    `DocMind E2E Python was not found at ${python}. Create the repository .venv first.`,
  );
}

const databasePath = join(runtimeRoot, "docmind.sqlite3");
const environment = {
  ...process.env,
  DOCMIND_ENV_FILE: join(runtimeRoot, "isolated.env"),
  DATABASE_URL: `sqlite+aiosqlite:///${databasePath}`,
  SECRET_KEY: "docmind-fullstack-e2e-secret-not-for-production",
  OPENAI_API_KEY: "sk-docmind-fullstack-e2e-placeholder",
  TASK_EXECUTION_MODE: "local",
  LOCAL_TASK_WORKERS: "1",
  QDRANT_PATH: join(runtimeRoot, "qdrant"),
  UPLOAD_DIR: join(runtimeRoot, "uploads"),
  SKILL_WORKSPACE_DIR: join(runtimeRoot, "skill-workspaces"),
  AGENT_CHECKPOINT_PATH: join(runtimeRoot, "agent-checkpoints.sqlite3"),
  DOCMIND_DESKTOP: "false",
  DOCMIND_FRONTEND_DIR: resolve(repoRoot, "frontend"),
  ALEMBIC_CONFIG: resolve(repoRoot, "alembic.ini"),
  RERANKER_MODE: "off",
  SKILL_SHELL_ENABLED: "false",
  SKILL_PACKAGE_SCRIPTS_ENABLED: "false",
  SKILL_REPOSITORY_ENABLED: "false",
  SKILL_MCP_ENABLED: "false",
  SKILL_BROWSER_ENABLED: "false",
  SKILL_APP_ENABLED: "false",
  LOG_LEVEL: "WARNING",
  LOG_JSON: "false",
  PYTHONUNBUFFERED: "1",
};

const child = spawn(
  python,
  [
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--log-level",
    "warning",
  ],
  {
    cwd: repoRoot,
    env: environment,
    stdio: "inherit",
  },
);

process.stdout.write(`DocMind full-stack E2E: http://127.0.0.1:${port}\n`);

let stopping = false;
let forceTimer;

function stopBackend() {
  if (stopping) return;
  stopping = true;
  child.kill("SIGTERM");
  forceTimer = setTimeout(() => child.kill("SIGKILL"), 10_000);
  forceTimer.unref();
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, stopBackend);
}

child.once("error", (error) => {
  rmSync(runtimeRoot, { recursive: true, force: true });
  process.stderr.write(`Unable to start DocMind full-stack E2E backend: ${error.message}\n`);
  process.exit(1);
});

child.once("exit", (code, signal) => {
  if (forceTimer) clearTimeout(forceTimer);
  rmSync(runtimeRoot, { recursive: true, force: true });
  if (stopping) process.exit(0);
  process.stderr.write(
    `DocMind full-stack E2E backend exited unexpectedly (${signal || code}).\n`,
  );
  process.exit(code && code > 0 ? code : 1);
});
