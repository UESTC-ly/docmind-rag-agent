# DocMind Desktop 3.1 release runtime

The installed Tauri application is self-contained. It starts one signed
`docmind-sidecar` executable that embeds Python, FastAPI and the backend Python
packages. Desktop storage is SQLite plus Qdrant local mode, and background jobs
run through DocMind's in-process task executor. Docker, PostgreSQL, Redis,
Qdrant Server, `uv` and a system Python installation are not consulted at
runtime.

The AI/embedding provider remains an external service by design. On first
launch, edit the generated `desktop.env` in the platform application-data
directory with the provider credentials, then restart DocMind.

## Runtime layout

The Tauri host creates this per-user, uninstall-safe layout:

```text
<app-config>/
  desktop.env
  data/
    docmind.db
    agent-checkpoints.sqlite3
    qdrant/
    uploads/
    skill_workspaces/
    secrets/jwt-signing-key
  logs/sidecar.log
```

The database, vector index, uploads and signing key never live in the signed
application bundle. Upgrades therefore replace immutable application files and
retain user data. For rollback, quit DocMind, back up the application-data
directory, install the previous signed package, and restore the backup only if
its schema is compatible with that release.

## Native build

Desktop bundles must be built on a native runner for every OS/architecture.
The runner needs Node 22, Rust 1.77.2+ and Python 3.12+; end users need none of
these toolchains. The macOS bundle declares macOS 11 as its minimum because the
embedded CPython native runtime also targets macOS 11; the host metadata must
never advertise an older version than the sidecar can start on.

```bash
cd desktop
npm ci
npm run build
```

`npm run build` performs the following release gates:

1. creates a private build-only virtual environment;
2. installs the pinned desktop-only backend set and PyInstaller build
   dependency (no PostgreSQL drivers or test tooling);
3. builds a one-file sidecar for Rust's native target triple;
4. executes the frozen binary's `check` command and proves that SQLite,
   Qdrant local, and local task execution are selected;
5. executes one signed package Python helper, rejects an arbitrary script path,
   then starts the frozen API and waits for `/health` after first-run migration;
6. copies the verified target-suffixed binary into Tauri `externalBin`;
7. bundles the desktop API client and bridge, then executes REST and SSE calls
   against the origin returned by a mocked Tauri `backend_origin` command;
8. merges `src-tauri/tauri.sidecar.conf.json` and builds the frontend and native
   installer. Keeping this release-only overlay separate lets Rust unit tests run
   without checking a generated binary into source control.

Set `DOCMIND_BUILD_PYTHON=/absolute/path/to/python` when the release runner has
multiple Python installations. Cross-compiling the sidecar is intentionally
rejected because PyInstaller freezes native extensions for its host platform.

The checked-in macOS configuration uses Tauri's documented ad-hoc identity
(`-`) so Apple Silicon bundles have a complete local signature even when no
Apple certificate is present. Ad-hoc signing does not establish publisher
identity or satisfy Gatekeeper notarization: a public release runner must still
inject a Developer ID identity and Apple notarization credentials, then verify
and staple the resulting package before distribution.

The hardened runtime remains enabled. Its single explicit entitlement disables
library validation for the managed sidecar because PyInstaller one-file mode
extracts its already signed Python/native libraries at process startup. The API
still binds only to loopback, and no arbitrary executable path is accepted.

`.github/workflows/desktop-release.yml` runs the same frozen-binary checks and
installer build on a native macOS runner. Pull requests that change a desktop
runtime input exercise that build; manual runs retain the macOS bundle as a CI
verification artifact. Windows and Linux packages are outside the v3.1 acceptance
scope. These artifacts are intentionally not called releases: a protected publishing
job must inject the platform signing/notarization credentials and verify the
native signature before distributing the same build.

Useful verification commands:

```bash
npm run build:sidecar
npm run test:sidecar
npm run test:rust
node scripts/build-sidecar.mjs --print-target
```

Before publishing, install the artifact on a clean machine/VM without Docker,
`uv`, Python, PostgreSQL, Redis or Qdrant Server. Verify first launch, document
upload and completion after an application restart, retrieval after restart,
and preservation of the application-data directory across an upgrade.

## Security and operations

- The API binds only to `127.0.0.1:8000`; an already occupied port fails closed.
- A random JWT key is created with owner-only permissions and reused across
  upgrades. It is never printed by sidecar self-checks.
- Runtime-owned storage/task settings override dotenv values so a stale desktop
  config cannot silently reconnect to network infrastructure.
- Desktop forces `AGENT_RUN_LOCK_BACKEND=sqlite`, so concurrent mutations of the
  same Agent `run_id` share a renewable lease in `agent-checkpoints.sqlite3`
  without requiring Redis. Web/celery deployments use Redis instead.
- FastAPI lifespan runs the same bounded checkpoint-retention job as Web. A
  maintenance lease prevents overlapping cleanup passes, and per-run leases
  prevent deletion while a run is being advanced or resumed.
- The frontend obtains the loopback API origin from the Tauri host only after
  readiness. The same built-code smoke verifies both REST and authenticated SSE;
  browser deployments keep relative URLs.
- The sidecar watches the Tauri parent PID. During normal shutdown Tauri first
  sends SIGTERM and waits for FastAPI lifespan cleanup; only a child that does
  not exit within the bounded timeout is force-killed.
- Startup failures include the sidecar exit status and point to
  `logs/sidecar.log`.
