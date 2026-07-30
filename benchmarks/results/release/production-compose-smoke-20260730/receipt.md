# DocMind Production Compose Smoke Receipt

- Run: `production-compose-smoke-20260730`
- Completed (UTC): `2026-07-30T07:38:45.090948+00:00`
- Source: `feature/agentic-rag-evalops-evidence` @ `b4b1b654d231e3d5d9574af0590b11ca96cbfbe8` (dirty working tree: `true`)
- Release source fingerprint: `d78636ddb25d47d02c7fe5e34c15f6230a4fac69d6c95a811567361c70084cec`
- Runtime: `local Colima Docker` / `aarch64/linux`
- Image: `sha256:6443c241164af4af72e10f80a60ef837f6f380c8fe44e89633cab733efc8104b`
- Result: **16/16 checks passed**

| Check | Result | Evidence |
|---|---:|---|
| `migration_completed` | PASS | status=exited; exit_code=0 |
| `api_healthy` | PASS | status=running; health=healthy |
| `api_non_root` | PASS | uid=10001 |
| `api_readonly_rootfs` | PASS | ReadonlyRootfs=true |
| `api_write_boundaries` | PASS | app_root=read-only; data=writable; tmp=writable |
| `worker_healthy` | PASS | status=running; health=healthy |
| `worker_non_root` | PASS | uid=10001 |
| `worker_readonly_rootfs` | PASS | ReadonlyRootfs=true |
| `worker_write_boundaries` | PASS | app_root=read-only; data=writable; tmp=writable |
| `postgres_not_host_published` | PASS | host_ports=none |
| `redis_not_host_published` | PASS | host_ports=none |
| `qdrant_not_host_published` | PASS | host_ports=none |
| `health_endpoint` | PASS | http_status=200; payload_status=ok |
| `product_ui_served` | PASS | http_status=200; agent_workspace_marker=true |
| `registration_flow` | PASS | http_status=201; id_present=True; password_fields_absent=True |
| `login_flow` | PASS | http_status=200; jwt_present=True; token_type=bearer |

## Evidence boundary

- Proves migration gating, API/worker health, non-root execution, read-only application roots, private infrastructure ports, frontend serving, registration, and JWT login in the local Compose topology.
- Generated credentials and JWTs were not printed or persisted.
- Does not prove TLS/domain configuration, secret-manager integration, backup/restore, multi-host failover, or external model-provider availability.
