"""Static release contracts for the self-hosted production stack."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_runtime_image_is_minimal_non_root_and_health_checked():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.12-slim@sha256:")
    assert "COPY . ." not in dockerfile
    assert "USER docmind" in dockerfile
    assert dockerfile.index("USER docmind") < dockerfile.index("ENTRYPOINT")
    assert "HEALTHCHECK" in dockerfile
    assert "tini" in dockerfile
    assert "/var/lib/docmind" in dockerfile


def test_runtime_pins_container_compatible_cryptography_build():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    sidecar_requirements = (
        ROOT / "desktop" / "sidecar" / "requirements-runtime.txt"
    ).read_text(encoding="utf-8")

    assert "cryptography==46.0.3" in requirements
    assert "cryptography==46.0.3" in sidecar_requirements


def test_web_image_context_excludes_secrets_generated_state_and_desktop():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert ".env.*" in dockerignore
    assert ".git" in dockerignore
    assert ".venv" in dockerignore
    assert "learn" in dockerignore
    assert "benchmarks/results" in dockerignore
    assert "\ndesktop\n" in f"\n{dockerignore}"


def test_production_compose_is_private_pinned_and_migration_gated():
    compose = (ROOT / "docker-compose.production.yml").read_text(encoding="utf-8")

    assert compose.count("ports:") == 1
    assert "internal: true" in compose
    assert "provider-egress" in compose
    assert compose.count("- provider-egress") == 1
    assert "read_only: true" in compose
    assert "service_completed_successfully" in compose
    assert "AGENT_RUN_LOCK_BACKEND" not in compose
    assert "@sha256:" in compose
    assert ":latest" not in compose
    assert "${DOCMIND_ENV_FILE:-.env.production}" in compose
    assert (
        "celery -A app.celery_app inspect ping -d "
        "celery@$$HOSTNAME --timeout=5 | grep -q pong"
    ) in compose


def test_production_example_fails_safe_for_external_capabilities():
    env = (ROOT / ".env.production.example").read_text(encoding="utf-8")

    assert "SKILL_PACKAGE_SCRIPTS_ENABLED=false" in env
    assert "SKILL_REPOSITORY_ENABLED=false" in env
    assert "SKILL_MCP_ENABLED=false" in env
    assert "SKILL_BROWSER_ENABLED=false" in env
    assert "SKILL_APP_ENABLED=false" in env
    assert "AGENT_RUN_LOCK_BACKEND=redis" in env
