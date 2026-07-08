#!/usr/bin/env bash
# DocMind 通用 Docker / Docker Desktop 启动脚本
#
# 适用场景:
# - Linux + Docker Engine
# - macOS + Docker Desktop
# - 其它已可直接运行 docker compose 的环境
#
# 不依赖 colima；如果你使用 macOS + colima，优先用 scripts/start-macos-colima.sh。

set -euo pipefail

BLUE='\033[0;34m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info() { echo -e "${BLUE}▶${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CELERY_LOG="/tmp/docmind_celery.log"
PG_DSN="postgresql://docmind:docmind123@127.0.0.1:5432/docmind_db"

cd "$DIR"

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        die "未找到 docker compose 或 docker-compose。请先安装 Docker Compose。"
    fi
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die "未找到 docker 命令。请先安装并启动 Docker。"
    docker info >/dev/null 2>&1 || die "Docker daemon 不可用。请先启动 Docker Desktop 或 Docker Engine。"
}

pg_reachable_from_host() {
    uv run python - "$PG_DSN" <<'PY' 2>/dev/null
import sys, asyncio, asyncpg

async def main():
    conn = await asyncpg.connect(sys.argv[1], timeout=5)
    await conn.close()

try:
    asyncio.run(main())
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

info "清理旧进程..."
if pkill -f "uvicorn app.main" 2>/dev/null; then
    warn "已停止旧 uvicorn"
fi
if pkill -f "celery -A app.celery_app" 2>/dev/null; then
    warn "已停止旧 Celery"
    sleep 1
fi

require_docker

info "清理旧容器..."
compose down --remove-orphans 2>/dev/null || true

info "启动 PostgreSQL / Redis / Qdrant..."
compose up -d
ok "容器已启动"

info "等待 PostgreSQL 从宿主机可达..."
for i in $(seq 1 30); do
    if pg_reachable_from_host; then
        ok "PostgreSQL 宿主机可达（${i}s）"
        break
    fi
    if [ "$i" -eq 30 ]; then
        die "PostgreSQL 30s 内仍不可达。请检查 5432 端口占用或 Docker 容器日志。"
    fi
    sleep 1
done

info "启动 Celery worker..."
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    uv run celery -A app.celery_app worker \
    --loglevel=info --pool=solo \
    >"$CELERY_LOG" 2>&1 &
CELERY_PID=$!

sleep 3
if ! kill -0 "$CELERY_PID" 2>/dev/null; then
    die "Celery 启动失败。查看日志: tail $CELERY_LOG"
fi
ok "Celery 已启动 (PID=$CELERY_PID)"
info "Celery 日志: tail -f $CELERY_LOG"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  DocMind 已就绪                           ║${NC}"
echo -e "${GREEN}║  http://localhost:8000                    ║${NC}"
echo -e "${GREEN}║                                           ║${NC}"
echo -e "${GREEN}║  Ctrl+C  停止 API                         ║${NC}"
echo -e "${GREEN}║  再次运行此脚本可完整重启                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""

exec uv run uvicorn app.main:app --reload --port 8000
