#!/usr/bin/env bash
# DocMind macOS launcher.
# Requires: uv, Docker, Docker Compose, Colima.

set -euo pipefail

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${BLUE}▶${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CELERY_LOG="${TMPDIR:-/tmp}/docmind_celery.log"
PG_DSN="postgresql://docmind:docmind123@127.0.0.1:5432/docmind_db"
cd "$ROOT_DIR"

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        die "未找到 Docker Compose。请安装 Docker Desktop/Compose plugin 或 docker-compose。"
    fi
}

ensure_prerequisites() {
    command -v uv >/dev/null 2>&1 || die "未找到 uv。安装：https://docs.astral.sh/uv/"
    command -v docker >/dev/null 2>&1 || die "未找到 docker。请先安装 Docker。"
    command -v colima >/dev/null 2>&1 || die "未找到 colima。macOS 推荐：brew install colima docker docker-compose"
    if [ ! -f .env ]; then
        cp .env.example .env
        warn "未发现 .env，已从 .env.example 创建。请填入真实 OPENAI_API_KEY 后再使用 AI 功能。"
    fi
}

ensure_python_env() {
    if [ ! -d .venv ]; then
        info "创建 Python 虚拟环境..."
        uv venv --python 3.12 .venv
    fi
    info "安装/同步 Python 依赖..."
    uv pip install -r requirements.txt >/dev/null
    ok "Python 依赖已就绪"
}

pg_reachable_from_host() {
    .venv/bin/python - "$PG_DSN" <<'PY' 2>/dev/null
import sys, asyncio, asyncpg
async def main():
    c = await asyncpg.connect(sys.argv[1], timeout=5)
    await c.close()
try:
    asyncio.run(main()); sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

heal_colima_forwards() {
    warn "检测到宿主机无法连通 PG，清理僵尸 SSH 转发并重建 colima..."
    pkill -f "_lima/colima/ssh.sock \[mux\]" 2>/dev/null || true
    sleep 2
    colima restart >/dev/null 2>&1 || die "colima 重启失败"
    sleep 3
    docker start docmind_postgres docmind_redis docmind_qdrant >/dev/null 2>&1 || true
    sleep 5
}

info "DocMind macOS 启动流程"
ensure_prerequisites
ensure_python_env

info "清理旧进程..."
pkill -f "uvicorn app.main" 2>/dev/null || true
pkill -f "celery -A app.celery_app" 2>/dev/null || true
sleep 1

info "清理旧容器..."
compose down --remove-orphans 2>/dev/null || true

if ! colima status 2>/dev/null | grep -q "Running"; then
    info "启动 colima..."
    colima start || die "colima 启动失败"
else
    ok "colima 已在运行"
fi

info "启动 PostgreSQL / Redis / Qdrant..."
compose up -d
ok "容器已启动"

info "等待 PostgreSQL 从宿主机可达..."
healed=0
for i in $(seq 1 30); do
    if pg_reachable_from_host; then
        ok "PostgreSQL 宿主机可达（${i}s）"
        break
    fi
    if [ "$i" -eq 8 ] && [ "$healed" -eq 0 ]; then
        heal_colima_forwards
        healed=1
    fi
    if [ "$i" -eq 30 ]; then
        die "PostgreSQL 30s 内仍不可达。手动检查: lsof -nP -iTCP:5432 -sTCP:LISTEN"
    fi
    sleep 1
done

info "启动 Celery worker..."
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    uv run celery -A app.celery_app worker --loglevel=info --pool=solo \
    >"$CELERY_LOG" 2>&1 &
CELERY_PID=$!
sleep 3
if ! kill -0 "$CELERY_PID" 2>/dev/null; then
    die "Celery 启动失败。查看日志: tail $CELERY_LOG"
fi
ok "Celery 已启动 (PID=$CELERY_PID)"
info "Celery 日志: tail -f $CELERY_LOG"

cat <<EOF

${GREEN}╔══════════════════════════════════════════╗${NC}
${GREEN}║  DocMind 已就绪                         ║${NC}
${GREEN}║  http://localhost:8000                  ║${NC}
${GREEN}║  Ctrl+C 停止 API；再次运行可完整重启     ║${NC}
${GREEN}╚══════════════════════════════════════════╝${NC}

EOF

exec uv run uvicorn app.main:app --reload --port 8000
