#!/usr/bin/env bash
# DocMind 一键启动脚本
# 用法: ./start.sh
# 每次运行都会先清理旧进程，再干净启动。

set -euo pipefail

# ── 颜色 ─────────────────────────────────────────────────────────
BLUE='\033[0;34m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info() { echo -e "${BLUE}▶${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELERY_LOG="/tmp/docmind_celery.log"
PG_DSN="postgresql://docmind:docmind123@127.0.0.1:5432/docmind_db"

cd "$DIR"

# 从宿主机真实探测 PG（走 host→colima→container，能发现僵尸 SSH 转发）。
# docker exec psql 只走容器内 socket，探不出这个问题，所以必须用这个。
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

# 杀掉 colima 遗留的僵尸 SSH mux 端口转发，再重启 colima 重建。
# 根因：docker-compose 重建容器后，旧 mux 仍霸占 5432/6379/6333，
# 把连接转给已销毁的旧容器 → 一发数据就 ConnectionReset。
heal_colima_forwards() {
    warn "检测到宿主机无法连通 PG，清理僵尸 SSH 转发并重建 colima..."
    pkill -f "_lima/colima/ssh.sock \[mux\]" 2>/dev/null || true
    sleep 2
    colima restart >/dev/null 2>&1 || die "colima 重启失败"
    sleep 3
    docker start docmind_postgres docmind_redis docmind_qdrant >/dev/null 2>&1 || true
    sleep 5
}

# ── 1. 停止旧进程 ─────────────────────────────────────────────────
info "清理旧进程..."

if pkill -f "uvicorn app.main" 2>/dev/null; then
    warn "已停止旧 uvicorn"
fi
if pkill -f "celery -A app.celery_app" 2>/dev/null; then
    warn "已停止旧 Celery"
    sleep 1  # 等 worker 释放连接
fi

# ── 2. 停止旧容器 ─────────────────────────────────────────────────
info "清理旧容器..."
docker-compose down --remove-orphans 2>/dev/null || true

# ── 3. 确保 colima 在跑 ────────────────────────────────────────────
if ! colima status 2>/dev/null | grep -q "Running"; then
    info "启动 colima..."
    colima start || die "colima 启动失败"
else
    ok "colima 已在运行"
fi

# ── 4. 启动容器 ───────────────────────────────────────────────────
info "启动 PostgreSQL / Redis / Qdrant..."
docker-compose up -d
ok "容器已启动"

# ── 5. 等 PostgreSQL 从宿主机真正可达（含僵尸转发自愈）───────────
info "等待 PostgreSQL 从宿主机可达..."
healed=0
for i in $(seq 1 30); do
    if pg_reachable_from_host; then
        ok "PostgreSQL 宿主机可达（${i}s）"
        break
    fi
    # 8s 还连不上，多半是僵尸 SSH 转发，自愈一次
    if [ "$i" -eq 8 ] && [ "$healed" -eq 0 ]; then
        heal_colima_forwards
        healed=1
    fi
    if [ "$i" -eq 30 ]; then
        die "PostgreSQL 30s 内仍不可达。手动检查: lsof -nP -iTCP:5432 -sTCP:LISTEN"
    fi
    sleep 1
done

# ── 6. 启动 Celery（后台，日志写文件）────────────────────────────
info "启动 Celery worker..."
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    uv run celery -A app.celery_app worker \
    --loglevel=info --pool=solo \
    >"$CELERY_LOG" 2>&1 &
CELERY_PID=$!

# 等 3s 确认进程没有立即崩溃
sleep 3
if ! kill -0 "$CELERY_PID" 2>/dev/null; then
    die "Celery 启动失败。查看日志: tail $CELERY_LOG"
fi
ok "Celery 已启动 (PID=$CELERY_PID)"
info "Celery 日志: tail -f $CELERY_LOG"

# ── 7. 启动 uvicorn（前台）────────────────────────────────────────
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
