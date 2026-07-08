#!/usr/bin/env bash
# DocMind 跨平台启动入口
#
# 用法:
#   ./start.sh                    # 自动选择启动后端
#   ./start.sh --macos-colima      # 强制使用已验证的 macOS + colima 脚本
#   ./start.sh --generic-docker    # 强制使用通用 Docker / Docker Desktop 脚本
#   ./start.sh --help
#
# 设计目标:
# - 根脚本只做 fresh clone 前置检查、依赖安装和平台分发。
# - 已验证的 macOS + colima 成功版本保留在 scripts/start-macos-colima.sh。
# - 新增跨平台逻辑放在 scripts/start-generic-docker.sh，避免覆盖成功版本。

set -euo pipefail

BLUE='\033[0;34m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info() { echo -e "${BLUE}▶${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

usage() {
    cat <<'EOF'
DocMind 启动脚本

用法:
  ./start.sh                 自动选择启动方式
  ./start.sh --macos-colima   使用 macOS + colima 稳定脚本
  ./start.sh --generic-docker 使用通用 Docker / Docker Desktop 脚本
  ./start.sh --skip-env-check 跳过 .env 占位符检查（高级用法）
  ./start.sh --help

首次 clone 后:
  1. 安装 uv、Docker（macOS + colima 或 Docker Desktop / Linux Docker）
  2. 运行 ./start.sh
  3. 若脚本创建了 .env，请填入真实 OPENAI_API_KEY 等配置后重新运行
EOF
}

MODE="auto"
SKIP_ENV_CHECK=0
for arg in "$@"; do
    case "$arg" in
        --macos-colima)
            MODE="macos-colima"
            ;;
        --generic-docker)
            MODE="generic-docker"
            ;;
        --skip-env-check)
            SKIP_ENV_CHECK=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数: $arg。运行 ./start.sh --help 查看用法"
            ;;
    esac
done

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1。请先安装后重试。"
}

ensure_env_file() {
    if [ -f .env ]; then
        return
    fi
    if [ ! -f .env.example ]; then
        die "缺少 .env，且未找到 .env.example。请创建 .env 后重试。"
    fi

    cp .env.example .env
    warn "已从 .env.example 复制生成 .env"
    cat <<'EOF'

请先编辑 .env，至少填入：
  - SECRET_KEY：一串长随机字符串
  - OPENAI_API_KEY：真实可用的对话模型 key
  - OPENAI_BASE_URL：你的 OpenAI 兼容服务地址；若使用官方服务可留空

如果不单独使用 embedding 服务，请把 EMBEDDING_API_KEY / EMBEDDING_BASE_URL 留空，
让系统复用 OPENAI_API_KEY / OPENAI_BASE_URL。

填好后重新运行：
  ./start.sh
EOF
    exit 1
}

env_value() {
    local key="$1"
    grep -E "^${key}=" .env 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//' | xargs
}

validate_env_placeholders() {
    if [ "$SKIP_ENV_CHECK" -eq 1 ]; then
        warn "已跳过 .env 占位符检查"
        return
    fi

    local secret_key openai_key openai_base embedding_key
    secret_key="$(env_value SECRET_KEY || true)"
    openai_key="$(env_value OPENAI_API_KEY || true)"
    openai_base="$(env_value OPENAI_BASE_URL || true)"
    embedding_key="$(env_value EMBEDDING_API_KEY || true)"

    [ -n "$secret_key" ] || die ".env 缺少 SECRET_KEY"
    [ "$secret_key" != "change-this-to-a-long-random-string" ] || die ".env 中 SECRET_KEY 仍是示例值，请改成随机字符串"

    [ -n "$openai_key" ] || die ".env 缺少 OPENAI_API_KEY"
    [ "$openai_key" != "sk-your-chat-api-key" ] || die ".env 中 OPENAI_API_KEY 仍是示例值，请填真实 key"

    if [ "$openai_base" = "https://your-relay-or-official/v1" ]; then
        die ".env 中 OPENAI_BASE_URL 仍是示例值；请改成真实地址，或使用官方服务时留空"
    fi

    if [ "$embedding_key" = "sk-your-embedding-api-key" ]; then
        die ".env 中 EMBEDDING_API_KEY 仍是示例值；请填真实 embedding key，或留空以复用 OPENAI_API_KEY"
    fi
}

ensure_python_env() {
    require_cmd uv

    if [ ! -d .venv ] || [ ! -x .venv/bin/python ]; then
        info "创建 Python 虚拟环境 .venv..."
        uv venv
    fi

    if [ ! -f .venv/.docmind-deps-installed ] || [ requirements.txt -nt .venv/.docmind-deps-installed ]; then
        info "安装 / 更新 Python 依赖..."
        uv pip install -r requirements.txt
        touch .venv/.docmind-deps-installed
    else
        ok "Python 依赖已就绪"
    fi
}

select_mode() {
    if [ "$MODE" != "auto" ]; then
        return
    fi

    # macOS + colima 是当前已验证成功路径；检测到 colima 时优先使用它。
    if [ "$(uname -s)" = "Darwin" ] && command -v colima >/dev/null 2>&1; then
        MODE="macos-colima"
    else
        MODE="generic-docker"
    fi
}

ensure_env_file
validate_env_placeholders
ensure_python_env
select_mode

case "$MODE" in
    macos-colima)
        info "启动方式: macOS + colima 稳定版"
        exec "$DIR/scripts/start-macos-colima.sh"
        ;;
    generic-docker)
        info "启动方式: 通用 Docker / Docker Desktop"
        exec "$DIR/scripts/start-generic-docker.sh"
        ;;
    *)
        die "内部错误：未知启动方式 $MODE"
        ;;
esac
