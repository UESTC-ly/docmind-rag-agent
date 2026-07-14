# DocMind Windows launcher.
# Requires: uv, Docker Desktop with Compose.

$ErrorActionPreference = "Stop"

function Write-Info($Message) { Write-Host "▶ $Message" -ForegroundColor Blue }
function Write-Ok($Message) { Write-Host "✓ $Message" -ForegroundColor Green }
function Write-Warn($Message) { Write-Host "! $Message" -ForegroundColor Yellow }
function Fail($Message) { Write-Host "✗ $Message" -ForegroundColor Red; exit 1 }

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$CeleryOut = Join-Path $env:TEMP "docmind_celery.log"
$CeleryErr = Join-Path $env:TEMP "docmind_celery.err.log"
$PgDsn = "postgresql://docmind:docmind123@127.0.0.1:5432/docmind_db"
Set-Location $Root

function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ComposeArgs)
    if ($script:ComposeMode -eq "plugin") {
        & docker compose @ComposeArgs
    } else {
        & docker-compose @ComposeArgs
    }
}

function Ensure-Prerequisites {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "未找到 uv。安装：https://docs.astral.sh/uv/" }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "未找到 docker。请先安装并启动 Docker Desktop。" }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) { Fail "Docker Desktop 未运行，或当前终端无法访问 Docker。" }

    & docker compose version *> $null
    if ($LASTEXITCODE -eq 0) {
        $script:ComposeMode = "plugin"
    } elseif (Get-Command docker-compose -ErrorAction SilentlyContinue) {
        $script:ComposeMode = "legacy"
    } else {
        Fail "未找到 Docker Compose。请安装 Docker Desktop 或 docker-compose。"
    }

    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Warn "未发现 .env，已从 .env.example 创建。请填入真实 OPENAI_API_KEY 后再使用 AI 功能。"
    }
}

function Ensure-PythonEnv {
    if (-not (Test-Path ".venv")) {
        Write-Info "创建 Python 虚拟环境..."
        & uv venv --python 3.12 .venv
    }
    Write-Info "安装/同步 Python 依赖..."
    & uv pip install -r requirements.txt *> $null
    if ($LASTEXITCODE -ne 0) { Fail "Python 依赖安装失败" }
    Write-Ok "Python 依赖已就绪"
}

function Test-PostgresReady {
    $code = @'
import sys, asyncio, asyncpg
async def main():
    c = await asyncpg.connect(sys.argv[1], timeout=5)
    await c.close()
try:
    asyncio.run(main()); sys.exit(0)
except Exception:
    sys.exit(1)
'@
    $code | & uv run python - $PgDsn *> $null
    return $LASTEXITCODE -eq 0
}

Write-Info "DocMind Windows 启动流程"
Ensure-Prerequisites
Ensure-PythonEnv

Write-Info "清理旧进程..."
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match "uvicorn app.main|celery -A app.celery_app" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Write-Info "清理旧容器..."
Invoke-Compose down --remove-orphans *> $null

Write-Info "启动 PostgreSQL / Redis / Qdrant..."
Invoke-Compose up -d
if ($LASTEXITCODE -ne 0) { Fail "容器启动失败" }
Write-Ok "容器已启动"

Write-Info "等待 PostgreSQL 从宿主机可达..."
$ready = $false
for ($i = 1; $i -le 30; $i++) {
    if (Test-PostgresReady) {
        Write-Ok "PostgreSQL 宿主机可达（${i}s）"
        $ready = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $ready) { Fail "PostgreSQL 30s 内仍不可达。请检查 docker ps 与端口 5432。" }

Write-Info "应用数据库迁移..."
& uv run alembic upgrade head
if ($LASTEXITCODE -ne 0) { Fail "数据库迁移失败" }
Write-Ok "数据库 schema 已升级"

Write-Info "启动 Celery worker..."
$celery = Start-Process -FilePath "uv" `
    -ArgumentList @("run", "celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--pool=solo") `
    -RedirectStandardOutput $CeleryOut `
    -RedirectStandardError $CeleryErr `
    -PassThru `
    -WindowStyle Hidden
Start-Sleep -Seconds 3
if ($celery.HasExited) { Fail "Celery 启动失败。查看日志: $CeleryOut / $CeleryErr" }
Write-Ok "Celery 已启动 (PID=$($celery.Id))"
Write-Info "Celery 日志: $CeleryOut / $CeleryErr"

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  DocMind 已就绪                         ║" -ForegroundColor Green
Write-Host "║  http://localhost:8000                  ║" -ForegroundColor Green
Write-Host "║  Ctrl+C 停止 API；再次运行可完整重启     ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

& uv run uvicorn app.main:app --reload --port 8000
