# DocMind — 评测驱动、证据闭环的文档任务 Agent

DocMind 是面向文档与知识工作的 **Agent 系统**；RAG 是 Agent 可选择、替换和回归验证的
知识能力，而不是产品本身。用户给出目标后，Agent 形成显式计划，选择工具与经公开基准
评测的 `PipelineSpec`，在弱证据时改写查询或切换管线，并对最终回答/报告执行逐结论
依据检查、修复或拒答。

产品由三个相互连接的界面组成：**Agent 工作台**（计划、执行、审批、恢复、交付）、
**RAG Evaluation & Badcase Lab**（公开数据集、基线/候选、回归门禁、失败归因与选择性
重跑）和 **Evidence Navigator**（逐结论引用、原文跳转、版本/时效、冲突与推测标签）。

配套一个 **原生单页前端**（编辑/瑞士极简风，FastAPI 直接托管、运行时零构建），覆盖
登录、流式对话、文档管理、Skills/审批、评估看板界面。v3.1.0 在 v3.0 LangGraph
外层编排、人工审批和 checkpoint 恢复基础上，增加了**跨 worker 运行租约**与**定期清理**：
Web 使用 Redis 对同一 `run_id` 互斥，桌面/本地形态使用 SQLite owner-token lease；完成态和
未完成态 checkpoint 按不同保留期分批清理。桌面版
保留同一套前端与 FastAPI API，通过 PyInstaller sidecar 内置 Python 后端，使用 SQLite、
Qdrant local 与本地任务执行器；安装后的终端用户不需要 Docker、PostgreSQL、Redis、
Qdrant Server、uv 或系统 Python。

**差异化主线**：

1. **Task-driven Agentic RAG**：`agent_plan_v1`、计划步与工具 trace 关联、弱证据自适应、
   人工审批、checkpoint 恢复、执行回执和质量门组成同一任务生命周期。
2. **Evaluation-driven RAG EvalOps**：RAG 组件可插拔，但只有在同一公开语料快照上完成
   可复现对比、Badcase 回归与门禁后才参与自动选择。
3. **Claim-level Evidence Closure**：事实/推测逐结论检查，引用可跳到原文块；过期来源被
   阻断，冲突必须披露，语义检查不可用或证据不足时 fail closed。

## 目录

- [架构总览](#架构总览)
- [技术栈](#技术栈)
- [核心设计](#核心设计)
- [快速启动](#快速启动)
- [桌面 App（v3.1）](#桌面-appv31)
- [Web 开发启动](#web-开发启动)
- [使用流程](#使用流程)
- [API 一览](#api-一览)
- [Skills 技能系统](#skills-技能系统)
- [多路召回](#多路召回)
- [流式输出](#流式输出)
- [RAG Evaluation & Badcase Lab](#rag-evaluation--badcase-lab)
- [前端界面](#前端界面)
- [测试与 CI](#测试与-ci)
- [配置项](#配置项)
- [项目结构](#项目结构)
- [新增一个 Skill](#新增一个-skill)

## 架构总览

```
                   ┌───────────────────────────────────────────┐
  HTTP (JWT 鉴权)  │            FastAPI (async)                 │
 ───────────────► │  routers → services → models / schemas     │
                   └──────┬──────────────────────┬──────────────┘
                          │                       │
           ┌──────────────▼────────┐   ┌──────────▼─────────────┐
           │  LangGraph Outer Agent │   │  RAG 问答 (chat)        │
           │  supervisor / approval │   │  检索→拼Prompt→LLM      │
           │ lease/checkpoint/resume│   └──────────┬─────────────┘
           └──────────────┬────────┘              │
                          │ 自主调度               │
           ┌──────────────▼─────────────────────┐ │
           │  Skills（可插拔，装饰器注册）        │ │
           │  Python Skills + Codex-style packages │ │
           │  kb / mindmap / graph / report /      │ │
           │  weekly / ppt / web / generic runner  │ │
           └──────────────┬─────────────────────┘ │
                          │                         │
   ┌──────────────────────▼─────────────────────────▼─────────────┐
   │  共享检索：dense + DB keyword → scored RRF → reranker          │
   └──────────────────────────────┬────────────────────────────────┘
                                  │
              ┌───────────────────┴────────────────────┐
              │ Web：PostgreSQL + Qdrant + Redis/Celery │
              │      + Redis run lease + SQLite checkpoint│
              │ Desktop：SQLite + Qdrant local + local  │
              │          task executor                  │
              └─────────────────────────────────────────┘
```

文档解析与评估运行都**不阻塞创建请求**。Web 部署把工作派发给 Celery；桌面部署把同一
任务提交给受控本地线程执行器。前端分别轮询文档或评估状态，直到 completed / failed。

## 技术栈

- **后端**：FastAPI + SQLAlchemy 2.0 (async) + Pydantic v2 + pydantic-settings
- **AI / Agent**：OpenAI 兼容 API（默认 `gpt-4o-mini` + `text-embedding-3-small`）+
  Function Calling + LangGraph 1.2（interrupt / checkpoint / resume）
- **Web 存储**：PostgreSQL（元数据 + 分块/全文检索）+ Qdrant Server（向量）+ Redis
- **桌面存储**：SQLite + Qdrant local（均在用户应用数据目录）
- **后台任务**：Web 使用 Celery；桌面使用本地任务执行器
- **鉴权**：JWT（python-jose）+ bcrypt 密码哈希（passlib）
- **文档解析**：pymupdf（PDF）+ python-docx（Word）+ 纯文本
- **联网搜索**：ddgs（DuckDuckGo，无需 API key）
- **评估数据集**：HuggingFace `datasets`（MS MARCO / CMRC 2018 / RAGTruth）

## 核心设计

### 双数据库引擎（async + sync）与两种部署配置

FastAPI 用异步驱动，后台任务用同步 Session——所以 `app/database.py` 同时暴露两套引擎：

| 引擎 | 驱动 | 用途 |
|---|---|---|
| `AsyncSessionLocal` | asyncpg / aiosqlite | FastAPI 路由（经 `get_db` 依赖注入） |
| `SyncSessionLocal` | psycopg2 / sqlite | Celery worker、桌面本地任务、评估/Skill 同步逻辑 |

同步 URL 由 `settings.sync_database_url` 自动从异步 URL 去掉 `+asyncpg` / `+aiosqlite` 得到。
**不要在异步路由里用同步 Session，反之亦然。**

### 异步 / 同步边界

OpenAI 调用（LLM、embedding）与 Agent 主循环仍是**同步阻塞**的，在异步服务层用
`asyncio.to_thread(...)` 隔离。评估运行不再留在请求线程：创建 run 后由 Celery 或桌面
本地任务执行器完成。

### LangGraph durable Agent 与审批边界

`app/agent/orchestrator.py` 的外层状态图由 `supervisor → select_tool →
approval_gate/execute_tool` 组成。每个节点只做一种工作，工具一次执行一个；高风险调用在
任何副作用前 `interrupt()`，`POST /agent/resume` 用同一 `thread_id` 继续。Web 默认把
checkpoint 写到 `./data/agent-checkpoints.sqlite3`，桌面 sidecar 强制写入用户数据目录。
完成态 Assistant 消息使用 `agent_run_id` 幂等落库；工具执行回执则在 checkpoint 节点重放时
复用已完成结果，并把结果不确定的副作用重新交给人工决定，而不是静默重试。

本版迁移范围是**外层 Agent**。Generic Skill 内部仍使用既有 ReAct runner；若 package
声明脚本、MCP、浏览器、App 或仓库能力，会在进入整个 package 前审批，但尚不是内部每个
action 的独立审批/checkpoint。Generic Skill 子图和完整 Multi-Agent 拆分留待后续版本。
详细状态、恢复 API 和已知边界见
[`doc/09-v3.0.0-LangGraph实施与发布.md`](doc/09-v3.0.0-LangGraph实施与发布.md)。

### 跨 worker 运行租约与 checkpoint 清理

所有会改变图状态的入口（首次执行、审批恢复、崩溃恢复）先获取同一 `run_id` 的独占租约。
Web 默认使用共享 Redis；桌面或 `TASK_EXECUTION_MODE=local` 默认使用 checkpoint SQLite
文件中的进程共享租约。租约采用随机 owner token、TTL、心跳续租和条件释放：另一个 worker
持有时 API 返回 `423 Locked`，锁服务故障或执行中丢失所有权时返回 `503`，两者都带
`Retry-After`。状态查询保持只读，不需要拿锁。

FastAPI lifespan 同时启动有界维护循环。默认每小时最多清理 100 个 run：完成态保留 30 天，
等待审批、崩溃或其他未完成态保留 90 天；删除前再次获取该 run 的正常租约，正运行的任务会跳过。
升级时，v3.0 的既有 checkpoint 先建立生命周期索引并从升级时刻开始计时，不会立刻被删除。

这里的 Redis 只解决“谁可以修改同一个 run”。LangGraph saver 仍是 SQLite；跨主机任意节点恢复
仍需 sticky routing、共享磁盘或未来的生产共享 checkpointer。租约也不能替代外部系统的幂等键/
fencing token。实现、配置、故障语义与运维边界见
[`doc/10-v3.1.0-分布式运行锁与生命周期维护.md`](doc/10-v3.1.0-分布式运行锁与生命周期维护.md)。

### 数据隔离

- 每个 Qdrant point 的 payload 都带 `user_id`，所有向量检索强制按 `user_id` 过滤。
- 所有按 id 查询的接口都校验资源归属（`WHERE user_id = current_user.id`），防止越权。

## 快速启动

### Web 前置要求

下表只针对源码运行的 Web 开发/服务部署；安装后的桌面 App 不需要这些运行时。

| 平台 | 必需软件 | 说明 |
|---|---|---|
| macOS | Homebrew、Colima、Docker CLI/Compose、uv、Python 3.12 | 推荐用根目录 `./start.sh`，会自动转到 `scripts/start-macos-colima.sh` |
| Linux | Docker Engine、Docker Compose plugin 或 `docker-compose`、uv、Python 3.12 | 推荐用根目录 `./start.sh`，会自动转到 `scripts/start-linux-docker.sh` |
| Windows | Docker Desktop、PowerShell 5+、uv、Python 3.12 | 使用根目录 `start.ps1` 或 `scripts/start-windows.ps1` |

安装 `uv` 可参考：<https://docs.astral.sh/uv/>。Windows 建议在 PowerShell 中执行；
Linux 用户需确保当前用户有 Docker 权限，或自行在 Docker 命令前加 `sudo`。

## 桌面 App（v3.1）

桌面版使用 **Tauri 2** 把现有单页前端放进系统 WebView，不重写业务 UI。生产安装包携带
一个由 PyInstaller 冻结的 `docmind-sidecar`，其中包含 Python 解释器、FastAPI 和后端依赖。
Rust 宿主只启动这个 sidecar；桌面数据层使用 SQLite + Qdrant local，文档解析与评估使用
本地任务执行器，Agent run lease 也强制写入同一个用户数据目录的 SQLite checkpoint 文件。
终端用户无需安装 Docker、PostgreSQL、Redis、Qdrant Server、uv 或 Python。

桌面基础运行完全本地，但聊天和 embedding 仍需要可访问的 OpenAI 兼容服务及相应凭据；
显式启用的 MCP、浏览器或 App bridge 也可能需要各自的服务、浏览器运行时或系统权限。

### 日常使用

从 GitHub Release 下载与当前操作系统/架构匹配的安装包后直接启动 `DocMind`。桌面 App 会：

1. 创建独立、卸载不删除的用户数据目录与随机 JWT 密钥；
2. 首次生成 `desktop.env`，并强制注入本地 SQLite、Qdrant path 与 local task 配置；
3. 启动只监听 `127.0.0.1:8000` 的内置 sidecar；
4. 在接受请求前执行 Alembic migration；
5. 等待 `/health` 就绪后显示登录页。

在 `desktop.env` 中填入真实 `OPENAI_API_KEY`（以及可选的独立 embedding 配置）后重启
App 即可使用 AI 功能。数据库、向量、上传文件、Skill 工作区、密钥和日志都不放在只读安装包内。

| 平台 | 配置目录 |
|---|---|
| macOS | `~/Library/Application Support/com.docmind.desktop/` |
| Windows | `%APPDATA%\\com.docmind.desktop\\` |
| Linux | `~/.config/com.docmind.desktop/` |

目录中包含 `desktop.env`、`data/docmind.db`、`data/agent-checkpoints.sqlite3`、
`data/qdrant/`、`data/uploads/`、`data/skill_workspaces/`、`data/secrets/` 和
`logs/sidecar.log`。删除或升级安装包不会主动
删除这些用户数据。

### 开发与打包

```bash
cd desktop
npm ci
npm run dev       # 构建本机 sidecar 并启动 Tauri 开发窗口
npm run build     # 验证 sidecar 后在当前操作系统生成安装包
```

构建机（不是终端用户）需要 Node 22、Rust 和 Python 3.12+。`npm run build` 会建立私有构建
环境、冻结并自检 sidecar、运行本地 SQLite/Qdrant/任务 smoke，再把目标三元组命名的
可执行文件写入 Tauri `externalBin`。PyInstaller 原生扩展不能跨平台冻结，因此 macOS、
Windows、Linux/不同架构都必须使用对应原生 runner，并配置平台签名/公证凭据。
v3.1 的自动验收只运行 macOS arm64：默认 ad-hoc 签名通过 bundle 完整性检查，公开分发时
仍须用 Developer ID 覆盖该身份并完成 Apple notarization/stapling。Linux/Windows 不在本次
验证范围。

桌面端保留网页开发入口：执行根目录 `./start.sh` 或 `start.ps1` 后仍可通过
`http://127.0.0.1:8000/` 使用。Web 开发栈仍需要 Docker/uv/Python/PG/Redis/Qdrant，
与桌面 sidecar 是两套存储配置；两者也不应同时占用同一台机器的 `8000` 端口。

### 原生交互

- 文档页提供系统文件选择器，仍兼容网页拖拽/选择上传。
- 删除文档改用系统确认框；解析完成会发送系统通知。
- 周报、PPT 与通用 Skill 文件产出会打开系统“另存为”对话框，而不是浏览器下载栏。

## Web 开发启动

### 1. 克隆项目

```bash
git clone https://github.com/UESTC-ly/docmind-rag-agent.git
cd docmind-rag-agent
```

### 2. 配置环境变量

首次运行启动脚本时，如果没有 `.env`，脚本会自动从 `.env.example` 复制一份。你也可以手动执行：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

至少需要确认这些值：

```env
DATABASE_URL=postgresql+asyncpg://docmind:docmind123@localhost:5432/docmind_db
SECRET_KEY=change-this-to-a-long-random-string
OPENAI_API_KEY=replace-with-your-chat-api-key
OPENAI_BASE_URL=https://your-relay-or-official/v1
CHAT_MODEL=gpt-4o-mini
```

如果暂时没有真实 `OPENAI_API_KEY`，服务仍可启动，但上传后的问答、Agent、embedding 等 AI
功能会在调用外部模型时失败；这不是启动脚本问题。

### 3. 一键启动（推荐）

macOS / Linux：

```bash
./start.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

启动脚本会自动完成：

1. 检查 `uv`、Docker/Compose 等前置依赖；
2. 若缺少 `.env`，从 `.env.example` 创建；
3. 创建 `.venv` 并 `uv pip install -r requirements.txt`；
4. 清理旧的 `uvicorn` / Celery 进程；
5. 启动 PostgreSQL / Redis / Qdrant；
6. 等待 PostgreSQL 从宿主机可达；
7. 执行 `alembic upgrade head`；
8. 后台启动 Celery worker（统一 `--pool=solo`，macOS 额外设置 fork 安全环境变量）；
9. 前台启动 `uvicorn app.main:app --reload --port 8000`。

启动成功后访问：

```text
http://localhost:8000/
```

Celery 日志：

| 平台 | 日志位置 |
|---|---|
| macOS/Linux | `/tmp/docmind_celery.log` 或 `$TMPDIR/docmind_celery.log` |
| Windows | `%TEMP%\docmind_celery.log` 与 `%TEMP%\docmind_celery.err.log` |

停止 API：在运行 `uvicorn` 的终端按 `Ctrl+C`。如需完整重启，直接再次运行启动脚本。

### 4. 手动启动（排查问题时使用）

#### 4.1 启动依赖服务

macOS 如使用 Colima：

```bash
colima start
```

通用 Docker Compose：

```bash
docker compose up -d      # Compose v2 推荐
# 或 docker-compose up -d # 旧版 Compose
docker compose ps
```

`docker-compose.yml` 会拉起：PostgreSQL 16（`docmind` / `docmind123` / `docmind_db`，端口 5432）、
Redis 7（6379）、Qdrant（6333 REST / 6334 gRPC）。

#### 4.2 安装 Python 依赖

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

#### 4.3 启动 Celery worker

先把 schema 升级到当前版本：

```bash
uv run alembic upgrade head
```

macOS：

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
  uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

Linux / Windows：

```bash
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

Web 模式下文档解析和评估运行都依赖 Celery，**必须启动**；否则任务会停在 `pending`
或派发失败。桌面模式由本地任务执行器接管，不启动 Celery。

#### 4.4 启动 API 服务

```bash
uv run uvicorn app.main:app --reload --port 8000
```

浏览器访问：

- 前端：<http://127.0.0.1:8000/>
- Swagger API 文档：<http://127.0.0.1:8000/docs>

FastAPI lifespan 会在接受请求前再次幂等执行 Alembic upgrade；schema 生命周期不再使用
`Base.metadata.create_all`。已有 v2.1 数据库的首次接管步骤见
[`doc/08-v2.2.0发布说明.md`](doc/08-v2.2.0发布说明.md)。

### 5. 常见启动问题

| 现象 | 处理 |
|---|---|
| `OPENAI_API_KEY` 报错或 AI 调用失败 | 检查 `.env` 是否填入真实 key / base_url |
| 上传文档一直 `pending` | Celery worker 没启动或启动失败，查看 Celery 日志 |
| PostgreSQL 端口 5432 冲突 | 停掉本机已有 PG，或修改 `docker-compose.yml` 与 `.env` 端口 |
| Linux Docker 权限不足 | 将用户加入 docker 组后重新登录，或手动用 sudo 启动依赖服务 |
| Windows 脚本执行策略阻止 | 使用 `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| macOS Colima 端口转发异常 | `scripts/start-macos-colima.sh` 会尝试自愈；仍失败时重启 Colima/Docker |

## 使用流程

1. `POST /auth/register` 注册 → `POST /auth/login` 拿 token → 右上角 Authorize 填 token
2. `POST /documents/upload` 上传文档（支持 `.pdf` / `.docx` / `.doc` / `.txt` / `.md`）
3. `GET /documents/{id}` 轮询，等 `status` 变 `completed`
4. `POST /agent/chat` 与 Agent 对话，Agent 自主选工具：
   - "这份文档讲了什么？" → `search_knowledge_base`
   - "把文档 1 生成思维导图" → `generate_mindmap`
   - "文档 1 里各概念的关系图" → `generate_relation_graph`
   - "就 XX 主题写份报告" → `generate_report`
   - "根据材料写一份本周周报" → `generate_weekly_report`
   - "根据这份材料制作汇报 PPT" → `generate_presentation`
   - "查一下最新的 XX" → `web_search`
5. 技能卡的“选择并填入模板”会把 `skill_name` 一并提交，首轮强制调用该技能，避免
   LLM 只在聊天窗口模拟产出；若选了 `document_id` 且未锁定技能，首轮强制调用
   `search_knowledge_base`，先取得 RAG 片段再继续编排。
6. 若返回 `status=waiting_approval`，检查 `approval.tool / args / reason / scope`，在 UI
   选择“批准并继续”或“拒绝本次调用”；也可调用 `POST /agent/resume`。刷新或重启后用
   `GET /agent/runs/{run_id}` 找回当前状态。
7. 前端在发送前生成并保存 `run_id`；请求响应丢失时，同一 ID 可幂等查询。若状态为
   `running + recoverable=true`，调用 `POST /agent/runs/{run_id}/recover` 从未完成节点继续。
8. 完成响应含 `answer`、`artifacts`、`trace`，并始终带本次 `run_id/thread_id`。
9. `GET /agent/skills` 查看技能、运行时兼容状态、grounding 模式与下载能力。

`POST /chat/` 是**不带 Agent 编排的纯 RAG 问答**：直接检索 → 拼 Prompt → 生成，返回
`answer` + `sources`（引用片段）。适合只需要问答、不需要工具调度的场景。

## API 一览

| 分组 | 方法 & 路径 | 说明 |
|---|---|---|
| 认证 | `POST /auth/register` | 注册新用户 |
| 认证 | `POST /auth/login` | 登录，返回 JWT |
| 认证 | `GET /auth/me` | 当前登录用户信息 |
| 文档 | `POST /documents/upload` | 上传文档，异步解析 |
| 文档 | `GET /documents/` | 我的文档列表 |
| 文档 | `GET /documents/{id}` | 查询单个文档解析状态 |
| 文档 | `PATCH /documents/{id}/metadata` | 更新来源版本、生效期、权威性与 supersession |
| 文档 | `GET /documents/{id}/chunks/{chunk_index}` | 打开引用对应的精确原文块 |
| 文档 | `DELETE /documents/{id}` | 删除文档（连带删 Qdrant 向量 + 级联删分块） |
| 问答 | `POST /chat/` | 纯 RAG 问答（无 Agent） |
| 问答 | `POST /chat/stream` | RAG 流式问答（SSE，逐 token） |
| 问答 | `GET /chat/conversations` | 我的对话列表 |
| 问答 | `GET /chat/conversations/{id}` | 某对话的消息历史 |
| Agent | `POST /agent/chat` | 与 Agent 对话；可传 `skill_name` 锁定首轮技能 |
| Agent | `POST /agent/resume` | 批准/拒绝并恢复等待中的 LangGraph run；可修订参数 |
| Agent | `GET /agent/runs/{run_id}` | 按当前用户读取 checkpoint 的公开状态 |
| Agent | `POST /agent/runs/{run_id}/recover` | 从非 interrupt 的未完成节点继续；执行回执防静默重复 |
| Agent | `GET /agent/skills` | 列出技能及 available / grounding / download 元数据 |
| 评估 | `POST /eval/datasets` | 从文档 LLM 反向出题生成数据集 |
| 评估 | `GET /eval/datasets` | 我的评估数据集列表 |
| 评估 | `GET /eval/pipelines` | 管线目录与不可变 fingerprint |
| 评估 | `POST /eval/pipelines/validate` | 校验内置/插件管线及索引兼容性 |
| 评估 | `POST /eval/runs` | 创建 pending run，派发后台任务并返回 `202 Accepted` |
| 评估 | `POST /eval/experiments` | 同一快照上的多管线基线/候选实验 |
| 评估 | `GET /eval/runs/{id}` | 轮询 pending/running/completed/failed 与聚合指标 |
| 评估 | `GET /eval/runs/{id}/details` | 逐条样本明细分数 |
| 评估 | `GET /eval/runs/{id}/metrics` | 版本化的 run/sample 指标与 judge 理由 |
| 评估 | `POST /eval/regression-gates` | 配置绝对或相对回归门禁 |
| 评估 | `GET /eval/runs/{id}/regression` | 查看候选运行的持久化门禁结论 |
| 评估 | `GET /eval/runs/{id}/badcases` | 根因分类、检索轨迹与原文跳转 |
| 评估 | `GET /eval/runs/{id}/badcase-diff` | 新增、修复和持续 Badcase |
| 评估 | `POST /eval/runs/{id}/rerun` | 只重跑所选样本；诊断运行不参与发布选择 |
| 健康 | `GET /health` | 健康检查 |

除 `register` / `login` / `health` 外，所有端点都需 `Authorization: Bearer <token>`。

## Skills 技能系统

技能是 Agent 的能力单元。v3.1.0 保持**两条执行路径并存**：

```text
Python-backed Skill：BaseSkill 子类 + run()，适合强确定性/强业务边界
Codex-style Generic Skill：只需 SKILL.md，可选 templates / references / scripts / assets
```

也就是说，DocMind 现在既保留原来的 Python 技能，也能扫描
`app/skills/packages/<slug>/SKILL.md` 这种主流 Agent Skills 文件夹。若 package 没有
对应 Python 类，系统会注册为 `GenericPackageSkill`，由内部 ReAct runner 按
Markdown 指令规划并调用受控工具执行。**`status: ready` 只代表包已审计，不代表获得
宿主权限**：`docmind.json.requires` 还要通过运行时 capability 检查；未满足时会返回
明确配置原因，且不会暴露给 Function Calling 或被技能卡锁定调用。

通用 runner 当前内置的动作能力：

- `read_skill_reference` / `read_skill_template`：渐进读取 package 资料；
- `search_uploaded_documents`：复用向量 + 关键词 RRF 的 DocMind RAG 检索；
- `list_files` / `read_file` / `write_file`：读写用户隔离工作区；
- `modify_code`：在隔离工作区写入/替换代码文件；
- `read_skill_asset` / `copy_skill_asset`：受控读取或复制 package 文本/二进制资源；
- `run_skill_script`：只执行已审计的 package 固定脚本；除解释器、环境变量和超时 allowlist
  外，还要求 macOS `sandbox-exec` 写入隔离探测成功，拒绝绝对路径、`..`、符号链接和
  workspace 外产物；
- `list_repository_files` / `read_repository_file` / `search_repository` / `git_history`：
  只读访问显式挂载的仓库根目录；
- `call_mcp`：生产 Streamable HTTP MCP adapter，服务、工具、认证和超时都由宿主配置；
- `use_browser_tool`：生产 Playwright adapter，按 execution 隔离会话、限制域名，截图只写工作区；
- `use_app_tool`：带 token 的 loopback HTTP bridge，仅暴露应用/动作 allowlist。

adapter 统一返回 `status / summary / next_actions / artifacts`。未配置、越权、超时或外部
服务失败时给出可恢复诊断，不把失败伪装成普通聊天成功。

安全边界：每个工具同时通过“宿主授权 ∩ package 声明”的双重 capability gate，工具发现和
实际执行各检查一次，伪造 tool call 不能绕过。通用技能的文件读写只发生在
`skill_workspaces/user_<id>/<skill_slug>/`；生产运行时不开放任意 shell，需执行动作时使用
受审计 package script 或专用 adapter，避免把 `/agent/chat` 变成远程代码执行入口。
此外，外层 LangGraph 会在进入声明高风险 capability 的 Generic package 前请求人工审批。
这个审批覆盖本次 package 调用，并不等价于内部 action 级审批；后者将在 runner 子图化后迁移。

| 技能名 | Grounding | 作用 | 下载 |
|---|---|---|---|
| `search_knowledge_base` | Hybrid RAG | 向量 + 关键词 RRF 检索，回答文档问题优先用 | — |
| `generate_mindmap` | 顺序正文前 8000 字 | 抽取层级结构，输出 Mermaid mindmap | `.mmd` |
| `generate_relation_graph` | 顺序正文前 8000 字 | 抽取实体+关系，输出 nodes/edges + Mermaid | `.json` |
| `generate_report` | Hybrid RAG | 检索→大纲→逐节生成→汇总成文 | `.md` |
| `generate_weekly_report` | Hybrid RAG | 根据检索材料生成结构化中文周报 | `.md` |
| `generate_presentation` | Hybrid RAG | 根据检索材料生成演示文稿 | `.pptx` |
| `web_search` | 联网 | DuckDuckGo 搜索外部信息 | — |
| `codex_note` | 选中文档时 Hybrid RAG | 按 SKILL.md 写 Markdown 笔记 | `.zip` |

产出 `type` 属于 `mindmap` / `relation_graph` / `report` / `weekly_report` /
`presentation`，或带 `artifact_kind=file` 的通用技能结果，会被收进响应的
`artifacts`。通用技能即使只返回聊天正文、没有主动调用 `write_file`，runner 也会把
最终正文落成 Markdown 并打 ZIP；该 fallback 仅对同时声明并获授 `workspace` 的文件型
package 生效，纯文本 package 不会获得隐式文档检索或写盘权限。

### 通用包兼容性隔离

每个通用包可增加 `docmind.json`：

```json
{
  "status": "ready",
  "reason": "已适配 DocMind 的受控运行时。",
  "requires": ["repository", "git", "package_scripts"]
}
```

- `ready`：包已审计；工具只在宿主 capability 与 `requires` 的交集内暴露和执行；
- `blocked`：包本身仍存在明确的不兼容实现（不是缺凭据/缺配置的常态）；
- 缺少文件：视为 `unreviewed`，默认隔离，防止复制一个 Codex 包后“看起来可用、实际只会聊天回答”。

当前审计结果：

- Ready（9）：`codex-note`、`jupyter-notebook`、`openai-docs`、`playwright`、
  `presentation`、`screenshot`、`security-best-practices`、
  `security-threat-model`、`weekly-report`；
- Blocked（4）：`gh-fix-ci`（缺 GitHub CLI/认证/写入工作流）、`pdf`（尚无 PDF
  action/scripts）、`playwright-interactive`（依赖 `js_repl`/Electron/danger-full-access）、
  `security-ownership-map`（依赖未提供的直接仓库脚本与图分析运行时）。

Ready 不等于始终可调用：若宿主未启用 package scripts、MCP、浏览器或仓库等声明能力，
对应 package 会显示 missing capability 并保持隔离。Blocked 则是明确实现不兼容，不能靠扩大
默认权限“解锁”。修改 manifest 本身也不能绕过宿主 capability gate。

### 两种 Skill Package 目录结构

Python-backed package（例如周报）：

```text
app/skills/packages/weekly-report/
├── SKILL.md                         给 Agent/开发者看的技能说明
├── skill.json                       Function Calling metadata（name/description/parameters）
├── docmind.json                     审计状态 + requires capability 清单
├── templates/
│   └── prompt.md                    LLM 提示词模板
└── references/
    └── writing-guide.md             写作规范、参考说明
```

Codex-style generic package（例如 `codex-note`）：

```text
app/skills/packages/codex-note/
├── SKILL.md                         必需，frontmatter 提供 name/description
├── docmind.json                     兼容审计通过后设为 ready
├── templates/                       可选
│   └── note.md
├── references/                      可选
│   └── style.md
├── scripts/                         可选，由受控 package script executor 运行
└── assets/                          可选
```

最小 `SKILL.md`：

```md
---
name: my_generic_skill
description: Explain exactly when this skill should be used.
---

# Workflow

Follow these steps and write final files into outputs/.
```

### 外层 Agent 状态图（`app/agent/orchestrator.py`）

LangGraph 状态图最多执行 `agent_max_steps`（默认 6）轮模型决策：

1. `supervisor` 把系统提示、历史、用户消息和已通过审计的工具定义交给 LLM；技能卡
   传 `skill_name` 时强制首轮指定技能，选定文档且未锁定技能时首轮强制 RAG；
2. `select_tool` 把同轮多个 `tool_calls` 拆成一次一个可 checkpoint 的调用；
3. 低风险调用进入 `execute_tool`；高风险调用先进入 `approval_gate` 并 `interrupt()`；
4. 批准后执行（可替换参数），拒绝则形成 `role: tool` observation，不产生副作用；
5. 每个节点完成后 saver 保存状态，再处理下一调用或回到 `supervisor`，直至最终答案/上限。

`run_id` 就是 LangGraph `thread_id`。HTTP 层总是使用持久化 SQLite saver；一次性直接调用
`run_agent()` 默认使用内存 saver，避免测试脚本污染仓库。

外层工具另有 `run_id + tool_call_id` 执行回执：工具已返回但图 checkpoint 尚未写入时崩溃，
恢复会复用已保存结果；若崩溃发生在工具执行期间、外部副作用是否完成无法判断，图会再次
interrupt 请求“是否重试”，不会静默执行第二次。

对话历史由 `app/agent/memory.py` 管理，只带入最近 `WINDOW_SIZE`（10）条消息控制 token。

## RAG Evaluation & Badcase Lab

这里不是偶尔调 Prompt 的打分页，而是可重复运行的 EvalOps 闭环。发布指标只允许来自
保留 URL、版本、许可证、split、原始快照 SHA-256、转换契约和语料 fingerprint 的公开
数据集；LLM 反向出题只能做补充诊断，不能替代公开 qrels/人工标签。

### 两类数据集来源

- **LLM 反向出题**（`dataset_gen.py`）：可生成诊断样本，但标记为 synthetic，
  `release_eligible=false`。
- **公开基准导入**（`dataset_import.py`）：导入业界标准数据集的 ground truth。
  关键在于评估检索指标依赖 Qdrant 里真实存在的向量，所以导入会走完整链路：
  `passages → Document + DocumentChunk(PG) → embed → upsert Qdrant → EvalDataset + EvalSample`。
  - **MS MARCO v2.1**：validation 前缀中每题 passages 的 `is_selected` 标注
  - **CMRC 2018**：中文阅读理解，去重 context 建段落池，每题指向自己的段落
  - **BEIR 标准目录**：`corpus.jsonl + queries.jsonl + qrels/<split>.tsv`，支持 graded qrels；
    上游来源和许可证必须显式提供，不能套用 BEIR 的代码许可证
  - **RAGTruth**：官方 `source_info.jsonl + response.jsonl` 不进入检索索引，而是校准
    faithfulness judge；`implicit_true` 会保留为“可能真实但没有上下文依据”

### 指标

`POST /eval/runs` 只创建 pending run 并返回 `202 Accepted`。Web 模式由 Celery task、桌面
模式由本地任务执行器调用同一个 runner；前端轮询 pending、running、completed 或 failed。
runner 使用条件 UPDATE/CAS 领取带 token 的租约并发送独立 heartbeat，终态写入也校验 token；
临时失败回到 pending，Celery 使用 late ack/worker-lost 重投，桌面重启会重派 pending 与中断的
running。数据库唯一约束保证同一 `(run_id, sample_id)` 不会重复追加明细。

评估运行（`runner.py`）遍历样本，对每条复用线上 dense + keyword + RRF + reranker 检索，
再计算：

| 层级 | 指标/检查 | 语义 |
|---|---|---|
| Document Retrieval | Hit@K、MRR、Recall@K、Precision@K、MAP@K、nDCG@K | 仅在真实 document qrels 存在时计算；缺失保持 unavailable |
| Passage/Chunk Retrieval | Hit@K、MRR、Recall@K、Precision@K、MAP@K、nDCG@K | 使用公开 chunk qrels/graded relevance |
| Generation | Faithfulness、Answer Relevance | 版本化 LLM judge；异常返回 `score=null`，不伪造 0 分 |
| Evidence | Groundedness、Citation Correctness、Citation Completeness | 原子 claim 到具体证据的语义与结构检查 |
| Policy | Refusal Accuracy、Conflict Awareness、Freshness Compliance | 可答/不可答、冲突披露、文档版本时效 |
| Agent | Verified Task Completion Rate | 任务、轨迹、产物和证据门同时通过 |

每个 run 固化 `PipelineSpec`、pipeline/index fingerprint、模型、prompt/rubric 版本、
代码 revision、环境 fingerprint、延迟、逐样本排名与 judge 理由。`EvalMetricResult`
保存版本化指标；聚合值可从逐样本结果独立重算。候选与命名基线之间会生成门禁结论，以及
`newly introduced / fixed / persistent` Badcase。Badcase 按 document miss、chunk boundary、
reranking、noise、stale source、unsupported claim、错误引用、错误拒答、冲突遗漏、
推测未标记和 evaluator disagreement 等层级归因。

诊断型 subset rerun 会复用来源 run 的不可变管线，只跑所选样本，并显式标记
`evaluation_scope=subset`；它不会进入发布回归或 Agent 的自动管线选择。

### 导入公开数据集（CLI）

```bash
# 1. 数据准备依赖是可选项，不进入应用运行时依赖
uv pip install datasets pandas pyarrow
uv run python data/download_datasets.py --only ms_marco --ms-marco-limit 3000
uv run python data/download_datasets.py --only cmrc2018

# RAGTruth 直接下载官方 JSONL，不依赖 HuggingFace datasets
uv run python data/download_datasets.py --only ragtruth

# 2. 导入为某用户的评估数据集（需 PG / Qdrant / embedding 服务在线）
uv run python data/import_datasets.py --user-id 1
uv run python data/import_datasets.py --user-id 1 --only ms_marco --limit 50

# 3. BEIR 标准目录导入；许可证必须按具体上游数据集填写
uv run python data/import_datasets.py --user-id 1 --only beir \
  --beir-dir data/beir/<dataset> --beir-name <name> \
  --source-name <upstream-name> --source-uri <public-url> \
  --source-version <snapshot> --license-name <verified-license>

# 4. 用 RAGTruth test split 校准 faithfulness judge
uv run python scripts/calibrate_ragtruth.py \
  --source-info data/ragtruth/source_info.jsonl \
  --responses data/ragtruth/response.jsonl \
  --output /tmp/ragtruth-calibration.json \
  --markdown /tmp/ragtruth-calibration.md

# 5. 对同一已导入快照运行可独立重算的检索对比
uv run python scripts/benchmark_retrieval.py \
  --dataset-id <id> --pipelines dense,hybrid,hybrid-rerank \
  --output /tmp/retrieval-evidence.json
```

RAGTruth 校准当前只证明 generation-level faithfulness judge；claim/citation 和
answer-relevance judge 的完整公共校准尚未完成前，`grounded_generation` 自动发布选择会
fail closed。脚本生成文件本身也不是通过证据，必须读取其中的 coverage、混淆矩阵、
balanced accuracy 与 `release_gate_eligible`。

## 多路召回

`app/services/retrieval.py` 是在线问答、Skills 与 evaluation runner 的共享入口。单一稠密
向量检索抓不住型号、专有名词等精确关键词，所以并联数据库关键词召回，再用
**RRF（Reciprocal Rank Fusion，倒数排名融合）**合并两路候选：
`RRF(d) = Σ 1/(k + rank_i(d))`（默认 k=60）。融合结果进入有界二阶段 reranker。

- 融合身份用 `(document_id, chunk_index)` 复合键，避免跨文档同序号块被误合并。
- `RETRIEVAL_MODE=hybrid`（默认）走多路；`dense` 退回纯向量检索。
- PostgreSQL 用 `to_tsvector('simple', content)`、`websearch_to_tsquery` 和 `ts_rank_cd`，
  v2.2 migration 建立对应 GIN expression index。
- 桌面 SQLite/测试使用 SQL `CASE + LIKE` 兼容评分；两种方言都在同一查询中应用
  user/document 条件、排序和 `LIMIT`，不会把用户全部 chunks 读进 Python。
- `RERANKER_MODE=local` 默认组合 RRF、dense、keyword 与直接词项重合信号，采用稳定
  tie-break；`http` 可接 cross-encoder provider，超时或响应异常自动回退 local；`off`
  保留融合顺序。

## 流式输出

`POST /chat/stream` 以 **Server-Sent Events** 逐 token 推送答案。事件流：
`event: meta`（会话 id + 来源）→ 多个 `event: token`（逐字）→ `event: done`，
流结束后落库。底层复用中转本就强制流式的特性（`llm_service.chat_completion_stream`
生成器逐块 yield）。前端用 `fetch` + `ReadableStream` 手解析 SSE（`EventSource`
无法带 `Authorization` 头，而本端点要 JWT）。

## 前端界面

`frontend/`，原生 HTML/CSS/JS 单页，**运行时零构建**，由 FastAPI `StaticFiles` 直接
托管（API 路由优先匹配，`/` 兜底返回单页）。Node/Playwright 只用于自动化测试与桌面
资源构建，不是 Web 页面运行依赖。编辑/瑞士极简风格：
纯白底 + 单一瑞士红强调 + sans 标题/衬线正文配对，跟随系统明暗。

四大界面：
- **登录/注册**：JWT 存 localStorage，401 自动清 token 回登录屏。
- **流式对话**：SSE 逐字浮现（带光标动画）、来源引用卡、左侧会话历史、Enter 发送。
- **文档管理**：拖拽上传、解析状态轮询（pending→completed 脉冲动画）、删除。
- **评估看板**：列数据集、一键入队、轮询 pending/running/completed/failed、6 指标可视化。

工程：模块化 JS（`api/ui/chat/docs/eval/main`）、CSS 按 surface 分文件、
compositor 友好动画、`prefers-reduced-motion` 降级、键盘焦点环、无 `innerHTML` 注入。
启动 API 后浏览器访问 http://127.0.0.1:8000/ 即用。

## 测试与 CI

自动化门禁覆盖 Python、浏览器、JavaScript 和 Rust；不要在文档中写死容易漂移的用例数
或一次性覆盖率快照，以当前 CI 输出为准。

- **纯函数单测**：检索指标、RRF 融合、密码哈希/JWT、数据集解析、分块——无 I/O，秒级。
- **服务单测**：共享检索/reranker、评估 runner/Celery task、Alembic、Skills adapters、
  capability gate、LangGraph interrupt/checkpoint/restart resume、LLM 流式聚合——mock 外部边界。
- **路由集成测**：用内存 sqlite 替 PG、mock 掉 Celery/embedding/LLM，真实 HTTP 打
  auth/documents/chat/agent/eval 全部端点（含 SSE 流式、评估双路径）。
- **真实全栈 E2E**：独立 Playwright 配置启动真实 FastAPI，使用临时 SQLite、Qdrant local
  和本地任务执行器，验证迁移、健康检查、401、注册/JWT、文档/会话/Skills API，以及页面
  reload 后登录态；它与 mock/视觉套件分开运行，不调用外部 LLM/adapters。

`.coveragerc` 配置并发追踪。`.github/workflows/ci.yml` 在 push/PR 时执行 Python
runtime-error lint、模型/API contract typecheck、pytest 覆盖率门槛、JavaScript 语法检查，
以及 Rust fmt/clippy/test。`.github/workflows/frontend-e2e.yml` 在 macOS Chromium 上分别运行
Page Object 驱动的 mock/视觉回归与真实 FastAPI 全栈路径；失败时上传 HTML report、trace、
截图、视频和 JUnit 结果。E2E 使用稳定 API fixture 与网络/状态等待，不依赖固定 sleep。
Linux/Windows 安装包不属于 v3.1 本地验收矩阵。

```bash
uv run pytest --cov=app --cov-report=term-missing   # 本地跑测 + 覆盖率

cd frontend
npm ci
npm run test:e2e                                   # Chromium E2E + 视觉回归
npm run test:e2e:fullstack                         # 真实 FastAPI + SQLite/Qdrant local
```

## 配置项

全部集中在 `app/config.py`（pydantic-settings，从 `.env` 读取，大小写不敏感）。

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | Web 用 `postgresql+asyncpg://`；桌面宿主注入 `sqlite+aiosqlite:///...` |
| `SECRET_KEY` | ✅ | — | JWT 签名密钥 |
| `OPENAI_API_KEY` | ✅ | — | 对话 LLM key（未单独配 embedding 时也复用它） |
| `REDIS_URL` | | `redis://localhost:6379/0` | Celery broker |
| `ALGORITHM` | | `HS256` | JWT 算法 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | | `60` | token 有效期 |
| `OPENAI_BASE_URL` | | 官方 | 兼容中转/代理地址 |
| `CHAT_MODEL` | | `gpt-4o-mini` | 对话模型 |
| `EMBEDDING_API_KEY` | | 复用对话 key | embedding 可用独立服务 |
| `EMBEDDING_BASE_URL` | | 复用对话 URL | embedding 独立地址 |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | embedding 模型 |
| `EMBEDDING_DIM` | | `1536` | **必须与模型实际维度一致** |
| `EMBEDDING_BATCH_SIZE` | | `10` | 部分服务（如 DashScope）单次批量有上限 |
| `QDRANT_URL` | | `http://localhost:6333` | Qdrant 地址 |
| `QDRANT_PATH` | | — | 设置后使用 Qdrant local；桌面宿主强制注入用户数据目录 |
| `QDRANT_COLLECTION` | | `docmind_chunks` | 向量集合名 |
| `UPLOAD_DIR` | | `./uploads` | 上传文件落盘目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | | `800` / `100` | 分块字符数 / 相邻块重叠 |
| `RETRIEVAL_TOP_K` | | `5` | 检索返回块数 |
| `RETRIEVAL_MODE` | | `hybrid` | `hybrid`=向量+关键词 RRF；`dense`=纯向量 |
| `RRF_K` | | `60` | RRF 融合常数 |
| `DENSE_CANDIDATES` / `KEYWORD_CANDIDATES` | | `20` / `20` | 融合前两路候选上限 |
| `RERANKER_MODE` | | `local` | `local` / `http`（失败回退 local）/ `off` |
| `RERANKER_CANDIDATE_LIMIT` | | `40` | 二阶段重排最大候选数 |
| `RERANKER_HTTP_*` | | — | 可选 cross-encoder URL、key、model、timeout |
| `RAG_PLUGIN_MODULES` | | 空 | 部署方信任的管线插件模块列表；不扫描用户可写目录 |
| `GROUNDING_VERIFICATION_MODE` | | `llm` | `llm`=逐结论语义检查；`off` 只能做结构诊断 |
| `GROUNDING_FAIL_CLOSED` | | `true` | 语义检查失败时拒答/阻断证据型交付 |
| `TASK_EXECUTION_MODE` | | `celery` | Web 用 `celery`；桌面宿主强制使用 `local` |
| `LOCAL_TASK_WORKERS` | | `2` | 本地任务执行器线程数 |
| `EVALUATION_LEASE_SECONDS` | | `2100` | 评估 worker 租约过期/崩溃接管窗口 |
| `EVALUATION_TASK_*_TIME_LIMIT_SECONDS` | | `1740/1800` | Celery 评估任务软/硬时限，短于租约 |
| `AGENT_MAX_STEPS` | | `6` | Agent 主循环最大步数 |
| `AGENT_PLANNING_MODE` | | `explicit` | 复杂任务先形成持久化 `agent_plan_v1` |
| `AGENT_CHECKPOINT_PATH` | | `./data/agent-checkpoints.sqlite3` | LangGraph SQLite checkpoint；桌面宿主覆盖到用户数据目录 |
| `AGENT_RUN_LOCK_BACKEND` | | `auto` | Web/celery 自动解析为 `redis`；desktop/local 解析为 `sqlite`；`off` 仅用于显式测试/诊断 |
| `AGENT_RUN_LOCK_TTL_SECONDS` | | `300` | 运行租约失效时间；必须大于心跳间隔 |
| `AGENT_RUN_LOCK_HEARTBEAT_SECONDS` | | `60` | 持有者续租间隔 |
| `AGENT_RUN_LOCK_NAMESPACE` | | `docmind:agent-run` | Redis/SQLite 租约键命名空间，不同环境应隔离 |
| `AGENT_CHECKPOINT_CLEANUP_ENABLED` | | `true` | 是否启动 checkpoint 生命周期维护循环 |
| `AGENT_CHECKPOINT_CLEANUP_INTERVAL_SECONDS` | | `3600` | 维护周期；多个 worker 由维护租约去重 |
| `AGENT_CHECKPOINT_COMPLETED_RETENTION_DAYS` | | `30` | 完成态 checkpoint/回执保留天数 |
| `AGENT_CHECKPOINT_INCOMPLETE_RETENTION_DAYS` | | `90` | 未完成/待审批 checkpoint 保留天数，不能短于完成态 |
| `AGENT_CHECKPOINT_CLEANUP_BATCH_SIZE` | | `100` | 每轮最多处理的 run 数，限制 SQLite 写锁时间 |
| `AGENT_HIGH_RISK_SKILLS` | | 空 | 必须审批的外层 Skill 名称，逗号分隔 |
| `AGENT_HIGH_RISK_CAPABILITIES` | | `package_scripts,mcp,browser,app,repository` | Generic package 声明这些能力时整次调用需审批 |
| `SKILL_RUNNER_MAX_STEPS` | | `8` | Codex-style 通用 skill 内部工具循环最大步数 |
| `SKILL_WORKSPACE_DIR` | | `./skill_workspaces` | 通用 skill 的用户隔离文件工作区 |
| `SKILL_PACKAGE_SCRIPTS_ENABLED` | | `false` | 是否启用已审计固定脚本；还需 package 声明和 macOS confinement |
| `SKILL_PACKAGE_SCRIPT_*` | | 见 `.env.example` | 解释器、环境变量 allowlist 与超时；任意 shell 始终不可用 |
| `SKILL_REPOSITORY_ENABLED` / `ROOT` | | `false` / — | 显式挂载只读仓库能力 |
| `SKILL_MCP_ENABLED` / `SERVERS_JSON` | | `false` / `{}` | MCP URL、工具 allowlist、token env 等配置 |
| `SKILL_BROWSER_ENABLED` / `ALLOWED_HOSTS` | | `false` / loopback | Playwright adapter 与域名 allowlist |
| `SKILL_APP_ENABLED` / `BRIDGE_*` | | `false` / — | loopback bridge URL/token 与应用动作 allowlist |
| `LOG_LEVEL` | | `INFO` | 日志级别 |
| `LOG_JSON` | | `true` | `true`=结构化 JSON（生产）；`false`=彩色文本（本地） |

> 对话与 embedding 可用不同服务商（如对话用 GPT、embedding 用 DashScope）；
> embedding 配置留空则复用对话的 key / base_url。
> **`bcrypt` 钉死在 4.0.1**：passlib 1.7.4 与 bcrypt 5.x 不兼容，升级时两者需同步。

## 项目结构

```
app/
├── main.py            FastAPI 入口（注册路由、lifespan 执行 Alembic/维护循环）
├── config.py          全局配置单例
├── database.py        异步 + 同步双引擎，get_db 依赖
├── celery_app.py      Celery 实例
├── agent/
│   ├── orchestrator.py       LangGraph 外层状态图与 checkpoint 生命周期记录
│   ├── planning.py           agent_plan_v1、计划归一化与评测选择步骤
│   ├── run_lock.py           Redis/SQLite owner-token 运行租约与心跳
│   ├── checkpoint_store.py   DocMind 生命周期/回执/维护元数据表
│   └── checkpoint_cleanup.py 定期保留策略、迁移回填与有界清理
├── models/            ORM 模型（user / document / conversation / evaluation）
├── schemas/           Pydantic 请求/响应模型
├── routers/           路由层（auth / documents / chat / agent / evaluation）
├── services/          业务逻辑
│   ├── auth / document / rag / agent / embedding / llm / vector_store
│   ├── retrieval.py   数据库关键词 + scored RRF 的共享召回
│   ├── reranker.py    本地二阶段排序 + HTTP provider 回退
│   ├── task_dispatcher.py  Celery / desktop local task 路由
│   ├── skill_retrieval.py  同步 Skills 复用 Hybrid RAG 的入口
│   └── evaluation/    dataset import / runner / metrics / regression /
│                      badcases / grounding / judge calibration / pipeline selection
├── agent/             LangGraph 外层编排、interrupt/checkpoint/resume、memory 对话记忆
├── skills/            Python Skills + generic runner + capability gate + adapters
│   ├── adapters/      MCP / Playwright / loopback App bridge
│   └── packages/      SKILL.md / docmind.json / templates / references / scripts / assets
├── tasks/             文档解析与评估 Celery tasks
└── utils/             工具（security JWT / deps / file_parser / logging / middleware）

alembic/               baseline + 检索 trace、EvalOps、来源版本和 subset rerun migrations
frontend/              原生单页前端 + Playwright E2E/视觉基线
├── index.html
├── styles/            tokens / base / layout / components（按 surface 分文件）
├── js/                api / ui / chat / docs / eval / main（ES modules）
└── e2e/               Page Object、API fixtures、视觉测试

desktop/               Tauri 2 宿主 + PyInstaller sidecar + 打包/smoke 脚本
tests/                 pytest + 内存 SQLite + mock 外部边界
data/                  评估数据集下载 / 导入脚本 + parquet 缓存 + manifest.json
.github/workflows/     Python/JS/Rust CI + Chromium E2E/视觉 CI
```

## 新增一个 Skill（体现可插拔）

现在有两种扩展方式。

### 方式 A：复制 Codex-style 通用 Skill 包

适合 prompt 工作流、文档产出、需要模板/参考资料的通用能力。

1. 在 `app/skills/packages/<slug>/` 下放入至少一个 `SKILL.md`：
   ```md
   ---
   name: my_generic_skill
   description: 用户什么时候应该调用这个技能。
   ---

   # Workflow
   1. 读取需要的 references。
   2. 按步骤完成任务。
   3. 如需文件产出，写入 outputs/。
   ```
2. 可选增加：
   - `templates/`：提示词、文档骨架；
   - `references/`：写作规范、API 说明、业务资料；
   - `scripts/`：由固定路径的 package script executor 执行，受解释器/环境/超时、
     参数逃逸校验、macOS 写入 sandbox 和 artifact 边界限制；
   - `assets/`：模板文件、图片等资源。
3. 增加 `docmind.json`。新复制的包默认是 `unreviewed`；审计其依赖后写明 `requires`，
   例如 `{"status":"ready","requires":["repository"]}`。包标为 ready 后仍必须由宿主
   显式开启相应 capability，manifest 不能自我授权。
4. 重启服务，`GET /agent/skills` 会看到该技能、兼容状态和执行模式。

### 方式 B：写 Python-backed Skill

适合需要强权限边界、强确定性、数据库/向量库访问或复杂文件生成的能力。

1. 在 `app/skills/packages/<slug>/` 下创建 `SKILL.md`、`skill.json`、`templates/`、`references/`。
2. 在 `app/skills/` 新建执行层文件，写一个继承 `BaseSkill` 的类：
   ```python
   @register_skill
   class MySkill(BaseSkill):
       package_slug = "<slug>"

       def run(self, context, **kwargs):
           ...
   ```
3. 在 `app/skills/__init__.py` 加一行 import（触发装饰器注册）。
4. 补测试，跑 `uv run pytest`。

经验原则：先用 Codex-style package 快速验证工作流；当它需要稳定 I/O、严格权限、复杂
文件格式或性能优化时，再沉淀为 Python-backed Skill。
