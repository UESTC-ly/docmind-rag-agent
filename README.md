# DocMind — Agentic 文档智能助手

一个基于 **RAG + Agent 编排**的文档智能系统。用户上传文档后，Agent 通过 OpenAI
Function Calling 自主判断该调用哪些技能（Skills）来完成任务：知识库问答、思维导图、
关系图谱、报告生成、周报/PPT 文件产出、联网搜索，以及 v1.0.0 正式包含的
**Codex-style 通用 Skills 包**。系统还内置一套 **RAG 评估模块**，用检索指标
（hit_rate / MRR / recall / precision）和 LLM-as-judge 生成指标
（faithfulness / answer_relevancy）量化问答质量。

配套一个 **原生单页前端**（编辑/瑞士极简风，FastAPI 直接托管、零构建），覆盖
登录、流式对话、文档管理、评估看板四大界面。v2.0 额外提供 **Tauri 桌面 App**：保留
同一套前端与 FastAPI API，但以原生窗口、系统文件选择/保存和系统通知交互。后端
**191 个测试、94% 覆盖率**，接了 GitHub Actions CI。

**亮点**：ReAct Agent 编排 · Codex-style 通用 Skills · 可插拔 Python Skills ·
多路召回（向量 + 关键词 RRF 融合）· SSE 流式输出 · RAG 评估闭环 ·
结构化日志（request_id 全链路追踪）。

## 目录

- [架构总览](#架构总览)
- [技术栈](#技术栈)
- [核心设计](#核心设计)
- [快速启动](#快速启动)
- [桌面 App（v2.0）](#桌面-appv20)
- [使用流程](#使用流程)
- [API 一览](#api-一览)
- [Skills 技能系统](#skills-技能系统)
- [多路召回](#多路召回)
- [流式输出](#流式输出)
- [RAG 评估模块](#rag-评估模块)
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
           │  Agent Orchestrator    │   │  RAG 问答 (chat)        │
           │  ReAct 主循环          │   │  检索→拼Prompt→LLM      │
           │  Function Calling      │   └──────────┬─────────────┘
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
   │  PostgreSQL(元数据/分块)   Qdrant(向量)   Redis(队列)          │
   └───────────────────────────────────┬───────────────────────────┘
                                        │ .delay()
                            ┌───────────▼────────────┐
                            │  Celery Worker          │
                            │  解析→分块→向量化→写库   │
                            └─────────────────────────┘
```

文档上传后**不阻塞请求**：API 只落库并派发 Celery 任务，解析与向量化在后台完成，
前端轮询文档状态（pending → processing → completed / failed）。

## 技术栈

- **后端**：FastAPI + SQLAlchemy 2.0 (async) + Pydantic v2 + pydantic-settings
- **AI**：OpenAI 兼容 API（默认 `gpt-4o-mini` + `text-embedding-3-small`）+ Function Calling
- **存储**：PostgreSQL（元数据 + 分块文本）+ Qdrant（向量）+ Redis（Celery broker）
- **异步**：Celery（文档解析不阻塞 API）
- **鉴权**：JWT（python-jose）+ bcrypt 密码哈希（passlib）
- **文档解析**：pymupdf（PDF）+ python-docx（Word）+ 纯文本
- **联网搜索**：ddgs（DuckDuckGo，无需 API key）
- **评估数据集**：HuggingFace `datasets`（MS MARCO / CMRC 2018 / RAGTruth）

## 核心设计

### 双数据库引擎（async + sync）

FastAPI 用异步驱动，Celery 任务是同步函数——所以 `app/database.py` 同时暴露两套引擎：

| 引擎 | 驱动 | 用途 |
|---|---|---|
| `AsyncSessionLocal` | asyncpg | FastAPI 路由（经 `get_db` 依赖注入） |
| `SyncSessionLocal` | psycopg2 | Celery worker、评估脚本、`to_thread` 里的同步逻辑 |

同步 URL 由 `settings.sync_database_url` 自动从异步 URL 去掉 `+asyncpg` 得到。
**不要在异步路由里用同步 Session，反之亦然。**

### 异步 / 同步边界

所有 OpenAI 调用（LLM、embedding）与 Agent 主循环都是**同步阻塞**的。在异步路由里
必须用 `asyncio.to_thread(...)` 包一层，避免阻塞事件循环。参考 `services/rag_service.py`、
`services/agent_service.py`、`routers/evaluation.py`。

### 数据隔离

- 每个 Qdrant point 的 payload 都带 `user_id`，所有向量检索强制按 `user_id` 过滤。
- 所有按 id 查询的接口都校验资源归属（`WHERE user_id = current_user.id`），防止越权。

## 快速启动

### 0. 前置要求

| 平台 | 必需软件 | 说明 |
|---|---|---|
| macOS | Homebrew、Colima、Docker CLI/Compose、uv、Python 3.12 | 推荐用根目录 `./start.sh`，会自动转到 `scripts/start-macos-colima.sh` |
| Linux | Docker Engine、Docker Compose plugin 或 `docker-compose`、uv、Python 3.12 | 推荐用根目录 `./start.sh`，会自动转到 `scripts/start-linux-docker.sh` |
| Windows | Docker Desktop、PowerShell 5+、uv、Python 3.12 | 使用根目录 `start.ps1` 或 `scripts/start-windows.ps1` |

安装 `uv` 可参考：<https://docs.astral.sh/uv/>。Windows 建议在 PowerShell 中执行；
Linux 用户需确保当前用户有 Docker 权限，或自行在 Docker 命令前加 `sudo`。

## 桌面 App（v2.0）

桌面版使用 **Tauri 2** 把现有单页前端放进系统 WebView，不重写业务 UI；Rust 宿主负责
启动和停止 Docker Compose、Celery 与 FastAPI。首次运行需要本机已有 Docker/Compose、uv 和
Python 3.12；它们仍是 PostgreSQL、Redis、Qdrant 与 Python AI 后端的运行时依赖。

### 日常使用

从 GitHub Release 下载与当前操作系统匹配的桌面安装包后直接启动 `DocMind`。桌面 App 会：

1. 在 macOS 尝试启动 Colima（仅当 Docker 尚未运行且已安装 Colima）；
2. 启动 PostgreSQL / Redis / Qdrant；
3. 首次创建独立 Python 环境并安装 `requirements.txt`；
4. 用 `--pool=solo` 启动 Celery，再启动仅监听 `127.0.0.1:8000` 的 FastAPI；
5. 等待 `/health` 端口就绪后显示登录页。

配置与可写数据不放在安装包内：桌面端首次启动会从 `.env.example` 创建自己的 `.env`。填入真实
`OPENAI_API_KEY` 后重启 App 即可使用 AI 功能。

| 平台 | 配置目录 |
|---|---|
| macOS | `~/Library/Application Support/com.docmind.desktop/` |
| Windows | `%APPDATA%\\com.docmind.desktop\\` |
| Linux | `~/.config/com.docmind.desktop/` |

目录中包含 `.env`、`data/uploads/`、`data/skill_workspaces/` 和 `logs/api.log` / `logs/celery.log`。
删除安装包不会删除这些用户数据；数据库、Redis 与 Qdrant 则继续由 Docker volume 持久化。

### 开发与打包

```bash
cd desktop
npm install
npm run dev       # 启动 Tauri 窗口与受管后端
npm run build     # 在当前操作系统生成安装包
```

Tauri 只能在目标操作系统上原生签名/打包：macOS 生成 `.app/.dmg`，Windows 生成 `.msi/.exe`，
Linux 生成 `.AppImage/.deb`。发布工程可在对应操作系统或 CI runner 执行 `npm run build`。

桌面端保留网页入口：执行根目录 `./start.sh` 或 `start.ps1` 后仍可通过
`http://127.0.0.1:8000/` 使用。两种入口共享同一个后端，不应同时占用同一台机器的 `8000` 端口。

### 原生交互

- 文档页提供系统文件选择器，仍兼容网页拖拽/选择上传。
- 删除文档改用系统确认框；解析完成会发送系统通知。
- 周报、PPT 与通用 Skill 文件产出会打开系统“另存为”对话框，而不是浏览器下载栏。

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
7. 后台启动 Celery worker（统一 `--pool=solo`，macOS 额外设置 fork 安全环境变量）；
8. 前台启动 `uvicorn app.main:app --reload --port 8000`。

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

macOS：

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
  uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

Linux / Windows：

```bash
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

文档解析靠 Celery，**必须启动**，否则上传文档会停在 `pending`。

#### 4.4 启动 API 服务

```bash
uv run uvicorn app.main:app --reload --port 8000
```

浏览器访问：

- 前端：<http://127.0.0.1:8000/>
- Swagger API 文档：<http://127.0.0.1:8000/docs>

启动时会自动建表（`Base.metadata.create_all`，仅开发用；生产应改用 Alembic 迁移）。

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
5. 返回体含 `answer`（最终回复）、`artifacts`（思维导图/图谱/报告/周报/PPT 等结构化或文件产出）、
   `trace`（每步调了哪个技能、传了什么参数，用于展示"思考过程"）
6. `GET /agent/skills` 查看当前所有可用技能

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
| 文档 | `DELETE /documents/{id}` | 删除文档（连带删 Qdrant 向量 + 级联删分块） |
| 问答 | `POST /chat/` | 纯 RAG 问答（无 Agent） |
| 问答 | `POST /chat/stream` | RAG 流式问答（SSE，逐 token） |
| 问答 | `GET /chat/conversations` | 我的对话列表 |
| 问答 | `GET /chat/conversations/{id}` | 某对话的消息历史 |
| Agent | `POST /agent/chat` | 与 Agent 对话（Function Calling 编排） |
| Agent | `GET /agent/skills` | 列出所有可用技能 |
| 评估 | `POST /eval/datasets` | 从文档 LLM 反向出题生成数据集 |
| 评估 | `GET /eval/datasets` | 我的评估数据集列表 |
| 评估 | `POST /eval/runs` | 触发一次评估运行（同步执行） |
| 评估 | `GET /eval/runs/{id}` | 某次运行的聚合指标 |
| 评估 | `GET /eval/runs/{id}/details` | 逐条样本明细分数 |
| 健康 | `GET /health` | 健康检查 |

除 `register` / `login` / `health` 外，所有端点都需 `Authorization: Bearer <token>`。

## Skills 技能系统

技能是 Agent 的能力单元。v1.0.0 正式版采用**两条执行路径并存**：

```text
Python-backed Skill：BaseSkill 子类 + run()，适合强确定性/强业务边界
Codex-style Generic Skill：只需 SKILL.md，可选 templates / references / scripts / assets
```

也就是说，DocMind 现在既保留原来的 Python 技能，也能扫描
`app/skills/packages/<slug>/SKILL.md` 这种主流 Agent Skills 文件夹。若 package 没有
对应 Python 类，系统会自动注册为 `GenericPackageSkill`，由内部 ReAct runner 按
Markdown 指令规划并调用受控工具执行。

通用 runner 当前内置的动作能力：

- `read_skill_reference` / `read_skill_template`：渐进读取 package 资料；
- `list_files` / `read_file` / `write_file`：读写用户隔离工作区；
- `modify_code`：在隔离工作区写入/替换代码文件；
- `run_shell`：默认关闭，开启后仍需命令 allowlist，且不使用 `shell=True`；
- `call_mcp` / `use_browser_tool` / `use_app_tool`：预留 adapter 扩展点，未配置时安全失败。

安全边界：通用技能的文件读写默认只发生在
`skill_workspaces/user_<id>/<skill_slug>/`，不会直接改 DocMind 仓库源码；shell 默认关闭，
避免把 `/agent/chat` 变成远程代码执行入口。

| 技能名 | 执行模式 | 作用 | 产出 `type` |
|---|---|---|---|
| `search_knowledge_base` | Python | Qdrant 语义检索，回答文档问题优先用 | `kb_search` |
| `generate_mindmap` | Python | 抽取层级结构，输出 Mermaid mindmap | `mindmap` |
| `generate_relation_graph` | Python | 抽取实体+关系（GraphRAG），输出 nodes/edges + Mermaid | `relation_graph` |
| `generate_report` | Python | 多步：检索→大纲→逐节生成→汇总成文 | `report` |
| `generate_weekly_report` | Python + Package | 根据材料生成结构化中文周报，并提供 Markdown 下载 | `weekly_report` |
| `generate_presentation` | Python + Package | 根据材料生成演示文稿结构，并提供 `.pptx` 下载 | `presentation` |
| `web_search` | Python | DuckDuckGo 联网搜索，知识库答不了时补充 | `web_search` |
| `codex_note` | Generic Package | 示例 Codex-style 通用技能：按 SKILL.md 写 Markdown 笔记并打包下载 | `generic_skill` |

产出 `type` 属于 `mindmap` / `relation_graph` / `report` / `weekly_report` /
`presentation`，或带 `artifact_kind=file` 的通用技能结果，会被收进响应的
`artifacts`，供前端渲染或下载。

### 两种 Skill Package 目录结构

Python-backed package（例如周报）：

```text
app/skills/packages/weekly-report/
├── SKILL.md                         给 Agent/开发者看的技能说明
├── skill.json                       Function Calling metadata（name/description/parameters）
├── templates/
│   └── prompt.md                    LLM 提示词模板
└── references/
    └── writing-guide.md             写作规范、参考说明
```

Codex-style generic package（例如 `codex-note`）：

```text
app/skills/packages/codex-note/
├── SKILL.md                         必需，frontmatter 提供 name/description
├── templates/                       可选
│   └── note.md
├── references/                      可选
│   └── style.md
├── scripts/                         可选，shell 开启且 allowlist 命中时才可运行
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

### Agent 主循环（`app/agent/orchestrator.py`）

ReAct 式循环，最多 `agent_max_steps`（默认 6）步防死循环：

1. 把 系统提示 + 历史 + 用户消息 + 所有工具定义 交给 LLM
2. LLM 要么直接回答（结束），要么返回 `tool_calls`
3. 逐个执行技能，结果作为 `role: tool` 消息塞回上下文
4. 回到第 1 步，直到 LLM 给出最终答案或达到最大步数

对话历史由 `app/agent/memory.py` 管理，只带入最近 `WINDOW_SIZE`（10）条消息控制 token。

## RAG 评估模块

量化"检索准不准、答得好不好"，是本项目的一大特色。

### 两类数据集来源

- **LLM 反向出题**（`dataset_gen.py`）：随机抽文档分块，让 LLM 就该块内容生成
  `(question, answer, relevant_chunk_ids)` 三元组。走 `POST /eval/datasets`。
- **公开基准导入**（`dataset_import.py`）：导入业界标准数据集的 ground truth。
  关键在于评估检索指标依赖 Qdrant 里真实存在的向量，所以导入会走完整链路：
  `passages → Document + DocumentChunk(PG) → embed → upsert Qdrant → EvalDataset + EvalSample`。
  - **MS MARCO v2.1**：每题 passages 带 `is_selected` 标记，汇成语料池做检索评估
  - **CMRC 2018**：中文阅读理解，去重 context 建段落池，每题指向自己的段落
  - **RAGTruth**：幻觉标注数据集，用于单独验证 faithfulness judge，**不走检索流程**

### 指标

评估运行（`runner.py`）遍历样本，对每条跑一次 RAG，计算：

| 类别 | 指标 | 含义 |
|---|---|---|
| 检索 | `hit_rate` | top-k 内是否命中至少一个相关块 |
| 检索 | `mrr` | 第一个命中位置倒数的均值 |
| 检索 | `recall` | 命中相关块数 / 全部相关块数 |
| 检索 | `precision` | 命中相关块数 / 返回块数 |
| 生成 | `faithfulness` | LLM-as-judge：答案是否忠于检索片段（不编造） |
| 生成 | `answer_relevancy` | LLM-as-judge：答案是否切题、完整 |

检索指标是纯函数（`retrieval_metrics.py`，无 I/O，易单测）；生成指标用 LLM 打分
（`generation_judge.py`，温度 0 求稳定）。逐条明细存 `EvalResult`，聚合均值存 `EvalRun`。

### 导入公开数据集（CLI）

```bash
# 1. 下载 parquet（需先装 datasets pandas pyarrow）
uv run python data/download_datasets.py --ms-marco-limit 3000

# 2. 导入为某用户的评估数据集（需 PG / Qdrant / embedding 服务在线）
uv run python data/import_datasets.py --user-id 1
uv run python data/import_datasets.py --user-id 1 --only ms_marco --limit 50
```

导入完成后，用返回的 `dataset_id` 调 `POST /eval/runs` 触发评估。

## 多路召回

`app/services/retrieval.py`。单一稠密向量检索抓不住精确关键词（型号、专有名词），
所以并联一路**关键词检索**，用 **RRF（Reciprocal Rank Fusion，倒数排名融合）**
把两路结果融合：`RRF(d) = Σ 1/(k + rank_i(d))`（k=60）。RRF 只看排名不看原始分数
量纲，天然免归一化，是混合检索业界标配。

- 融合身份用 `(document_id, chunk_index)` 复合键，避免跨文档同序号块被误合并。
- `RETRIEVAL_MODE=hybrid`（默认）走多路；`dense` 退回纯向量检索。
- 关键词路用命中词数打分，实现简单、sqlite 也能跑（便于测试）；生产可换 PG 全文
  检索或 BM25，接口不变。

## 流式输出

`POST /chat/stream` 以 **Server-Sent Events** 逐 token 推送答案。事件流：
`event: meta`（会话 id + 来源）→ 多个 `event: token`（逐字）→ `event: done`，
流结束后落库。底层复用中转本就强制流式的特性（`llm_service.chat_completion_stream`
生成器逐块 yield）。前端用 `fetch` + `ReadableStream` 手解析 SSE（`EventSource`
无法带 `Authorization` 头，而本端点要 JWT）。

## 前端界面

`frontend/`，原生 HTML/CSS/JS 单页，**零构建、无 Node 工具链**，由 FastAPI
`StaticFiles` 直接托管（API 路由优先匹配，`/` 兜底返回单页）。编辑/瑞士极简风格：
纯白底 + 单一瑞士红强调 + sans 标题/衬线正文配对，跟随系统明暗。

四大界面：
- **登录/注册**：JWT 存 localStorage，401 自动清 token 回登录屏。
- **流式对话**：SSE 逐字浮现（带光标动画）、来源引用卡、左侧会话历史、Enter 发送。
- **文档管理**：拖拽上传、解析状态轮询（pending→completed 脉冲动画）、删除。
- **评估看板**：列数据集、一键触发运行、6 指标条形可视化。

工程：模块化 JS（`api/ui/chat/docs/eval/main`）、CSS 按 surface 分文件、
compositor 友好动画、`prefers-reduced-motion` 降级、键盘焦点环、无 `innerHTML` 注入。
启动 API 后浏览器访问 http://127.0.0.1:8000/ 即用。

## 测试与 CI

`tests/`，**191 个测试、覆盖率 94%**（`pytest` + `pytest-asyncio` + `pytest-cov`）。

- **纯函数单测**：检索指标、RRF 融合、密码哈希/JWT、数据集解析、分块——无 I/O，秒级。
- **服务单测**：评估 runner、dataset_gen、Celery 任务、8 个 Skills、LLM 流式聚合——
  mock 外部边界。
- **路由集成测**：用内存 sqlite 替 PG、mock 掉 Celery/embedding/LLM，真实 HTTP 打
  auth/documents/chat/agent/eval 全部端点（含 SSE 流式、评估双路径）。

`.coveragerc` 配了 `concurrency=greenlet,thread`，正确追踪 async 端点在事件循环里
执行的行。CI（`.github/workflows/ci.yml`）在 push/PR 时用 uv + Python 3.12 跑
`pytest --cov-fail-under=90`；测试全 mock + 内存库，**CI 无需 PG/Redis/Qdrant/
embedding**，必填配置喂假值即可。

```bash
uv run pytest --cov=app --cov-report=term-missing   # 本地跑测 + 覆盖率
```

## 配置项

全部集中在 `app/config.py`（pydantic-settings，从 `.env` 读取，大小写不敏感）。

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | 须用 `postgresql+asyncpg://` scheme |
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
| `QDRANT_COLLECTION` | | `docmind_chunks` | 向量集合名 |
| `UPLOAD_DIR` | | `./uploads` | 上传文件落盘目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | | `800` / `100` | 分块字符数 / 相邻块重叠 |
| `RETRIEVAL_TOP_K` | | `5` | 检索返回块数 |
| `RETRIEVAL_MODE` | | `hybrid` | `hybrid`=向量+关键词 RRF；`dense`=纯向量 |
| `RRF_K` | | `60` | RRF 融合常数 |
| `KEYWORD_CANDIDATES` | | `20` | 关键词召回候选数 |
| `AGENT_MAX_STEPS` | | `6` | Agent 主循环最大步数 |
| `SKILL_RUNNER_MAX_STEPS` | | `8` | Codex-style 通用 skill 内部工具循环最大步数 |
| `SKILL_WORKSPACE_DIR` | | `./skill_workspaces` | 通用 skill 的用户隔离文件工作区 |
| `SKILL_SHELL_ENABLED` | | `false` | 是否允许通用 skill 调用 shell；默认关闭 |
| `SKILL_SHELL_ALLOWED_COMMANDS` | | `echo,cat,...` | shell 开启后允许执行的命令名 allowlist |
| `SKILL_SHELL_TIMEOUT_SECONDS` | | `10` | 单次 shell 命令超时秒数 |
| `LOG_LEVEL` | | `INFO` | 日志级别 |
| `LOG_JSON` | | `true` | `true`=结构化 JSON（生产）；`false`=彩色文本（本地） |

> 对话与 embedding 可用不同服务商（如对话用 GPT、embedding 用 DashScope）；
> embedding 配置留空则复用对话的 key / base_url。
> **`bcrypt` 钉死在 4.0.1**：passlib 1.7.4 与 bcrypt 5.x 不兼容，升级时两者需同步。

## 项目结构

```
app/
├── main.py            FastAPI 入口（注册路由、启动建表）
├── config.py          全局配置单例
├── database.py        异步 + 同步双引擎，get_db 依赖
├── celery_app.py      Celery 实例
├── models/            ORM 模型（user / document / conversation / evaluation）
├── schemas/           Pydantic 请求/响应模型
├── routers/           路由层（auth / documents / chat / agent / evaluation）
├── services/          业务逻辑
│   ├── auth / document / rag / agent / embedding / llm / vector_store
│   ├── retrieval.py   多路召回（RRF 融合 + 关键词检索）
│   └── evaluation/    评估子模块（dataset_gen / dataset_import / runner /
│                      retrieval_metrics / generation_judge）
├── agent/             Agent 编排（orchestrator 主循环 / memory 对话记忆）
├── skills/            可插拔技能（Python Skills + Codex-style generic runner）
│   └── packages/      目录化 Skill Package：SKILL.md / templates / references / scripts / assets
├── tasks/             Celery 异步任务（document_tasks 解析流水线）
└── utils/             工具（security JWT / deps / file_parser / logging / middleware）

frontend/              原生单页前端（FastAPI 托管，零构建）
desktop/               Tauri 2 桌面壳、原生能力与打包配置
├── index.html
├── styles/            tokens / base / layout / components（按 surface 分文件）
└── js/                api / ui / chat / docs / eval / main（ES modules）

tests/                 191 个测试，pytest + 内存 sqlite + mock 外部边界
data/                  评估数据集下载 / 导入脚本 + parquet 缓存 + manifest.json
.github/workflows/     CI（pytest + 90% 覆盖率门槛）
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
   - `scripts/`：需要 shell 开启且命令 allowlist 命中后才能运行；
   - `assets/`：模板文件、图片等资源。
3. 重启服务，`GET /agent/skills` 会看到该技能，执行模式为 `generic_package`。

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
