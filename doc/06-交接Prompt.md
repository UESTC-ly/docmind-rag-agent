# 项目交接 Prompt

> 用途：开启一个**新的 AI 对话**接手 DocMind 时，把下面「Prompt 正文」整段复制粘贴给它。
> 它会引导新 AI 先读关键文档和代码、建立全貌，再按作者的协作偏好开展工作。
>
> 使用建议：粘贴前把最后的「本次任务」一行改成你当下要做的事。

---

## Prompt 正文（复制以下全部）

````
你将接手一个已有项目 DocMind，请先建立全貌再动手。不要凭猜测回答，一切以代码为准。

【项目位置】本地 clone 后的仓库目录（例如 ./docmind-rag-agent）
【仓库】https://github.com/UESTC-ly/docmind-rag-agent （公开，分支 main）

【第一步：按顺序读这些文件建立全貌，读完再开始任何工作】
1. doc/05-项目交接文档.md   —— 项目全貌、架构、当前状态、已知缺口、环境坑（最重要，先读它）
2. README.md                —— 架构、配置、启动方式、API 一览
3. app/database.py          —— async+sync 双 DB 引擎（理解并发模型的基础）
4. app/services/rag_service.py       —— RAG 核心链路（检索→Prompt→生成→流式）
5. app/agent/orchestrator.py         —— LangGraph 外层编排 + interrupt/checkpoint/resume
6. app/agent/run_lock.py             —— Redis/SQLite 运行租约 + TTL/heartbeat
7. app/agent/checkpoint_cleanup.py   —— checkpoint 生命周期与定期有界清理
8. app/services/retrieval.py         —— 多路召回（RRF 融合 + 关键词检索）
9. app/services/evaluation/runner.py —— 评估编排（检索指标 + LLM-judge）
如需更细的模块地图和"每个文件读什么"，见 doc/04-项目学习路线指引.md。

【一句话定位】
Agentic RAG 文档智能问答系统：上传文档→Agent 用 OpenAI Function Calling 自主编排
Python-backed 与 Codex-style 通用技能（问答/思维导图/关系图谱/报告/周报/PPT/联网等）
→流式回答附来源。内置 RAG 评估闭环（检索指标 hit_rate/MRR/recall/precision +
LLM-as-judge faithfulness/relevancy）。在线问答、Skills 与评估共用数据库关键词 + dense
候选、scored RRF 与 reranker。
技术栈：Python 3.12 · FastAPI async · SQLAlchemy 2.0 · PostgreSQL · Qdrant · Celery ·
Redis · 原生单页前端 + Tauri 自包含桌面运行时。GitHub Actions 覆盖 Python（90% 覆盖率门槛）、
JavaScript、Rust、Playwright E2E 与视觉回归；测试数量和即时结果以当前分支 CI 为准。
外层 Agent 已迁移到 LangGraph；v3.1 增加 Web Redis/桌面 SQLite 的跨 worker run lease，
并定期分批清理 checkpoint。Generic Skill 内部 ReAct 子图和完整 Multi-Agent 尚未迁移。

【必须遵守的关键约定（否则会引入 bug）】
- 双 DB 引擎：FastAPI 用 AsyncSessionLocal(asyncpg)，Celery 用 SyncSessionLocal(psycopg2)，
  绝不混用。
- 所有同步调用（OpenAI/embedding/Qdrant）在 async 路由里必须 asyncio.to_thread 包一层。
- Celery worker 在 macOS 必须 --pool=solo（否则原生扩展 fork 会 SIGSEGV）。
- Alembic 是唯一运行时 schema 管理路径；应用启动前执行 migration，禁止恢复 `create_all`。
- LLM 统一走 stream=True（中转 gzxsy.vip 强制流式，非流式会返回 str 报错）。
- 高风险副作用必须发生在 LangGraph `interrupt()` 之后；恢复节点会从头执行，禁止在
  interrupt 前放不可幂等写入。checkpoint 查询/恢复必须校验当前 user_id。
- 所有会推进 Agent graph 的 chat/resume/recover 必须先获取同一 run_id 的租约；423 是可重试
  冲突，503 表示锁后端不可用或所有权丢失。状态 GET 保持只读。清理前也必须获取 run lease。
- 改动后跑 `uv run pytest`（用 uv，不用 pip/conda）；push 前先 source .venv/bin/activate
  （pre-push hook 用 venv 的 pytest）。
- 数据隔离：所有 id 查询校验 user_id 归属，向量检索强制按 user_id 过滤。
- `.env` 含真实密钥、不入库；adapter、repository、script 等高权限能力默认关闭并使用 allowlist。

【启动方式】
macOS/Linux：./start.sh；Windows：powershell -ExecutionPolicy Bypass -File .\\start.ps1；浏览器 http://localhost:8000/

【作者与协作偏好——非常重要】
作者是 Python 初级、备战 2026 秋招，做这个项目是为了简历 + 面试讲解，要求真正掌握
每一行代码、能独立复述和修改。请：
- 概念先讲清原理，涉及核心逻辑时引导作者动手写，你负责审阅纠错，不要默默替他写完了事。
- 分段小步推进，每步给可运行的验证点，等反馈再继续，不要一次性倾倒大量内容。
- 报错时先要完整 Traceback 再对症下药，不盲改。排查接口用 curl -i 让输出清晰。
- 关键概念（JWT/异步/RAG/Agent/RRF）随代码顺带解释，并适时埋面试题。
- 修改前先读相关代码和测试，遵循既有分层与风格；改完补/更新对应测试。

【当前已知边界（若要做增强，从这里挑）】
脑图/关系图谱的超长文档全文 map-reduce；外部 MCP/browser/App bridge/HTTP reranker 的部署与
凭据轮换；Generic Skill 子图与完整 Multi-Agent 迁移；Web 多主机生产共享 checkpointer；macOS
对外安装包的 Developer ID 签名、公证和干净机器验收。Linux/Windows 安装包不在 v3.1
本地验收范围。已完成能力与运维责任详见 doc/05 第 7 节、
doc/09-v3.0.0-LangGraph实施与发布.md 和 doc/10-v3.1.0-分布式运行锁与生命周期维护.md。

【本次任务】
<在这里写清你这次要做什么，例如："带我精读 Agent 编排模块" 或 "给检索加 rerank">

先确认你已读完上面第一步列出的文件、能复述项目全貌和四条关键数据流（上传/问答/Agent/评估），
再开始本次任务。
````

---

## 使用说明

1. 打开一个全新的 AI 对话（新窗口 / 新 session）。
2. 复制上面 ```` 代码块内的全部内容粘贴进去。
3. 把最后「本次任务」那行改成你当前的具体需求。
4. 新 AI 会先读文档建立全貌、复述验证，再干活。

**为什么这样设计**：交接的核心风险是新 AI"不看代码就凭训练记忆瞎答"。这个 prompt 强制它
① 先读 `doc/05` 交接文档（全貌）→ ② 读 5 个核心代码文件（细节）→ ③ 复述验证（确认真读懂）
→ ④ 才动手。同时把"作者要真掌握、要小步带教"的协作偏好前置，避免新 AI 一上来大包大揽。
