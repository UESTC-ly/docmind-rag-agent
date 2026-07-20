import os

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("DOCMIND_ENV_FILE", ".env"),
        case_sensitive=False,
    )

    # 数据库（异步，供 FastAPI 用）
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # 对话 LLM（OpenAI 兼容）
    openai_api_key: str
    openai_base_url: str | None = None  # 兼容中转，None 用官方地址
    chat_model: str = "gpt-4o-mini"

    # Embedding（可用与对话不同的独立服务）
    # 留空则复用上面的对话 API（key / base_url）
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # 必须与 embedding 模型实际维度一致
    embedding_batch_size: int = 10  # 部分服务(如 DashScope)单次批量有上限

    # Qdrant 向量库
    qdrant_url: str = "http://localhost:6333"
    # 桌面生产包使用 Qdrant 本地持久化；设置后不连接外部 Qdrant 服务。
    qdrant_path: str | None = None
    qdrant_collection: str = "docmind_chunks"

    # 文档处理
    upload_dir: str = "./uploads"
    chunk_size: int = 800  # 每块字符数
    chunk_overlap: int = 100  # 相邻块重叠字符数

    # RAG 检索
    retrieval_top_k: int = 5  # 检索返回的块数
    # 多路召回：dense=仅向量检索；hybrid=向量+关键词，RRF 融合
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60  # RRF 融合常数，业界经验值 60，越大越弱化排名差异
    dense_candidates: int = 20  # 稠密召回候选数（融合前）
    keyword_candidates: int = 20  # 关键词召回候选数（融合前）

    # 二阶段排序：local=确定性混合信号；http=远端 cross-encoder（失败回退 local）；
    # off=仅保留 RRF 顺序。候选上限防止远端请求和本地打分无界增长。
    reranker_mode: str = "local"
    reranker_candidate_limit: int = 40
    reranker_http_url: str | None = None
    reranker_http_api_key: str | None = None
    reranker_http_model: str | None = None
    reranker_http_timeout_seconds: int = 10
    reranker_rrf_weight: float = 0.45
    reranker_dense_weight: float = 0.25
    reranker_keyword_weight: float = 0.10
    reranker_lexical_weight: float = 0.20

    # 后台任务：Web 默认 Celery；桌面 bundle 使用进程内受控线程执行器。
    task_execution_mode: str = "celery"
    local_task_workers: int = 2
    evaluation_lease_seconds: int = 2100
    evaluation_task_soft_time_limit_seconds: int = 1740
    evaluation_task_time_limit_seconds: int = 1800

    # Agent
    agent_max_steps: int = 6  # Agent 主循环最大步数，防死循环
    # LangGraph SQLite checkpointer。Web 可改为独立持久化路径；桌面 sidecar 会把
    # 它覆盖到用户 data_dir，确保重启应用后仍可恢复等待审批的 Agent。
    agent_checkpoint_path: str = "./data/agent-checkpoints.sqlite3"
    # 逗号分隔的外层 Skill 名称；用于部署方追加必须人工审批的具体工具。
    agent_high_risk_skills: str = ""
    # Generic Skill 只要声明下列宿主能力，就在进入整个 package 前暂停审批。
    # 内部逐动作审批要等 Generic Skill 迁移为 LangGraph 子图后才能可靠实现。
    agent_high_risk_capabilities: str = "package_scripts,mcp,browser,app,repository"
    # Codex-style 通用 Skill runner
    skill_runner_max_steps: int = 8  # 单个通用 skill 内部工具循环最大步数
    skill_workspace_dir: str = "./skill_workspaces"  # 通用 skill 文件读写工作区
    # Legacy compatibility only. Production capability discovery always rejects
    # arbitrary shell access; audited package scripts use macOS confinement.
    skill_shell_enabled: bool = False
    skill_shell_allowed_commands: str = "echo,cat,ls,pwd,grep,sed"
    skill_shell_timeout_seconds: int = 10

    # Generic Skill capability gates.  Implemented capability does not imply
    # authority: host access remains opt-in for Web deployments.
    skill_package_scripts_enabled: bool = False
    skill_package_script_interpreters: str = "python,node,bash,powershell,swift"
    skill_package_script_env_allowlist: str = "PATH,HOME,TMPDIR,TEMP"
    skill_package_script_timeout_seconds: int = 30
    skill_repository_enabled: bool = False
    skill_repository_root: str | None = None
    skill_repository_max_chars: int = 40000

    # Production external adapters.
    # MCP JSON: {"server": {"url": "...", "tools": ["tool"],
    #                       "token_env": "MCP_TOKEN", "headers": {}}}
    skill_mcp_enabled: bool = False
    skill_mcp_servers_json: str = "{}"
    skill_mcp_timeout_seconds: int = 30
    skill_browser_enabled: bool = False
    skill_browser_headless: bool = True
    skill_browser_allowed_hosts: str = "127.0.0.1,localhost"
    skill_browser_timeout_seconds: int = 20
    skill_app_enabled: bool = False
    skill_app_bridge_url: str | None = None
    skill_app_bridge_token: str | None = None
    skill_app_allowed_actions_json: str = "{}"
    skill_app_timeout_seconds: int = 20
    skill_adapter_observation_max_chars: int = 40000
    skill_artifact_max_bytes: int = 20 * 1024 * 1024

    # 日志
    log_level: str = "INFO"
    log_json: bool = (
        True  # True=结构化 JSON 日志（生产/可观测）；False=彩色文本（本地开发）
    )

    # 桌面 sidecar / frozen runtime。
    docmind_desktop: bool = False
    docmind_frontend_dir: str | None = None
    alembic_config: str | None = None

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"dense", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be dense or hybrid")
        return normalized

    @field_validator("reranker_mode")
    @classmethod
    def validate_reranker_mode(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"off", "local", "http"}:
            raise ValueError("RERANKER_MODE must be off, local, or http")
        return normalized

    @field_validator("task_execution_mode")
    @classmethod
    def validate_task_execution_mode(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in {"celery", "local"}:
            raise ValueError("TASK_EXECUTION_MODE must be celery or local")
        return normalized

    @property
    def sync_database_url(self) -> str:
        """把 API 的异步 URL 映射为后台任务使用的同步驱动 URL。"""
        return self.database_url.replace("postgresql+asyncpg", "postgresql").replace(
            "sqlite+aiosqlite", "sqlite"
        )

    @property
    def resolved_embedding_key(self) -> str:
        """embedding 的 key，未单独配置则复用对话的 key。"""
        return self.embedding_api_key or self.openai_api_key

    @property
    def resolved_embedding_base_url(self) -> str | None:
        """embedding 的 base_url，未单独配置则复用对话的。"""
        return self.embedding_base_url or self.openai_base_url

    @property
    def skill_shell_allowed_command_set(self) -> set[str]:
        """把逗号分隔的 allowlist 转成集合，供受控 shell executor 使用。"""
        return {
            item.strip()
            for item in self.skill_shell_allowed_commands.split(",")
            if item.strip()
        }

    @property
    def skill_package_script_interpreter_set(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.skill_package_script_interpreters.split(",")
            if item.strip()
        }

    @property
    def skill_package_script_env_allowlist_set(self) -> set[str]:
        return {
            item.strip()
            for item in self.skill_package_script_env_allowlist.split(",")
            if item.strip()
        }

    @property
    def skill_browser_allowed_host_set(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.skill_browser_allowed_hosts.split(",")
            if item.strip()
        }

    @property
    def agent_high_risk_skill_set(self) -> set[str]:
        return {
            item.strip()
            for item in self.agent_high_risk_skills.split(",")
            if item.strip()
        }

    @property
    def agent_high_risk_capability_set(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.agent_high_risk_capabilities.split(",")
            if item.strip()
        }


# 全局单例，其他模块直接 from app.config import settings
settings = Settings()
