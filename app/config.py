from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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
    keyword_candidates: int = 20  # 关键词召回候选数（融合前）

    # Agent
    agent_max_steps: int = 6  # Agent 主循环最大步数，防死循环
    # Codex-style 通用 Skill runner
    skill_runner_max_steps: int = 8  # 单个通用 skill 内部工具循环最大步数
    skill_workspace_dir: str = "./skill_workspaces"  # 通用 skill 文件读写工作区
    skill_shell_enabled: bool = False  # shell 工具默认关闭，避免聊天入口变成 RCE
    skill_shell_allowed_commands: str = "echo,cat,ls,pwd,grep,sed,python,python3,node,npm,uv"
    skill_shell_timeout_seconds: int = 10

    # 日志
    log_level: str = "INFO"
    log_json: bool = True  # True=结构化 JSON 日志（生产/可观测）；False=彩色文本（本地开发）

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def sync_database_url(self) -> str:
        """Celery worker 用同步驱动（psycopg2），从异步 URL 转换而来。"""
        return self.database_url.replace("+asyncpg", "")

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


# 全局单例，其他模块直接 from app.config import settings
settings = Settings()
