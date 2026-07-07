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

    # Agent
    agent_max_steps: int = 6  # Agent 主循环最大步数，防死循环

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


# 全局单例，其他模块直接 from app.config import settings
settings = Settings()
