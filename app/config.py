from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ecom-cs-agent"
    app_env: str = "dev"
    default_shop_id: str = "demo-shop"
    kb_path: str = "app/knowledge_base/products.json"
    router_model_name: str = "Qwen3-1.7B"
    answer_model_name: str = "Qwen3-4B"
    embedding_model_name: str = "Qwen3-Embedding-0.6B"
    reranker_model_name: str = "Qwen3-Reranker-0.6B"
    max_tool_chain_steps: int = 4

    @property
    def resolved_kb_path(self) -> Path:
        path = Path(self.kb_path)
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parent.parent
        return project_root / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = _load_env_file()
    return Settings(
        app_name=env.get("APP_NAME", "ecom-cs-agent"),
        app_env=env.get("APP_ENV", "dev"),
        default_shop_id=env.get("DEFAULT_SHOP_ID", "demo-shop"),
        kb_path=env.get("KB_PATH", "app/knowledge_base/products.json"),
        router_model_name=env.get("ROUTER_MODEL_NAME", "Qwen3-1.7B"),
        answer_model_name=env.get("ANSWER_MODEL_NAME", "Qwen3-4B"),
        embedding_model_name=env.get("EMBEDDING_MODEL_NAME", "Qwen3-Embedding-0.6B"),
        reranker_model_name=env.get("RERANKER_MODEL_NAME", "Qwen3-Reranker-0.6B"),
        max_tool_chain_steps=int(env.get("MAX_TOOL_CHAIN_STEPS", "4")),
    )


def _load_env_file() -> dict[str, str]:
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    loaded = dict(os.environ)
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded.setdefault(key.strip(), value.strip())
    return loaded
