"""Shared retrieval configuration for the API and evaluation CLI."""

from functools import lru_cache
import os

from .credentials import transform
from .knowledge import Retriever
from .model import Model


LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"


def local_embedding(cache_dir):
    @lru_cache(maxsize=1)
    def load():
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise RuntimeError("本地向量模型需要安装 requirements-rag.txt。") from None
        return TextEmbedding(model_name=LOCAL_MODEL, cache_dir=str(cache_dir), threads=2)

    def embed(texts):
        return list(load().embed(texts, batch_size=16))

    return embed


def saved_key(store, provider):
    saved = store.model_settings()
    if saved and saved["provider"] == provider and saved["protected_key"]:
        return transform(saved["protected_key"], protect=False).decode("ascii")
    return ""


def build_retriever(store, index_filename="knowledge.db"):
    uri = os.getenv("DEEPVOCAL_MILVUS_URI")
    local_uri = str((store.root / index_filename).resolve())
    if uri == "lite":
        uri = local_uri
    name = os.getenv("DEEPVOCAL_EMBEDDING_MODEL")
    provider = os.getenv("DEEPVOCAL_EMBEDDING_PROVIDER", "openai")
    local = None
    if os.getenv("DEEPVOCAL_LOCAL_EMBEDDING") == "1" or provider == "local":
        local = (local_embedding(store.root / "embedding-models"), LOCAL_MODEL, 512)

    embed = embedding_id = dimensions = None
    if provider == "local":
        embed, embedding_id, dimensions = local
        uri = local_uri
    elif uri and name:
        dimensions = int(value) if (value := os.getenv("DEEPVOCAL_EMBEDDING_DIMENSIONS")) else None
        key = os.getenv("DEEPVOCAL_EMBEDDING_API_KEY", "")
        if not key and os.getenv("DEEPVOCAL_EMBEDDING_USE_SAVED_KEY") == "1":
            key = saved_key(store, provider)
        if key or provider == "custom":
            model = Model(provider, name, key)
            embedding_id = name if provider == "dashscope" else f"{provider}:{name}"

            def embed(texts):
                return [vector for offset in range(0, len(texts), 10)
                        for vector in model.embed(texts[offset:offset + 10], dimensions)]

    reranker = None
    if rerank_name := os.getenv("DEEPVOCAL_RERANK_MODEL"):
        key = os.getenv("DEEPVOCAL_RERANK_API_KEY", "")
        if not key and os.getenv("DEEPVOCAL_RERANK_USE_SAVED_KEY") == "1":
            key = saved_key(store, "dashscope")
        if key:
            reranker = Model("dashscope", rerank_name, key)

    return Retriever(uri, os.getenv("DEEPVOCAL_MILVUS_TOKEN"), embed, embedding_id,
                     corpus_dir=store.root / "knowledge", dimensions=dimensions,
                     local_embed=local, local_uri=local_uri if local else None, reranker=reranker)
