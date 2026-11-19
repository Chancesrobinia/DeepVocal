"""Load local knowledge sources and their reviewed graph relations."""

import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ChunkId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")]
Label = Annotated[str, Field(min_length=1, max_length=160)]


class Chunk(BaseModel):
    id: ChunkId
    section: Label | None = None
    text: str = Field(min_length=1, max_length=2000)
    page: int | None = Field(default=None, ge=1)
    page_label: Label | None = None
    entities: list[Label] = Field(default_factory=list)


class Relation(BaseModel):
    head: Label
    relation: Label
    tail: Label
    evidence_id: ChunkId


class Document(BaseModel):
    id: ChunkId
    title: Label
    edition: Label | None = None
    source_url: str | None = None
    use: Literal["local", "cloud"] = "local"
    chunks: list[Chunk] = Field(min_length=1)
    relations: list[Relation] = Field(default_factory=list)

    @field_validator("source_url")
    @classmethod
    def public_url(cls, value):
        if value:
            url = urlsplit(value)
            if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password:
                raise ValueError("Source URL must be a public HTTP or HTTPS address.")
        return value

    def passages(self):
        for chunk in self.chunks:
            parts = [part for part in (self.title, self.edition, chunk.section) if part]
            if chunk.page is not None:
                parts.append(f"PDF 第 {chunk.page} 页")
                if chunk.page_label and chunk.page_label != str(chunk.page):
                    parts.append(f"页标 {chunk.page_label}")
            yield {**chunk.model_dump(), "document_id": self.id, "title": chunk.section or self.title,
                   "source": " · ".join(parts), "source_url": self.source_url, "allow_cloud": self.use == "cloud"}


def load_corpus(directory):
    if directory is None:
        return [], []
    docs = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            docs.append(Document.model_validate(json.loads(path.read_text(encoding="utf-8-sig"))))
        except (ValueError, OSError):
            raise ValueError(f"知识库文件格式错误：{path.name}") from None
    doc_ids = [doc.id for doc in docs]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("知识库里存在重复的文档 ID。")
    passages = [passage for doc in docs for passage in doc.passages()]
    ids = {row["id"] for row in passages}
    if len(ids) != len(passages):
        raise ValueError("知识库里存在重复的段落 ID。")
    edges = []
    for doc in docs:
        for edge in doc.relations:
            if edge.evidence_id not in ids:
                raise ValueError(f"关系引用了不存在的段落：{edge.evidence_id}")
            edges.append((edge.head, edge.relation, edge.tail, edge.evidence_id))
    return passages, edges
