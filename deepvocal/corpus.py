"""Load local knowledge sources and their reviewed graph relations."""

from datetime import date
import json
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ChunkId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")]
Label = Annotated[str, Field(min_length=1, max_length=160)]


class Figure(BaseModel):
    id: ChunkId
    caption: Label
    description: str = Field(min_length=1, max_length=700)
    credit: str | None = Field(default=None, max_length=300)


class Chunk(BaseModel):
    id: ChunkId
    section: Label | None = None
    text: str = Field(min_length=1, max_length=2000)
    page: int | None = Field(default=None, ge=1)
    page_label: Label | None = None
    page_end: int | None = Field(default=None, ge=1)
    page_end_label: Label | None = None
    entities: list[Label] = Field(default_factory=list)
    summary: str | None = Field(default=None, min_length=1, max_length=700)
    figures: list[Figure] = Field(default_factory=list)

    @field_validator("figures")
    @classmethod
    def unique_figures(cls, figures):
        if len({figure.id for figure in figures}) != len(figures):
            raise ValueError("段落里存在重复的图版 ID。")
        return figures

    @model_validator(mode="after")
    def page_span(self):
        if self.page_end is not None and (self.page is None or self.page_end <= self.page):
            raise ValueError("跨页段落的结束页必须晚于起始页。")
        if self.page_end_label is not None and (self.page_end is None or self.page_label is None):
            raise ValueError("跨页页标需要起始页标与结束页。")
        return self


class Relation(BaseModel):
    head: Label
    relation: Label
    tail: Label
    evidence_id: ChunkId
    scope: Label | None = None
    valid_from: date | None = None
    valid_to: date | None = None

    @model_validator(mode="after")
    def validate_interval(self):
        if self.valid_from and self.valid_to and self.valid_from >= self.valid_to:
            raise ValueError("关系的起始日必须早于结束日。")
        return self


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
                pages = f"{chunk.page}–{chunk.page_end}" if chunk.page_end else str(chunk.page)
                parts.append(f"PDF 第 {pages} 页")
                if chunk.page_label and chunk.page_label != str(chunk.page):
                    label = f"{chunk.page_label}–{chunk.page_end_label}" if chunk.page_end_label else chunk.page_label
                    parts.append(f"页标 {label}")
            fields = chunk.model_dump(exclude_none=True, exclude={"figures"})
            if chunk.figures:
                fields["figures"] = [
                    {**figure.model_dump(exclude_none=True),
                     "url": f"/api/knowledge/figures/{chunk.id}/{figure.id}"}
                    for figure in chunk.figures
                ]
            yield {**fields, "document_id": self.id, "title": chunk.section or self.title,
                   "source": " · ".join(parts), "source_url": self.source_url, "allow_cloud": self.use == "cloud"}


def load_corpus(directory, *, structured_relations=False):
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
            edges.append(edge if structured_relations else (edge.head, edge.relation, edge.tail, edge.evidence_id))
    return passages, edges
