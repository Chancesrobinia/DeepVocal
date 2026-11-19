"""Convert an authorized PDF, text or Markdown file into the DeepVocal corpus format."""

import argparse
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from .corpus import Chunk, DATA_DIR, Document


def page_numbers(spec, count):
    if not spec:
        return list(range(1, count + 1))
    pages = set()
    for part in spec.split(","):
        bounds = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part.strip())
        if not bounds:
            raise ValueError("页码应为 1,3-5 这类格式。")
        start = int(bounds[1])
        end = int(bounds[2] or start)
        if not 1 <= start <= end <= count:
            raise ValueError(f"页码超出范围（PDF 共 {count} 页）。")
        pages.update(range(start, end + 1))
    return sorted(pages)


def sections(path, page_spec=None, default_section=None):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError:
            raise ValueError("读取 PDF 需要先安装 requirements-pdf.txt。") from None
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError("请使用已授权且可读取的 PDF。")
            labels = reader.page_labels
            for number in page_numbers(page_spec, len(reader.pages)):
                yield default_section, number, labels[number - 1], reader.pages[number - 1].extract_text()
        except PdfReadError:
            raise ValueError("无法读取 PDF 文本。") from None
        return
    if suffix not in (".txt", ".md"):
        raise ValueError("支持 PDF、TXT 和 Markdown 文件。")
    if page_spec:
        raise ValueError("--pages 仅用于 PDF。")
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".txt":
        yield default_section, None, None, text
        return
    buffers = []
    title = default_section
    for line in text.splitlines():
        match = re.fullmatch(r"#{1,6}\s+(.+)", line)
        if match:
            if buffers:
                yield title, None, None, "\n".join(buffers)
            title = match[1].strip()
            buffers = []
        else:
            buffers.append(line)
    if buffers:
        yield title, None, None, "\n".join(buffers)


def split_text(text, size=500, overlap=60):
    text = re.sub(r"[^\S\n]+", " ", text.replace("\r", "")).strip()
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind(mark, start + size * 2 // 3, end) for mark in "。！？；\n")
            if boundary >= 0:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end == len(text):
            break
        start = end - overlap


def write_document(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False) as pending:
        temp = Path(pending.name)
        pending.write(document.model_dump_json(indent=2))
        pending.write("\n")
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Import a document into DeepVocal.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--id", required=True, help="Stable lowercase document identifier")
    parser.add_argument("--title", required=True)
    parser.add_argument("--edition")
    parser.add_argument("--source-url")
    parser.add_argument("--section")
    parser.add_argument("--pages", help="One-based PDF page positions, for example 12-35,39")
    parser.add_argument("--allow-cloud", action="store_true", help="Allow external embedding and generation over these passages")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("DEEPVOCAL_DATA_DIR") or DATA_DIR) / "knowledge")
    args = parser.parse_args()
    skipped = []
    try:
        chunks = []
        for title, page, label, text in sections(args.path, args.pages, args.section):
            if page is not None and not text.strip():
                skipped.append(page)
            for piece in split_text(text):
                chunk_id = f"{args.id}-{len(chunks) + 1:05}"
                chunks.append(Chunk(id=chunk_id, text=piece, section=title, page=page, page_label=label))
        if not chunks:
            raise ValueError("没有可提取的文字。扫描版请先做 OCR 并核对页码。")
        doc = Document(id=args.id, title=args.title, edition=args.edition, source_url=args.source_url,
                       use="cloud" if args.allow_cloud else "local", chunks=chunks)
        output = args.output_dir / (doc.id + ".json")
        if output.exists():
            raise ValueError("文档 ID 已存在，另选 ID，或先备份并移走旧版本，以免覆盖已审阅的图谱关系。")
        write_document(output, doc)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"已写入 {len(chunks)} 个段落：{output.resolve()}")
    if skipped:
        print("未提取到文字的 PDF 页：" + ", ".join(map(str, skipped)))


if __name__ == "__main__":
    main()
