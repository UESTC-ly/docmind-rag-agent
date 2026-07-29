"""Document extraction and chunking with stable source coordinates.

The legacy ``extract_text`` / ``split_text`` helpers remain available for
callers that only need text.  New ingestion uses the structured variants so a
chunk can be traced to extracted-text character offsets, paragraph numbers,
and, for PDFs, the page returned by PyMuPDF.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf
from docx import Document as DocxDocument


LOCATOR_VERSION = "extracted_text_v1"


@dataclass(frozen=True)
class ParsedParagraph:
    """One non-empty extracted paragraph in the canonical text coordinate space."""

    content: str
    paragraph_index: int
    char_start: int
    char_end: int
    page_number: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Canonical extracted text plus its human-readable source coordinates."""

    text: str
    paragraphs: tuple[ParsedParagraph, ...]
    locator_version: str = LOCATOR_VERSION


@dataclass(frozen=True)
class LocatedChunk:
    """A retrieval chunk with optional page/paragraph/character provenance."""

    content: str
    page_start: int | None
    page_end: int | None
    paragraph_start: int | None
    paragraph_end: int | None
    char_start: int | None
    char_end: int | None
    locator_version: str = LOCATOR_VERSION


def extract_text(file_path: str) -> str:
    """根据扩展名提取文档纯文本。不支持的类型抛 ValueError。"""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix in (".docx", ".doc"):
        return _extract_docx(file_path)
    if suffix in (".txt", ".md"):
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"不支持的文件类型: {suffix}")


def extract_text_with_locations(file_path: str) -> ParsedDocument:
    """Extract canonical text with coordinates usable by the Evidence Navigator.

    Character offsets are offsets in the normalized extracted text returned by
    this function, not byte offsets in the original binary.  PDF page numbers
    are exact PyMuPDF page numbers (one based); its paragraphs are text blocks,
    which is the finest stable structure exposed by the current extractor.
    """

    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        source_paragraphs = _extract_pdf_paragraphs(file_path)
    elif suffix in (".docx", ".doc"):
        source_paragraphs = _extract_docx_paragraphs(file_path)
    elif suffix in (".txt", ".md"):
        source_paragraphs = _text_paragraphs(
            Path(file_path).read_text(encoding="utf-8", errors="ignore")
        )
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
    return _build_parsed_document(source_paragraphs)


def _extract_pdf(file_path: str) -> str:
    parts: list[str] = []
    with fitz.open(file_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def _extract_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_pdf_paragraphs(file_path: str) -> list[tuple[str, int | None]]:
    """Return PDF text blocks with their one-based PyMuPDF page number."""

    paragraphs: list[tuple[str, int | None]] = []
    with fitz.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks")
            for block in blocks:
                text = _normalize_paragraph(str(block[4]))
                if text:
                    paragraphs.append((text, page_number))
            if not blocks:
                paragraphs.extend(
                    (text, page_number)
                    for text, _ in _text_paragraphs(page.get_text())
                )
    return paragraphs


def _extract_docx_paragraphs(file_path: str) -> list[tuple[str, int | None]]:
    document = DocxDocument(file_path)
    return [
        (text, None)
        for paragraph in document.paragraphs
        if (text := _normalize_paragraph(paragraph.text))
    ]


def _text_paragraphs(text: str) -> list[tuple[str, int | None]]:
    paragraphs: list[tuple[str, int | None]] = []
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)
            continue
        if lines:
            paragraphs.append(("\n".join(lines), None))
            lines = []
    if lines:
        paragraphs.append(("\n".join(lines), None))
    return paragraphs


def _normalize_paragraph(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _build_parsed_document(
    source_paragraphs: list[tuple[str, int | None]],
) -> ParsedDocument:
    paragraphs: list[ParsedParagraph] = []
    parts: list[str] = []
    offset = 0
    for raw_content, page_number in source_paragraphs:
        content = _normalize_paragraph(raw_content)
        if not content:
            continue
        if parts:
            offset += 1  # The canonical document joins paragraphs with one newline.
        char_start = offset
        char_end = char_start + len(content)
        parts.append(content)
        paragraphs.append(
            ParsedParagraph(
                content=content,
                paragraph_index=len(paragraphs) + 1,
                char_start=char_start,
                char_end=char_end,
                page_number=page_number,
            )
        )
        offset = char_end
    return ParsedDocument(text="\n".join(parts), paragraphs=tuple(paragraphs))


def split_text(
    text: str, chunk_size: int = 800, chunk_overlap: int = 100
) -> list[str]:
    """把长文本切成带重叠的块。

    先按段落合并，尽量在自然边界切分；单段超长时按字符硬切。
    """
    if not text.strip():
        return []

    # 先按段落粗切
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # 当前块加上这段仍不超限 → 合并
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # 单段就超长，硬切
            if len(para) > chunk_size:
                chunks.extend(_hard_split(para, chunk_size, chunk_overlap))
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def split_text_with_locations(
    document: ParsedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[LocatedChunk]:
    """Split a parsed document while preserving the source span of every chunk.

    The content boundary behavior intentionally mirrors ``split_text``.  A
    character span may include the original newline between two paragraphs;
    paragraph and page ranges make that explicit to callers.
    """

    if not document.paragraphs:
        return []

    chunks: list[LocatedChunk] = []
    current: list[ParsedParagraph] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current, current_length
        if not current:
            return
        page_numbers = [p.page_number for p in current if p.page_number is not None]
        chunks.append(
            LocatedChunk(
                content="\n".join(paragraph.content for paragraph in current),
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                paragraph_start=current[0].paragraph_index,
                paragraph_end=current[-1].paragraph_index,
                char_start=current[0].char_start,
                char_end=current[-1].char_end,
                locator_version=document.locator_version,
            )
        )
        current = []
        current_length = 0

    for paragraph in document.paragraphs:
        candidate_length = current_length + len(paragraph.content) + (1 if current else 0)
        if candidate_length <= chunk_size:
            current.append(paragraph)
            current_length = candidate_length
            continue

        flush_current()
        if len(paragraph.content) > chunk_size:
            step = max(chunk_size - chunk_overlap, 1)
            for start in range(0, len(paragraph.content), step):
                content = paragraph.content[start : start + chunk_size]
                chunks.append(
                    LocatedChunk(
                        content=content,
                        page_start=paragraph.page_number,
                        page_end=paragraph.page_number,
                        paragraph_start=paragraph.paragraph_index,
                        paragraph_end=paragraph.paragraph_index,
                        char_start=paragraph.char_start + start,
                        char_end=paragraph.char_start + start + len(content),
                        locator_version=document.locator_version,
                    )
                )
        else:
            current = [paragraph]
            current_length = len(paragraph.content)

    flush_current()
    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符窗口硬切，步长 = chunk_size - overlap。"""
    step = max(chunk_size - overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]
