"""文档解析与分块工具。

支持 PDF(pymupdf)、Word(python-docx)、纯文本。
分块策略：按字符数滑动窗口切分，相邻块重叠一部分，避免语义在边界被切断。
"""

from pathlib import Path

import fitz  # pymupdf
from docx import Document as DocxDocument


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


def _extract_pdf(file_path: str) -> str:
    parts: list[str] = []
    with fitz.open(file_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def _extract_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


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


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符窗口硬切，步长 = chunk_size - overlap。"""
    step = max(chunk_size - overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]
