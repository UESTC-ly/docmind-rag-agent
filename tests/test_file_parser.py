"""文档解析与分块工具测试。

txt 用真实临时文件；pdf/docx 的第三方库调用打桩；重点覆盖 split_text 的
段落合并、硬切、重叠、空输入等分支。
"""

import pytest

from app.utils import file_parser
from app.utils.file_parser import extract_text, split_text


class TestSplitText:
    def test_empty_returns_empty(self):
        assert split_text("") == []
        assert split_text("   \n  ") == []

    def test_short_text_single_chunk(self):
        assert split_text("短文本", chunk_size=800) == ["短文本"]

    def test_paragraphs_merged_within_limit(self):
        # 两段合计不超限 → 合并成一块，用换行连接
        text = "第一段\n第二段"
        chunks = split_text(text, chunk_size=800)
        assert chunks == ["第一段\n第二段"]

    def test_paragraphs_split_when_exceeding_limit(self):
        # 每段 6 字，chunk_size=8 → 放不下两段，切成两块
        chunks = split_text("aaaaaa\nbbbbbb", chunk_size=8)
        assert chunks == ["aaaaaa", "bbbbbb"]

    def test_oversized_paragraph_hard_split_with_overlap(self):
        # 单段 20 字符，chunk_size=10 overlap=2 → 步长 8，窗口硬切
        text = "x" * 20
        chunks = split_text(text, chunk_size=10, chunk_overlap=2)
        # 每块最长 10；覆盖到全部内容
        assert all(len(c) <= 10 for c in chunks)
        assert chunks[0] == "x" * 10


class TestExtractText:
    def test_txt_reads_content(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello 文档", encoding="utf-8")
        assert extract_text(str(f)) == "hello 文档"

    def test_md_reads_content(self, tmp_path):
        f = tmp_path / "a.md"
        f.write_text("# 标题", encoding="utf-8")
        assert extract_text(str(f)) == "# 标题"

    def test_unsupported_type_raises(self, tmp_path):
        f = tmp_path / "a.xyz"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            extract_text(str(f))

    def test_pdf_dispatch(self, monkeypatch):
        # 打桩 _extract_pdf，验证按扩展名正确分派
        monkeypatch.setattr(file_parser, "_extract_pdf", lambda p: "PDF内容")
        assert extract_text("doc.pdf") == "PDF内容"

    def test_docx_dispatch(self, monkeypatch):
        monkeypatch.setattr(file_parser, "_extract_docx", lambda p: "DOCX内容")
        assert extract_text("doc.docx") == "DOCX内容"
