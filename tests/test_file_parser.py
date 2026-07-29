"""文档解析与分块工具测试。

txt 用真实临时文件；pdf/docx 的第三方库调用打桩；重点覆盖 split_text 的
段落合并、硬切、重叠、空输入等分支。
"""

import pytest

from app.utils import file_parser
from app.utils.file_parser import (
    extract_text,
    extract_text_with_locations,
    split_text,
    split_text_with_locations,
)


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


class TestSourceLocations:
    def test_text_locations_keep_paragraphs_and_character_spans(self, tmp_path):
        path = tmp_path / "source.md"
        path.write_text(
            "第一段第一行\n第一段第二行\n\n第二段",
            encoding="utf-8",
        )

        parsed = extract_text_with_locations(str(path))
        chunks = split_text_with_locations(parsed, chunk_size=20)

        assert parsed.text == "第一段第一行\n第一段第二行\n第二段"
        assert [(p.paragraph_index, p.char_start, p.char_end) for p in parsed.paragraphs] == [
            (1, 0, 13),
            (2, 14, 17),
        ]
        assert len(chunks) == 1
        assert chunks[0].paragraph_start == 1
        assert chunks[0].paragraph_end == 2
        assert parsed.text[chunks[0].char_start : chunks[0].char_end] == chunks[0].content

    def test_pdf_locations_report_real_one_based_pages(self, tmp_path):
        path = tmp_path / "source.pdf"
        document = file_parser.fitz.open()
        first = document.new_page()
        first.insert_text((72, 72), "first source")
        second = document.new_page()
        second.insert_text((72, 72), "second source")
        document.save(path)
        document.close()

        parsed = extract_text_with_locations(str(path))
        chunks = split_text_with_locations(parsed, chunk_size=20, chunk_overlap=0)

        assert [paragraph.page_number for paragraph in parsed.paragraphs] == [1, 2]
        assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [
            (1, 1),
            (2, 2),
        ]
        assert all(
            parsed.text[chunk.char_start : chunk.char_end] == chunk.content
            for chunk in chunks
        )
