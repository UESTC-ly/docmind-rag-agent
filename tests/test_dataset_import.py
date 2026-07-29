"""数据集导入器的纯解析逻辑测试。

只测 parse_*_rows（无 DB / embedding），用公开数据集格式的最小夹具验证：
语料池构建、relevant_chunk_ids 映射、去重、过滤无效样本、limit 截断。
"""

from app.services.evaluation.dataset_import import (
    corpus_fingerprint,
    parse_beir_rows,
    parse_cmrc_rows,
    parse_ms_marco_rows,
    source_snapshot_fingerprint,
)


def _ms_row(query, answers, texts, flags):
    return {
        "query": query,
        "answers": answers,
        "passages": {"passage_text": texts, "is_selected": flags},
    }


class TestParseMsMarco:
    def test_maps_relevant_to_global_index(self):
        # 单题：3 段 passage，第 2 段(下标1)被选中 → relevant=[1]
        rows = [_ms_row("q1", ["ans"], ["p0", "p1", "p2"], [0, 1, 0])]
        corpus, pending = parse_ms_marco_rows(rows, limit=10)
        assert corpus == ["p0", "p1", "p2"]
        assert pending[0]["relevant"] == [1]
        assert pending[0]["question"] == "q1"
        assert pending[0]["answer"] == "ans"

    def test_global_index_accumulates_across_questions(self):
        # 第二题的相关下标应基于已累积的语料池偏移
        rows = [
            _ms_row("q1", ["a1"], ["p0", "p1"], [1, 0]),
            _ms_row("q2", ["a2"], ["p2", "p3"], [0, 1]),
        ]
        corpus, pending = parse_ms_marco_rows(rows, limit=10)
        assert len(corpus) == 4
        assert pending[0]["relevant"] == [0]  # p0
        assert pending[1]["relevant"] == [3]  # p3，偏移 2 + 下标 1

    def test_skips_question_without_selected_passage(self):
        # 无 is_selected 的题应被跳过（否则 recall 恒 0）
        rows = [_ms_row("q1", ["a"], ["p0", "p1"], [0, 0])]
        corpus, pending = parse_ms_marco_rows(rows, limit=10)
        assert pending == []
        assert corpus == []

    def test_skips_question_without_answer(self):
        rows = [_ms_row("q1", [], ["p0"], [1])]
        _, pending = parse_ms_marco_rows(rows, limit=10)
        assert pending == []

    def test_respects_limit(self):
        rows = [_ms_row(f"q{i}", ["a"], ["p"], [1]) for i in range(5)]
        _, pending = parse_ms_marco_rows(rows, limit=2)
        assert len(pending) == 2


def _cmrc_row(question, context, answers):
    return {
        "question": question,
        "context": context,
        "answers": {"text": answers},
    }


class TestParseCmrc:
    def test_single_question_maps_to_context(self):
        rows = [_cmrc_row("谁开发的？", "某游戏由A和B开发", ["A和B"])]
        corpus, pending = parse_cmrc_rows(rows, limit=10)
        assert corpus == ["某游戏由A和B开发"]
        assert pending[0]["relevant"] == [0]
        assert pending[0]["answer"] == "A和B"

    def test_shared_context_is_deduplicated(self):
        # 两题共享同一 context → 语料池只存 1 份，都指向下标 0
        rows = [
            _cmrc_row("q1", "共享段落", ["a1"]),
            _cmrc_row("q2", "共享段落", ["a2"]),
        ]
        corpus, pending = parse_cmrc_rows(rows, limit=10)
        assert corpus == ["共享段落"]
        assert pending[0]["relevant"] == [0]
        assert pending[1]["relevant"] == [0]

    def test_distinct_contexts_get_distinct_indices(self):
        rows = [
            _cmrc_row("q1", "段落甲", ["a1"]),
            _cmrc_row("q2", "段落乙", ["a2"]),
        ]
        corpus, pending = parse_cmrc_rows(rows, limit=10)
        assert corpus == ["段落甲", "段落乙"]
        assert pending[1]["relevant"] == [1]

    def test_skips_empty_context_or_answer(self):
        rows = [
            _cmrc_row("q1", "", ["a"]),        # 空 context
            _cmrc_row("q2", "有内容", []),      # 空答案
        ]
        _, pending = parse_cmrc_rows(rows, limit=10)
        assert pending == []

    def test_respects_limit(self):
        rows = [_cmrc_row(f"q{i}", f"ctx{i}", ["a"]) for i in range(5)]
        _, pending = parse_cmrc_rows(rows, limit=3)
        assert len(pending) == 3


class TestParseBeir:
    def test_preserves_full_corpus_and_graded_qrels(self):
        corpus, pending = parse_beir_rows(
            [
                {"_id": "d1", "title": "Doc 1", "text": "alpha"},
                {"_id": "d2", "title": "", "text": "beta"},
                {"_id": "d3", "title": "Distractor", "text": "gamma"},
            ],
            [
                {"_id": "q1", "text": "alpha question"},
                {"_id": "q2", "text": "beta question"},
            ],
            [
                {"query-id": "q1", "corpus-id": "d1", "score": "2"},
                {"query-id": "q1", "corpus-id": "d2", "score": "1"},
                {"query-id": "q2", "corpus-id": "missing", "score": "1"},
            ],
            limit=10,
        )

        assert corpus == ["Doc 1\nalpha", "beta", "Distractor\ngamma"]
        assert len(pending) == 1
        assert pending[0]["external_id"] == "q1"
        assert pending[0]["relevant"] == [0, 1]
        assert pending[0]["chunk_qrels"] == {"0": 2, "1": 1}
        assert pending[0]["metadata"]["public_qrels"] == {"d1": 2, "d2": 1}

    def test_corpus_fingerprint_is_order_and_boundary_sensitive(self):
        first = corpus_fingerprint(["ab", "c"])
        assert first == corpus_fingerprint(["ab", "c"])
        assert first != corpus_fingerprint(["a", "bc"])
        assert first != corpus_fingerprint(["c", "ab"])

    def test_raw_source_snapshot_fingerprint_is_order_sensitive(
        self,
        tmp_path,
    ):
        first = tmp_path / "corpus.jsonl"
        second = tmp_path / "queries.jsonl"
        first.write_text("corpus", encoding="utf-8")
        second.write_text("queries", encoding="utf-8")

        fingerprint = source_snapshot_fingerprint([first, second])

        assert len(fingerprint) == 64
        assert fingerprint == source_snapshot_fingerprint([first, second])
        assert fingerprint != source_snapshot_fingerprint([second, first])
