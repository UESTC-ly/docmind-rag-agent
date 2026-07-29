"""把外部标准数据集导入为 DocMind 的评估数据集。

与 dataset_gen.py（LLM 反向出题）互补：这里导入业界基准的 ground truth，
简历上"用公开基准评估检索质量"比"自生成"更有说服力。

落地方式（关键）：评估 runner 的检索指标依赖 Qdrant 里真实存在的向量，
所以导入必须走完整链路：
    passages → Document + DocumentChunk(PG) → embed → upsert Qdrant
              → EvalDataset + EvalSample(relevant_chunk_ids 指向相关块)

支持的数据集：
    - MS MARCO v2.1 : 每题 passages 带 is_selected 标记 → 语料级检索评估
    - CMRC 2018     : 多题共享 context → 去重段落，每题指向自己的段落

RAGTruth 不在此列：它是"给定 context + 已有回答 + 幻觉标注"，属于验证
faithfulness judge 的独立任务，不走检索流程。见 validate_faithfulness_judge()。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.evaluation import EvalCorpusDocument, EvalDataset, EvalSample
from app.services.embedding_service import embed_texts
from app.services.vector_store import upsert_chunks


@dataclass(frozen=True)
class PublicDatasetSpec:
    """Auditable provenance for a supported public benchmark snapshot."""

    key: str
    source_name: str
    source_uri: str
    source_version: str
    license_name: str
    language: str
    domain: str
    task_type: str
    authority: str


MS_MARCO_V21 = PublicDatasetSpec(
    key="ms-marco-v2.1",
    source_name="MS MARCO",
    source_uri="https://microsoft.github.io/msmarco/",
    source_version="2.1",
    license_name="MS MARCO non-commercial research terms",
    language="en",
    domain="web search",
    task_type="rag_qa",
    authority="Microsoft",
)

CMRC_2018 = PublicDatasetSpec(
    key="cmrc2018",
    source_name="CMRC 2018",
    source_uri="https://ymcui.com/cmrc2018/",
    source_version="2018",
    license_name="CC BY-SA 4.0",
    language="zh",
    domain="Wikipedia",
    task_type="rag_qa",
    authority="CMRC 2018 organizers",
)


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def corpus_fingerprint(chunk_texts: list[str]) -> str:
    """Hash exact ordered corpus contents, including chunk boundaries."""

    digest = hashlib.sha256()
    for index, text in enumerate(chunk_texts):
        payload = f"{index}:{len(text)}:".encode("utf-8") + text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def source_snapshot_fingerprint(paths: list[Path]) -> str:
    """Hash exact raw source files and logical boundaries in stable order."""

    digest = hashlib.sha256()
    for path in paths:
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(len(block).to_bytes(8, "big"))
                digest.update(block)
    return digest.hexdigest()


def parse_ms_marco_rows(rows, limit: int) -> tuple[list[str], list[dict]]:
    """纯解析：MS MARCO 原始行 → (corpus 语料池, pending 问题列表)。

    无 DB / embedding 副作用，可独立单测。规则见 import_ms_marco 文档。
    """
    corpus: list[str] = []      # 全局 passage 池，下标即 chunk_index
    pending: list[dict] = []    # 暂存问题，等语料池建好再落库

    for row_index, row in enumerate(rows):
        if len(pending) >= limit:
            break
        answers = [a for a in (row.get("answers") or []) if a.strip()]
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") or []
        flags = passages.get("is_selected") or []
        if not answers or not texts:
            continue

        # 该题相关 passage 的全局下标；无 selected 则跳过
        relevant = [len(corpus) + i for i, sel in enumerate(flags) if sel == 1]
        if not relevant:
            continue

        corpus.extend(texts)  # 所有 passage 都进池（含干扰项，才是真检索）
        pending.append(
            {
                "question": row["query"].strip(),
                "answer": answers[0],
                "relevant": relevant,
                "external_id": str(
                    row.get("query_id", row.get("id", row_index))
                ),
            }
        )

    return corpus, pending


def parse_cmrc_rows(rows, limit: int) -> tuple[list[str], list[dict]]:
    """纯解析：CMRC 原始行 → (corpus 去重段落池, pending 问题列表)。

    多题共享 context 时按文本去重，每题 relevant 指向自己的段落下标。
    """
    corpus: list[str] = []
    ctx_index: dict[str, int] = {}  # context 文本 → chunk_index，去重
    pending: list[dict] = []

    for row_index, row in enumerate(rows):
        if len(pending) >= limit:
            break
        context = (row.get("context") or "").strip()
        answers = (row.get("answers") or {}).get("text") or []
        answers = [a for a in answers if a.strip()]
        if not context or not answers:
            continue

        if context not in ctx_index:
            ctx_index[context] = len(corpus)
            corpus.append(context)

        pending.append(
            {
                "question": row["question"].strip(),
                "answer": answers[0],
                "relevant": [ctx_index[context]],
                "external_id": str(row.get("id", row_index)),
            }
        )

    return corpus, pending


def parse_beir_rows(
    corpus_rows,
    query_rows,
    qrel_rows,
    limit: int,
) -> tuple[list[str], list[dict]]:
    """Parse the standard BEIR JSONL/TSV contract without extra dependencies.

    All corpus rows are retained because cropping distractors changes the
    retrieval task. Only the deterministic query prefix is limited.
    """

    corpus: list[str] = []
    corpus_index: dict[str, int] = {}
    for row in corpus_rows:
        public_id = str(row.get("_id", row.get("id", ""))).strip()
        text = str(row.get("text", "")).strip()
        title = str(row.get("title", "")).strip()
        content = "\n".join(part for part in (title, text) if part)
        if not public_id or not content or public_id in corpus_index:
            continue
        corpus_index[public_id] = len(corpus)
        corpus.append(content)

    relevance: dict[str, dict[str, int]] = {}
    for row in qrel_rows:
        query_id = str(
            row.get("query-id", row.get("query_id", row.get("qid", "")))
        ).strip()
        corpus_id = str(
            row.get("corpus-id", row.get("corpus_id", row.get("docid", "")))
        ).strip()
        try:
            score = int(row.get("score", 0))
        except (TypeError, ValueError):
            continue
        if query_id and corpus_id in corpus_index and score > 0:
            relevance.setdefault(query_id, {})[corpus_id] = score

    pending: list[dict] = []
    for row in query_rows:
        if len(pending) >= limit:
            break
        query_id = str(row.get("_id", row.get("id", ""))).strip()
        question = str(row.get("text", row.get("query", ""))).strip()
        graded_qrels = relevance.get(query_id, {})
        if not query_id or not question or not graded_qrels:
            continue
        local_qrels = {
            str(corpus_index[public_id]): grade
            for public_id, grade in graded_qrels.items()
        }
        pending.append(
            {
                "question": question,
                "answer": "",
                "relevant": [int(index) for index in local_qrels],
                "external_id": query_id,
                "chunk_qrels": local_qrels,
                "metadata": {"public_qrels": graded_qrels},
            }
        )
    return corpus, pending


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def _read_qrels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        return [
            dict(zip(header, line.rstrip("\n").split("\t"), strict=False))
            for line in stream
            if line.strip()
        ]


def _materialize_corpus(
    db: Session,
    user_id: int,
    filename: str,
    chunk_texts: list[str],
) -> Document:
    """把一批文本落成 Document + DocumentChunk，向量化后写入 Qdrant。

    chunk_index 就是 chunk_texts 的下标（0-based），与后续 EvalSample
    的 relevant_chunk_ids 保持同一套编号。返回已 flush、带 id 的 Document。
    """
    if not chunk_texts:
        raise ValueError("语料为空，无法建文档")

    # 1. 建 Document（虚拟来源，file_path 标注数据集出处，便于溯源）
    document = Document(
        user_id=user_id,
        filename=filename,
        file_path=f"dataset://{filename}",
        status=DocumentStatus.COMPLETED,
        chunk_count=len(chunk_texts),
    )
    db.add(document)
    db.flush()  # 拿到 document.id

    # 2. 建 DocumentChunk（文本存 PG）
    for idx, content in enumerate(chunk_texts):
        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=idx,
                content=content,
            )
        )
    db.flush()

    # 3. 向量化 + 写 Qdrant（chunks 与 vectors 同序，chunk_index 即下标）
    vectors = embed_texts(chunk_texts)
    upsert_chunks(document.id, user_id, chunk_texts, vectors)

    return document


def _mark_public_document(
    document: Document,
    spec: PublicDatasetSpec,
    corpus: list[str],
) -> None:
    document.source_uri = spec.source_uri
    document.source_version = spec.source_version
    document.authority = spec.authority
    document.content_fingerprint = corpus_fingerprint(corpus)
    document.source_status = "current"


def import_ms_marco(
    db: Session,
    user_id: int,
    parquet_path: str,
    dataset_name: str = "MS MARCO v2.1",
    limit: int = 30,
    source_split: str = "validation",
) -> EvalDataset:
    """导入 MS MARCO：所有选中题的 passages 汇成一个语料池做检索评估。

    只保留「有答案 且 至少一个 passage 被标记 is_selected」的题——否则
    relevant_chunk_ids 为空，recall 恒 0，样本无意义。
    """
    from datasets import load_dataset  # 惰性导入：纯解析逻辑与测试不依赖此重型库

    raw = load_dataset("parquet", data_files=parquet_path, split="train")
    corpus, pending = parse_ms_marco_rows(raw, limit)

    if not pending:
        raise ValueError("MS MARCO 没有可用样本（缺答案或缺 selected passage）")

    document = _materialize_corpus(db, user_id, dataset_name, corpus)
    _mark_public_document(document, MS_MARCO_V21, corpus)
    dataset = _build_dataset(
        db,
        user_id,
        document.id,
        dataset_name,
        pending,
        corpus=corpus,
        spec=MS_MARCO_V21,
        split=source_split,
        transform_contract="ms_marco_passages_v1",
        raw_snapshot_fingerprint=source_snapshot_fingerprint(
            [Path(parquet_path)]
        ),
    )
    db.commit()
    db.refresh(dataset)
    return dataset


def import_cmrc2018(
    db: Session,
    user_id: int,
    parquet_path: str,
    dataset_name: str = "CMRC 2018",
    limit: int = 30,
    source_split: str = "validation",
) -> EvalDataset:
    """导入 CMRC 2018：去重 context 建段落池，每题指向自己的段落。"""
    from datasets import load_dataset  # 惰性导入：纯解析逻辑与测试不依赖此重型库

    raw = load_dataset("parquet", data_files=parquet_path, split="train")
    corpus, pending = parse_cmrc_rows(raw, limit)

    if not pending:
        raise ValueError("CMRC 2018 没有可用样本")

    document = _materialize_corpus(db, user_id, dataset_name, corpus)
    _mark_public_document(document, CMRC_2018, corpus)
    dataset = _build_dataset(
        db,
        user_id,
        document.id,
        dataset_name,
        pending,
        corpus=corpus,
        spec=CMRC_2018,
        split=source_split,
        transform_contract="cmrc_context_dedupe_v1",
        raw_snapshot_fingerprint=source_snapshot_fingerprint(
            [Path(parquet_path)]
        ),
    )
    db.commit()
    db.refresh(dataset)
    return dataset


def import_beir(
    db: Session,
    user_id: int,
    dataset_dir: str,
    *,
    dataset_name: str,
    source_name: str,
    source_uri: str,
    source_version: str,
    license_name: str,
    split: str = "test",
    language: str = "en",
    domain: str = "mixed",
    limit: int = 100,
) -> EvalDataset:
    """Import a complete standard-format BEIR corpus and a query prefix.

    Dataset-specific provenance is mandatory because BEIR redistributes many
    upstream corpora under different terms and explicitly does not grant one
    umbrella data license.
    """

    root = Path(dataset_dir)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    missing = [
        str(path)
        for path in (corpus_path, queries_path, qrels_path)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"BEIR 数据文件缺失: {', '.join(missing)}")
    if not all(
        value.strip()
        for value in (
            source_name,
            source_uri,
            source_version,
            license_name,
        )
    ):
        raise ValueError("BEIR 导入必须声明上游来源、版本和许可证")

    corpus, pending = parse_beir_rows(
        _read_jsonl(corpus_path),
        _read_jsonl(queries_path),
        _read_qrels(qrels_path),
        limit,
    )
    if not pending:
        raise ValueError("BEIR 没有可用的 query/qrels 样本")

    spec = PublicDatasetSpec(
        key=f"beir:{source_name}",
        source_name=source_name,
        source_uri=source_uri,
        source_version=source_version,
        license_name=license_name,
        language=language,
        domain=domain,
        task_type="retrieval",
        authority="upstream dataset owner",
    )
    document = _materialize_corpus(db, user_id, dataset_name, corpus)
    _mark_public_document(document, spec, corpus)
    dataset = _build_dataset(
        db,
        user_id,
        document.id,
        dataset_name,
        pending,
        corpus=corpus,
        spec=spec,
        split=split,
        transform_contract="beir_standard_v1",
        raw_snapshot_fingerprint=source_snapshot_fingerprint(
            [corpus_path, queries_path, qrels_path]
        ),
    )
    db.commit()
    db.refresh(dataset)
    return dataset


def _build_dataset(
    db: Session,
    user_id: int,
    document_id: int,
    name: str,
    pending: list[dict],
    *,
    corpus: list[str] | None = None,
    spec: PublicDatasetSpec | None = None,
    split: str | None = None,
    transform_contract: str | None = None,
    raw_snapshot_fingerprint: str | None = None,
) -> EvalDataset:
    """用暂存的问题列表建 EvalDataset + EvalSample。"""
    fingerprint = corpus_fingerprint(corpus) if corpus is not None else None
    transform_spec = (
        _canonical_json(
            {
                "contract": transform_contract,
                "corpus_items": len(corpus or []),
                "sample_count": len(pending),
            }
        )
        if transform_contract
        else None
    )
    dataset = EvalDataset(
        user_id=user_id,
        document_id=document_id,
        name=name,
        source_name=spec.source_name if spec else None,
        source_uri=spec.source_uri if spec else None,
        source_version=spec.source_version if spec else None,
        license_name=spec.license_name if spec else None,
        split=split,
        corpus_fingerprint=fingerprint,
        source_snapshot_fingerprint=raw_snapshot_fingerprint,
        transform_spec=transform_spec,
        language=spec.language if spec else None,
        domain=spec.domain if spec else None,
        task_type=spec.task_type if spec else None,
        label_source="public_ground_truth" if spec else "synthetic",
        release_eligible=spec is not None,
        metadata_json=(
            _canonical_json(
                {
                    "source_key": spec.key,
                    "authority": spec.authority,
                    "import_contract": "public_eval_dataset_v1",
                }
            )
            if spec
            else None
        ),
    )
    db.add(dataset)
    db.flush()

    for item in pending:
        relevant = item["relevant"]
        db.add(
            EvalSample(
                dataset_id=dataset.id,
                question=item["question"],
                ground_truth_answer=item["answer"],
                relevant_chunk_ids=_canonical_json(relevant),
                external_id=item.get("external_id"),
                chunk_qrels=_canonical_json(
                    item.get(
                        "chunk_qrels",
                        {str(chunk_id): 1 for chunk_id in relevant},
                    )
                ),
                answerable=bool(item.get("answerable", True)),
                expected_citations=_canonical_json(relevant),
                slice_tags=_canonical_json(
                    [spec.language, spec.task_type] if spec else []
                ),
                metadata_json=(
                    _canonical_json(item["metadata"])
                    if item.get("metadata")
                    else None
                ),
            )
        )
    if spec and fingerprint:
        db.add(
            EvalCorpusDocument(
                dataset_id=dataset.id,
                document_id=document_id,
                public_id=f"{spec.key}:{split or 'unspecified'}:corpus",
                source_uri=spec.source_uri,
                source_version=spec.source_version,
                content_fingerprint=fingerprint,
                authority=spec.authority,
                source_status="current",
                metadata_json=_canonical_json(
                    {
                        "license_name": spec.license_name,
                        "chunk_count": len(corpus or []),
                    }
                ),
            )
        )
    return dataset
