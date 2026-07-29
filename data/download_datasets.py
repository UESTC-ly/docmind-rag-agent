"""Download auditable public RAG evaluation snapshots to local-only storage.

- MS MARCO v2.1: passage retrieval labels, deterministic validation prefix.
- CMRC 2018: Chinese extractive QA with supporting contexts.
- RAGTruth: official source/response JSONL with human hallucination spans.

The application runtime does not depend on HuggingFace ``datasets``.  It is an
optional data-preparation dependency used only for the parquet sources.  The
RAGTruth path uses Python's standard library and downloads the official
repository files directly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent
RAGTRUTH_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "ParticleMedia/RAGTruth/main/dataset"
)
ALCE_BASE_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/ALCE/"
    "main/human_eval"
)
ARES_BASE_URL = (
    "https://raw.githubusercontent.com/stanford-futuredata/ARES/"
    "main/datasets/example_files"
)
BEIR_DATASETS = {
    "scifact": {
        "download_uri": (
            "https://public.ukp.informatik.tu-darmstadt.de/"
            "thakur/BEIR/datasets/scifact.zip"
        ),
        "md5": "5f7d1de60b170fc8027bb7898e2efca1",
        "source_name": "SciFact",
        "source_uri": "https://huggingface.co/datasets/BeIR/scifact",
        "source_version": "BEIR scifact snapshot",
        "license_name": "CC BY-SA 4.0",
        "split": "test",
    }
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _save(rows, out_dir: Path, split: str) -> dict:
    """Save HuggingFace rows as parquet and retain the exact file checksum."""

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "MS MARCO/CMRC 下载需要可选依赖："
            "uv pip install datasets pandas pyarrow"
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = rows if isinstance(rows, Dataset) else Dataset.from_list(rows)
    path = out_dir / f"{split}.parquet"
    dataset.to_parquet(str(path))
    print(f"    -> {path.relative_to(DATA_DIR)}  ({len(dataset)} 条)")
    return {
        "path": str(path.relative_to(DATA_DIR)),
        "count": len(dataset),
        "sha256": _sha256(path),
    }


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "MS MARCO/CMRC 下载需要可选依赖："
            "uv pip install datasets pandas pyarrow"
        ) from exc
    return load_dataset(*args, **kwargs)


def _download(url: str, path: Path, *, force: bool) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and path.is_file():
        print(f"    -> 复用 {path.relative_to(DATA_DIR)}")
        return {
            "url": url,
            "path": str(path.relative_to(DATA_DIR)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    request = Request(url, headers={"User-Agent": "DocMind-EvalOps/1.0"})
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(request, timeout=180) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    print(f"    -> {path.relative_to(DATA_DIR)}  ({path.stat().st_size} bytes)")
    return {
        "url": url,
        "path": str(path.relative_to(DATA_DIR)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a public snapshot without permitting path traversal."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(
                    f"unsafe archive member: {member.filename}"
                )
        bundle.extractall(destination)


def download_beir(dataset_name: str, *, force: bool) -> dict:
    """Download one allowlisted BEIR snapshot with upstream license metadata."""

    spec = BEIR_DATASETS[dataset_name]
    print(f"[BEIR] {dataset_name} ({spec['split']})")
    archive = DATA_DIR / "beir" / f"{dataset_name}.zip"
    snapshot = _download(
        str(spec["download_uri"]),
        archive,
        force=force,
    )
    actual_md5 = _md5(archive)
    if actual_md5 != spec["md5"]:
        raise ValueError(
            f"BEIR {dataset_name} checksum mismatch: "
            f"expected {spec['md5']}, got {actual_md5}"
        )

    root = DATA_DIR / "beir" / dataset_name
    if force and root.exists():
        shutil.rmtree(root)
    if not all(
        (root / relative).is_file()
        for relative in (
            "corpus.jsonl",
            "queries.jsonl",
            f"qrels/{spec['split']}.tsv",
        )
    ):
        _safe_extract_zip(archive, DATA_DIR / "beir")
    files = {
        relative: {
            "path": str((root / relative).relative_to(DATA_DIR)),
            "bytes": (root / relative).stat().st_size,
            "sha256": _sha256(root / relative),
        }
        for relative in (
            "corpus.jsonl",
            "queries.jsonl",
            f"qrels/{spec['split']}.tsv",
        )
    }
    return {
        **spec,
        "beir_name": dataset_name,
        "archive": {**snapshot, "md5": actual_md5},
        "files": files,
    }


def download_ms_marco(limit: int) -> dict:
    """Stream a deterministic validation prefix instead of several GB."""

    print(f"[MS MARCO] v2.1 validation，前 {limit} 条原始记录")
    rows = list(
        itertools.islice(
            _load_dataset(
                "microsoft/ms_marco",
                "v2.1",
                split="validation",
                streaming=True,
            ),
            limit,
        )
    )
    snapshot = _save(rows, DATA_DIR / "ms_marco", "validation")
    return {
        "source_name": "MS MARCO",
        "source_uri": "https://microsoft.github.io/msmarco/",
        "source_version": "2.1",
        "repository": "microsoft/ms_marco",
        "config": "v2.1",
        "split": "validation",
        "snapshot": snapshot,
    }


def download_cmrc2018() -> dict:
    """Download public train and validation splits for Chinese QA."""

    print("[CMRC 2018] train + validation")
    snapshots = {}
    for split in ("train", "validation"):
        snapshots[split] = _save(
            _load_dataset("hfl/cmrc2018", split=split),
            DATA_DIR / "cmrc2018",
            split,
        )
    return {
        "source_name": "CMRC 2018",
        "source_uri": "https://ymcui.com/cmrc2018/",
        "source_version": "2018",
        "repository": "hfl/cmrc2018",
        "snapshots": snapshots,
    }


def download_ragtruth(*, force: bool) -> dict:
    """Download the two official RAGTruth JSONL files without repackaging."""

    print("[RAGTruth] official repository JSONL")
    files = {
        filename: _download(
            f"{RAGTRUTH_BASE_URL}/{filename}",
            DATA_DIR / "ragtruth" / filename,
            force=force,
        )
        for filename in ("source_info.jsonl", "response.jsonl")
    }
    return {
        "source_name": "RAGTruth",
        "source_uri": "https://github.com/ParticleMedia/RAGTruth",
        "source_version": "repository snapshot",
        "license_name": "MIT",
        "files": files,
    }


def download_alce(*, force: bool) -> dict:
    """Download ALCE's released human citation and utility annotations."""

    print("[ALCE] official human evaluation JSON")
    files = {
        filename: _download(
            f"{ALCE_BASE_URL}/{filename}",
            DATA_DIR / "alce" / filename,
            force=force,
        )
        for filename in (
            "human_eval_citations_completed.json",
            "human_eval_utility_completed.json",
        )
    }
    return {
        "source_name": "ALCE",
        "source_uri": "https://github.com/princeton-nlp/ALCE",
        "source_version": "official main human_eval snapshot",
        "license_name": "MIT",
        "files": files,
    }


def download_ares(*, force: bool) -> dict:
    """Download ARES's released NQ validation labels for evaluator calibration."""

    print("[ARES] official NQ labeled evaluator validation TSV")
    filename = "nq_labeled_output.tsv"
    file_info = _download(
        f"{ARES_BASE_URL}/{filename}",
        DATA_DIR / "ares" / filename,
        force=force,
    )
    return {
        "source_name": "ARES",
        "source_uri": "https://github.com/stanford-futuredata/ARES",
        "source_version": "official main example_files snapshot",
        "license_name": "Apache-2.0",
        "files": {filename: file_info},
    }


def _load_existing_manifest_datasets(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(existing, dict):
        return {}
    datasets = existing.get("datasets")
    return dict(datasets) if isinstance(datasets, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="下载公开 RAG 评估数据集")
    parser.add_argument(
        "--only",
        choices=[
            "all",
            "ms_marco",
            "cmrc2018",
            "beir",
            "ragtruth",
            "alce",
            "ares",
        ],
        default="all",
    )
    parser.add_argument(
        "--beir-dataset",
        choices=sorted(BEIR_DATASETS),
        default="scifact",
        help="允许下载的 BEIR 子集（默认 scifact）",
    )
    parser.add_argument(
        "--ms-marco-limit",
        type=int,
        default=3000,
        help="MS MARCO validation 原始记录前缀（默认 3000）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖本地已有的 RAGTruth 原始快照",
    )
    args = parser.parse_args()
    if args.ms_marco_limit <= 0:
        parser.error("--ms-marco-limit 必须为正整数")

    started = time.time()
    manifest_path = DATA_DIR / "manifest.json"
    manifest = {
        "contract": "docmind_public_download_manifest_v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datasets": _load_existing_manifest_datasets(manifest_path),
    }
    targets = (
        ["ms_marco", "cmrc2018", "beir", "ragtruth", "alce", "ares"]
        if args.only == "all"
        else [args.only]
    )
    for target in targets:
        if target == "ms_marco":
            value = download_ms_marco(args.ms_marco_limit)
        elif target == "cmrc2018":
            value = download_cmrc2018()
        elif target == "beir":
            value = download_beir(args.beir_dataset, force=args.force)
        elif target == "alce":
            value = download_alce(force=args.force)
        elif target == "ares":
            value = download_ares(force=args.force)
        else:
            value = download_ragtruth(force=args.force)
        manifest["datasets"][target] = value

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"\n完成，用时 {time.time() - started:.1f}s。"
        f"清单: {manifest_path.relative_to(DATA_DIR)}"
    )


if __name__ == "__main__":
    main()
