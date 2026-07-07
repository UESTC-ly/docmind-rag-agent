"""下载 RAG 评估数据集到本地 (parquet 格式)。

- MS MARCO v2.1  : 检索指标 (hit_rate / MRR / recall) —— 流式取子集
- CMRC 2018      : 中文阅读理解，带证据段落 —— 全量
- RAGTruth       : 幻觉标注，验证 faithfulness judge —— 全量

用法:
    uv run python data/download_datasets.py
    uv run python data/download_datasets.py --ms-marco-limit 5000
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from datasets import Dataset, load_dataset

DATA_DIR = Path(__file__).resolve().parent


def _save(rows: list[dict] | Dataset, out_dir: Path, split: str) -> int:
    """把一组样本存成 parquet，返回条数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = rows if isinstance(rows, Dataset) else Dataset.from_list(rows)
    path = out_dir / f"{split}.parquet"
    ds.to_parquet(str(path))
    print(f"    -> {path.relative_to(DATA_DIR)}  ({len(ds)} 条)")
    return len(ds)


def download_ms_marco(limit: int) -> dict:
    """流式取 validation 子集，避免拉全量 (数 GB)。"""
    print(f"[1/3] MS MARCO v2.1 (validation, 取前 {limit} 条)")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    rows = list(itertools.islice(ds, limit))
    n = _save(rows, DATA_DIR / "ms_marco", "validation")
    return {"repo": "microsoft/ms_marco", "config": "v2.1", "split": "validation", "count": n}


def download_cmrc2018() -> dict:
    """中文阅读理解，全量 (train + validation)。"""
    print("[2/3] CMRC 2018 (train + validation, 全量)")
    counts = {}
    for split in ("train", "validation"):
        ds = load_dataset("hfl/cmrc2018", split=split)
        counts[split] = _save(ds, DATA_DIR / "cmrc2018", split)
    return {"repo": "hfl/cmrc2018", "splits": counts}


def download_ragtruth() -> dict:
    """幻觉标注数据集，全量。"""
    print("[3/3] RAGTruth (wandb/RAGTruth-processed, 全量)")
    counts = {}
    dd = load_dataset("wandb/RAGTruth-processed")
    for split in dd:
        counts[split] = _save(dd[split], DATA_DIR / "ragtruth", split)
    return {"repo": "wandb/RAGTruth-processed", "splits": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 RAG 评估数据集")
    parser.add_argument("--ms-marco-limit", type=int, default=3000,
                        help="MS MARCO validation 取样条数 (默认 3000)")
    args = parser.parse_args()

    t0 = time.time()
    manifest = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}

    manifest["datasets"]["ms_marco"] = download_ms_marco(args.ms_marco_limit)
    manifest["datasets"]["cmrc2018"] = download_cmrc2018()
    manifest["datasets"]["ragtruth"] = download_ragtruth()

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成，用时 {time.time() - t0:.1f}s。清单: {manifest_path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
