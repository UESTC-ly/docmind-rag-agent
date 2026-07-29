"""把下载好的 parquet 数据集导入为某用户的评估数据集（CLI）。

前置：先跑 download_datasets.py 下载 parquet；确保 PG / Qdrant / embedding
服务可用（导入会真实向量化并写 Qdrant）。

用法：
    uv run python data/import_datasets.py --user-id 1
    uv run python data/import_datasets.py --user-id 1 --only ms_marco --limit 50
    uv run python data/import_datasets.py --user-id 1 --only beir \
      --beir-dir data/beir/scifact --beir-name SciFact \
      --source-name SciFact --source-uri https://github.com/allenai/scifact \
      --source-version BEIR-snapshot --license-name MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许以脚本方式直接运行时找到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SyncSessionLocal  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.evaluation.dataset_import import (  # noqa: E402
    import_beir,
    import_cmrc2018,
    import_ms_marco,
)

DATA_DIR = Path(__file__).resolve().parent

IMPORTERS = {
    "ms_marco": (
        import_ms_marco,
        "ms_marco/validation.parquet",
        "MS MARCO v2.1",
        "validation",
    ),
    "cmrc2018": (
        import_cmrc2018,
        "cmrc2018/validation.parquet",
        "CMRC 2018",
        "validation",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="导入评估数据集")
    parser.add_argument("--user-id", type=int, required=True, help="目标用户 id")
    parser.add_argument(
        "--only",
        choices=[*IMPORTERS, "beir"],
        help="只导入指定数据集，默认全部（ms_marco + cmrc2018）",
    )
    parser.add_argument(
        "--limit", type=int, default=30, help="每个数据集导入的问题数（默认 30）"
    )
    parser.add_argument("--beir-dir", type=Path)
    parser.add_argument("--beir-name")
    parser.add_argument("--source-name")
    parser.add_argument("--source-uri")
    parser.add_argument("--source-version")
    parser.add_argument("--license-name")
    parser.add_argument("--split", default="test", help="BEIR qrels split")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="mixed")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit 必须为正整数")

    targets = [args.only] if args.only else list(IMPORTERS)

    with SyncSessionLocal() as db:
        if db.get(User, args.user_id) is None:
            sys.exit(f"用户 id={args.user_id} 不存在，请先注册或换一个 id")

        for key in targets:
            if key == "beir":
                required = {
                    "--beir-dir": args.beir_dir,
                    "--beir-name": args.beir_name,
                    "--source-name": args.source_name,
                    "--source-uri": args.source_uri,
                    "--source-version": args.source_version,
                    "--license-name": args.license_name,
                }
                missing = [flag for flag, value in required.items() if not value]
                if missing:
                    parser.error(
                        "BEIR 导入缺少参数: " + ", ".join(missing)
                    )
                print(
                    f"[导入] {args.beir_name} "
                    f"(split={args.split}, limit={args.limit}) …"
                )
                ds = import_beir(
                    db,
                    args.user_id,
                    str(args.beir_dir),
                    dataset_name=args.beir_name,
                    source_name=args.source_name,
                    source_uri=args.source_uri,
                    source_version=args.source_version,
                    license_name=args.license_name,
                    split=args.split,
                    language=args.language,
                    domain=args.domain,
                    limit=args.limit,
                )
                print(
                    f"    -> EvalDataset id={ds.id}，"
                    f"document_id={ds.document_id}，{len(ds.samples)} 条样本"
                )
                continue

            fn, rel_path, name, source_split = IMPORTERS[key]
            path = DATA_DIR / rel_path
            if not path.exists():
                print(f"[跳过] {name}: 找不到 {path}，请先运行 download_datasets.py")
                continue
            print(f"[导入] {name} (limit={args.limit}) …")
            ds = fn(
                db,
                args.user_id,
                str(path),
                name,
                args.limit,
                source_split,
            )
            n = len(ds.samples)
            print(f"    -> EvalDataset id={ds.id}，document_id={ds.document_id}，{n} 条样本")

    print("完成。可在评估接口用上述 dataset_id 触发运行。")


if __name__ == "__main__":
    main()
