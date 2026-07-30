# DocMind 检索回归评测报告

- 数据集：MS MARCO 2.1 (validation)
- 语料指纹：`7aa0afdbae10d474b5e3a02d91a4fd673c202e417315a828d0a7cf6f429160a0`
- 实验指纹：`2e936adced698d718ddfd1d6691781ec9efd0bab967415da4018467f9e942ab1`
- 样本数：30

## 汇总

| Pipeline | Hit@K | MRR | Recall@K | Precision@K | MAP@K | nDCG@K | Badcase | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.9667 | 0.5228 | 0.9278 | 0.1933 | 0.5017 | 0.6133 | 10 | 339.8 |
| hybrid | 0.9333 | 0.5161 | 0.8944 | 0.1867 | 0.4950 | 0.6004 | 10 | 260.4 |
| hybrid-rerank | 0.9333 | 0.5411 | 0.8944 | 0.1867 | 0.5200 | 0.6194 | 9 | 263.2 |

## Badcase 分类

- **dense**：incomplete_recall=3, low_precision=1, poor_ranking=9, retrieval_miss=1
- **hybrid**：incomplete_recall=4, low_precision=2, poor_ranking=9, retrieval_miss=2
- **hybrid-rerank**：incomplete_recall=4, low_precision=2, poor_ranking=8, retrieval_miss=2
