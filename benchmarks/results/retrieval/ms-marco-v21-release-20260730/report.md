# DocMind 检索回归评测报告

- 数据集：MS MARCO 2.1 (validation)
- 语料指纹：`6c7b8e1118a0feda46290223c475896eb4c1df7a0f18fd6492dc773a82e0740c`
- 实验指纹：`94b0dfbd9b82180967f1310305fd8e45fe0b315b478071d8f124fcb0ddb3dc1e`
- 样本数：100

## 汇总

| Pipeline | Hit@K | MRR | Recall@K | Precision@K | MAP@K | nDCG@K | Badcase | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.8700 | 0.5383 | 0.8383 | 0.1780 | 0.5182 | 0.6033 | 37 | 272.2 |
| hybrid | 0.8900 | 0.5400 | 0.8633 | 0.1820 | 0.5273 | 0.6150 | 34 | 275.6 |
| hybrid-rerank | 0.9100 | 0.5412 | 0.8833 | 0.1860 | 0.5300 | 0.6221 | 29 | 271.4 |

## Badcase 分类

- **dense**：incomplete_recall=19, low_precision=13, poor_ranking=34, retrieval_miss=13
- **hybrid**：incomplete_recall=16, low_precision=11, poor_ranking=33, retrieval_miss=11
- **hybrid-rerank**：incomplete_recall=14, low_precision=9, poor_ranking=28, retrieval_miss=9
