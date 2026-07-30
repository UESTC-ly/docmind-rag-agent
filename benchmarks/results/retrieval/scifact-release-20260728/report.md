# DocMind 检索回归评测报告

- 数据集：SciFact BEIR scifact md5:5f7d1de60b170fc8027bb7898e2efca1 (test)
- 语料指纹：`556ea36c6e516071fb5851f44fe8e7a137dde697e0ddd8288bc44060a7db3771`
- 实验指纹：`86c011a475722b0a34f843b58f43b04ddde6ab8b3a0b48fd257ea0d744a33324`
- 样本数：100

## 汇总

| Pipeline | Hit@K | MRR | Recall@K | Precision@K | MAP@K | nDCG@K | Badcase | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.8900 | 0.7270 | 0.8677 | 0.1960 | 0.7086 | 0.7534 | 24 | 266.9 |
| hybrid | 0.8900 | 0.7220 | 0.8677 | 0.1960 | 0.7036 | 0.7498 | 24 | 275.1 |
| hybrid-rerank | 0.8700 | 0.7285 | 0.8452 | 0.1900 | 0.7081 | 0.7483 | 25 | 286.0 |

## Badcase 分类

- **dense**：incomplete_recall=16, low_precision=11, poor_ranking=21, retrieval_miss=11
- **hybrid**：incomplete_recall=16, low_precision=11, poor_ranking=21, retrieval_miss=11
- **hybrid-rerank**：incomplete_recall=19, low_precision=13, poor_ranking=21, retrieval_miss=13
