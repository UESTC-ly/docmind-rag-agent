# DocMind 检索回归评测报告

- 数据集：CMRC 2018 2018 (validation)
- 语料指纹：`9588f1f3a236b3497c736fcd540d45d64222c87290341511b94ae28b6d7bdb3a`
- 实验指纹：`38803ae514f6f7be5a197dcf63546498efcb16560cfeeee72e906fde7dec8cb2`
- 样本数：30

## 汇总

| Pipeline | Hit@K | MRR | Recall@K | Precision@K | MAP@K | nDCG@K | Badcase | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | 294.8 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | 312.3 |
| hybrid-rerank | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | 267.7 |

## Badcase 分类

- **dense**：无
- **hybrid**：无
- **hybrid-rerank**：无
