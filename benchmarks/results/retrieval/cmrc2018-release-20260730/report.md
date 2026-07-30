# DocMind 检索回归评测报告

- 数据集：CMRC 2018 2018 (validation)
- 语料指纹：`f3563fb04a1f1aec86c9452ae3e63dd789ec83650b15e5e842246d4f1612f077`
- 实验指纹：`0df13dc770fe7c4d73dbbbb1397839f1d187652ad64f1f6a4920e05daa1a3528`
- 样本数：100

## 汇总

| Pipeline | Hit@K | MRR | Recall@K | Precision@K | MAP@K | nDCG@K | Badcase | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | 266.0 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | 270.7 |
| hybrid-rerank | 1.0000 | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 0 | 266.5 |

## Badcase 分类

- **dense**：无
- **hybrid**：无
- **hybrid-rerank**：无
