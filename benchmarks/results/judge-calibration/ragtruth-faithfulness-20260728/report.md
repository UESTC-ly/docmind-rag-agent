# DocMind RAGTruth Judge 校准报告

- 结论：**不可作为发布门禁证据**
- Judge：`gpt-5.6-luna` / `faithfulness_v2`
- Split：`test`
- 原始快照：`d77689ec4b5f94382316320c832d20a747af85f31addce5d673f1e47ff4f8acc`
- 报告指纹：`a741fa589c94ea6681920d7b3044b1c28935b185511e039be372ec7c82c1468f`

## 汇总

| 指标 | 数值 |
|---|---:|
| 样本数 | 100 |
| 可用 Judge 结果 | 44 |
| Judge 不可用 | 56 |
| Coverage | 0.44 |
| Accuracy | 0.863636 |
| Precision | 0.75 |
| Recall | 0.6 |
| Specificity | 0.941176 |
| Balanced Accuracy | 0.770588 |

## 说明

- 本报告校准 generation-level faithfulness judge，不代表 claim citation judge 已校准。
- `implicit_true` 表示可能真实但未出现在上下文；DocMind 严格依据性契约仍判为 unsupported。
- 缺失或异常的 Judge 返回保持 unavailable，不会折算为 0 分或通过。
