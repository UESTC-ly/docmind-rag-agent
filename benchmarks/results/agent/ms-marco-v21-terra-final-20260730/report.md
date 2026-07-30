# DocMind Agent 任务评测报告

> 本报告评估 Agent 任务闭环，不把检索指标替代为 Agent 成功率。

## 汇总

| 指标 | 数值 |
|---|---:|
| 场景数 | 2 |
| Verified Task Completion Rate | 1.0000 |
| 任务成功率 | 1.0000 |
| 计划完成率 | 1.0000 |
| 工具选择召回率 | 1.0000 |
| 可验证产物率 | 1.0000 |
| 平均人工介入次数 | 0.0000 |
| 平均延迟 | 181.300s |
| P95 延迟 | 185.607s |
| 并发配置 / 实测峰值 | 1 / 1 |
| 端到端吞吐 | 0.0055 cases/s |
| Provider Token Usage | observed |
| Provider Cost | unavailable |
| Failure Classes | passed=2 |
| Provider Model | gpt-5.6-terra |
| Configured Request Model | gpt-5.6-terra |
| 公开证据来源合格 | 是 |
| Agent 发布门禁 | 通过 |

说明：Token 用量和货币成本只统计 Agent API 明确返回的 provider telemetry；缺失时保持 unavailable，不从模型名、提示词或延迟推算。

## 场景

- **ms-marco-corporation-verified-report**：通过（`passed`）
- **ms-marco-bradford-verified-report**：通过（`passed`）
