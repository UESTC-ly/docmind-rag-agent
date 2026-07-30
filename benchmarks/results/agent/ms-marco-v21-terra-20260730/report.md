# DocMind Agent 任务评测报告

> 本报告评估 Agent 任务闭环，不把检索指标替代为 Agent 成功率。

## 汇总

| 指标 | 数值 |
|---|---:|
| 场景数 | 2 |
| Verified Task Completion Rate | 0.5000 |
| 任务成功率 | 0.5000 |
| 计划完成率 | 0.8333 |
| 工具选择召回率 | 0.7500 |
| 可验证产物率 | 0.5000 |
| 平均人工介入次数 | 0.0000 |
| 平均延迟 | 223.274s |
| P95 延迟 | 295.973s |
| 并发配置 / 实测峰值 | 1 / 1 |
| 端到端吞吐 | 0.0045 cases/s |
| Provider Token Usage | observed |
| Provider Cost | unavailable |
| Failure Classes | agent_outcome_failure=1, passed=1 |
| Provider Model | provider-unavailable |
| Configured Request Model | unavailable |
| 公开证据来源合格 | 是 |
| Agent 发布门禁 | 未通过 |

说明：Token 用量和货币成本只统计 Agent API 明确返回的 provider telemetry；缺失时保持 unavailable，不从模型名、提示词或延迟推算。

## 场景

- **ms-marco-corporation-verified-report**：通过（`unknown`）
- **ms-marco-lightning-verified-report**：失败（`unknown`）
  - 缺少成功 Skill：generate_verified_research_report
  - 必需产物未通过质量门
  - 任务没有通过证据闭环质量门
  - Skill 执行顺序不符合要求：select_evaluated_rag_pipeline → generate_verified_research_report
