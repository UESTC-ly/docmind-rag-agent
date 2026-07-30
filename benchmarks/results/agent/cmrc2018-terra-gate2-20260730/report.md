# DocMind Agent 任务评测报告

> 本报告评估 Agent 任务闭环，不把检索指标替代为 Agent 成功率。

## 汇总

| 指标 | 数值 |
|---|---:|
| 场景数 | 2 |
| Verified Task Completion Rate | 0.0000 |
| 任务成功率 | 0.0000 |
| 计划完成率 | 0.1667 |
| 工具选择召回率 | 0.0000 |
| 可验证产物率 | 0.0000 |
| 平均人工介入次数 | 0.0000 |
| 平均延迟 | 16.288s |
| P95 延迟 | 23.497s |
| 并发配置 / 实测峰值 | 1 / 1 |
| 端到端吞吐 | 0.0614 cases/s |
| Provider Token Usage | partial |
| Provider Cost | unavailable |
| Failure Classes | agent_outcome_failure=2 |
| Provider Model | gpt-5.6-terra |
| Configured Request Model | gpt-5.6-terra |
| 公开证据来源合格 | 是 |
| Agent 发布门禁 | 未通过 |

说明：Token 用量和货币成本只统计 Agent API 明确返回的 provider telemetry；缺失时保持 unavailable，不从模型名、提示词或延迟推算。

## 场景

- **cmrc-sengoku-musou-verified-report**：失败（`unknown`）
  - 模型服务不可用，Agent 已安全终止
  - status=failed，期望 completed
  - 缺少成功 Skill：generate_verified_research_report、select_evaluated_rag_pipeline
  - 缺少产物：verified_research_report
  - 必需产物未通过质量门
  - 任务没有通过证据闭环质量门
  - Skill 执行顺序不符合要求：select_evaluated_rag_pipeline → generate_verified_research_report
  - 回答未覆盖公开标注答案关键词组：光荣 / Koei；ω-force / ω-Force / Omega Force
- **cmrc-gongchepu-verified-report**：失败（`unknown`）
  - 模型服务不可用，Agent 已安全终止
  - status=failed，期望 completed
  - 缺少成功 Skill：generate_verified_research_report、select_evaluated_rag_pipeline
  - 缺少产物：verified_research_report
  - 必需产物未通过质量门
  - 任务没有通过证据闭环质量门
  - Skill 执行顺序不符合要求：select_evaluated_rag_pipeline → generate_verified_research_report
  - 回答未覆盖公开标注答案关键词组：打击乐 / percussion；记谱 / notation
