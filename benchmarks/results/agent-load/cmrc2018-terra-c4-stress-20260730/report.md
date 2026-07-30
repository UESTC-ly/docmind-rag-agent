# DocMind Agent 并发与压力评测报告

> 仅统计 Agent API 明确返回的 token/cost telemetry；未返回时保持 unavailable。

| 并发配置 / 实测峰值 | 场景执行数 | VTCR | P95 延迟 | 吞吐 | 请求 / 重试 | Provider Token | Provider Cost | Provider Model | 失败分类 | Gate |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 4 / 4 | 4 | 1.0000 | 119.926s | 0.0334 cases/s | 38.0 / 0.0 | 168686.0 (observed) | unavailable | gpt-5.6-terra | 无 | 通过 |

**压力发布门禁：通过**

## 失败凭据

- 无。
