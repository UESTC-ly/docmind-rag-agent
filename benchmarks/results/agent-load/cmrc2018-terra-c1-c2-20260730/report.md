# DocMind Agent 并发与压力评测报告

> 仅统计 Agent API 明确返回的 token/cost telemetry；未返回时保持 unavailable。

| 并发配置 / 实测峰值 | 场景执行数 | VTCR | P95 延迟 | 吞吐 | 请求 / 重试 | Provider Token | Provider Cost | Provider Model | 失败分类 | Gate |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 1 / 1 | 2 | 1.0000 | 176.677s | 0.0082 cases/s | 21.0 / 0.0 | 99942.0 (observed) | unavailable | gpt-5.6-terra | 无 | 通过 |
| 2 / 2 | 2 | 1.0000 | 142.667s | 0.0140 cases/s | 21.0 / 0.0 | 98192.0 (observed) | unavailable | gpt-5.6-terra | 无 | 通过 |

**压力发布门禁：通过**

## 失败凭据

- 无。
