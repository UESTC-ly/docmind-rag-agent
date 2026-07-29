# Agent 任务评测

该评测衡量任务成功、计划完成、工具选择、人工介入、恢复事件、产物完成和质量门，
不使用 RAG 命中率代替 Agent 成功率。

场景文件必须使用公开数据集或公开文档，保留来源 URL、版本、许可证、split、
语料 SHA-256、转换说明和公开样本 ID。`scenarios.example.json` 固定为实际
SciFact 公开快照的 provenance；它不包含本机 `document_id`。运行时会请求
`/eval/datasets`，只接受与 `document_selector` 完全匹配、且标记为
`public_ground_truth` / `release_eligible` 的唯一语料。零哈希、硬编码文档 ID、
缺失 provenance 或多份同指纹语料都会 fail closed。

```bash
DOCMIND_BENCHMARK_TOKEN="<登录后 JWT>" \
uv run python scripts/benchmark_agent.py \
  --scenarios benchmarks/agent/scenarios.example.json \
  --concurrency 2 \
  --output /tmp/docmind-agent-eval.json \
  --markdown /tmp/docmind-agent-eval.md
```

结果 JSON 保存场景套件指纹、公开来源、每个场景的压缩请求、Agent plan、trace、
产物验证和失败原因，并报告 Verified Task Completion Rate、逐场景延迟、端到端
吞吐和实际 in-flight 并发。下载内容会被剔除，避免把大型 base64 文件写入评测
证据。

内置两个不同的 SciFact 人工标注 query，且不锁定具体 Skill；门禁要求 Agent 先调用
公开回归评测管线选择器，再生成经过证据门验证的研究报告。因此该演示评测的是
Agent 自主规划和证据交付闭环，而不是直接调用一个固定 RAG 函数。

`release_evidence_eligible=true` 只表示公开数据 provenance 完整；
`release_gate_passed=true` 才表示所有场景均完成并通过证据门。兼容字段
`release_eligible` 与严格的 `release_gate_passed` 保持一致。

Provider token usage 和货币成本只有在 `/agent/chat` 明确返回对应 telemetry 时才会
汇总。当前标准 Agent API 不提供该字段，因此报告会输出 `unavailable`；脚本不会从
模型名、tokenizer 估算或请求耗时反推出成本。

完整公开演示前置为已启动 FastAPI、Qdrant、embedding 服务，以及已导入与上述指纹
一致的 SciFact 语料。导入完成后，上述单条命令会自动定位该用户的公开文档并运行
Agent 验收，不需要手填数据库 document ID。
