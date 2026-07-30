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
  --timeout-seconds 600 \
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

仓库同时提供两组扩展任务，均绑定本地导入时产生的公开语料指纹，不写死数据库 ID：

| 场景文件 | 公开来源 | 任务数 | Agent 验收重点 |
|---|---|---:|---|
| `scenarios.example.json` | BEIR SciFact test | 2 | 科学声明核验、管线选择、逐结论证据门 |
| `scenarios.ms_marco_v21.json` | MS MARCO v2.1 validation | 2 | 英文公开问答、标准答案关键词、证据不足拒答 |
| `scenarios.cmrc2018.json` | CMRC 2018 validation | 2 | 中文公开问答、标准答案关键词、中文证据交付 |

MS MARCO 与 CMRC 并不是把检索命中直接计作 Agent 成功。每个任务仍必须完成
`select_evaluated_rag_pipeline → generate_verified_research_report`，通过产物和
证据门，并覆盖公开人工标注答案；任一条件失败都会进入 Badcase。

`release_evidence_eligible=true` 只表示公开数据 provenance 完整；
`release_gate_passed=true` 才表示所有场景均完成并通过证据门。兼容字段
`release_eligible` 与严格的 `release_gate_passed` 保持一致。

Provider token usage、请求/重试计数、provider-reported model 和货币成本只有在
`/agent/chat` 返回对应 telemetry 时才会汇总。Agent API 会透传运行期实际观测值；
若上游响应没有 token、模型名或带币种的成本字段，对应指标保持 `unavailable` 或
`partial`。脚本不会从模型名、tokenizer 估算，也不会从请求耗时反推出成本。

完整公开演示前置为已启动 FastAPI、Qdrant、embedding 服务，以及已导入与上述指纹
一致的 SciFact 语料。导入完成后，上述单条命令会自动定位该用户的公开文档并运行
Agent 验收，不需要手填数据库 document ID。

`--timeout-seconds` 是单个端到端 Agent 请求的等待上限；它应覆盖多轮规划、检索、
长上下文报告生成与逐结论依据性校验，而不是只按一次模型调用的耗时设置。

## 并发与压力门禁

同一套冻结场景可以通过黑盒 HTTP 压力工具分阶段运行：

```bash
DOCMIND_BENCHMARK_TOKEN="<登录后 JWT>" \
uv run python scripts/benchmark_agent_load.py \
  --scenarios benchmarks/agent/scenarios.example.json \
  --concurrency 1,2,4 \
  --repetitions 2 \
  --warmup-repetitions 1 \
  --min-vtcr 1 \
  --max-p95-seconds 600 \
  --require-token-usage \
  --output /tmp/docmind-agent-load.json \
  --markdown /tmp/docmind-agent-load.md
```

每一级都会保留配置并发、实测最大 in-flight、P50/P95/P99、吞吐、VTCR、逐例失败、
provider 明确回传的 Token/请求/重试/模型和货币成本。若 provider 没有回传货币成本，
结果保持 `unavailable`；只有明确需要把它作为发布阻断项时才传
`--require-provider-cost`。工具不根据模型名或耗时估算费用。
