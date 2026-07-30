# Agent Pipeline Selection Audit

- Agent report: `benchmarks/results/agent/cmrc2018-terra-final-20260730/report.json`
- Dataset: `CMRC 2018` / `validation`
- Selected run: `42`
- Pipeline: `hybrid-rerank`
- Pipeline fingerprint: `fc8177cc1896635b36c0fef33b14f8305d3b66316c52aca63b97a60f012207f8`
- Release status: `approved`
- Applicable error gates: `5/5 passed`
- Agent fingerprint match: `true`

## Evidence boundary

- This is a post-run database snapshot because the historical Agent response did not yet embed the compact selection receipt.
- It verifies dataset and pipeline-fingerprint continuity; it is not a standalone product-release approval.
