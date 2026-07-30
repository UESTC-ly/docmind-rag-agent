# ALCE / ARES Judge Calibration

> Public-label calibration evidence. These judges are not eligible to approve a release.

- Generated: `2026-07-28T16:08:16.027817+00:00`
- All release-gate eligible: `false`

| Target | Cases | Coverage | Balanced accuracy | Release-gate eligible |
|---|---:|---:|---:|---:|
| Citation Correctness (ALCE) | 100 | 1.0000 | 0.6800 | no |
| Answer Relevance (ARES) | 100 | 0.8100 | 0.8875 | no |

## Decision boundary

- The full JSON retains per-case public-label evidence, judge status, model, rubric version, and input fingerprints.
- A failed or incomplete calibration remains fail-closed; these scores must not be used to auto-approve a release.
