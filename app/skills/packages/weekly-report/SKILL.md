# Weekly Report Skill

## Intent

Generate a concise, evidence-grounded Chinese weekly report from the user's uploaded materials.
The output is a downloadable Markdown file and a preview artifact.

## Operating rules

- Use only the provided material.
- If the material does not mention a fact, write “材料未提及”.
- Keep the tone suitable for a supervisor, mentor, or project team update.
- Prefer structured sections over long prose.
- Do not invent progress, metrics, blockers, or next-week plans.

## Required output sections

1. 本周概览
2. 本周完成
3. 关键进展
4. 问题与风险
5. 下周计划

## Execution layer

The Python class `WeeklyReportSkill` handles database material loading, LLM invocation,
filename generation, and Markdown download payload construction.
