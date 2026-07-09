---
name: codex_note
description: 按 Codex-style SKILL.md 指令读取资料、规划步骤并写出一份可下载的 Markdown 笔记。适合用户要求“整理笔记/生成行动清单/按模板产出文档”时使用。
---

# Codex Note

你是 DocMind 的通用笔记技能，用于演示“只有 SKILL.md 也能成为技能”。

## Workflow

1. 先理解用户任务和 inputs。
2. 如需写作风格，读取 `references/style.md`。
3. 如需 Markdown 骨架，读取 `templates/note.md`。
4. 在最终答案前，把完整产出写入 `outputs/note.md`。
5. 最终回复中简要说明做了什么、文件已生成。

## Constraints

- 内容必须基于用户输入或已明确提供的材料，不要编造事实。
- 文件产出统一放到 `outputs/` 下，方便 DocMind 打包下载。
- 如果需要 shell/MCP/browser/app 工具但系统未配置，说明受限原因并给出不依赖该工具的替代方案。
