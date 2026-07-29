"""导入所有技能模块，触发 @register_skill 注册。

Agent 启动时 import app.skills，所有技能就自动进入 Registry。
新增技能：在这里加一行 import 即可。
"""

from app.skills import (  # noqa: F401
    evaluation_advisor,
    graph,
    kb_search,
    mindmap,
    presentation,
    research_report,
    report,
    weekly_report,
    web_search,
)
from app.skills.registry import register_generic_package_skills
from app.skills.adapters import install_runtime_adapters

# Python-backed 技能导入完成后，再扫描纯 SKILL.md package。
# 已经由 Python 类注册的 package 会跳过；未注册的 Codex-style package 会变成
# GenericPackageSkill，体现 v0.5 的“通用 skills 包”能力。
install_runtime_adapters()
register_generic_package_skills()
