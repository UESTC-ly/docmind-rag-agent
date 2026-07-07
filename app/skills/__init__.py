"""导入所有技能模块，触发 @register_skill 注册。

Agent 启动时 import app.skills，所有技能就自动进入 Registry。
新增技能：在这里加一行 import 即可。
"""

from app.skills import (  # noqa: F401
    graph,
    kb_search,
    mindmap,
    report,
    web_search,
)
