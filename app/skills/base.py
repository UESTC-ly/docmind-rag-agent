"""Skill 抽象基类与上下文。

每个 Skill 是一个能力单元，声明：
  - name：唯一标识，也是 function calling 里的函数名
  - description：给 LLM 看的说明，LLM 据此决定要不要调用
  - parameters：JSON Schema，描述调用参数（function calling 规范）
  - run()：真正的执行逻辑

新增技能 = 写一个继承 BaseSkill 的类 + @register_skill 注册，核心编排零改动。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SkillContext:
    """执行技能时的上下文，携带用户身份和作用范围。"""

    user_id: int
    document_id: int | None = None


class BaseSkill(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema
    package_slug: str | None = None  # 可选：绑定 app/skills/packages/<slug>
    execution_mode: str = "python"
    _package: Any = None

    def load_package(self):
        """懒加载 Skill Package。

        Python 类仍然是执行层；package 目录提供说明、metadata、模板与 references。
        这里延迟导入，避免 base.py 和 package_loader.py 形成循环导入。
        """
        if not self.package_slug:
            return None
        if self._package is None:
            from app.skills.package_loader import load_skill_package

            self._package = load_skill_package(self.package_slug)
        return self._package

    def apply_package_metadata(self) -> None:
        """用 package metadata 覆盖 function-calling 暴露信息。"""
        package = self.load_package()
        if package is None:
            return
        self.name = package.name
        self.description = package.description
        self.parameters = package.parameters

    def package_template(self, name: str, fallback: str) -> str:
        """读取 package 模板；未绑定 package 或模板不存在时使用 fallback。"""
        package = self.load_package()
        if package is None:
            return fallback
        return package.templates.get(name, fallback)

    def package_reference(self, name: str, fallback: str = "") -> str:
        """读取 package reference；未绑定 package 或 reference 不存在时返回 fallback。"""
        package = self.load_package()
        if package is None:
            return fallback
        return package.references.get(name, fallback)

    @abstractmethod
    def run(self, context: SkillContext, **kwargs) -> dict:
        """执行技能，返回结构化结果（dict，会被序列化回传给 LLM 和前端）。"""
        ...

    def to_tool(self) -> dict:
        """转成 OpenAI function calling 的 tool 定义。"""
        self.apply_package_metadata()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
