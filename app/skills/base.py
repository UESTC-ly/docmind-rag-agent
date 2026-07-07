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


@dataclass
class SkillContext:
    """执行技能时的上下文，携带用户身份和作用范围。"""

    user_id: int
    document_id: int | None = None


class BaseSkill(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema

    @abstractmethod
    def run(self, context: SkillContext, **kwargs) -> dict:
        """执行技能，返回结构化结果（dict，会被序列化回传给 LLM 和前端）。"""
        ...

    def to_tool(self) -> dict:
        """转成 OpenAI function calling 的 tool 定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
