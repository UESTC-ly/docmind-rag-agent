"""Skills 注册表。

用装饰器把 Skill 类登记进全局字典。Agent 从这里发现所有可用技能。
这就是"可插拔"的实现：新技能只要被 import + 装饰，就自动出现在 Agent 的工具箱里。
"""

from app.skills.base import BaseSkill

_REGISTRY: dict[str, BaseSkill] = {}


def register_skill(cls: type[BaseSkill]) -> type[BaseSkill]:
    """类装饰器：实例化并登记一个 Skill。"""
    instance = cls()
    if instance.name in _REGISTRY:
        raise ValueError(f"Skill 名称重复: {instance.name}")
    _REGISTRY[instance.name] = instance
    return cls


def get_skill(name: str) -> BaseSkill | None:
    return _REGISTRY.get(name)


def all_skills() -> list[BaseSkill]:
    return list(_REGISTRY.values())


def all_tools() -> list[dict]:
    """所有技能的 function calling 定义，喂给 LLM。"""
    return [skill.to_tool() for skill in _REGISTRY.values()]
