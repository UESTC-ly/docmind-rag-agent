"""Skills 注册表。

用装饰器把 Skill 类登记进全局字典。Agent 从这里发现所有可用技能。
这就是"可插拔"的实现：新技能只要被 import + 装饰，就自动出现在 Agent 的工具箱里。
"""

from app.skills.base import BaseSkill

_REGISTRY: dict[str, BaseSkill] = {}


def register_skill_instance(instance: BaseSkill) -> BaseSkill:
    """登记一个已经实例化的 Skill。

    Python-backed 技能用类装饰器；Codex-style 通用技能需要按 package 动态构造
    实例，因此共用这个底层入口。
    """
    instance.apply_package_metadata()
    if instance.name in _REGISTRY:
        raise ValueError(f"Skill 名称重复: {instance.name}")
    _REGISTRY[instance.name] = instance
    return instance


def register_skill(cls: type[BaseSkill]) -> type[BaseSkill]:
    """类装饰器：实例化并登记一个 Skill。"""
    instance = cls()
    register_skill_instance(instance)
    return cls


def get_skill(name: str) -> BaseSkill | None:
    return _REGISTRY.get(name)


def all_skills() -> list[BaseSkill]:
    return list(_REGISTRY.values())


def all_tools() -> list[dict]:
    """所有技能的 function calling 定义，喂给 LLM。"""
    return [skill.to_tool() for skill in _REGISTRY.values() if skill.available]


def register_generic_package_skills() -> None:
    """自动登记没有 Python 执行类兜底的 Codex-style package。

    已有 Python-backed skill 的 package（例如 weekly-report / presentation）会因
    name 已存在而跳过；只有纯 SKILL.md package 才变成 GenericPackageSkill。
    """
    from app.skills.generic import GenericPackageSkill
    from app.skills.package_loader import list_skill_packages

    for package in list_skill_packages():
        if package.name in _REGISTRY:
            continue
        register_skill_instance(GenericPackageSkill(package))
