"""反射模型基类：统一契约。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ReflexModel(ABC):
    """所有反射模型的统一接口。

    decide() 返回 (action, confidence∈[0,1])。置信度必须是"校准过的可信概率"，
    否则 gate 的阈值毫无意义（见 calibrate.py）。
    """

    name: str = "base"

    @abstractmethod
    def decide(self, input_fp: dict, options: list[str], criteria: dict | None = None) -> tuple[str, float]:
        """criteria：选项 -> 描述（可选）。Laya 用它把"裸标签"变"带语义的选项"，
        零样本准确率能差一截；RuleModel 忽略它。"""
