"""角色C：边界闸门。置信度三段门控。

  c >= tau_auto          -> AUTO       真反射，自动执行
  tau_review <= c < tau_auto -> ESCALATE_LLM  上抛 System 2
  c < tau_review          -> HUMAN      直接给人

阈值来自 DecisionType（按风险档设默认）。前提是置信度已被 calibrate.py 校准——
校准过的置信度才有意义，闸门才敢放行。
"""
from __future__ import annotations

from enum import Enum

from .schema import DecisionType


class Route(str, Enum):
    AUTO = "auto"
    ESCALATE_LLM = "llm"
    HUMAN = "human"


def route(dt: DecisionType, confidence: float) -> Route:
    if confidence >= dt.tau_auto:
        return Route.AUTO
    if confidence >= dt.tau_review:
        return Route.ESCALATE_LLM
    return Route.HUMAN
