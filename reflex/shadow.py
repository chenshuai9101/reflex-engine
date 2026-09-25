"""影子 / 建议 / 自动 三模式：边界是"挣"来的，不是"定"来的。

  shadow  只记录"我会怎么做 + 置信度"，不行动   -> 对比你真实动作，算一致率
  suggest 给建议，你一键确认/纠正（纠正自动回写） -> 看采纳率
  auto    自动执行（仅低风险档 + 高置信）        -> 看纠正率，超阈值自动降级

每级用数据证明"它比你准"，才升下一级。
"""
from __future__ import annotations

from enum import Enum

from .gate import Route, route
from .schema import DecisionType
from .store import DecisionStore


class Mode(str, Enum):
    SHADOW = "shadow"
    SUGGEST = "suggest"
    AUTO = "auto"


class Runner:
    """一条决策的完整处理：decide -> 按模式 + 闸门决定下一步。"""

    def __init__(self, model, store: DecisionStore):
        self.model = model
        self.store = store

    def step(self, dt: DecisionType, input_fp: dict, mode: Mode = Mode.SHADOW) -> tuple[str, float, Route]:
        """返回 (action, confidence, route)。实际"执行动作"的副作用留 TODO 接入。"""
        action, conf = self.model.decide(input_fp, dt.options, dt.criteria)
        r = route(dt, conf)

        if mode == Mode.SHADOW:
            # 只记录，不行动。后续脚本会把"模型预测 vs 你真实动作"做比对。
            pass
        elif mode == Mode.SUGGEST:
            # 给建议，等用户确认/纠正（纠正走 correct.record_correction）。
            pass
        elif mode == Mode.AUTO:
            # 仅 r == Route.AUTO 时执行；否则上抛。
            if r == Route.AUTO:
                self._execute(dt, action)
        return action, conf, r

    def _execute(self, dt: DecisionType, action: str) -> None:
        """执行真实动作（归档邮件 / 打分 / 路由）。TODO(细节): 接具体源。"""
        raise NotImplementedError(f"TODO: 执行 {dt.name} -> {action}")
