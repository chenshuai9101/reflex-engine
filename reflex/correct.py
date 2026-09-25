"""角色B：修正回流。谁修？你被动修（零摩擦）+ 定时批量修（10 分钟）。

铁律：绝不逐条打断用户弹窗确认——那会杀死反射的意义。
"""
from __future__ import annotations

from .store import DecisionStore


def record_correction(store: DecisionStore, decision_id: str, corrected_action: str) -> None:
    """被动修正：你 undo / 改判（如把已归档邮件拖回"待办"），把纠正写回日志。

    这个"移动/撤销"动作本身就是一条金标签，零额外摩擦。
    """
    store.mark_outcome(decision_id, outcome="wrong", corrected_to=corrected_action)


def review_queue(store: DecisionStore, decision_type: str, limit: int = 50) -> list[dict]:
    """批量修正：拉最拿不准的 limit 条，每周花 10 分钟一次性点对/错。

    返回的是系统最纠结的样本 = 主动学习里收益最大的那批。
    """
    return store.query_low_confidence(decision_type, limit)
