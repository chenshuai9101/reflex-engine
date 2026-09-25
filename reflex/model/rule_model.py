"""关键词规则模型：冷启动基线 + 兜底。零依赖，可立即跑。

用途：
  1. 在没有任何训练数据时，先给一个能跑通的基线（Phase 0 就能测影子模式）。
  2. 作为模型失败时的兜底（永远有东西可路由）。

注意：置信度用"命中关键词的比例"粗略估计，不是校准过的——正式上自动执行前，
必须换成蒸馏出的分类器 + calibrate.py 校准。
"""
from __future__ import annotations

from .base import ReflexModel


class RuleModel(ReflexModel):
    name = "rule"

    def __init__(self, rules: dict[str, list[str]] | None = None):
        # action -> 触发关键词
        self.rules = rules or {
            "archive": ["newsletter", "digest", "notification", "weekly", "unsubscribe"],
            "reply_now": ["urgent", "asap", "deadline", "review", "confirm"],
            "delegate": ["forward", "fyi", "loop", "team"],
            "mark_todo": ["action", "todo", "please", "follow"],
        }

    def decide(self, input_fp: dict, options: list[str], criteria: dict | None = None) -> tuple[str, float]:
        del criteria  # RuleModel 用硬编码关键词，忽略 criteria 描述
        tokens = set(input_fp.get("tokens", []))
        best_action, best_hits = None, 0
        for action, kws in self.rules.items():
            if action not in options:
                continue
            hits = sum(1 for k in kws if k in tokens)
            if hits > best_hits:
                best_action, best_hits = action, hits
        if best_action is None:
            best_action = options[0]
        # 命中比例作为粗略置信度：0 命中 -> 低置信（会上抛），全命中 -> 偏高置信
        conf = 0.5 + 0.4 * (best_hits / max(len(tokens), 1))
        return best_action, min(conf, 0.9)
