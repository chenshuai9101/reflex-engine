"""Phase 3：影子模式。模型只记录"我会怎么做 + 置信度"，不行动。

默认用 LayaModel（本地 MLX 决策模型，~5ms 校准决策）；laya-mlx 不可用时自动回退
RuleModel（关键词规则）。运行（需 laya-mlx 的 venv）：

    ~/laya-mlx/.venv/bin/python scripts/shadow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reflex.schema import load_decision_types, fingerprint
from reflex.store import DecisionStore
from reflex.gate import route
from reflex.model import build_model
from reflex.shadow import Runner, Mode


def main() -> None:
    dts = load_decision_types("config/decisions.json")
    store = DecisionStore("data/decisions.db")
    model = build_model()  # 默认 laya，缺失回退 rule
    runner = Runner(model, store)
    print(f"[model] {model.name}")

    # 演示：对一条样例邮件文本跑一遍影子模式
    sample = "URGENT: please review the Q3 budget by Friday"
    dt = dts["email_triage"]
    fp = fingerprint(sample, keep_text=True)  # keep_text：让 Laya 读到原文
    action, conf, r = runner.step(dt, fp, mode=Mode.SHADOW)
    print(f"样例邮件: {sample!r}")
    print(f"  影子预测: {action}  (置信度 {conf:.3f})  -> 路由 {r.value}")
    print(f"  (闸门阈值: tau_auto={dt.tau_auto}, tau_review={dt.tau_review}, 风险档={dt.risk_tier.value})")
    print(f"\n[影子] 链路跑通（模型={model.name}）。下一步: 接真实捕获源 + 用数据比对影子准确率。")


if __name__ == "__main__":
    main()
