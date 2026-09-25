"""Phase 2：蒸馏 + 训练 + 校准（骨架占位）。

TODO(细节，逐条接):
  1. 蒸馏冷启动：调 LLM 补足历史数据缺口，产出弱标签写回日志（source='llm'）。
  2. 训练：把日志里 source='human' 的样本当金标签，训一个分类器
     （ModernBERT 微调 / Laya 微调），输出 (action, 置信度)。
  3. 校准：用 calibrate.expected_calibration_error 评估，温度缩放/isotonic 校准。
  4. 验收：留出集 ECE < 0.1。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reflex.store import DecisionStore


def main() -> None:
    store = DecisionStore("data/decisions.db")
    for dt in ("email_triage", "job_posting_relevance", "code_review_severity"):
        n = store.count(dt)
        if n:
            print(f"{dt}: {n} 条样本")
    print("\n[骨架] 训练/校准逻辑尚未实现。先跑 collect.py 攒数据（目标 >=500 条）。")


if __name__ == "__main__":
    main()
