"""Phase 1：纯被动收集。装好捕获器，跑这个把"你本来就在做的动作"导进日志。

用法（骨架阶段，Gmail 未接入前先演示形态）:
    python scripts/collect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reflex.store import DecisionStore
from reflex.capture.email_gmail import GmailCapture


def main() -> None:
    store = DecisionStore("data/decisions.db")
    capture = GmailCapture()
    # TODO(细节): 按 config/decisions.json 里启用的源，实例化多个 capture 循环 ingest
    try:
        n = capture.ingest(store)
    except NotImplementedError as e:
        print("[骨架] Gmail 捕获源尚未接入。")
        print(e)
        print(f"当前日志条数: {store.count()}")
        return
    print(f"新增 {n} 条，当前日志总数 {store.count()}")


if __name__ == "__main__":
    main()
