"""Phase 1：被动收集。接一个捕获源，把"你本来就在做的动作"导进日志。

用法（零依赖文件源，立即可跑）:
    python3 scripts/collect.py --source file --path examples/seed_decisions.jsonl

用法（Gmail，需 App Password，见 capture/email_gmail.py）:
    export GMAIL_USER=... GMAIL_APP_PASSWORD=...
    python3 scripts/collect.py --source gmail
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reflex.store import DecisionStore


def build_source(args):
    if args.source == "file":
        from reflex.capture.file_source import FileSource
        return FileSource(args.path)
    if args.source == "gmail":
        from reflex.capture.email_gmail import GmailCapture
        return GmailCapture()
    raise SystemExit(f"未知源: {args.source}（可选 file | gmail）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("file", "gmail"), default="file")
    parser.add_argument("--path", default="examples/seed_decisions.jsonl", help="file 源的 JSONL 路径")
    parser.add_argument("--db", default="data/decisions.db")
    args = parser.parse_args()

    store = DecisionStore(args.db)
    capture = build_source(args)
    before = store.count()
    try:
        n = capture.ingest(store)
    except RuntimeError as e:
        print(f"[Phase 1] 捕获源未就绪：{e}")
        return
    print(f"[Phase 1] 源={capture.name}  新增 {n} 条  日志总数 {store.count()}（原 {before}）")


if __name__ == "__main__":
    main()
