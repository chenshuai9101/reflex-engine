"""查看决策日志：按类型计数 + 最近 N 条（隐私：不存原文，只显示指纹关键词）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reflex.store import DecisionStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/decisions.db")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    store = DecisionStore(args.db)
    total = store.count()
    print(f"日志总数: {total}\n")

    with store._conn() as c:
        rows = c.execute(
            "SELECT decision_type, COUNT(*) n FROM decisions GROUP BY decision_type"
        ).fetchall()
        print("按类型:")
        for r in rows:
            print(f"  {r['decision_type']:24} {r['n']}")

        print(f"\n最近 {args.limit} 条:")
        recent = c.execute(
            "SELECT ts, decision_type, action, input_fp FROM decisions ORDER BY ts DESC LIMIT ?",
            (args.limit,),
        ).fetchall()
        for r in recent:
            fp = json.loads(r["input_fp"])
            toks = " ".join(fp.get("tokens", [])[:8])
            print(f"  {r['ts'][:19]}  {r['decision_type']:22} -> {r['action']:16}  | {toks}")


if __name__ == "__main__":
    main()
