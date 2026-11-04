"""循环测试：对一批带标签样本跑 Laya 分类，比对真值，输出准确率 + 混淆 + 误判明细。

这是 reflex-engine 的"循环"在语料上的体现：跑一轮 → 看误判在哪 → 改 criteria →
再跑一轮比对提升。数据来自 capture（这里用 JSONL 语料代替真实来源）。

用法:
    ~/laya-mlx/.venv/bin/python scripts/loop_test.py --jsonl examples/tumor_type_seed.jsonl --type tumor_type
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reflex.model.laya_model import LayaModel
from reflex.schema import load_decision_types


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="带标签样本 JSONL（每行 text/action/decision_type）")
    ap.add_argument("--type", default="tumor_type")
    ap.add_argument("--config", default="config/decisions.json")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    dt = load_decision_types(args.config)[args.type]
    model = LayaModel(args.model) if args.model else LayaModel()
    model.warmup()

    samples = [
        json.loads(l) for l in Path(args.jsonl).read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]

    ok = 0
    conf_matrix = defaultdict(lambda: defaultdict(int))
    gold_total = defaultdict(int)
    gold_hit = defaultdict(int)
    wrong: list[tuple[str, str, str, float]] = []
    for s in samples:
        o = model.decide_full({"text": s["text"]}, dt.options, dt.criteria)
        pred, gold = o["action"], s["action"]
        conf_matrix[gold][pred] += 1
        gold_total[gold] += 1
        if pred == gold:
            ok += 1
            gold_hit[gold] += 1
        else:
            wrong.append((s["text"], gold, pred, o["confidence"]))

    n = len(samples)
    print(f"\n=== {args.type} · 循环测试 ===  {ok}/{n} = {ok/n:.1%}\n")

    print("分标签召回率:")
    for label in dt.options:
        tot = gold_total.get(label, 0)
        if tot == 0:
            continue
        print(f"  {label:16} {gold_hit.get(label, 0)}/{tot} = {gold_hit.get(label, 0) / tot:.0%}")

    print("\n混淆矩阵（行=真值，列=预测）:")
    labels = dt.options
    print(" " * 16 + "".join(f"{l[:8]:>9}" for l in labels))
    for gold in labels:
        row = conf_matrix[gold]
        cells = "".join(f"{row.get(pred, 0):>9}" for pred in labels)
        print(f"{gold:16}{cells}")

    print(f"\n误判 {len(wrong)} 条（按置信度降序，看是否「高置信还错」）:")
    for text, gold, pred, conf in sorted(wrong, key=lambda x: -x[3]):
        print(f"  conf={conf:.2f}  真值={gold:14} 预测={pred:14} | {text[:50]}")


if __name__ == "__main__":
    main()
