#!/usr/bin/env python3
"""Phase 2 微调：把带标签日志蒸馏成一个分类器（多语种 BERT 基座）。

对应 train.py 的 TODO：拿 source='human' 的金标签，微调一个分类器。
注意：这是"把日志蒸馏成分类器"，不是复现 Laya 的 RLCD 训练（那是上游的事）。

用法（先下载基座）:
    hf download bert-base-multilingual-cased --local-dir models/bert-base-multilingual-cased
    ~/laya-mlx/.venv/bin/python scripts/finetune.py \
        --jsonl examples/tumor_type_train.jsonl \
        --base models/bert-base-multilingual-cased --out models/tumor_classifier
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base", default="bert-base-multilingual-cased")
    ap.add_argument("--out", default="models/tumor_classifier")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    rows = [json.loads(l) for l in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = sorted({r["action"] for r in rows})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print(f"样本 {len(rows)} · 类 {labels}")

    # 分层切分（每类 20% 进 val，保证每类都留出）
    random.shuffle(rows)
    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["action"], []).append(r)
    train, val = [], []
    for l, rs in by_label.items():
        k = max(1, int(len(rs) * 0.2))
        val += rs[:k]
        train += rs[k:]
    print(f"train {len(train)} · val {len(val)}")

    tok = AutoTokenizer.from_pretrained(args.base)

    def encode(rs):
        x = tok([r["text"] for r in rs], truncation=True, padding=True, max_length=128, return_tensors="pt")
        y = torch.tensor([label2id[r["action"]] for r in rs])
        return x, y

    x_train, y_train = encode(train)
    x_val, y_val = encode(val)

    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = (len(train) + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    def move(x):
        return {k: v.to(device) for k, v in x.items()}

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(y_train))
        total = correct = 0
        for i in range(0, len(y_train), args.batch):
            idx = perm[i:i + args.batch]
            xb = move({k: v[idx] for k, v in x_train.items()})
            yb = y_train[idx].to(device)
            out = model(**xb, labels=yb)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            pred = out.logits.argmax(-1)
            total += yb.numel()
            correct += (pred == yb).sum().item()

        model.eval()
        with torch.no_grad():
            out = model(**move(x_val))
            pred = out.logits.argmax(-1).cpu()
            val_acc = (pred == y_val).float().mean().item()
        print(f"epoch {ep + 1}/{args.epochs}  train_acc={correct / total:.3f}  val_acc={val_acc:.3f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "labels.json").write_text(json.dumps(id2label, ensure_ascii=False))
    print(f"saved → {args.out}")
    print("labels:", id2label)


if __name__ == "__main__":
    main()
