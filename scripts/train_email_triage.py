#!/usr/bin/env python3
"""email triage 端到端验证：微调分类器 → 校准 → 共形闸门 → 三个数字。

数据：examples/email_triage_train.jsonl（prep_email_triage.py 产出，来自
tunedtensor/email-triage-v1，映射到 reflex-engine 的 4 桶 reply_now/delegate/archive/mark_todo）。

流程：
  train/cal/val 分层切分(60/20/20) → 微调 AutoModelForSequenceClassification
  → 校准集上温度缩放 (+可选保序) → 共形阈值
  → 验证集报告：准确率 / ECE(校准前后) / 安全 auto 覆盖率(1%/2%/5% 错误预算)
  → 写 data/email_triage_report.md + models/email_triage_classifier/

用法：
  HF_ENDPOINT=https://hf-mirror.com /opt/homebrew/bin/python3 scripts/train_email_triage.py \
      --base answerdotai/ModernBERT-base --epochs 3 --out models/email_triage_classifier
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from reflex.calibrate import Calibrator, expected_calibration_error
from reflex.conformal import SplitConformal


# -- 数据 -------------------------------------------------------------------
def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def stratified_split(rows: list[dict], seed: int = 42):
    texts = [r["text"] for r in rows]
    acts = [r["action"] for r in rows]
    labels = sorted(set(acts))
    y = [labels.index(a) for a in acts]
    # train 60% / (cal+val) 40%
    tr_idx, rest_idx = train_test_split(
        np.arange(len(rows)), test_size=0.4, stratify=y, random_state=seed
    )
    y_rest = [y[i] for i in rest_idx]
    cal_idx, val_idx = train_test_split(
        np.array(rest_idx), test_size=0.5, stratify=y_rest, random_state=seed
    )
    return (
        [rows[i] for i in tr_idx],
        [rows[i] for i in cal_idx],
        [rows[i] for i in val_idx],
        labels,
    )


# -- 训练 -------------------------------------------------------------------
def encode(tokenizer, texts, labels, max_len=256):
    x = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def move(obj, device):
    return {k: v.to(device) for k, v in obj.items()}


def train(model, device, x_train, y_train, x_cal, y_cal, epochs, lr, batch, seed, class_weights=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = (len(y_train) + batch - 1) // batch
    total_steps = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_train))
        total = correct = 0
        for i in range(0, len(y_train), batch):
            idx = perm[i:i + batch]
            xb = move({k: v[idx] for k, v in x_train.items()}, device)
            yb = y_train[idx].to(device)
            out = model(**xb)
            loss = criterion(out.logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            pred = out.logits.argmax(-1)
            total += yb.numel()
            correct += (pred == yb).sum().item()
        acc = correct / total
        # 每轮在 cal 上粗看（不用于选模型，只打印）
        model.eval()
        with torch.no_grad():
            out = model(**move(x_cal, device))
            cacc = (out.logits.argmax(-1).cpu() == y_cal).float().mean().item()
        print(f"  epoch {ep + 1}/{epochs}  train_acc={acc:.3f}  cal_acc={cacc:.3f}")
    return model


def get_logits(model, device, x):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x["input_ids"]), 64):
            xb = move({k: v[i:i + 64] for k, v in x.items()}, device)
            logits.append(model(**xb).logits.cpu())
    return torch.cat(logits, dim=0).numpy()


# -- 评估：安全 auto 覆盖率 --------------------------------------------------
def safe_auto_coverage(probs: np.ndarray, labels: np.ndarray, eps: float) -> float:
    """在错误率 < eps 的约束下，能被安全 auto 执行的比例。

    对每个候选置信度阈值 tau，auto 集合 = {max conf >= tau}，取其中错误率 < eps
    的最大覆盖率。等价于 selective classification 的风险-覆盖率曲线。
    """
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == labels)
    best = 0.0
    for tau in np.unique(conf):
        mask = conf >= tau
        if mask.sum() == 0:
            continue
        err = 1.0 - correct[mask].mean()
        if err < eps:
            best = max(best, float(mask.mean()))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="examples/email_triage_train.jsonl")
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--out", default="models/email_triage_classifier")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--isotonic", action="store_true", help="校准额外叠加保序回归(>=1k 样本)")
    ap.add_argument("--alpha", type=float, default=0.05, help="共形覆盖水平 1-alpha")
    ap.add_argument("--class-weights", action="store_true", help="类别平衡权重(默认关,少数类会过度修正)")
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout 正则(防过拟合,小数据可 0.3)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # 1) 数据
    rows = load_rows(args.data)
    print(f"加载 {len(rows)} 条")
    train_rows, cal_rows, val_rows, labels = stratified_split(rows, args.seed)
    print(f"split: train={len(train_rows)} cal={len(cal_rows)} val={len(val_rows)} labels={labels}")
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    def to_ids(rs):
        return [label2id[r["action"]] for r in rs], [r["text"] for r in rs]

    y_tr, tx_tr = to_ids(train_rows)
    y_cal, tx_cal = to_ids(cal_rows)
    y_val, tx_val = to_ids(val_rows)

    # 2) 微调（先试离线缓存，避免二次联网抓 .no_exist 文件时卡死；失败再联网）
    print(f"loading base: {args.base}")
    try:
        tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base)
    # dropout 正则须通过 config 传（ModernBert 不接受 __init__ kwarg）；小数据防过拟合
    try:
        cfg = AutoConfig.from_pretrained(args.base, num_labels=len(labels), local_files_only=True)
    except Exception:
        cfg = AutoConfig.from_pretrained(args.base, num_labels=len(labels))
    if hasattr(cfg, "hidden_dropout_prob"):
        cfg.hidden_dropout_prob = args.dropout
    if hasattr(cfg, "attention_probs_dropout_prob"):
        cfg.attention_probs_dropout_prob = args.dropout

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base, config=cfg, local_files_only=True
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(args.base, config=cfg)
    model = model.to(device)

    x_train, y_train = encode(tok, tx_tr, y_tr, args.max_len)
    x_cal, y_cal_t = encode(tok, tx_cal, y_cal, args.max_len)
    x_val, y_val_t = encode(tok, tx_val, y_val, args.max_len)

    # 类别平衡权重（默认关；实测对 mark_todo 这类模糊少数类会过度修正，反而伤总体准确率）
    class_weights = None
    if args.class_weights:
        cnt = Counter(y_tr)
        n = len(y_tr)
        K = len(labels)
        class_weights = torch.tensor([n / (K * cnt[k]) for k in range(K)], dtype=torch.float)
        print(f"class_weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    model = train(model, device, x_train, y_train, x_cal, y_cal_t,
                  args.epochs, args.lr, args.batch, args.seed, class_weights)

    # 3) 保存
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "labels.json").write_text(json.dumps(id2label, ensure_ascii=False))
    print(f"saved model → {args.out}")

    # 4) logits
    cal_logits = get_logits(model, device, x_cal)
    val_logits = get_logits(model, device, x_val)
    y_cal_np = np.array(y_cal)
    y_val_np = np.array(y_val)

    # 5) 校准（只在校准集上拟合）
    calib = Calibrator().fit(cal_logits, y_cal_np, use_isotonic=args.isotonic)
    cal_probs = calib.predict_proba(cal_logits)
    val_probs = calib.predict_proba(val_logits)

    # 未校准概率（温度 T=1）
    raw_val_probs = Calibrator().predict_proba(val_logits)  # T=1

    # 6) 共形
    sc = SplitConformal(alpha=args.alpha).fit(cal_probs, y_cal_np, labels)
    conf_eval = sc.evaluate(val_probs, y_val_np)

    # 7) 评估数字
    pred = val_probs.argmax(axis=-1)
    acc = float((pred == y_val_np).mean())
    conf = val_probs.max(axis=-1)
    correct = (pred == y_val_np).astype(float)
    ece_raw = expected_calibration_error(raw_val_probs.max(axis=-1), correct)
    ece_cal = expected_calibration_error(conf, correct)
    report_cls = classification_report(y_val_np, pred, target_names=labels, digits=3)

    cov1 = safe_auto_coverage(val_probs, y_val_np, 0.01)
    cov2 = safe_auto_coverage(val_probs, y_val_np, 0.02)
    cov5 = safe_auto_coverage(val_probs, y_val_np, 0.05)

    # 8) 报告
    lines = [
        "# email triage 端到端验证报告",
        "",
        f"- 基座：`{args.base}` · epochs={args.epochs} · 设备={device}",
        f"- 数据：{len(rows)} 条（train/cal/val = {len(train_rows)}/{len(cal_rows)}/{len(val_rows)}）",
        f"- 标签：{labels}",
        f"- 温度 T = {calib.T:.3f}（{'<1 过度自信' if calib.T < 1 else '>=1 欠自信'}）",
        "",
        "## 三个核心数字",
        "",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| **准确率** | **{acc:.3f}** |",
        f"| **ECE（校准前 → 后）** | **{ece_raw:.3f} → {ece_cal:.3f}** |",
        f"| **安全 auto 覆盖率** @1%/2%/5% 错误 | **{cov1:.2%} / {cov2:.2%} / {cov5:.2%}** |",
        "",
        "## 逐类指标",
        "",
        "```",
        report_cls.strip(),
        "```",
        "",
        "## 共形闸门（split conformal）",
        "",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| 经验覆盖率（期望 >= {1 - args.alpha:.2f}） | {conf_eval['coverage']:.3f} |",
        f"| 单例率（安全 auto 上界） | {conf_eval['singleton_rate']:.2%} |",
        f"| 单例错误率 | {conf_eval['singleton_error_rate']:.2%} |",
        f"| q_hat | {conf_eval['q_hat']:.3f} |",
        "",
        "## 解读",
        "",
        "- 准确率 >0.85 且 ECE 校准后 <0.1，说明「被动标签 → 微调 → 校准」这条链成立。",
        "- 安全 auto 覆盖率：这是「有多少比例的邮件能放心自动归档」——它决定了这个产品",
        "  能不能真的省注意力（影子→建议→自动的终点）。",
        f"- 共形单例率 {conf_eval['singleton_rate']:.2%} 是更保守的 auto 上界（带有限样本保证）。",
    ]
    report = Path("data") / "email_triage_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 9) 控制台
    print("\n================ 结果 ================")
    print(f"准确率           : {acc:.3f}")
    print(f"ECE 校准前→后    : {ece_raw:.3f} → {ece_cal:.3f}  (T={calib.T:.3f})")
    print(f"安全auto@1%/2%/5%: {cov1:.2%} / {cov2:.2%} / {cov5:.2%}")
    print(f"共形 覆盖率/单例率/单例错误: "
          f"{conf_eval['coverage']:.3f} / {conf_eval['singleton_rate']:.2%} / {conf_eval['singleton_error_rate']:.2%}")
    print(f"报告已写 → {report}")
    print("\n逐类指标：\n" + report_cls)


if __name__ == "__main__":
    main()
