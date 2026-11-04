#!/usr/bin/env python3
"""下载并预处理 email-triage-v1 冷启动数据 → reflex-engine 的 FileSource 格式。

tunedtensor/email-triage-v1 的 triage 标签 5 类 → reflex-engine email_triage 4 桶：
    reply    → reply_now
    escalate → delegate
    ignore   → archive
    archive  → archive
    review   → mark_todo

输出 examples/email_triage_train.jsonl，每行：
    {"text": "Subject: ... Body: ...", "action": "<4桶之一>", "decision_type": "email_triage"}
（text 已剥掉数据集自带的 prompt 前缀，只保留邮件内容本身。）
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

TRIAGE_MAP = {
    "reply": "reply_now",
    "escalate": "delegate",
    "ignore": "archive",
    "archive": "archive",
    "review": "mark_todo",
}


def extract_text(inp: str) -> str:
    """剥掉 prompt 前缀，从第一个 'Subject:' 起保留邮件原文。"""
    idx = inp.find("Subject:")
    if idx != -1:
        return inp[idx:].strip()
    return inp.strip()


def main() -> None:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("tunedtensor/email-triage-v1", "data/train.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]

    out = []
    skipped = 0
    for r in rows:
        try:
            o = json.loads(r["output"])
        except (json.JSONDecodeError, KeyError, TypeError):
            skipped += 1
            continue
        triage = o.get("triage")
        if triage not in TRIAGE_MAP:
            skipped += 1
            continue
        out.append({
            "text": extract_text(r["input"]),
            "action": TRIAGE_MAP[triage],
            "decision_type": "email_triage",
        })

    dest = Path(__file__).resolve().parent.parent / "examples" / "email_triage_train.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")

    dist = Counter(x["action"] for x in out)
    print(f"wrote {len(out)} rows → {dest}")
    print(f"skipped {skipped} rows (bad/unknown label)")
    print("distribution:", dict(dist))


if __name__ == "__main__":
    main()
