"""JSONL 文件捕获源：零依赖、无需外部凭证，Phase 1 立即可跑。

格式（每行一个 JSON）：
    {"text": "邮件/工单原文", "action": "archive", "decision_type": "email_triage"}

  * action 必须属于 config/decisions.json 里对应 decision_type 的 options
  * 以 # 开头的行是注释，跳过
  * id 由 (decision_type, action, text) 哈希而成，天然幂等——重复 ingest 不重复写

这是"真实捕获源"里最轻的一种：你把任何来源的真实决定导成 JSONL，
（Gmail 导出、工单导出、手标数据），它就能把 (输入 → 你的动作) 灌进日志。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .base import Capture
from ..schema import Decision, fingerprint


class FileSource(Capture):
    name = "file_jsonl"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def poll(self) -> list[Decision]:
        if not self.path.is_file():
            return []
        decisions = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            text = rec["text"]
            action = rec["action"]
            dt = rec.get("decision_type", "email_triage")
            decisions.append(
                Decision(
                    decision_type=dt,
                    input_fp=fingerprint(text),
                    action=action,
                    source="human",  # 你的真实动作 = 金标签
                    id=hashlib.sha256(f"{dt}\0{action}\0{text}".encode()).hexdigest()[:32],
                )
            )
        return decisions
