"""Gmail 捕获源（IMAP + 应用专用密码）。推荐的第一个试点源。

为什么先做邮件：白领决策里频率最高、动作最可枚举、标签现成（归档/待办/星标
就是现成的 (邮件 → 动作) 样本）、风险不对称最低。

前置准备（一次性）：
  1. Google 账号开启两步验证。
  2. 生成「应用专用密码」（Google 账号 → 安全性 → 应用专用密码）。
  3. 在 Gmail 里给要捕获的动作建好标签/星标。

用环境变量传凭据（不要写进代码/仓库）：
    export GMAIL_USER='you@gmail.com'
    export GMAIL_APP_PASSWORD='xxxx xxxx xxxx xxxx'
    python3 scripts/collect.py --source gmail

标签 → 动作映射（默认，可自定）：Archive→archive、Followup→mark_todo、
Starred→reply_now。没命中任何已知标签的邮件跳过（不产生样本）。

⚠️ 本实现用 Gmail 的 X-GM-LABELS 扩展直接读标签，逻辑完整但**未经真机验证**——
需要你的 App Password 才能真正跑一次。
"""
from __future__ import annotations

import hashlib
import imaplib
import os
import re
from email.header import decode_header
from pathlib import Path

from .base import Capture
from ..schema import Decision, fingerprint

# 标签 → 动作。键是 Gmail 标签名（大小写敏感），值必须是 email_triage 的 options。
LABEL_ACTION = {
    "Archive": "archive",
    "Followup": "mark_todo",
    "Starred": "reply_now",
}


def _decode(v: bytes | str | None) -> str:
    if v is None:
        return ""
    parts = decode_header(v)
    out = []
    for text, enc in parts:
        if isinstance(text, bytes):
            out.append(text.decode(enc or "utf-8", errors="ignore"))
        else:
            out.append(text)
    return " ".join(out)


class GmailCapture(Capture):
    name = "email_gmail"

    def __init__(
        self,
        user: str | None = None,
        app_password: str | None = None,
        cursor_path: str | Path = "data/gmail_cursor.txt",
        label_action: dict | None = None,
    ):
        self.user = user or os.environ.get("GMAIL_USER")
        self.app_password = app_password or os.environ.get("GMAIL_APP_PASSWORD")
        self.cursor_path = Path(cursor_path)
        self.label_action = label_action or LABEL_ACTION

    # -- 游标（UID）持久化：增量 + 幂等 --------------------------------
    def _last_uid(self) -> int:
        if self.cursor_path.is_file():
            return int(self.cursor_path.read_text().strip() or 0)
        return 0

    def _save_uid(self, uid: int) -> None:
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_path.write_text(str(uid))

    def _connect(self):
        if not self.user or not self.app_password:
            raise RuntimeError(
                "缺少 GMAIL_USER / GMAIL_APP_PASSWORD 环境变量（Google 应用专用密码）"
            )
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        imap.login(self.user, self.app_password)
        return imap

    @staticmethod
    def _parse_labels(fetch_extra) -> list[str]:
        # X-GM-LABELS 形如 b'(\\Inbox "Followup" "Archive")'，用引号提取
        raw = fetch_extra[0] if isinstance(fetch_extra, tuple) else b""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "ignore")
        return re.findall(r'"([^"]+)"', raw)

    def poll(self) -> list[Decision]:
        imap = self._connect()
        try:
            imap.select('"[Gmail]/All Mail"', readonly=True)
            last = self._last_uid()
            typ, data = imap.uid("SEARCH", None, f"UID {last + 1}:*")
            uids = data[0].split() if typ == "OK" and data and data[0] else []
            decisions = []
            new_last = last
            for raw_uid in uids:
                uid = int(raw_uid)
                typ, msg = imap.uid("FETCH", uid, "(X-GM-LABELS BODY.PEEK[HEADER.FIELDS (SUBJECT FROM)])")
                if typ != "OK" or not msg or not msg[0]:
                    continue
                labels = self._parse_labels(msg[0])
                action = next((self.label_action[l] for l in labels if l in self.label_action), None)
                if action is None:
                    continue
                # 主题 + 发件人当输入文本（不存原文，只做指纹）
                text = _decode(msg[1] if len(msg) > 1 else b"")
                decisions.append(
                    Decision(
                        decision_type="email_triage",
                        input_fp=fingerprint(text),
                        action=action,
                        source="human",
                        id=hashlib.sha256(f"gmail\0{uid}\0{action}".encode()).hexdigest()[:32],
                    )
                )
                new_last = max(new_last, uid)
            if new_last > last:
                self._save_uid(new_last)
            return decisions
        finally:
            try:
                imap.logout()
            except Exception:
                pass
