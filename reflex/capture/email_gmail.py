"""邮件源捕获器（示例，也是推荐的第一个试点源）。

为什么先做邮件：白领决策里频率最高、动作最可枚举、标签现成（归档/垃圾箱/标签/转发
就是现成的 (邮件 -> 动作) 样本）、风险不对称最低。

TODO(细节，逐条接):
  1. 认证：Gmail API OAuth 或 App Password（本机 claude-proxy 环境注意代理）。
  2. 拉取：messages.list / threads.list，游标增量。
  3. 动作映射：labelId 对应 archive/reply_now/delegate/mark_todo。
     已归档 -> archive；在垃圾箱 -> 可判 irrelevant/spam；带"待办"标签 -> mark_todo。
  4. 游标持久化：把 last_history_id 存进 data/，避免重复拉。
"""
from __future__ import annotations

from .base import Capture
from ..schema import Decision, fingerprint


class GmailCapture(Capture):
    name = "email_gmail"

    def __init__(self, options: list[str] | None = None, cursor_path: str = "data/gmail_cursor.txt"):
        self.options = options or ["archive", "reply_now", "delegate", "mark_todo"]
        self.cursor_path = cursor_path

    def poll(self) -> list[Decision]:
        raise NotImplementedError(
            "TODO: 接 Gmail API。见 docstring 的 4 步。骨架只定义了数据形态：\n"
            f"  每个样本 = Decision(decision_type='{self.name}', input_fp=fingerprint(邮件主题+正文摘要), action=<你实际做的动作>)\n"
            "  source='human'（你的真实动作是金标签）"
        )
