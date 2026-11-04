"""数据模型：决策类型（配置）与决策日志条目（事实）。"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class RiskTier(str, Enum):
    """风险档：决定反射有多保守。风险越高，自动执行阈值越高。"""
    LOW = "low"
    MED = "med"
    HIGH = "high"


def risk_defaults(tier: RiskTier | str) -> tuple[float, float]:
    """风险档 -> (tau_auto, tau_review)。越危险越难自动执行。"""
    return {
        "low": (0.85, 0.55),
        "med": (0.95, 0.60),
        "high": (0.99, 0.70),
    }[tier]


@dataclass
class DecisionType:
    """一个可反射化的决策类型（来自 config/decisions.json）。"""
    name: str                      # 例 "email_triage"
    options: list[str]             # 可枚举的动作集合
    risk_tier: RiskTier = RiskTier.LOW
    tau_auto: float = 0.0          # 自动执行阈值；0 表示用风险档默认值
    tau_review: float = 0.0        # 低于此值直接给人；0 表示用默认值
    criteria: Optional[dict] = None  # 选项 -> 描述（喂给 Laya 的 choice，提升零样本准确率）

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionType":
        dt = cls(name=d["name"], options=d["options"],
                 risk_tier=RiskTier(d.get("risk_tier", "low")),
                 tau_auto=d.get("tau_auto", 0.0),
                 tau_review=d.get("tau_review", 0.0),
                 criteria=d.get("criteria"))
        # 未显式指定阈值时，用风险档默认值
        ta, tr = risk_defaults(dt.risk_tier)
        if dt.tau_auto <= 0:
            dt.tau_auto = ta
        if dt.tau_review <= 0:
            dt.tau_review = tr
        return dt


@dataclass
class Decision:
    """一条决策日志（append-only，永不删除）。"""
    decision_type: str
    input_fp: dict                  # 输入指纹：不存原文，见 fingerprint()
    action: str                     # 最终动作，属于 options
    source: str = "human"           # human | llm | rule
    confidence: Optional[float] = None
    outcome: Optional[str] = None   # 后续回填 correct | wrong
    corrected_to: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


_embedder = None  # 惰性加载的句子嵌入器；False 表示不可用


def _get_embedder():
    """惰性加载 sentence-transformers 嵌入器（真实语义嵌入，确定性）。

    失败（无 sentence-transformers / 模型未下载）时返回 None，指纹回退到哈希词袋。
    """
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            import os
            model_name = os.environ.get("REFLEX_EMBED_MODEL", "all-MiniLM-L6-v2")
            _embedder = SentenceTransformer(model_name)
        except Exception:
            _embedder = False
    return _embedder if _embedder else None


def fingerprint(text: str, dim: int = 128, keep_tokens: int = 50, keep_text: bool = False) -> dict:
    """把原文压成指纹：哈希 + 语义嵌入 + 少量关键词。

    设计取舍：默认不存原文（隐私），但保留 top-k 关键词，让规则模型/可解释性可用。
    若场景敏感，把 keep_tokens 设 0 即可只留哈希+嵌入。

    emb 优先用真实句子嵌入（sentence-transformers，确定性、含语义）；不可用时
    回退到哈希词袋。注意：Python 内建 hash() 对字符串是进程级盐化的（跨进程
    不确定），所以词袋只作降级兜底，不作为持久化语义。

    keep_text=True 会在指纹里附带原文 text 字段，供模型在推理期读到完整上下文。
    注意：这只用于推理（shadow/suggest/auto），持久化日志请保持 False，否则原文
    会写进 decisions.db 破坏"不存原文"的隐私默认。
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    toks = text.lower().split()

    emb = None
    model = _get_embedder()
    if model is not None:
        try:
            emb = model.encode(text, normalize_embeddings=True).astype(float).tolist()
        except Exception:
            emb = None
    if emb is None:
        emb = [0.0] * dim
        for t in toks:
            emb[hash(t) % dim] += 1.0

    # top-k 关键词（按出现频次）
    from collections import Counter
    tokens = [w for w, _ in Counter(toks).most_common(keep_tokens)]
    out = {"text_hash": h, "emb": emb, "len": len(text), "tokens": tokens}
    if keep_text:
        out["text"] = text
    return out


def load_decision_types(path: str | Path) -> dict[str, DecisionType]:
    """从 config/decisions.json 加载所有决策类型定义。"""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, DecisionType] = {}
    for name, d in raw.items():
        d = dict(d); d.setdefault("name", name)
        out[name] = DecisionType.from_dict(d)
    return out
