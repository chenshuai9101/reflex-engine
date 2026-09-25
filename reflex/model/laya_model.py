"""Laya 决策模型适配器：本地 MLX 推理，~5-12ms 出校准后的结构化决策。

Laya = 开源 System-1 决策模型（Convai Innovations，Apache-2.0）。本适配器把
reflex-engine 的 decide() 契约映射到 Laya 的 choice 原语：

    decide(input_fp, options) -> (action, confidence)

  * action      = Laya 输出的选项标签（必属 options）
  * confidence  = argmax 概率（RLCD 校准过的 P(正确)），直接喂给 gate.py 三段门控

依赖 laya-mlx（本机装在 ~/laya-mlx/.venv），权重默认用多语种 checkpoint
（~5.5ms，中英皆可）；英文场景想更准可换英文 checkpoint（~12ms，路径
~/laya-mlx/models/laya-mlx）。用环境变量 LAYA_MODEL_PATH 覆盖。

注意（和 gate.py 阈值语义对齐的关键）：
  * Laya 的 "confidence" 字段是归一化香农熵（1 - H/log k），对 4 选 1 哪怕
    top=0.955 也只有 ~0.83，会卡在 tau_auto=0.85 之下、导致永远上抛。所以这里
    用 argmax 概率（校准后的 P(正确)），与 gate 的"P 正确"阈值语义一致。
  * 置信度已由 RLCD 校准；攒够真实标签后仍可用 calibrate.py 独立复检 ECE。
  * act_probability（Laya 的"该不该行动"独立头）留作将来的 abstain→HUMAN 信号。
"""
from __future__ import annotations

import os
from typing import Any

from .base import ReflexModel

DEFAULT_MODEL_PATH = os.environ.get(
    "LAYA_MODEL_PATH",
    os.path.expanduser("~/laya-mlx/models/laya-multilingual-mlx"),
)


class LayaModel(ReflexModel):
    name = "laya"

    def __init__(
        self,
        model_path: str | None = None,
        dtype: str = "float16",
        instructions: str | None = None,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.dtype = dtype
        self.instructions = instructions or (
            "Classify this input into exactly one of the given options."
        )
        self._agent: Any = None  # laya_mlx.Agent，惰性加载

    # -- 加载 ---------------------------------------------------------------
    def _load(self):
        if self._agent is not None:
            return self._agent
        try:
            import laya_mlx as laya
        except ImportError as e:  # pragma: no cover - 依赖缺失时的清晰报错
            raise ImportError(
                "laya-mlx 未安装。安装: pip install laya-mlx；"
                "或直接用本机 venv 运行: ~/laya-mlx/.venv/bin/python"
            ) from e
        self._agent = laya.load(self.model_path, dtype=self.dtype)
        return self._agent

    @property
    def agent(self):
        return self._load()

    def warmup(self) -> None:
        """预热：触发首次编译 + 缓存，之后 predict 才是稳态延迟（~5ms）。"""
        self.decide({"tokens": ["warmup"]}, ["warmup_a", "warmup_b"])

    # -- 决策 ---------------------------------------------------------------
    def decide(self, input_fp: dict, options: list[str], criteria: dict | None = None) -> tuple[str, float]:
        out = self.decide_full(input_fp, options, criteria)
        return out["action"], out["confidence"]

    def decide_full(self, input_fp: dict, options: list[str], criteria: dict | None = None) -> dict:
        """decide() 的富信息版：额外返回每个选项的概率与 act_probability（供 UI/诊断）。

        返回 {action, confidence, probabilities, act_probability}。
        """
        if not options:
            raise ValueError("options 不能为空")
        # criteria 优先用带描述的 dict（大幅提升零样本准确率）；键与 options 不一致时退回裸标签
        if isinstance(criteria, dict) and criteria and set(criteria.keys()) == set(options):
            crit = criteria
        else:
            crit = list(options)
        question = {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": crit,
        }
        result = self.agent.predict(self._state_from_fp(input_fp), {"triage": question})
        ans = result["answers"]["triage"]
        return {
            "action": ans["choice"],
            # 校准后的 P(正确) ≈ argmax 概率；喂给 gate.py 的三段门控
            "confidence": float(max(ans["probabilities"].values())),
            "probabilities": {k: float(v) for k, v in ans["probabilities"].items()},
            "act_probability": float(ans["action"]["act_probability"]),
        }

    @staticmethod
    def _state_from_fp(input_fp: dict) -> str:
        """从输入指纹重建模型可读的文本。

        优先用保留的原文（fingerprint(keep_text=True) 时存在）；否则退化为
        top-k 关键词拼接（保序能力差，但保住"不存原文"的隐私默认）。
        """
        text = input_fp.get("text") or input_fp.get("body")
        if text:
            return text
        tokens = input_fp.get("tokens") or []
        return " ".join(tokens)
