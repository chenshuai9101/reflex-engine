"""共形预测：给 auto 档一个有限样本的覆盖保证（distribution-free）。

为什么不能只用校准概率：校准只保证"平均而言 90% 的对"，不保证"这一条对不对"。
共形给的是集合级保证：P(真标签 ∈ 预测集) >= 1-alpha，且不假设分布。

用法（split conformal，最轻量，零依赖）：
    sc = SplitConformal(alpha=0.05).fit(cal_probs, cal_labels, label_set)
    sets = sc.predict_set(val_probs)          # list[set[str]]
    coverage = mean(true_label in set)        # 期望 >= 1-alpha
    singleton_auto = mean(len(set)==1)        # 安全 auto 覆盖率（上界）
"""
from __future__ import annotations

import numpy as np


class SplitConformal:
    """split-conformal 分类器。nonconformity 分数 = 1 - P(真类)。

    对一个新样本，预测集 = { y : 1 - P(y) <= q_hat }。
    集合是单例时才允许 auto 执行（那条唯一标签就是动作）。
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat: float | None = None
        self.labels: list[str] = []

    def fit(self, probs: np.ndarray, labels: np.ndarray, label_set: list[str]) -> "SplitConformal":
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)
        self.labels = list(label_set)
        # nonconformity = 1 - P(真类)，直接用整数 id 索引
        scores = 1.0 - probs[np.arange(len(labels)), labels]
        n = len(scores)
        # 有限样本修正：q_hat = 第 ceil((n+1)(1-alpha)) 个最小的分数
        q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha))) - 1
        q_idx = min(max(q_idx, 0), n - 1)
        self.q_hat = float(np.sort(scores)[q_idx])
        return self

    def predict_set(self, probs: np.ndarray) -> list[set[str]]:
        probs = np.asarray(probs, dtype=float)
        out = []
        for p in probs:
            s = {self.labels[j] for j in range(len(self.labels)) if (1.0 - p[j]) <= self.q_hat}
            out.append(s)
        return out

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        """在验证集上报告：经验覆盖率、单例率（安全 auto 上界）、单例错误率。

        labels 传整数 id（0..K-1）；内部用 self.labels[id] 转回名字判集合归属。
        """
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)
        sets = self.predict_set(probs)
        n = len(labels)
        covered = sum(1 for s, yi in zip(sets, labels) if self.labels[yi] in s)
        singleton = sum(1 for s in sets if len(s) == 1)
        # 单例里的错误：集合只含一个元素且它不是真类
        singleton_wrong = sum(
            1 for s, yi in zip(sets, labels)
            if len(s) == 1 and (next(iter(s)) != self.labels[yi])
        )
        return {
            "coverage": covered / n,
            "singleton_rate": singleton / n,
            "singleton_error_rate": (singleton_wrong / singleton) if singleton else 0.0,
            "q_hat": self.q_hat,
            "alpha": self.alpha,
        }
