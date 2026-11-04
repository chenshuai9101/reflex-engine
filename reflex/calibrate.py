"""角色A：统计校准。让"说 90% 有信心"真的意味着"90% 对"。

这是边界闸门能用的前提，也是最容易被跳过的坑：
Laya 类模型原始 ECE 可达 0.4+（置信度是假的），校准后才 <0.1。

真实实现（不再占位）：
  - expected_calibration_error / maximum_calibration_error：评估
  - temperature_scale：在 logits 上拟合单一温度 T（scipy L-BFGS-B 最小化 NLL）
  - isotonic_top：对 top-1 概率做保序回归（sklearn，适合 >=1k 校准样本）
  - Calibrator：封装 fit(logits, labels) -> predict_proba(logits) 输出校准概率

约定：labels 是整数 id（0..K-1）。temperature 作用于 logits（多分类），
isotonic 作用于 top-1 概率（单变量保序）。两者都只在校准集上拟合，绝不在验证集上。
"""
from __future__ import annotations

import numpy as np


# -- 评估 -------------------------------------------------------------------
def expected_calibration_error(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> float:
    """ECE：把置信度分箱，比较每箱"平均置信度" vs "实际准确率"。目标 < 0.1。

    conf: 模型预测的置信度（0~1）
    acc:  这些预测实际对不对（0/1）
    """
    conf = np.asarray(conf, dtype=float)
    acc = np.asarray(acc, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(conf[m].mean() - acc[m].mean())
    return float(ece)


def maximum_calibration_error(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> float:
    """MCE：最差那箱的偏差。"""
    conf = np.asarray(conf, dtype=float)
    acc = np.asarray(acc, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        mce = max(mce, abs(conf[m].mean() - acc[m].mean()))
    return float(mce)


# -- 校准方法 ---------------------------------------------------------------
def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """在 logits 上拟合单一温度 T，使 softmax(logits/T) 的置信度被校准。

    logits: (N, K) 未归一化分数
    labels: (N,) 整数 id
    返回 T（>0）。T>1 表示原模型过度自信，需要"降温"。
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if logits.ndim == 1:
        logits = logits[:, None]  # 二分类安全处理

    def nll(t) -> float:
        # scipy minimize 传进来的是 (1,) 数组，先压成标量
        t = float(np.asarray(t).reshape(-1)[0])
        t = max(t, 1e-6)
        scaled = logits / t
        m = scaled.max(axis=-1, keepdims=True)
        logp = scaled - m - np.log(np.exp(scaled - m).sum(axis=-1, keepdims=True))
        return float(-logp[np.arange(len(labels)), labels].mean())

    try:
        from scipy.optimize import minimize
        res = minimize(nll, 1.0, bounds=[(1e-3, 20.0)], method="L-BFGS-B")
        return float(res.x[0])
    except Exception:
        # 无 scipy 时退化为不缩放（T=1）
        return 1.0


def _softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits[None, :]
    scaled = logits / t
    m = scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled - m)
    return e / e.sum(axis=-1, keepdims=True)


def isotonic_top(probs: np.ndarray, labels: np.ndarray, top_id: np.ndarray) -> object:
    """对 top-1 概率做保序回归（单变量）。适合 >=1k 校准样本时作为温度缩放的补充。

    probs:  (N, K) 未校准概率
    labels: (N,) 整数 id
    top_id: (N,) 每条样本 argmax 的类别 id
    返回拟合好的 sklearn IsotonicRegression（对 top-1 概率校准）。
    """
    from sklearn.isotonic import IsotonicRegression

    conf = probs[np.arange(len(labels)), top_id]
    correct = (top_id == labels).astype(float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(conf, correct)
    return iso


# -- 封装 -------------------------------------------------------------------
class Calibrator:
    """在校准集上拟合温度 + 保序，然后给任意 logits 输出校准概率。

    fit(logits, labels) 之后：
      predict_proba(logits) -> (N, K) 校准概率（温度缩放后，top-1 再做保序）
    """

    def __init__(self):
        self.T: float = 1.0
        self.iso = None
        self.K: int = 0

    def fit(self, logits: np.ndarray, labels: np.ndarray, use_isotonic: bool = False) -> "Calibrator":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        self.K = logits.shape[-1]
        self.T = temperature_scale(logits, labels)
        if use_isotonic:
            probs = _softmax(logits, self.T)
            top_id = probs.argmax(axis=-1)
            self.iso = isotonic_top(probs, labels, top_id)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        probs = _softmax(logits, self.T)
        if self.iso is not None:
            # 只校准 top-1，其余类别按比例缩（保持和为 1）
            top_id = probs.argmax(axis=-1)
            top_conf = probs[np.arange(len(probs)), top_id]
            top_conf_cal = np.clip(self.iso.predict(top_conf), 0.0, 1.0)
            new = probs.copy()
            # 剩余质量按原比例分配给其余类
            rest = 1.0 - top_conf_cal
            rest_mask = np.ones_like(probs, dtype=bool)
            rest_mask[np.arange(len(probs)), top_id] = False
            rest_sum = probs[rest_mask].reshape(len(probs), -1).sum(axis=-1, keepdims=True)
            if np.any(rest_sum > 0):
                scale = rest / np.maximum(rest_sum, 1e-12)
                new[rest_mask] = (probs[rest_mask].reshape(len(probs), -1) * scale).reshape(-1)
            new[np.arange(len(probs)), top_id] = top_conf_cal
            probs = new / new.sum(axis=-1, keepdims=True)
        return probs
