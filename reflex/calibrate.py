"""角色A：统计校准。让"说 90% 有信心"真的意味着"90% 对"。

这是边界闸门能用的前提，也是最容易被跳过的坑：
Laya 类模型原始 ECE 可达 0.4+（置信度是假的），校准后才 <0.1。

骨架实现 ECE（评估）+ 温度缩放占位。isotonic regression 留 TODO（scikit-learn）。
"""
from __future__ import annotations

import numpy as np


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


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """在留出集上拟合温度 T，使 softmax(logits/T) 的置信度被校准。

    TODO(细节): 用 scipy.optimize.minimize 最小化负对数似然求 T。
    骨架返回 T=1（无校准），先让管线跑通，细节后补。
    """
    del logits, labels
    return 1.0
