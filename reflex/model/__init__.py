"""反射模型层：给定输入指纹 + 选项集合，返回 (动作, 置信度)。

默认模型是 LayaModel（本地 MLX，~5ms 校准决策）；laya-mlx 不可用时回退 RuleModel。
"""
from __future__ import annotations


def build_model(name: str = "laya", **kwargs):
    """构造反射模型。

    name='laya'（默认，本地 Laya 决策模型）| 'rule'（关键词规则基线）。
    'laya' 在 laya-mlx 缺失时回退 'rule' 并告警，保证骨架在裸环境也能跑通。
    """
    if name == "rule":
        from .rule_model import RuleModel
        return RuleModel(**kwargs)
    if name != "laya":
        raise ValueError(f"未知模型: {name!r}（可选 laya | rule）")
    try:
        import laya_mlx  # noqa: F401  —— 探测依赖是否存在
        from .laya_model import LayaModel
        return LayaModel(**kwargs)
    except ImportError:
        import warnings
        warnings.warn(
            "laya-mlx 未安装，已回退到 RuleModel（关键词规则）。"
            "要启用 Laya 本地决策模型：pip install laya-mlx，"
            "或用 ~/laya-mlx/.venv/bin/python 运行。"
        )
        from .rule_model import RuleModel
        return RuleModel(**kwargs)
