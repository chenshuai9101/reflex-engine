# reflex-engine：白领"肌肉记忆"决策工具链

把日常工作中**高频、可枚举、低风险**的小决策，被动录下来 → 蒸馏成会"条件反射"的小模型
→ 只在拿不准时才上抛给人 / LLM。

> 核心思想：LLM 管大规模复合思考（System 2），一个便宜的小决策模型管无数小范围确定性
> 决策（System 1 反射）。反射省的不是智能，是**注意力**。

---

## 架构（一个闭环，四个部件）

```
 日常行为 ─(capture 零摩擦)─▶ 决策日志(store, append-only) ─(train 蒸馏+校准)─▶ 反射模型(model)
      ▲                                                                             │
      └────(correct 修正回流: undo/改判)──── 边界闸门(gate) ◀──(置信度三段门控)── 上抛 LLM/人
```

## 对应三问

| 问题 | 模块 | 谁来做 |
|---|---|---|
| 1 如何收集标注 | `capture/` + `store.py` + `scripts/collect.py` | 被动捕获（你的动作就是标签）+ LLM 蒸馏冷启动 |
| 2 谁来校准修正 | `calibrate.py`(统计) + `correct.py`(修正) | 统计校准=机器自动；标签修正=你被动 undo + 每周批量 10 分钟 |
| 3 边界确认划分 | `gate.py` + `shadow.py` | 置信度三段门控；影子→建议→自动 阶梯"挣"边界 |

## 目录

```
config/decisions.json   决策类型定义（选项、风险档、阈值）
reflex/
  schema.py             决策类型 + 决策日志 + 输入指纹（隐私：不存原文）
  store.py              决策日志（SQLite，append-only，零依赖）
  gate.py               边界闸门（置信度三段门控）
  calibrate.py          统计校准（ECE + 温度缩放）
  correct.py            修正回流（被动 undo + 批量 review）
  shadow.py             影子/建议/自动 三模式 Runner
  capture/base.py       捕获器接口
  capture/email_gmail.py 邮件源（第一个试点，TODO 接 Gmail API）
  model/base.py         反射模型接口
  model/rule_model.py   关键词规则模型（冷启动基线 + 兜底）
  model/laya_model.py   Laya 决策模型适配器（本地 MLX，已接入，默认模型）
scripts/
  collect.py            Phase 1 纯被动收集
  train.py              Phase 2 蒸馏+训练+校准（占位）
  shadow.py             Phase 3 影子模式（可跑通）
```

## 快速上手（骨架现在能跑什么）

```bash
cd ~/reflex-engine
# 影子预测：默认用 LayaModel（本地 MLX 决策模型）；用 laya-mlx 的 venv 运行
~/laya-mlx/.venv/bin/python scripts/shadow.py
# 或（无 laya-mlx 时自动回退关键词规则模型）：
python3 scripts/shadow.py
python3 scripts/collect.py     # Gmail 未接入前，会提示"尚未接入"并显示日志计数
```

核心依赖仅 `numpy`；本地决策模型额外需要 `mlx` + `laya-mlx`（已装在
`~/laya-mlx/.venv`）。权重默认走多语种 checkpoint，可用 `LAYA_MODEL_PATH` 覆盖。

## Demos（网页演示）

`demos/` 下有三个网页演示 + 一个延迟基准，展示 `LayaModel` 的实际用法：

```bash
~/laya-mlx/.venv/bin/python demos/web_triage.py   # 医药分诊：纯 Laya vs reflex-engine 闸门
~/laya-mlx/.venv/bin/python demos/web_snake_reflex.py  # 贪吃蛇走 LayaModel + gate
```

详见 `demos/README.md`。

## 落地路径（Phase）

| 阶段 | 做什么 | 验收 |
|---|---|---|
| 0 | 选第一个反射 + 装捕获器 | 捕获器能导出 `(输入指纹, 动作)` |
| 1 | 纯被动收集 1-2 周 | 日志 >=500 条，且你没额外花一分钟 |
| 2 | LLM 蒸馏冷启动 + 训练 + 校准 | 留出集 ECE < 0.1 |
| 3 | 影子模式 | 影子一致率 > 90% |
| 4 | 建议模式 | 采纳率 > 85%，纠正率 < 10% |
| 5 | 自动执行（仅低风险） | 纠正率 < 5%，漂移监控上线 |
| 6 | 复制到下一个决策类型 | 复用捕获/日志/校准/闸门 |

## 三条铁律

1. **收集必须零摩擦**——骑在你本来就在做的动作上，任何额外一步都会死。
2. **置信度必须校准**——没校准的反射谁也不敢信（Laya 类原始 ECE 可达 0.4+）。
3. **边界是挣来的**——影子→建议→自动，每级用数据证明"它比你准"才升下一级。
