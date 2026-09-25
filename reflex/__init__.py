"""reflex-engine：白领"肌肉记忆"决策工具链（骨架）。

把日常工作中"高频、可枚举、低风险"的小决策，被动录下来 → 蒸馏成会"条件反射"的小模型
→ 只在拿不准时才上抛给人/LLM。

四个部件：
  capture/  捕获层（零摩擦：把你本来就在做的动作导成样本）
  schema/store 决策日志（append-only，唯一数据事实）
  model/    反射模型（规则/蒸馏出的分类器）
  gate/calibrate/correct/shadow  边界闸门 + 统计校准 + 修正回流 + 三模式

对应三问：
  问题1 数据收集  -> capture/ + store.py + scripts/collect.py
  问题2 谁来校准  -> calibrate.py(统计) + correct.py(修正)
  问题3 边界划分  -> gate.py + shadow.py
"""
__version__ = "0.1.0"
