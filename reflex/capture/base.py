"""捕获器基类：所有数据源的统一接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import Decision


class Capture(ABC):
    """捕获器契约：poll() 从源拉取新决策，ingest() 写入日志。

    关键设计：poll 必须增量 + 幂等（重复跑不重复写，靠 Decision.id 去重）。
    """

    name: str = "base"

    @abstractmethod
    def poll(self) -> list[Decision]:
        """从源拉取"上次游标之后"的新决策（输入指纹 + 你的动作）。"""
        ...

    def ingest(self, store) -> int:
        """poll -> 写日志，返回新增条数。"""
        ds = self.poll()
        n = 0
        for d in ds:
            store.append(d)
            n += 1
        return n
