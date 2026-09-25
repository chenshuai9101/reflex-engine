"""决策日志：SQLite，append-only，零依赖（stdlib）。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .schema import Decision


class DecisionStore:
    """唯一的数据事实来源。日志永不删除，只追加 + 回填 outcome/corrected_to。"""

    def __init__(self, path: str | Path = "data/decisions.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    ts TEXT,
                    decision_type TEXT,
                    input_fp TEXT,        -- JSON: {text_hash, emb, len, tokens}
                    action TEXT,
                    source TEXT,
                    confidence REAL,
                    outcome TEXT,
                    corrected_to TEXT
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_type ON decisions(decision_type)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_conf ON decisions(confidence)")

    def append(self, d: Decision) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
                (d.id, d.ts, d.decision_type, json.dumps(d.input_fp),
                 d.action, d.source, d.confidence, d.outcome, d.corrected_to),
            )

    def mark_outcome(self, decision_id: str, outcome: str, corrected_to: str | None = None) -> None:
        """角色B：回填对错（被动 undo / 批量 review 都走这里）。"""
        with self._conn() as c:
            c.execute(
                "UPDATE decisions SET outcome=?, corrected_to=? WHERE id=?",
                (outcome, corrected_to, decision_id),
            )

    def query_low_confidence(self, decision_type: str, limit: int = 50) -> list[dict]:
        """角色B批量修正：拉最拿不准的 limit 条（|置信度-0.5| 最大 = 最纠结）。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM decisions WHERE decision_type=? AND confidence IS NOT NULL "
                "ORDER BY ABS(confidence-0.5) DESC LIMIT ?",
                (decision_type, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def correction_rate(self, decision_type: str) -> float:
        """角色C漂移监控：自动执行项的纠正率。>阈值(如5%)应降级+重训。

        TODO(细节): 目前统计全部 source；应只统计 source='auto'（自动执行后的被动纠正）。
        """
        with self._conn() as c:
            r = c.execute(
                "SELECT COUNT(*) n, SUM(outcome='wrong') w FROM decisions WHERE decision_type=?",
                (decision_type,),
            ).fetchone()
        return (r["w"] or 0) / r["n"] if r["n"] else 0.0

    def count(self, decision_type: str | None = None) -> int:
        with self._conn() as c:
            if decision_type:
                return c.execute(
                    "SELECT COUNT(*) FROM decisions WHERE decision_type=?", (decision_type,)
                ).fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
