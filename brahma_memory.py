import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Optional

MEMORY_DB = "brahma_memory.db"


class BrahmaMemory:

    def __init__(self, db_path: str = MEMORY_DB):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id           TEXT PRIMARY KEY,
                    timestamp    TEXT NOT NULL,
                    goal         TEXT NOT NULL,
                    source_type  TEXT,
                    problem_type TEXT,
                    best_model   TEXT,
                    metrics      TEXT,
                    notes        TEXT
                )
            """)
            conn.commit()

    def save_run(
        self,
        goal: str,
        source_type: str,
        problem_type: Optional[str] = None,
        best_model: Optional[str] = None,
        metrics: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs
                   (id, timestamp, goal, source_type, problem_type, best_model, metrics, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    datetime.utcnow().isoformat(),
                    goal,
                    source_type,
                    problem_type or "",
                    best_model or "",
                    json.dumps(metrics or {}),
                    notes or "",
                ),
            )
            conn.commit()
        return run_id

    def get_recent_runs(self, limit: int = 10) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_similar_runs(self, goal: str, limit: int = 5) -> list:
        words = [w.lower() for w in goal.split() if len(w) > 3]
        if not words:
            return self.get_recent_runs(limit)
        conditions = " OR ".join(["LOWER(goal) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words] + [limit]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs WHERE {conditions} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, row) -> dict:
        keys = ["id", "timestamp", "goal", "source_type", "problem_type",
                "best_model", "metrics", "notes"]
        d = dict(zip(keys, row))
        d["metrics"] = json.loads(d["metrics"] or "{}")
        return d

    def extract_and_save(self, goal: str, source_type: str) -> Optional[str]:
        """Read leaderboard.csv produced by the pipeline and persist the best result."""
        leaderboard_path = "outputs/data/leaderboard.csv"
        if not os.path.exists(leaderboard_path):
            return self.save_run(goal=goal, source_type=source_type)
        try:
            import pandas as pd
            df = pd.read_csv(leaderboard_path)
            df = df[~df["model"].str.contains("Dummy", case=False, na=False)]
            if df.empty:
                return self.save_run(goal=goal, source_type=source_type)
            best = df.sort_values("auc_val", ascending=False).iloc[0]
            metrics = {}
            for col in ["auc_val", "f1_val", "recall_val", "precision_val", "auc_train"]:
                if col in best:
                    try:
                        metrics[col] = round(float(best[col]), 4)
                    except (ValueError, TypeError):
                        pass
            return self.save_run(
                goal=goal,
                source_type=source_type,
                best_model=str(best["model"]),
                metrics=metrics,
            )
        except Exception:
            return self.save_run(goal=goal, source_type=source_type)

    def format_for_prompt(self, goal: str) -> str:
        similar = self.get_similar_runs(goal, limit=5)
        recent = self.get_recent_runs(limit=3)

        seen = set()
        combined = []
        for r in similar + recent:
            if r["id"] not in seen:
                combined.append(r)
                seen.add(r["id"])

        if not combined:
            return ""

        lines = [
            "# BRAHMA MEMORY — Past Pipeline Runs",
            "Use these results to inform algorithm selection, expected metric ranges, "
            "and hyperparameter starting points. Prefer models that worked well on similar goals.\n",
        ]
        for r in combined:
            date = r["timestamp"][:10]
            m = r["metrics"]
            metrics_str = (
                ", ".join(f"{k}={v}" for k, v in m.items()) if m else "metrics not captured"
            )
            model_str = r["best_model"] or "unknown"
            lines.append(
                f"• [{date}] \"{r['goal']}\" | source={r['source_type']} "
                f"| best_model={model_str} | {metrics_str}"
            )
        return "\n".join(lines)
