"""Records every question the app answers, so we can measure how well it works.

One row per question, in a small SQLite database next to the code (runs.db).
"""
import datetime
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent.parent / "runs.db"   # same place, wherever you start the app

COLUMNS = """
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT,
    question        TEXT,
    model           TEXT,
    status          TEXT,     -- success | partial | failed | provider_error | error
    seconds         REAL,
    steps_total     INTEGER,
    steps_ok        INTEGER,
    failed_attempts INTEGER,  -- code attempts that failed (each one triggered a repair or the give-up)
    repaired_steps  INTEGER,  -- steps whose first code failed
    repaired_ok     INTEGER,  -- ...and that the model then fixed
    error           TEXT
"""


def run_status(result: dict) -> dict:
    """Summarise a Worker result as numbers that can be logged and averaged."""
    steps = result.get("steps", [])
    ran = [s for s in steps if not s.get("skipped")]
    steps_ok = sum(s["success"] for s in steps)
    repaired = [s for s in ran if s["failed_attempts"]]
    if result.get("error"):
        status = "error"
    elif steps and steps_ok == len(steps):
        status = "success"
    elif steps_ok > 0:
        status = "partial"
    else:
        status = "failed"
    return {
        "status": status,
        "steps_total": len(steps),
        "steps_ok": steps_ok,
        "failed_attempts": sum(len(s["failed_attempts"]) for s in ran),
        "repaired_steps": len(repaired),
        "repaired_ok": sum(s["success"] for s in repaired),
        "error": result.get("error") or result.get("stopped"),
    }


class RunLog:
    def __init__(self, db_path=DEFAULT_DB):
        self.db_path = str(db_path)
        self._execute(f"CREATE TABLE IF NOT EXISTS runs ({COLUMNS})")

    def _execute(self, sql, params=()):
        # A fresh connection per call: safe when several browser tabs use the app at once.
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:                      # commits on success, rolls back on error
                conn.execute(sql, params)
        finally:
            conn.close()

    def record(self, question, model, status, seconds, steps_total=0, steps_ok=0,
               failed_attempts=0, repaired_steps=0, repaired_ok=0, error=None):
        self._execute(
            "INSERT INTO runs (timestamp, question, model, status, seconds, steps_total, steps_ok, "
            "failed_attempts, repaired_steps, repaired_ok, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.datetime.now().isoformat(timespec="seconds"), question, model, status,
             round(seconds, 2), steps_total, steps_ok, failed_attempts, repaired_steps, repaired_ok, error),
        )

    def load(self, limit=1000) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC LIMIT ?", conn, params=(limit,))
        finally:
            conn.close()

    @staticmethod
    def summary(runs: pd.DataFrame) -> dict:
        """Headline numbers for the dashboard."""
        if runs.empty:
            return {}
        repaired = runs["repaired_steps"].sum()
        return {
            "runs": len(runs),
            "success_rate": (runs["status"] == "success").mean(),
            "partial_or_failed": runs["status"].isin(["partial", "failed"]).sum(),
            "provider_errors": (runs["status"] == "provider_error").sum(),
            "median_seconds": runs["seconds"].median(),
            # Of the steps whose first code failed, how many did self-repair rescue?
            "repair_success_rate": runs["repaired_ok"].sum() / repaired if repaired else None,
        }
