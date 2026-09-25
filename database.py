
"""
database.py
============
Lightweight SQLite persistence layer for storing prediction history.

The database is purely local and temporary (``temp_database.db``) -- there
is no authentication or multi-user support, consistent with the project's
"simple local tool" design goal.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from config import DATABASE_PATH, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Represents a single stored prediction."""

    id: int
    timestamp: str
    risk_band: str
    risk_score: float
    confidence: float
    patient_inputs: dict


class Database:
    """Thin wrapper around SQLite for prediction history management."""

    def __init__(self, db_path: str = DATABASE_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    risk_band TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    patient_inputs TEXT NOT NULL
                )
                """
            )
        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    def insert_prediction(
        self,
        risk_band: str,
        risk_score: float,
        confidence: float,
        patient_inputs: dict,
    ) -> int:
        """Insert a new prediction record and return its row id."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prediction_history
                    (timestamp, risk_band, risk_score, confidence, patient_inputs)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, risk_band, risk_score, confidence, json.dumps(patient_inputs)),
            )
            new_id = cursor.lastrowid
        logger.info("Inserted prediction record id=%s risk_band=%s", new_id, risk_band)
        return new_id

    def get_history(self, limit: int = 200) -> list[PredictionRecord]:
        """Return the most recent prediction records, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, risk_band, risk_score, confidence, patient_inputs
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            PredictionRecord(
                id=row[0],
                timestamp=row[1],
                risk_band=row[2],
                risk_score=row[3],
                confidence=row[4],
                patient_inputs=json.loads(row[5]),
            )
            for row in rows
        ]

    def delete_record(self, record_id: int) -> None:
        """Delete a single prediction record by id."""
        with self._connect() as conn:
            conn.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
        logger.info("Deleted prediction record id=%s", record_id)

    def clear_all(self) -> None:
        """Delete every stored prediction record."""
        with self._connect() as conn:
            conn.execute("DELETE FROM prediction_history")
        logger.info("Cleared all prediction history.")

    def count(self) -> int:
        """Return the total number of stored prediction records."""
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()
        return int(n)
