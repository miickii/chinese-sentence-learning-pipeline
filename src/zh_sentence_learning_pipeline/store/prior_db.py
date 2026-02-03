"""
store/prior_db.py

SQLite schema + helpers for the GLOBAL grammar prior database.

This DB is built from a huge corpus (e.g., 1M sentences) and stores:
- pattern_global_stats: global counts + (sampled) diversity
- pattern_global_realizations: example realizations per pattern (capped)

This DB is NOT learner-specific.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PRIOR_SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pattern_global_stats (
  pattern_id TEXT PRIMARY KEY,
  count      INTEGER NOT NULL,
  diversity  INTEGER NOT NULL,
  log_freq   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pattern_global_realizations (
  pattern_id   TEXT NOT NULL,
  realization  TEXT NOT NULL,
  PRIMARY KEY (pattern_id, realization)
);

CREATE INDEX IF NOT EXISTS idx_pgs_count ON pattern_global_stats(count);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
  conn = sqlite3.connect(str(db_path))
  conn.row_factory = sqlite3.Row
  return conn


def init_prior_db(conn: sqlite3.Connection) -> None:
  conn.executescript(DDL)
  conn.execute(
    "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
    ("prior_schema_version", str(PRIOR_SCHEMA_VERSION)),
  )
  conn.commit()
