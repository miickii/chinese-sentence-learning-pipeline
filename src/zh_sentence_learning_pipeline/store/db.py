"""
store/db.py

What this file does:
- Defines the SQLite schema for the project (bootstrap state DB).
- Provides connect() and init_db() helpers.

How it fits:
- Bootstrapper initializes the DB and writes:
  - sentences (jieba tokens + HSK tokens + char tokens + patterns + skeleton)
  - vocab_stats (based on HSK tokens)
  - pattern_stats + realizations (grammar Layer B+C)
  - meta (schema version + config + timestamps)

Notes:
- We keep schema stable (SCHEMA_VERSION=2).
- New config is stored in meta (anchors_json, extractor toggles, etc.).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  zh_text TEXT NOT NULL,
  tokens_jieba_json TEXT NOT NULL,
  tokens_hsk_json TEXT NOT NULL,
  tokens_char_json TEXT NOT NULL,
  patterns_json TEXT NOT NULL,
  skeleton TEXT,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(zh_text, source)
);

CREATE INDEX IF NOT EXISTS idx_sentences_source ON sentences(source);

-- Vocab stats are based on HSK-aligned tokens to support the HSK queue later
CREATE TABLE IF NOT EXISTS vocab_stats (
  word TEXT PRIMARY KEY,
  count INTEGER NOT NULL,
  mastery REAL NOT NULL,
  last_seen TEXT,
  hsk_level INTEGER,
  hsk_frequency INTEGER
);

CREATE TABLE IF NOT EXISTS pattern_stats (
  pattern_id TEXT PRIMARY KEY,
  count INTEGER NOT NULL,
  mastery REAL NOT NULL,
  diversity INTEGER NOT NULL,
  emerged INTEGER NOT NULL,
  last_seen TEXT
);

CREATE TABLE IF NOT EXISTS pattern_realizations (
  pattern_id TEXT NOT NULL,
  realization TEXT NOT NULL,
  PRIMARY KEY (pattern_id, realization)
);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  created_at TEXT NOT NULL,
  config_hash TEXT,
  anchors_hash TEXT,
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(kind);
"""

def connect(db_path: str | Path) -> sqlite3.Connection:
  conn = sqlite3.connect(str(db_path))
  conn.row_factory = sqlite3.Row
  return conn

def init_db(conn: sqlite3.Connection) -> None:
  conn.executescript(DDL)
  conn.execute(
    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
    ("schema_version", str(SCHEMA_VERSION)),
  )
  conn.commit()
