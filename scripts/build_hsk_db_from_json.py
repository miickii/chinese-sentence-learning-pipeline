#!/usr/bin/env python3
"""
scripts/build_hsk_db_from_json.py

Build a new HSK vocabulary SQLite database from complete_hsk.json.

Outputs a SQLite DB with table:
  chinese_words(simplified, level, frequency, pinyin, meanings, pos, traditional)

This is compatible with your existing loader:
  HSKLexicon.from_sqlite(... table="chinese_words")

Usage:
  PYTHONPATH=src python scripts/build_hsk_db_from_json.py \
    --json data/complete_hsk.json \
    --out  data/hsk_vocabulary_v2.db

Notes:
- If your JSON path is different (e.g. downloaded elsewhere), pass --json explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LEVEL_RE = re.compile(r"^(new|old)-(\d+)$")


DDL = """
PRAGMA journal_mode=WAL;

DROP TABLE IF EXISTS chinese_words;

CREATE TABLE chinese_words (
  simplified   TEXT PRIMARY KEY,
  level        INTEGER,
  frequency    INTEGER,
  pinyin       TEXT,
  meanings     TEXT,
  pos          TEXT,
  traditional  TEXT
);

CREATE INDEX idx_chinese_words_level ON chinese_words(level);
CREATE INDEX idx_chinese_words_frequency ON chinese_words(frequency);
"""


def parse_level(level_list: List[str]) -> Optional[int]:
    """
    Given a list like ["new-2","old-3"], choose:
      - smallest numeric among "new-*", else
      - smallest numeric among "old-*"
    Return int level, or None if cannot parse.
    """
    new_levels: List[int] = []
    old_levels: List[int] = []
    for s in level_list or []:
        m = LEVEL_RE.match(str(s).strip())
        if not m:
            continue
        scheme, num = m.group(1), int(m.group(2))
        if scheme == "new":
            new_levels.append(num)
        else:
            old_levels.append(num)

    if new_levels:
        return min(new_levels)
    if old_levels:
        return min(old_levels)
    return None


def uniq_join(items: Iterable[str], sep: str = "; ") -> Optional[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    if not out:
        return None
    return sep.join(out)


def extract_forms(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (traditional, pinyin, meanings) aggregated across forms.
    """
    forms = entry.get("forms") or []
    trads: List[str] = []
    pins: List[str] = []
    means: List[str] = []

    for f in forms:
        if not isinstance(f, dict):
            continue
        trads.append(str(f.get("traditional") or "").strip())

        trans = f.get("transcriptions") or {}
        if isinstance(trans, dict):
            pins.append(str(trans.get("pinyin") or "").strip())

        mlist = f.get("meanings") or []
        if isinstance(mlist, list):
            means.extend([str(m).strip() for m in mlist])

    return (
        uniq_join(trads, sep="; "),
        uniq_join(pins, sep="; "),
        uniq_join(means, sep="; "),
    )


def build_rows(json_path: Path) -> List[Tuple[str, Optional[int], Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of entries.")

    rows: List[Tuple[str, Optional[int], Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]] = []
    for e in data:
        if not isinstance(e, dict):
            continue
        simp = str(e.get("simplified") or "").strip()
        if not simp:
            continue

        level = parse_level(list(e.get("level") or []))
        freq_raw = e.get("frequency")
        freq = int(freq_raw) if isinstance(freq_raw, int) else (int(freq_raw) if str(freq_raw).isdigit() else None)

        pos_list = e.get("pos") or []
        pos = None
        if isinstance(pos_list, list):
            pos = uniq_join([str(p) for p in pos_list], sep=";")

        traditional, pinyin, meanings = extract_forms(e)

        rows.append((simp, level, freq, pinyin, meanings, pos, traditional))

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to complete_hsk.json")
    ap.add_argument("--out", required=True, help="Output SQLite DB path (e.g. data/hsk_vocabulary_v2.db)")
    args = ap.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    rows = build_rows(json_path)

    conn = sqlite3.connect(str(out_path))
    try:
        conn.executescript(DDL)
        conn.executemany(
            """
            INSERT INTO chinese_words(simplified, level, frequency, pinyin, meanings, pos, traditional)
            VALUES(?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    print(f"✅ Built DB: {out_path}")
    print(f"   rows inserted: {len(rows)}")
    print("   table: chinese_words (simplified, level, frequency, pinyin, meanings, pos, traditional)")


if __name__ == "__main__":
    main()
# PYTHONPATH=src python scripts/build_hsk_db_from_json.py --json data/complete_hsk.json --out data/hsk_vocabulary_v2.db