"""
utils/io.py

What this file does:
- Reads an Anki-exported CSV and extracts a single column into a list of strings.

How it fits:
- This is the only place where Anki/CSV format matters.
- After bootstrap, the system uses SQLite as the source of truth.
"""

from __future__ import annotations

import csv
from pathlib import Path

def read_csv_column(path: str | Path, column: str) -> list[str]:
    path = Path(path)
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row (no fieldnames).")
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found. Available: {reader.fieldnames}")
        for row in reader:
            txt = (row.get(column) or "").strip()
            if txt:
                out.append(txt)
    return out
