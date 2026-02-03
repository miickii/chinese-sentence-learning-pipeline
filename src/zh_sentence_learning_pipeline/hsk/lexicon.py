"""
hsk/lexicon.py

Purpose
-------
Load an HSK vocabulary lexicon from an SQLite database and provide:

1) Fast lookup: word -> metadata
2) Fast segmentation primitive: Trie-based "longest match" from any index in a string

New in this version
-------------------
- Backwards compatible with your older DB.
- Auto-detects optional columns (e.g. traditional, pos) if present in your v2 DB.

How it's used in the project
----------------------------
- Bootstrapper: tokenize known sentences into stable "HSK vocab units"
- HSK queue: pick the next target word
- Generator/mutator: enforce lexical constraints using HSK vocabulary units
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, Set


@dataclass(frozen=True)
class HSKEntry:
    simplified: str
    level: int
    frequency: Optional[int]
    pinyin: Optional[str]
    meanings: Optional[str]
    traditional: Optional[str] = None
    pos: Optional[str] = None


class TrieNode:
    __slots__ = ("children", "terminal_word")
    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.terminal_word: Optional[str] = None


class HSKLexicon:
    """
    Holds:
    - entries: word -> HSKEntry
    - trie: for longest-match segmentation
    """

    def __init__(self, entries: Dict[str, HSKEntry]) -> None:
        self.entries = entries
        self.trie = TrieNode()
        self._build_trie(entries.keys())

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}  # (cid, name, type, notnull, dflt_value, pk)

    @classmethod
    def from_sqlite(
        cls,
        db_path: str,
        max_level: int = 6,
        include_level7: bool = False,
        table: str = "chinese_words",
    ) -> "HSKLexicon":
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"HSK DB not found: {db_path}")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cols = cls._table_columns(conn, table)

            wanted = ["simplified", "level", "frequency", "pinyin", "meanings"]
            if "traditional" in cols:
                wanted.append("traditional")
            if "pos" in cols:
                wanted.append("pos")

            # include_level7 means allow level=7 as well; otherwise exclude it
            if include_level7:
                lvl_clause = " AND (level <= ? OR level = 7)"
                params = [max_level]
            else:
                lvl_clause = " AND level <= ? AND level != 7"
                params = [max_level]

            sql = f"""
            SELECT {", ".join(wanted)}
            FROM {table}
            WHERE simplified IS NOT NULL AND TRIM(simplified) != ''
            {lvl_clause}
            """
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        entries: Dict[str, HSKEntry] = {}
        for r in rows:
            w = (r["simplified"] or "").strip()
            if not w:
                continue

            lvl = int(r["level"]) if r["level"] is not None else 999

            freq = int(r["frequency"]) if ("frequency" in r.keys() and r["frequency"] is not None) else None
            pinyin = r["pinyin"] if "pinyin" in r.keys() else None
            meanings = r["meanings"] if "meanings" in r.keys() else None
            trad = r["traditional"] if "traditional" in r.keys() else None
            pos = r["pos"] if "pos" in r.keys() else None

            cand = HSKEntry(w, lvl, freq, pinyin, meanings, traditional=trad, pos=pos)

            # Keep best entry if duplicates:
            # prefer lower level, then better (lower) frequency rank if available
            if w in entries:
                prev = entries[w]
                prev_key = (prev.level, prev.frequency if prev.frequency is not None else 10**9)
                new_key = (cand.level, cand.frequency if cand.frequency is not None else 10**9)
                if new_key < prev_key:
                    entries[w] = cand
            else:
                entries[w] = cand

        return cls(entries)

    def _build_trie(self, words: Iterable[str]) -> None:
        """
        ---------------------------
        Trie ("prefix tree") overview
        ---------------------------
        A Trie stores many words by sharing common prefixes.
        
        Example words: ["学", "学校", "学生"]
        
        Root
         └─ '学' (terminal_word="学")
               ├─ '校' (terminal_word="学校")
               └─ '生' (terminal_word="学生")
        """
        for w in words:
            node = self.trie
            for ch in w:
                node = node.children.setdefault(ch, TrieNode())
            node.terminal_word = w

    def longest_match(self, text: str, start: int) -> Optional[Tuple[str, int]]:
        """
        Returns (word, end_index) for the longest word match starting at `start`, or None.
        end_index is exclusive.
        """
        node = self.trie
        best_word: Optional[str] = None
        best_end = start

        i = start
        while i < len(text):
            ch = text[i]
            if ch not in node.children:
                break
            node = node.children[ch]
            i += 1
            if node.terminal_word is not None:
                best_word = node.terminal_word
                best_end = i

        if best_word is None:
            return None
        return best_word, best_end

    def meta(self, word: str) -> Optional[HSKEntry]:
        return self.entries.get(word)