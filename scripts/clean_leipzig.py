#!/usr/bin/env python3
"""
clean_leipzig.py

Purpose:
  Clean a Leipzig "sentences" file into a sentence-level corpus compatible with the pipeline.

Input:
  - Leipzig sentences file, commonly either:
      * "ID<TAB>sentence"
      * or just "sentence"
    UTF-8-ish text

Output:
  - A UTF-8 text file: one sentence per line
  - Guarantees:
      * Strict simplified-only (drops sentences that OpenCC t2s would change)
      * No duplicates (global dedup within this corpus)
      * Basic cleaning + length/noise filtering

Usage:
  python scripts/clean_leipzig.py data/raw/leipzig/zho_news_2007-2009_1M-sentences.txt data/processed/leipzig_news.sentences.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from opencc import OpenCC


def normalize(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # Many Leipzig sentence files are "ID<TAB>sentence"
    if "\t" in s:
        s = s.split("\t", 1)[1].strip()
    # Remove whitespace inside Chinese sentences
    s = re.sub(r"\s+", "", s)
    # Remove common quote marks
    s = re.sub(r'[“”"\'‘’]', "", s)
    return s


def good_sentence_basic(s: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(s) <= max_len):
        return False
    if re.search(r"[A-Za-z0-9]{6,}", s):
        return False
    return True


def is_simplified_strict(s: str, cc_t2s: OpenCC) -> bool:
    return s == cc_t2s.convert(s)


def process_sentence(s: str, cc_t2s: OpenCC, min_len: int, max_len: int) -> Optional[str]:
    s = normalize(s)
    if not s:
        return None
    if not good_sentence_basic(s, min_len=min_len, max_len=max_len):
        return None
    if not is_simplified_strict(s, cc_t2s):
        return None
    return s


def main(in_path: str, out_path: str, min_len: int = 6, max_len: int = 60) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    cc_t2s = OpenCC("t2s")

    seen = set()
    kept = 0

    with in_path.open("r", encoding="utf-8", errors="ignore") as f, \
         out_path.open("w", encoding="utf-8") as out:
        for line in f:
            s = process_sentence(line, cc_t2s, min_len, max_len)
            if s and s not in seen:
                seen.add(s)
                out.write(s + "\n")
                kept += 1

    print(f"✅ wrote {kept} unique simplified sentences to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: clean_leipzig.py <in.txt> <out.txt>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
