#!/usr/bin/env python3
"""
wiki_to_sentences.py

Purpose:
  Convert WikiExtractor JSON output for Chinese Wikipedia into a clean, sentence-level corpus.

Input:
  - A directory produced by WikiExtractor --json (many files of JSON lines)
    Each line is a JSON object like: {"id": "...", "text": "..."}

Output:
  - A UTF-8 text file: one sentence per line
  - Guarantees:
      * Strict simplified-only (drops sentences that OpenCC t2s would change)
      * No duplicates (global dedup)
      * Basic cleaning + length/noise filtering

Usage:
  python scripts/wiki_to_sentences.py data/raw/wiki/extracted data/processed/wiki.sentences.txt
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from opencc import OpenCC

# Sentence split: keep end punctuation
SENT_SPLIT = re.compile(r"(?<=[。！？])")


def normalize(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # Remove whitespace inside Chinese sentences
    s = re.sub(r"\s+", "", s)
    # Remove common quote marks
    s = re.sub(r'[“”"\'‘’]', "", s)
    # Remove parentheticals (often citations/explanations)
    s = re.sub(r"（.*?）|\(.*?\)", "", s)
    # Remove citation markers like [1], [23]
    s = re.sub(r"\[[0-9]+\]", "", s)
    return s


def good_sentence_basic(s: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(s) <= max_len):
        return False
    # Avoid long latin/digit runs
    if re.search(r"[A-Za-z0-9]{6,}", s):
        return False
    return True


def is_simplified_strict(s: str, cc_t2s: OpenCC) -> bool:
    """
    Strict simplified check:
      If converting Traditional->Simplified changes the string,
      we treat it as containing Traditional and drop it.
    """
    return s == cc_t2s.convert(s)


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def process_sentence(
    sent: str,
    cc_t2s: OpenCC,
    min_len: int,
    max_len: int,
) -> Optional[str]:
    sent = normalize(sent)
    if not sent:
        return None
    if not good_sentence_basic(sent, min_len=min_len, max_len=max_len):
        return None
    if not is_simplified_strict(sent, cc_t2s):
        return None
    return sent


def main(extracted_dir: str, out_path: str, min_len: int = 6, max_len: int = 60) -> None:
    extracted_dir = Path(extracted_dir)
    out_path = Path(out_path)

    cc_t2s = OpenCC("t2s")

    seen = set()
    kept = 0

    with out_path.open("w", encoding="utf-8") as out:
        for fp in iter_jsonl_files(extracted_dir):
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        for sent in SENT_SPLIT.split(text):
                            cleaned = process_sentence(sent, cc_t2s, min_len, max_len)
                            if cleaned and cleaned not in seen:
                                seen.add(cleaned)
                                out.write(cleaned + "\n")
                                kept += 1
            except Exception:
                # skip any unreadable file / malformed JSON chunk
                continue

    print(f"✅ wrote {kept} unique simplified sentences to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: wiki_to_sentences.py <extracted_dir> <out.txt>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
