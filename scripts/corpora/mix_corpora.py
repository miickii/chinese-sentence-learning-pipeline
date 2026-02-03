#!/usr/bin/env python3
"""
mix_corpora.py

Purpose:
  Mix two already-clean sentence corpora (Wikipedia + Leipzig news) into one global prior file.

Inputs (must already be cleaned by previous scripts):
  - data/processed/wiki.sentences.txt
  - data/processed/leipzig_news.sentences.txt

Outputs:
  - data/processed/global_prior.sentences.txt
  - Guarantees:
      * Samples approximately wiki_n and news_n lines (subject to available lines)
      * Final global dedup (removes any duplicates across corpora)
      * Shuffled output

Important:
  - This script DOES NOT re-check traditional/simplified (to avoid redundancy).
    That is enforced in wiki_to_sentences.py and clean_leipzig.py.

Usage:
  python scripts/mix_corpora.py \
    --wiki data/processed/wiki.sentences.txt \
    --news data/processed/leipzig_news.sentences.txt \
    --out  data/processed/global_prior.sentences.txt \
    --wiki_n 600000 \
    --news_n 400000 \
    --seed 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List, Set


def iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def reservoir_sample(path: Path, k: int, rng: random.Random) -> List[str]:
    """
    Reservoir sample k items from an iterable without loading whole file.
    Assumes lines are already clean and deduplicated within the corpus.
    """
    res: List[str] = []
    n = 0
    for s in iter_lines(path):
        n += 1
        if len(res) < k:
            res.append(s)
        else:
            j = rng.randrange(n)
            if j < k:
                res[j] = s

    if len(res) < k:
        print(f"⚠️  {path} had only {len(res)} lines available (target {k}).", file=sys.stderr)
    return res


def write_shuffled_global_dedup(samples: List[str], out_path: Path, rng: random.Random) -> int:
    rng.shuffle(samples)
    seen: Set[str] = set()
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for s in samples:
            if s in seen:
                continue
            seen.add(s)
            out.write(s + "\n")
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--news", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki_n", type=int, required=True)
    ap.add_argument("--news_n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    wiki_path = Path(args.wiki)
    news_path = Path(args.news)
    out_path = Path(args.out)

    rng = random.Random(args.seed)

    wiki_sample = reservoir_sample(wiki_path, args.wiki_n, rng)
    news_sample = reservoir_sample(news_path, args.news_n, rng)

    combined = wiki_sample + news_sample
    written = write_shuffled_global_dedup(combined, out_path, rng)

    print(f"✅ wrote {written} unique mixed sentences to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
