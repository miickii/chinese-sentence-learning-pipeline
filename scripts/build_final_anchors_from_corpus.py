#!/usr/bin/env python3
"""
scripts/build_final_anchors_from_corpus.py

What this script does
---------------------
Build a *final* global anchor list by re-scoring / filtering anchor candidates against a large corpus.

Why this exists
---------------
Your current `scripts/build_global_anchors.py` builds anchors from a dictionary/HSK JSON by POS tags.
That gets you a reasonable *candidate* list, but it can still include "too-contenty" tokens (e.g. common
adjectives) and it doesn't know what is actually frequent / widely distributed in *your* corpus.

This script takes:
  1) a candidate anchor list (e.g. data/global_anchors.json)
  2) a large corpus text file (one sentence per line)

…and computes corpus statistics for each candidate:
  - DF: in how many sentences does the token appear?
  - TF: how many total occurrences?
  - neighbor entropy: how diverse are its immediate left/right contexts?

Then it outputs a refined anchor list that is:
  - widely distributed (high DF or DF-rate)
  - context-flexible (high neighbor entropy)

This tends to keep "true function-ish anchors" and drop topical/content-ish tokens.

Output
------
JSON of the form:
  {
    "anchors": ["的","了","在", ...],
    "meta": {... thresholds, corpus_size, created_at ...},
    "stats": {
       "的": {"df": 12345, "tf": 45678, "df_rate": 0.33, "H_lr": 4.12, "score": 2.91},
       ...
    }
  }

You can keep only "anchors" for runtime; "stats" is for sanity-checking and tuning.

Usage
-----
PYTHONPATH=src python scripts/build_final_anchors_from_corpus.py \
  --candidates data/global_anchors.core_candidates.json \
  --corpus data/combined_corpus_simplified.txt \
  --out data/global_anchors.final.json \
  --topk 600 \
  --min-df 50 \
  --min-df-rate 0.0005 \
  --min-entropy 2.0

Notes
-----
- Corpus should be one sentence per line (or at least one "unit" per line).
- Tokenization uses jieba if available, otherwise it falls back to character-level tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def try_import_jieba():
    try:
        import jieba  # type: ignore
        return jieba
    except Exception:
        return None


def entropy(counts: Counter[str]) -> float:
    """Shannon entropy in nats (natural log)."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def load_candidates(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "anchors" in data and isinstance(data["anchors"], list):
        return [str(x).strip() for x in data["anchors"] if str(x).strip()]
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    raise ValueError(f"Unrecognized candidate format: {path}")


def iter_corpus_lines(path: Path, limit: Optional[int] = None) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            s = line.strip()
            if not s:
                continue
            yield s


def tokenize_sentence(text: str, jieba_mod) -> List[str]:
    if jieba_mod is not None:
        # precise mode is fine for this statistical scoring
        return [t.strip() for t in jieba_mod.lcut(text) if t.strip()]
    # fallback: char tokens
    return [ch for ch in text if not ch.isspace()]


@dataclass
class AnchorStats:
    df: int = 0
    tf: int = 0
    left: Counter[str] = None  # type: ignore
    right: Counter[str] = None  # type: ignore

    def __post_init__(self):
        if self.left is None:
            self.left = Counter()
        if self.right is None:
            self.right = Counter()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Candidate anchors JSON (e.g. data/global_anchors.json)")
    ap.add_argument("--corpus", required=True, help="Large corpus txt (one sentence per line)")
    ap.add_argument("--out", required=True, help="Output JSON (e.g. data/global_anchors.final.json)")
    ap.add_argument("--topk", type=int, default=600, help="Keep top-K anchors after scoring (default: 600)")
    ap.add_argument("--max-len", type=int, default=4, help="Max anchor length in characters (default: 4)")
    ap.add_argument("--min-df", type=int, default=50, help="Minimum sentence DF count (default: 50)")
    ap.add_argument("--min-df-rate", type=float, default=0.0, help="Minimum DF-rate (df / num_sentences)")
    ap.add_argument("--min-entropy", type=float, default=1.8, help="Minimum neighbor entropy (default: 1.8 nats)")
    ap.add_argument("--limit-lines", type=int, default=0, help="Debug: only read first N lines (0 = all)")
    ap.add_argument("--no-stats", action="store_true", help="If set, omit per-anchor stats in output JSON")
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    corpus_path = Path(args.corpus)
    out_path = Path(args.out)

    if not cand_path.exists():
        raise FileNotFoundError(f"Candidates not found: {cand_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    candidates = load_candidates(cand_path)
    # de-dup while preserving order
    seen: Set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    # length filter early
    candidates = [c for c in candidates if len(c) <= args.max_len]

    cand_set = set(candidates)
    stats: Dict[str, AnchorStats] = {c: AnchorStats() for c in candidates}

    jieba_mod = try_import_jieba()
    if jieba_mod is None:
        print("⚠️  jieba not available; falling back to character tokenization", file=sys.stderr)

    limit = None if args.limit_lines == 0 else args.limit_lines
    num_sentences = 0

    for sent in iter_corpus_lines(corpus_path, limit=limit):
        num_sentences += 1
        toks = tokenize_sentence(sent, jieba_mod)
        present = set(toks) & cand_set
        for a in present:
            stats[a].df += 1

        # TF + neighbor context
        for i, t in enumerate(toks):
            if t not in cand_set:
                continue
            st = stats[t]
            st.tf += 1
            left_tok = toks[i - 1] if i - 1 >= 0 else "<BOS>"
            right_tok = toks[i + 1] if i + 1 < len(toks) else "<EOS>"
            st.left[left_tok] += 1
            st.right[right_tok] += 1

    if num_sentences == 0:
        raise ValueError("Corpus appears empty after stripping lines.")

    scored: List[Tuple[str, float, Dict[str, float]]] = []
    for a, st in stats.items():
        df_rate = st.df / num_sentences
        H_l = entropy(st.left)
        H_r = entropy(st.right)
        H_lr = 0.5 * (H_l + H_r)

        # A conservative, tunable score:
        # - distribution matters most (df_rate)
        # - context diversity helps separate function words from topical terms
        # - TF helps break ties (log-scaled)
        score = (2.0 * df_rate) + (0.25 * H_lr) + (0.10 * math.log1p(st.tf))

        aux = {
            "df": float(st.df),
            "tf": float(st.tf),
            "df_rate": float(df_rate),
            "H_l": float(H_l),
            "H_r": float(H_r),
            "H_lr": float(H_lr),
            "score": float(score),
        }
        scored.append((a, score, aux))

    # Apply hard filters
    filtered = []
    for a, score, aux in scored:
        if aux["df"] < args.min_df:
            continue
        if aux["df_rate"] < args.min_df_rate:
            continue
        if aux["H_lr"] < args.min_entropy:
            continue
        filtered.append((a, score, aux))

    # Sort by score desc
    filtered.sort(key=lambda x: x[1], reverse=True)
    kept = filtered[: max(args.topk, 0)]

    anchors = [a for a, _, _ in kept]

    out: Dict[str, object] = {
        "anchors": anchors,
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_candidates": str(cand_path),
            "source_corpus": str(corpus_path),
            "num_sentences": num_sentences,
            "thresholds": {
                "topk": args.topk,
                "max_len": args.max_len,
                "min_df": args.min_df,
                "min_df_rate": args.min_df_rate,
                "min_entropy": args.min_entropy,
            },
            "tokenizer": "jieba" if jieba_mod is not None else "char_fallback",
        },
    }

    if not args.no_stats:
        out_stats: Dict[str, Dict[str, float]] = {}
        for a, score, aux in kept:
            out_stats[a] = {k: float(v) for k, v in aux.items()}
        out["stats"] = out_stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote: {out_path}")
    print(f"   corpus sentences: {num_sentences}")
    print(f"   candidates: {len(candidates)}")
    print(f"   passed filters: {len(filtered)}")
    print(f"   kept topk: {len(anchors)}")
    print("   sample:", anchors[:25])


if __name__ == "__main__":
    main()

'''
Current used command:
PYTHONPATH=src python scripts/build_final_anchors_from_corpus.py \
  --candidates data/global_anchors.core_candidates.json \
  --corpus data/combined_corpus_simplified.txt \
  --out data/global_anchors.final.balanced.json \
  --topk 400 \
  --max-len 4 \
  --min-df 100 \
  --min-df-rate 0.0003 \
  --min-entropy 2.0
'''