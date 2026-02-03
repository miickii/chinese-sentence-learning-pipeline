#!/usr/bin/env python3
"""
scripts/inspect_bootstrap_state.py

Bootstrap inspection + evaluation script (single file).

What it does
------------
Reads your project state SQLite DB and prints a high-signal report to help you
validate whether bootstrap + anchors + pattern extraction are working as intended.

This version is UPDATED for your new anchor strategy:
- global_anchors.json is treated as an "allowed candidates" list (POS-derived).
- actual anchors are "activated locally" by DF/TF from your own deck and stored in meta.anchors_json.

Report sections
---------------
1) DB integrity: tables exist, row counts, meta keys
2) Anchor diagnostics:
   - activated anchors from meta
   - optional: global candidate anchors file (allowed set)
   - DF stats across corpus
   - sanity checks: did activated anchors come from candidates? are they high DF?
3) Random sentence spot-checks (jieba/hsk/char tokens + pattern count + skeleton)
4) Pattern health:
   - count distribution
   - emerged ratio
   - top emerged patterns + sample realizations
5) Vocab health (HSK tokens):
   - unknown/fallback ratio
   - top vocab tokens
   - single-char ratio
6) Length robustness demo:
   - compares pattern overlap for short vs long variants

Usage
-----
PYTHONPATH=src python scripts/inspect_bootstrap_state.py --db data/state.db --global-anchors data/global_anchors.json --pairs 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------
# Helpers: printing
# ---------------------------

def hr(ch: str = "=", n: int = 80) -> str:
    return ch * n

def fmt_pct(a: int, b: int) -> str:
    if b <= 0:
        return "0.0%"
    return f"{(100.0 * a / b):.1f}%"

def take(n: int, it: Sequence[Any]) -> Sequence[Any]:
    return it[:n]

def safe_json_loads(s: str, fallback: Any) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return fallback


# ---------------------------
# DB access
# ---------------------------

EXPECTED_TABLES = {
    "meta",
    "sentences",
    "vocab_stats",
    "pattern_personal_stats",
    "pattern_personal_realizations",
}

def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]

def table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])

def read_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {r["key"]: r["value"] for r in rows}

def sample_sentences(conn: sqlite3.Connection, k: int) -> List[sqlite3.Row]:
    # Random sample without loading all rows into python
    return conn.execute(
        """
        SELECT id, zh_text, tokens_jieba_json, tokens_hsk_json, tokens_char_json, patterns_json, skeleton
        FROM sentences
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (k,),
    ).fetchall()


# ---------------------------
# Anchor loading + DF stats
# ---------------------------

def load_global_candidates(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"global anchors file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = payload.get("anchors", None)
    if not isinstance(anchors, list):
        raise ValueError(f"Invalid global anchors payload in {path}: expected {{'anchors':[...]}}, got keys={list(payload.keys())}")
    out = {str(a).strip() for a in anchors if str(a).strip()}
    return out

def load_activated_anchors(meta: Dict[str, str]) -> Optional[Set[str]]:
    s = meta.get("anchors_json")
    if not s:
        return None
    arr = safe_json_loads(s, fallback=None)
    if not isinstance(arr, list):
        return None
    return {str(x) for x in arr}

def parse_int_meta(meta: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(meta.get(key, str(default)))
    except Exception:
        return default

def parse_bool_meta(meta: Dict[str, str], key: str, default: bool) -> bool:
    v = meta.get(key, None)
    if v is None:
        return default
    return str(v).strip() in {"1", "true", "True", "yes", "YES"}

def corpus_df(tokens_jieba_all: List[List[str]], max_len: Optional[int] = None) -> Counter[str]:
    df = Counter()
    for sent in tokens_jieba_all:
        if max_len is None:
            uniq = set(sent)
        else:
            uniq = {t for t in sent if len(t) <= max_len}
        df.update(uniq)
    return df

def corpus_tf(tokens_jieba_all: List[List[str]], max_len: Optional[int] = None) -> Counter[str]:
    tf = Counter()
    for sent in tokens_jieba_all:
        if max_len is None:
            tf.update(sent)
        else:
            tf.update([t for t in sent if len(t) <= max_len])
    return tf

def read_all_jieba_tokens(conn: sqlite3.Connection) -> List[List[str]]:
    rows = conn.execute("SELECT tokens_jieba_json FROM sentences").fetchall()
    out: List[List[str]] = []
    for r in rows:
        toks = safe_json_loads(r["tokens_jieba_json"], fallback=[])
        if isinstance(toks, list):
            out.append([str(x) for x in toks])
    return out


# ---------------------------
# Pattern health
# ---------------------------

@dataclass
class PatternRow:
    pattern_key: str
    count_seen: int
    distinct_sentence_count: int
    emerged: int

def top_patterns(conn: sqlite3.Connection, limit: int = 20, emerged_only: bool = True) -> List[PatternRow]:
    if emerged_only:
        rows = conn.execute(
            """
            SELECT pattern_key, count_seen, distinct_sentence_count, emerged
            FROM pattern_personal_stats
            WHERE emerged = 1
            ORDER BY count_seen DESC, distinct_sentence_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT pattern_key, count_seen, distinct_sentence_count, emerged
            FROM pattern_personal_stats
            ORDER BY count_seen DESC, distinct_sentence_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        PatternRow(
            r["pattern_key"],
            int(r["count_seen"]),
            int(r["distinct_sentence_count"]),
            int(r["emerged"]),
        )
        for r in rows
    ]

def sample_realizations(conn: sqlite3.Connection, pkey: str, k: int = 6) -> List[str]:
    rows = conn.execute(
        """
        SELECT realization
        FROM pattern_personal_realizations
        WHERE pattern_key = ?
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (pkey, k),
    ).fetchall()
    return [r["realization"] for r in rows]

def pattern_count_hist(conn: sqlite3.Connection) -> Counter[int]:
    rows = conn.execute("SELECT count_seen AS c FROM pattern_personal_stats").fetchall()
    hist = Counter(int(r["c"]) for r in rows)
    return hist


# ---------------------------
# Vocab health
# ---------------------------

def vocab_unknown_ratio(conn: sqlite3.Connection) -> Tuple[int, int]:
    total = table_count(conn, "vocab_stats")
    unk = int(conn.execute("SELECT COUNT(*) AS n FROM vocab_stats WHERE hsk_level IS NULL").fetchone()["n"])
    return unk, total

def top_vocab(conn: sqlite3.Connection, k: int = 20) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT word, count, hsk_level, hsk_frequency
        FROM vocab_stats
        ORDER BY count DESC
        LIMIT ?
        """,
        (k,),
    ).fetchall()

def single_char_vocab_count(conn: sqlite3.Connection) -> Tuple[int, int]:
    total = table_count(conn, "vocab_stats")
    one = int(conn.execute("SELECT COUNT(*) AS n FROM vocab_stats WHERE LENGTH(word) = 1").fetchone()["n"])
    return one, total


# ---------------------------
# Length robustness demo
# ---------------------------

def try_import_project_modules():
    """
    Try to import your project tokenizer and extractor.
    If it fails, we still run everything except the robustness demo.
    """
    try:
        from zh_sentence_learning_pipeline.grammar.tokenize import tokenize_words_jieba
        from zh_sentence_learning_pipeline.grammar.patterns import extract_patterns_from_tokens
        return tokenize_words_jieba, extract_patterns_from_tokens
    except Exception:
        return None, None

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to state.db")
    ap.add_argument("--global-anchors", default=None, help="Path to global_anchors.json (candidate whitelist)")
    ap.add_argument("--pairs", type=int, default=6, help="How many random sentences to spot-check")
    ap.add_argument("--seed", type=int, default=13, help="Random seed for repeatable output")
    ap.add_argument("--print-anchors", type=int, default=120, help="Print first N anchors (sorted)")
    ap.add_argument("--top-patterns", type=int, default=20, help="How many top patterns to print")
    ap.add_argument("--top-vocab", type=int, default=20, help="How many top vocab to print")

    # Optional: recompute what anchors would be (without changing DB)
    ap.add_argument("--recompute-anchors", action="store_true", help="Recompute anchors from corpus for comparison")
    ap.add_argument("--anchors-top-k", type=int, default=70, help="Top-K anchors to compute when recomputing")
    ap.add_argument("--anchor-method", choices=["df", "tf"], default="df", help="Method to compute anchors when recomputing")
    ap.add_argument("--anchor-max-len", type=int, default=2, help="Max token length for anchor candidates when recomputing")

    # Robustness demo knobs (if import works)
    ap.add_argument("--max-ngram-n", type=int, default=None, help="Override extractor max_ngram_n (else uses meta)")
    ap.add_argument("--span-max-gap", type=int, default=None, help="Override span_max_gap (else uses meta)")
    ap.add_argument("--skip-max-jump", type=int, default=None, help="Override skip_max_jump (else uses meta)")
    ap.add_argument("--no-skipgrams", action="store_true", help="Disable anchor skip-grams in the demo")
    ap.add_argument("--no-spans", action="store_true", help="Disable span patterns in the demo")
    ap.add_argument("--no-span-sigs", action="store_true", help="Disable span signatures in the demo")
    ap.add_argument("--no-cskel", action="store_true", help="Disable compressed skeleton in the demo")

    args = ap.parse_args()
    random.seed(args.seed)

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    global_candidates = load_global_candidates(Path(args.global_anchors)) if args.global_anchors else None

    conn = connect(db_path)
    try:
        print(hr("="))
        print("BOOTSTRAP INSPECTION REPORT (paste this back to ChatGPT)")
        print(hr("="))
        print()

        # --- DB integrity
        print("=== DB integrity & counts ===")
        tables = list_tables(conn)
        print("Tables present:", tables)

        missing = sorted(list(EXPECTED_TABLES - set(tables)))
        if missing:
            print("❌ Missing expected tables:", missing)
        else:
            print("✅ All expected tables exist.")

        for t in sorted(list(EXPECTED_TABLES)):
            if t in tables:
                print(f"{t:>20}: {table_count(conn, t)}")
        # extra
        for t in ["sqlite_sequence"]:
            if t in tables:
                print(f"{t:>20}: {table_count(conn, t)}")

        meta = read_meta(conn)
        print("\nMeta (selected):")
        for k in [
            "schema_version",
            "bootstrapped_at",
            "anchors_source",
            "global_anchors_path",
            "global_candidates_count",
            "anchors_top_k",
            "anchor_method",
            "anchor_max_token_len",
            "anchors_count",
            "hsk_db_path",
            "hsk_table",
            "hsk_max_level",
            "include_level7",
            "max_ngram_n",
            "add_tok_ngrams",
            "add_anchor_windows",
            "add_skeleton",
            "add_compressed_skeleton",
            "add_anchor_pairs",
            "add_anchor_skip2",
            "add_anchor_skip3",
            "add_anchor_sequence",
            "add_anchor_spans",
            "add_span_signatures",
            "span_max_gap",
            "skip_max_jump",
            "pattern_min_count_seen",
            "pattern_min_distinct_sentences",
        ]:
            if k in meta:
                print(f"  - {k}: {meta[k]}")

        # --- Anchors diagnostics
        print("\n=== Anchor diagnostics ===")
        activated = load_activated_anchors(meta)
        if activated is None:
            print("⚠️ No anchors_json found in meta. (Did you store anchors into meta?)")
        else:
            print(f"Activated anchors (from meta.anchors_json): {len(activated)}")
            shown = sorted(list(activated))[: args.print_anchors]
            print(f"first {len(shown)} sorted:", shown)

        if global_candidates is not None:
            print(f"\nGlobal anchor candidates file: {len(global_candidates)} candidates")

            if activated is not None:
                inter = activated & global_candidates
                bad = activated - global_candidates
                print(f"Activated ∩ candidates: {len(inter)} ({fmt_pct(len(inter), len(activated))})")
                print(f"Activated \\ candidates: {len(bad)} ({fmt_pct(len(bad), len(activated))})")
                if bad:
                    print("⚠️ Some activated anchors are not in candidate set (unexpected):", sorted(list(bad))[:50])

        # Compute DF over your deck (from stored jieba tokens)
        tokens_jieba_all = read_all_jieba_tokens(conn)
        df = corpus_df(tokens_jieba_all)
        tf = corpus_tf(tokens_jieba_all)
        total_docs = len(tokens_jieba_all)

        # show top df short tokens
        top_df = df.most_common(25)
        print("\nTop tokens by DF (coverage across sentences):")
        for t, c in top_df:
            print(f"  - {t}: df={c} ({fmt_pct(c, total_docs)}), tf={tf.get(t,0)}")

        if global_candidates is not None:
            # top df among candidates
            df_cand = [(t, df.get(t, 0)) for t in global_candidates]
            df_cand.sort(key=lambda x: x[1], reverse=True)
            print("\nTop candidate tokens by DF in YOUR deck:")
            for t, c in df_cand[:25]:
                print(f"  - {t}: df={c} ({fmt_pct(c, total_docs)})")

        if args.recompute_anchors:
            print("\n=== Anchor recomputation (comparison) ===")
            # recompute using settings, optionally restricted to candidates
            max_len = args.anchor_max_len
            method = args.anchor_method
            k = args.anchors_top_k

            if method == "df":
                df2 = corpus_df(tokens_jieba_all, max_len=max_len)
                items = df2.most_common(k)
                recomputed = {t for t, _ in items if (global_candidates is None or t in global_candidates)}
                # if candidates exist, we may have fewer than k due to restriction; fill by scanning
                if global_candidates is not None and len(recomputed) < k:
                    # scan next best until filled
                    for t, _ in df2.most_common(5000):
                        if len(t) <= max_len and t in global_candidates:
                            recomputed.add(t)
                            if len(recomputed) >= k:
                                break
            else:
                tf2 = corpus_tf(tokens_jieba_all, max_len=max_len)
                items = tf2.most_common(5000)
                recomputed = set()
                for t, _ in items:
                    if len(t) <= max_len and (global_candidates is None or t in global_candidates):
                        recomputed.add(t)
                        if len(recomputed) >= k:
                            break

            print(f"Recomputed anchors: {len(recomputed)} (method={method}, top_k={k}, max_len={max_len}, restricted_to_candidates={global_candidates is not None})")
            print("first 120 sorted:", sorted(list(recomputed))[:120])

            if activated is not None:
                inter = activated & recomputed
                print(f"Activated ∩ recomputed: {len(inter)} ({fmt_pct(len(inter), len(activated))})")
                print(f"Jaccard(activated, recomputed) = {jaccard(activated, recomputed):.3f}")

        # --- Random sentence spot-checks
        print("\n=== Random sentence spot-checks ===")
        rows = sample_sentences(conn, args.pairs)
        for r in rows:
            tj = safe_json_loads(r["tokens_jieba_json"], fallback=[])
            th = safe_json_loads(r["tokens_hsk_json"], fallback=[])
            tc = safe_json_loads(r["tokens_char_json"], fallback=[])
            pats = safe_json_loads(r["patterns_json"], fallback=[])
            print("\n--- id=%s ---" % r["id"])
            print("zh:   ", r["zh_text"])
            print("jieba:", tj)
            print("hsk:  ", th)
            print("char: ", tc)
            print("patterns:", len(pats), "| skeleton:", r["skeleton"])

        # --- Pattern health
        print("\n=== Pattern health ===")
        total_patterns = table_count(conn, "pattern_personal_stats")
        emerged_patterns = int(conn.execute("SELECT COUNT(*) AS n FROM pattern_personal_stats WHERE emerged=1").fetchone()["n"])
        print("patterns total:  ", total_patterns)
        print("patterns emerged:", emerged_patterns, f"({fmt_pct(emerged_patterns, total_patterns)})")

        hist = pattern_count_hist(conn)
        c1 = hist.get(1, 0)
        c2 = hist.get(2, 0)
        c5 = sum(v for k, v in hist.items() if k >= 5)
        print(f"count=1:   {c1} ({fmt_pct(c1, total_patterns)})")
        print(f"count=2:   {c2} ({fmt_pct(c2, total_patterns)})")
        print(f"count>=5:  {c5} ({fmt_pct(c5, total_patterns)})")

        print("\nTop emerged patterns (by count_seen) with sample realizations:")
        tops = top_patterns(conn, limit=args.top_patterns, emerged_only=True)
        for pr in tops:
            print(
                f"\n- {pr.pattern_key} | count_seen={pr.count_seen} "
                f"distinct_sentences={pr.distinct_sentence_count}"
            )
            exs = sample_realizations(conn, pr.pattern_key, k=6)
            for e in exs:
                print("    •", e)

        # --- Vocab health
        print("\n=== Vocab health (HSK tokens) ===")
        unk, total = vocab_unknown_ratio(conn)
        print(f"vocab total: {total}")
        print(f"vocab with NULL hsk_level (fallback/unknown): {unk} ({fmt_pct(unk, total)})")

        one, total2 = single_char_vocab_count(conn)
        print(f"Single-character tokens in vocab_stats: {one} ({fmt_pct(one, total2)})")

        print("\nTop vocab tokens by count:")
        for r in top_vocab(conn, k=args.top_vocab):
            print(f"  - {r['word']}: count={r['count']}, level={r['hsk_level']}, freq={r['hsk_frequency']}")

        # --- Length robustness demo
        print("\n=== Length robustness demo (pattern overlap) ===")
        tokenize_words_jieba, extract_patterns_from_tokens = try_import_project_modules()

        if tokenize_words_jieba is None or extract_patterns_from_tokens is None:
            print("⚠️ Could not import project modules for the robustness demo.")
            print("   Make sure you run with PYTHONPATH=src and that your package imports work.")
        elif activated is None:
            print("⚠️ No activated anchors available (meta.anchors_json missing), cannot run demo.")
        else:
            # use meta defaults unless overridden
            max_ngram_n = args.max_ngram_n if args.max_ngram_n is not None else parse_int_meta(meta, "max_ngram_n", 4)
            span_max_gap = args.span_max_gap if args.span_max_gap is not None else parse_int_meta(meta, "span_max_gap", 20)
            skip_max_jump = args.skip_max_jump if args.skip_max_jump is not None else parse_int_meta(meta, "skip_max_jump", 10)

            add_tok_ngrams = parse_bool_meta(meta, "add_tok_ngrams", False)
            add_anchor_windows = parse_bool_meta(meta, "add_anchor_windows", True)
            add_skeleton = parse_bool_meta(meta, "add_skeleton", True)
            add_cskel = not args.no_cskel and parse_bool_meta(meta, "add_compressed_skeleton", True)
            add_anchor_pairs = parse_bool_meta(meta, "add_anchor_pairs", True)
            add_anchor_skip2 = not args.no_skipgrams and parse_bool_meta(meta, "add_anchor_skip2", False)
            add_anchor_skip3 = not args.no_skipgrams and parse_bool_meta(meta, "add_anchor_skip3", False)
            add_anchor_sequence = parse_bool_meta(meta, "add_anchor_sequence", False)
            add_anchor_spans = not args.no_spans and parse_bool_meta(meta, "add_anchor_spans", False)
            add_span_sigs = not args.no_span_sigs and parse_bool_meta(meta, "add_span_signatures", False)

            def pats_of(s: str) -> Set[str]:
                tj = tokenize_words_jieba(s)
                pats, _ = extract_patterns_from_tokens(
                    tj,
                    anchors=activated,
                    max_ngram_n=max_ngram_n,
                    add_tok_ngrams=add_tok_ngrams,
                    add_anchor_windows=add_anchor_windows,
                    add_skeleton=add_skeleton,
                    add_compressed_skeleton=add_cskel,
                    add_anchor_pairs=add_anchor_pairs,
                    add_anchor_skip2=add_anchor_skip2,
                    add_anchor_skip3=add_anchor_skip3,
                    add_anchor_sequence=add_anchor_sequence,
                    add_anchor_spans=add_anchor_spans,
                    add_span_signatures=add_span_sigs,
                    span_max_gap=span_max_gap,
                    skip_max_jump=skip_max_jump,
                )
                return {p.pattern_key for p in pats}

            examples = [
                ("我把它扔掉", "我把你昨天用的杯子扔掉了"),
                ("他在学校学习", "他今天在学校认真地学习了"),
                ("我给你打电话", "我昨天晚上给你打了一个电话"),
            ]

            for A, B in examples:
                pa = pats_of(A)
                pb = pats_of(B)
                j = jaccard(pa, pb)
                shared = len(pa & pb)
                union = len(pa | pb)
                print(f"\nA: {A}")
                print(f"B: {B}")
                print(f"  Jaccard(pattern_keys) = {j:.3f} | shared={shared} union={union}")

            print("\nInterpretation:")
            print("- If Jaccard rises vs your previous report, your patterns are less length/local-context sensitive.")
            print("- Skip-grams and span patterns are the main drivers of length invariance.")

        print("\n" + hr("="))
        print("END REPORT")
        print(hr("="))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
