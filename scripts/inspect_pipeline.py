#!/usr/bin/env python3
"""
scripts/inspect_pipeline.py

Unified sanity check for:
- prior DB (global patterns)
- state DB (personal patterns)
- shared PatternKey stability

Usage:
  PYTHONPATH=src python scripts/inspect_pipeline.py \
    --prior-db data/chinese_prior.db \
    --state-db data/state.db \
    --anchors data/final_anchors.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------
# Helpers
# ---------------------------

def hr(ch: str = "=", n: int = 88) -> str:
    return ch * n

def fmt_int(n: int) -> str:
    return f"{n:,}"

def fmt_pct(a: int, b: int) -> str:
    if b <= 0:
        return "0.0%"
    return f"{(100.0 * a / b):.1f}%"

def safe_json_loads(s: str, fallback):
    try:
        return json.loads(s)
    except Exception:
        return fallback

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

def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]

def read_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    if "meta" not in set(list_tables(conn)):
        return {}
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {r["key"]: r["value"] for r in rows}

def family_from_key(key: str) -> str:
    if "|" in key:
        return key.split("|", 1)[0]
    if ":" in key:
        return key.split(":", 1)[0]
    return "unknown"


# ---------------------------
# Prior DB checks
# ---------------------------

def quantile_value(conn: sqlite3.Connection, table: str, col: str, q: float) -> int:
    total = table_count(conn, table)
    if total <= 0:
        return 0
    idx = max(0, int(math.ceil(total * q)) - 1)
    row = conn.execute(
        f"SELECT {col} AS c FROM {table} ORDER BY {col} LIMIT 1 OFFSET ?",
        (idx,),
    ).fetchone()
    return int(row["c"]) if row else 0

def prior_top_patterns(conn: sqlite3.Connection, limit: int = 20) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT pattern_key, count_sentences, count_occurrences, p_global
        FROM pattern_global_stats
        ORDER BY count_sentences DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


# ---------------------------
# State DB checks
# ---------------------------

def sample_sentence_patterns(conn: sqlite3.Connection, k: int) -> List[List[str]]:
    rows = conn.execute(
        """
        SELECT patterns_json
        FROM sentences
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (k,),
    ).fetchall()
    out: List[List[str]] = []
    for r in rows:
        arr = safe_json_loads(r["patterns_json"], fallback=[])
        if isinstance(arr, list):
            out.append([str(x) for x in arr])
    return out

def anchor_df_from_sentences(
    conn: sqlite3.Connection,
    anchors: Sequence[str],
    k: int,
) -> Counter[str]:
    rows = conn.execute(
        """
        SELECT tokens_jieba_json
        FROM sentences
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (k,),
    ).fetchall()
    df = Counter()
    anchors_set = set(anchors)
    for r in rows:
        toks = safe_json_loads(r["tokens_jieba_json"], fallback=[])
        if not isinstance(toks, list):
            continue
        present = {t for t in toks if t in anchors_set}
        df.update(present)
    return df


def load_anchors(path: Optional[Path], meta: Dict[str, str]) -> Optional[List[str]]:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Anchors file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        anchors = payload.get("anchors")
        if not isinstance(anchors, list):
            raise ValueError(f"Invalid anchors JSON. Expected {{'anchors':[...]}}, got keys={list(payload.keys())}")
        return [str(a).strip() for a in anchors if str(a).strip()]

    meta_anchors = meta.get("anchors_json")
    if meta_anchors:
        arr = safe_json_loads(meta_anchors, fallback=[])
        if isinstance(arr, list):
            return [str(a) for a in arr]
    return None


def pattern_key_sanity(conn: sqlite3.Connection, table: str, key_col: str, limit: int) -> Tuple[int, int]:
    rows = conn.execute(
        f"SELECT {key_col} AS k FROM {table} ORDER BY RANDOM() LIMIT ?",
        (limit,),
    ).fetchall()
    bad = 0
    total = 0
    for r in rows:
        total += 1
        k = str(r["k"])
        if not k or "|a=" not in k or "|p=" not in k:
            bad += 1
    return bad, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-db", required=True, help="Path to global prior DB (pattern_global_stats).")
    ap.add_argument("--state-db", required=True, help="Path to personal state DB (pattern_personal_stats).")
    ap.add_argument("--anchors", default="", help="Optional anchors JSON for checks.")
    ap.add_argument("--sample-sentences", type=int, default=200, help="Sample size for sentence checks.")
    ap.add_argument("--sample-keys", type=int, default=1000, help="Sample size for pattern_key sanity checks.")
    args = ap.parse_args()

    prior_path = Path(args.prior_db)
    state_path = Path(args.state_db)

    if not prior_path.exists():
        raise FileNotFoundError(f"Prior DB not found: {prior_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State DB not found: {state_path}")

    prior = connect(prior_path)
    state = connect(state_path)

    try:
        print(hr("="))
        print("PIPELINE INSPECTION")
        print(hr("="))

        # ---------------------------
        # Prior DB health
        # ---------------------------
        print("\n=== Prior DB health ===")
        prior_tables = set(list_tables(prior))
        for t in ["pattern_global_stats", "pattern_global_realizations", "meta"]:
            status = "OK" if t in prior_tables else "MISSING"
            n = table_count(prior, t) if t in prior_tables else 0
            print(f"{t:<30} {status:<8} rows={fmt_int(n)}")

        # ---------------------------
        # State DB health
        # ---------------------------
        print("\n=== State DB health ===")
        state_tables = set(list_tables(state))
        for t in ["sentences", "vocab_stats", "pattern_personal_stats", "pattern_personal_realizations", "meta"]:
            status = "OK" if t in state_tables else "MISSING"
            n = table_count(state, t) if t in state_tables else 0
            print(f"{t:<30} {status:<8} rows={fmt_int(n)}")

        # ---------------------------
        # Prior pattern distribution
        # ---------------------------
        print("\n=== Prior pattern distribution ===")
        total_patterns = table_count(prior, "pattern_global_stats")
        singletons = int(
            prior.execute(
                "SELECT COUNT(*) AS n FROM pattern_global_stats WHERE count_sentences=1"
            ).fetchone()["n"]
        )
        print(f"patterns total:  {fmt_int(total_patterns)}")
        print(f"singletons:      {fmt_int(singletons)} ({fmt_pct(singletons, total_patterns)})")
        print(f"p50 count_sentences: {quantile_value(prior, 'pattern_global_stats', 'count_sentences', 0.50)}")
        print(f"p90 count_sentences: {quantile_value(prior, 'pattern_global_stats', 'count_sentences', 0.90)}")
        print(f"p99 count_sentences: {quantile_value(prior, 'pattern_global_stats', 'count_sentences', 0.99)}")

        print("\nTop patterns by count_sentences:")
        for i, r in enumerate(prior_top_patterns(prior, limit=20), 1):
            pg_val = r["p_global"]
            pg = f"{float(pg_val):.4f}" if pg_val is not None else "-"
            print(
                f"{i:>2}. df={fmt_int(int(r['count_sentences'])):>9} "
                f"occ={fmt_int(int(r['count_occurrences'])):>9} "
                f"p={pg:>7}  {r['pattern_key']}"
            )

        # ---------------------------
        # Coverage mass
        # ---------------------------
        print("\n=== Coverage mass (emerged patterns) ===")
        emerged_rows = state.execute(
            "SELECT pattern_key FROM pattern_personal_stats WHERE emerged=1"
        ).fetchall()
        emerged_keys = [r["pattern_key"] for r in emerged_rows]
        emerged = len(emerged_keys)
        total_personal = table_count(state, "pattern_personal_stats")

        if emerged_keys:
            qmarks = ",".join(["?"] * len(emerged_keys))
            rows = prior.execute(
                f"""
                SELECT pattern_key, p_global
                FROM pattern_global_stats
                WHERE pattern_key IN ({qmarks})
                """,
                emerged_keys,
            ).fetchall()
            p_map = {r["pattern_key"]: float(r["p_global"] or 0.0) for r in rows}
            coverage = sum(p_map.get(k, 0.0) for k in emerged_keys)
        else:
            coverage = 0.0

        print(f"emerged patterns: {fmt_int(emerged)} / {fmt_int(total_personal)}")
        print(f"coverage_mass:    {coverage:.4f}")

        # ---------------------------
        # Sentence pattern coverage
        # ---------------------------
        print("\n=== Sentence pattern coverage (sample) ===")
        samples = sample_sentence_patterns(state, k=args.sample_sentences)
        any_patterns = 0
        any_skel = 0
        for pats in samples:
            if pats:
                any_patterns += 1
            if any(family_from_key(p) == "skel" for p in pats):
                any_skel += 1
        total_samples = max(1, len(samples))
        print(f"sentences sampled: {total_samples}")
        print(f"with >=1 pattern:  {any_patterns} ({fmt_pct(any_patterns, total_samples)})")
        print(f"with skel pattern: {any_skel} ({fmt_pct(any_skel, total_samples)})")

        # ---------------------------
        # Anchors checks
        # ---------------------------
        print("\n=== Anchor checks ===")
        state_meta = read_meta(state)
        anchors_path = Path(args.anchors) if args.anchors else None
        anchors = load_anchors(anchors_path, state_meta)
        if anchors:
            print(f"anchors count: {fmt_int(len(anchors))}")
            df = anchor_df_from_sentences(state, anchors, k=args.sample_sentences)
            top = df.most_common(20)
            print("top anchors by DF (sample):")
            for a, c in top:
                print(f"  {a}  df={fmt_int(c)}")
            zero = [a for a in anchors if df.get(a, 0) == 0]
            print(f"anchors with df=0 in sample: {fmt_int(len(zero))}")
            if zero:
                print("sample df=0 anchors:", zero[:40])
        else:
            print("No anchors provided (use --anchors or ensure anchors_json exists in state meta).")

        # ---------------------------
        # PatternKey sanity
        # ---------------------------
        print("\n=== PatternKey sanity ===")
        bad_prior, total_prior = pattern_key_sanity(prior, "pattern_global_stats", "pattern_key", args.sample_keys)
        bad_state, total_state = pattern_key_sanity(state, "pattern_personal_stats", "pattern_key", args.sample_keys)
        print(f"prior keys checked:  {fmt_int(total_prior)} | invalid format: {fmt_int(bad_prior)}")
        print(f"state keys checked:  {fmt_int(total_state)} | invalid format: {fmt_int(bad_state)}")

        print("\n" + hr("="))
        print("END REPORT")
        print(hr("="))

    finally:
        prior.close()
        state.close()


if __name__ == "__main__":
    main()
