#!/usr/bin/env python3
"""
scripts/inspect_prior_db.py

Inspect a "global prior" SQLite DB that stores global grammar pattern statistics.

This script prints a high-signal report:
- DB integrity (tables, columns, row counts)
- pattern stats summary (min/max/quantiles-ish via histogram)
- distribution diagnostics (how many patterns are rare / common)
- top patterns by count
- pattern family breakdown (anch_pair, anch_seq, cskel, skel, tok_ng, ...)
- sample realizations for selected patterns
- optional: sanity-check that your anchors show up in frequent patterns

Typical usage:
  PYTHONPATH=src python scripts/inspect_prior_db.py --db data/chinese_prior.db

With anchors check:
  PYTHONPATH=src python scripts/inspect_prior_db.py \
    --db data/chinese_prior.db \
    --anchors data/anchors_v1.json

If your DB uses different table/column names, pass:
  --stats-table pattern_global_stats --real-table pattern_global_realizations
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------
# Pretty printing helpers
# ---------------------------

def hr(ch: str = "=", n: int = 88) -> str:
    return ch * n

def fmt_int(n: int) -> str:
    return f"{n:,}"

def fmt_pct(a: int, b: int) -> str:
    if b <= 0:
        return "0.0%"
    return f"{(100.0 * a / b):.1f}%"

def safe_json_loads(s: str, fallback: Any = None) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return fallback


# ---------------------------
# DB helpers
# ---------------------------

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
    # columns: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]

def maybe_read_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    if "meta" not in set(list_tables(conn)):
        return {}
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {r["key"]: r["value"] for r in rows}


# ---------------------------
# Prior table expectations
# ---------------------------

DEFAULT_STATS_TABLE = "pattern_global_stats"
DEFAULT_REAL_TABLE = "pattern_global_realizations"

# We'll accept some alternates if your builder used different names
CANDIDATE_STATS_TABLES = [
    "pattern_global_stats",
    "pattern_stats_global",
    "prior_pattern_stats",
    "pattern_stats",
]
CANDIDATE_REAL_TABLES = [
    "pattern_global_realizations",
    "pattern_realizations_global",
    "prior_pattern_realizations",
    "pattern_realizations",
]


@dataclass
class PatternRow:
    pattern_id: str
    count: int
    diversity: int
    log_freq: Optional[float] = None


def pick_existing_table(conn: sqlite3.Connection, preferred: str, candidates: List[str]) -> Optional[str]:
    tables = set(list_tables(conn))
    if preferred in tables:
        return preferred
    for t in candidates:
        if t in tables:
            return t
    return None


def read_top_patterns(
    conn: sqlite3.Connection,
    stats_table: str,
    limit: int = 30,
) -> List[PatternRow]:
    cols = set(table_columns(conn, stats_table))

    # minimal columns required
    if "pattern_id" not in cols or "count" not in cols:
        raise ValueError(f"Stats table '{stats_table}' missing required columns. Has: {sorted(cols)}")

    has_div = "diversity" in cols
    has_log = "log_freq" in cols

    if has_div and has_log:
        rows = conn.execute(
            f"""
            SELECT pattern_id, count, diversity, log_freq
            FROM {stats_table}
            ORDER BY count DESC, diversity DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [PatternRow(r["pattern_id"], int(r["count"]), int(r["diversity"]), float(r["log_freq"])) for r in rows]

    if has_div:
        rows = conn.execute(
            f"""
            SELECT pattern_id, count, diversity
            FROM {stats_table}
            ORDER BY count DESC, diversity DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [PatternRow(r["pattern_id"], int(r["count"]), int(r["diversity"]), None) for r in rows]

    # no diversity/log_freq
    rows = conn.execute(
        f"""
        SELECT pattern_id, count
        FROM {stats_table}
        ORDER BY count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [PatternRow(r["pattern_id"], int(r["count"]), 0, None) for r in rows]


def count_histogram(conn: sqlite3.Connection, stats_table: str) -> Counter[int]:
    rows = conn.execute(f"SELECT count FROM {stats_table}").fetchall()
    return Counter(int(r["count"]) for r in rows)


def pattern_family(pid: str) -> str:
    """
    Infer family prefix from pattern_id.
    """
    if ":" not in pid:
        return "unknown"
    return pid.split(":", 1)[0]


def family_breakdown(conn: sqlite3.Connection, stats_table: str, top_k: int = 12) -> List[Tuple[str, int]]:
    rows = conn.execute(f"SELECT pattern_id FROM {stats_table}").fetchall()
    c = Counter(pattern_family(r["pattern_id"]) for r in rows)
    return c.most_common(top_k)


def sample_realizations(
    conn: sqlite3.Connection,
    real_table: str,
    pid: str,
    k: int = 6,
) -> List[str]:
    cols = set(table_columns(conn, real_table))
    if "pattern_id" not in cols or "realization" not in cols:
        raise ValueError(f"Realizations table '{real_table}' missing required columns. Has: {sorted(cols)}")

    rows = conn.execute(
        f"""
        SELECT realization
        FROM {real_table}
        WHERE pattern_id = ?
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (pid, k),
    ).fetchall()
    return [r["realization"] for r in rows]


def load_anchors(path: Optional[Path]) -> Optional[List[str]]:
    """
    Supports JSON formats:
      {"anchors": ["的","了",...]}  (your format)
    """
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Anchors file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = payload.get("anchors")
    if not isinstance(anchors, list):
        raise ValueError(f"Invalid anchors JSON. Expected {{'anchors':[...]}}, got keys={list(payload.keys())}")
    out = [str(a).strip() for a in anchors if str(a).strip()]
    return out or None


def anchors_presence_quickcheck(
    conn: sqlite3.Connection,
    stats_table: str,
    anchors: Sequence[str],
    sample_n: int = 80,
) -> None:
    """
    Quick heuristic: check if anchors appear inside top patterns (as substrings).
    This is not perfect, but catches obvious "wrong anchors file / wrong DB".
    """
    tops = read_top_patterns(conn, stats_table, limit=200)
    blob = "\n".join(p.pattern_id for p in tops)

    hit = []
    miss = []
    for a in anchors[:sample_n]:
        if a in blob:
            hit.append(a)
        else:
            miss.append(a)

    print("\n=== Anchor presence quick-check (heuristic) ===")
    print(f"Anchors provided: {len(anchors)} (checking first {min(sample_n, len(anchors))})")
    print(f"Found in top-200 pattern_ids: {len(hit)} ({fmt_pct(len(hit), min(sample_n, len(anchors)))})")
    if hit:
        print("Sample hits:", hit[:40])
    if miss:
        print("Sample misses:", miss[:40])
    if len(hit) < max(5, int(0.15 * min(sample_n, len(anchors)))):
        print("⚠️ This looks low. Possible causes:")
        print("   - DB was built with a different anchor set than this file")
        print("   - Your patterns don't embed anchors in IDs (less likely with your extractor)")
        print("   - Your top patterns are dominated by non-anchor families (tok_ng/skel/cskel) with placeholders")


def print_hist_summary(hist: Counter[int], total_patterns: int) -> None:
    """
    Summarize the count histogram in a readable way.
    """
    if total_patterns <= 0:
        return

    c1 = hist.get(1, 0)
    c2 = hist.get(2, 0)
    c3 = hist.get(3, 0)
    c5p = sum(v for k, v in hist.items() if k >= 5)
    c10p = sum(v for k, v in hist.items() if k >= 10)
    c100p = sum(v for k, v in hist.items() if k >= 100)

    print("Count distribution (pattern_global_stats.count):")
    print(f"  count=1:    {fmt_int(c1)} ({fmt_pct(c1, total_patterns)})")
    print(f"  count=2:    {fmt_int(c2)} ({fmt_pct(c2, total_patterns)})")
    print(f"  count=3:    {fmt_int(c3)} ({fmt_pct(c3, total_patterns)})")
    print(f"  count>=5:   {fmt_int(c5p)} ({fmt_pct(c5p, total_patterns)})")
    print(f"  count>=10:  {fmt_int(c10p)} ({fmt_pct(c10p, total_patterns)})")
    print(f"  count>=100: {fmt_int(c100p)} ({fmt_pct(c100p, total_patterns)})")

    # show some histogram buckets
    buckets = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 9),
        (10, 19),
        (20, 49),
        (50, 99),
        (100, 199),
        (200, 499),
        (500, 999),
        (1000, 10**12),
    ]
    print("\nHistogram buckets:")
    for lo, hi in buckets:
        n = sum(v for k, v in hist.items() if lo <= k <= hi)
        label = f"{lo}" if lo == hi else f"{lo}-{hi if hi < 10**11 else '∞'}"
        print(f"  {label:>8}: {fmt_int(n)} ({fmt_pct(n, total_patterns)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to prior SQLite DB (e.g. data/chinese_prior.db)")
    ap.add_argument("--anchors", default=None, help="Optional: path to final_anchors.json for sanity check")
    ap.add_argument("--stats-table", default=DEFAULT_STATS_TABLE, help="Pattern stats table name")
    ap.add_argument("--real-table", default=DEFAULT_REAL_TABLE, help="Pattern realizations table name")
    ap.add_argument("--top", type=int, default=30, help="How many top patterns to print")
    ap.add_argument("--sample-real", type=int, default=6, help="How many realizations to sample per pattern")
    ap.add_argument("--show-realizations", action="store_true", help="Actually print sample realizations")
    ap.add_argument("--family-top", type=int, default=15, help="How many pattern families to show")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = connect(db_path)
    try:
        print(hr("="))
        print("PRIOR DB INSPECTION REPORT (paste this back to ChatGPT if you want)")
        print(hr("="))
        print(f"DB: {db_path}")
        print()

        tables = list_tables(conn)
        print("=== Tables ===")
        for t in tables:
            print(" -", t)
        print()

        # Try to pick correct tables if user passed defaults but they don't exist
        stats_table = pick_existing_table(conn, args.stats_table, CANDIDATE_STATS_TABLES)
        real_table = pick_existing_table(conn, args.real_table, CANDIDATE_REAL_TABLES)

        if stats_table is None:
            print("❌ Could not find a stats table.")
            print("   Looked for:", [args.stats_table] + CANDIDATE_STATS_TABLES)
            return
        print(f"Using stats table: {stats_table}")

        if real_table is None:
            print("⚠️ Could not find a realizations table (ok if you didn't store realizations).")
            print("   Looked for:", [args.real_table] + CANDIDATE_REAL_TABLES)
        else:
            print(f"Using realizations table: {real_table}")

        print("\n=== Row counts ===")
        print(f"{stats_table:>28}: {fmt_int(table_count(conn, stats_table))}")
        if real_table is not None:
            print(f"{real_table:>28}: {fmt_int(table_count(conn, real_table))}")
        if "meta" in set(tables):
            print(f"{'meta':>28}: {fmt_int(table_count(conn, 'meta'))}")

        # Meta
        meta = maybe_read_meta(conn)
        if meta:
            print("\n=== Meta (selected) ===")
            for k in [
                "schema_version",
                "built_at",
                "prior_source",
                "input_path",
                "anchors_path",
                "anchors_count",
                "max_ngram_n",
                "span_max_gap",
                "skip_max_jump",
                "skip_add_trigrams",
                "add_compressed_skeleton",
                "add_span_patterns",
                "add_span_signatures",
                "add_anchor_skipgrams",
                "add_anchor_pairs",
                "add_anchor_sequence",
            ]:
                if k in meta:
                    print(f" - {k}: {meta[k]}")

        # Columns sanity
        print("\n=== Columns ===")
        print(f"{stats_table}: {table_columns(conn, stats_table)}")
        if real_table is not None:
            print(f"{real_table}: {table_columns(conn, real_table)}")

        # Pattern families
        print("\n=== Pattern family breakdown ===")
        fams = family_breakdown(conn, stats_table, top_k=args.family_top)
        total_patterns = table_count(conn, stats_table)
        for fam, c in fams:
            print(f" - {fam:<12} {fmt_int(c):>10}  ({fmt_pct(c, total_patterns)})")

        # Count histogram
        print("\n=== Pattern count distribution ===")
        hist = count_histogram(conn, stats_table)
        print_hist_summary(hist, total_patterns)

        # Top patterns
        print("\n=== Top patterns by count ===")
        tops = read_top_patterns(conn, stats_table, limit=args.top)
        for i, pr in enumerate(tops, 1):
            lf = f"{pr.log_freq:.3f}" if pr.log_freq is not None else "-"
            print(f"{i:>2}. count={fmt_int(pr.count):>9}  div={fmt_int(pr.diversity):>6}  log_freq={lf:>7}   {pr.pattern_id}")

        # Show realizations if requested and table exists
        if args.show_realizations and real_table is not None:
            print("\n=== Sample realizations (random) ===")
            for pr in tops[: min(12, len(tops))]:
                print(f"\n- {pr.pattern_id} | count={fmt_int(pr.count)} div={fmt_int(pr.diversity)}")
                exs = sample_realizations(conn, real_table, pr.pattern_id, k=args.sample_real)
                for e in exs:
                    print("    •", e)

        # Anchors check (optional)
        anchors_path = Path(args.anchors) if args.anchors else None
        anchors = load_anchors(anchors_path)
        if anchors:
            anchors_presence_quickcheck(conn, stats_table, anchors, sample_n=80)

        print("\n" + hr("="))
        print("END REPORT")
        print(hr("="))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
