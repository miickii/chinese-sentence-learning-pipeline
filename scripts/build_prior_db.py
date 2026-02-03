#!/usr/bin/env python3
"""
scripts/build_chinese_prior_db.py

Build a GLOBAL grammar prior DB from a large corpus using frozen anchors.

Input:
  - corpus: one sentence per line text file (your case)
  - anchors: JSON file containing {"anchors":[...]} OR a raw list [...]

Output:
  - SQLite DB: data/chinese_prior.db

Stores:
  - pattern_global_stats(pattern_key, family, count_sentences, count_occurrences, distinct_realization_count, p_global)
      * distinct_realization_count is "distinct realizations observed up to cap"
  - pattern_global_realizations(pattern_key, realization)
      * capped per pattern for size control
  - meta: config + provenance

Usage example:
  PYTHONPATH=src python scripts/build_prior_db.py \
    --corpus data/processed/global_prior.sentences.txt \
    --anchors data/final_anchors.json \
    --out data/chinese_prior.db \
    --max-ngram-n 4 \
    --span-max-gap 20 \
    --skip-max-jump 10 \
    --add-anchor-skip3 0 \
    --cap-realizations-per-pattern 30 \
    --commit-every 5000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Set

from zh_sentence_learning_pipeline.grammar.tokenize import tokenize_words_jieba
from zh_sentence_learning_pipeline.grammar.patterns import extract_patterns_from_tokens
from zh_sentence_learning_pipeline.grammar.pattern_key import family_from_key
from zh_sentence_learning_pipeline.store.prior_db import connect, init_prior_db


def load_anchors(path: str | Path) -> Set[str]:
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"Anchors file not found: {p}")
  data = json.loads(p.read_text(encoding="utf-8"))
  if isinstance(data, dict) and "anchors" in data:
    arr = data["anchors"]
  else:
    arr = data
  if not isinstance(arr, list) or not arr:
    raise ValueError(f"Invalid anchors JSON in {p}. Expected {{'anchors':[...]}}, or a raw list.")
  out = {str(x).strip() for x in arr if str(x).strip()}
  if not out:
    raise ValueError(f"Anchors JSON produced empty set: {p}")
  return out


def meta_put(conn, key: str, value: object) -> None:
  conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value)))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--corpus", required=True, help="Text file: one Chinese sentence per line.")
  ap.add_argument("--anchors", required=True, help="Frozen anchors JSON path.")
  ap.add_argument("--out", required=True, help="Output SQLite DB path (e.g. data/chinese_prior.db)")

  ap.add_argument("--max-ngram-n", type=int, default=4)
  ap.add_argument("--span-max-gap", type=int, default=20)
  ap.add_argument("--skip-max-jump", type=int, default=10)

  ap.add_argument("--add-tok-ngrams", type=int, default=0)
  ap.add_argument("--add-anchor-windows", type=int, default=1)
  ap.add_argument("--add-skeleton", type=int, default=1)
  ap.add_argument("--add-cskel", type=int, default=1)
  ap.add_argument("--add-anchor-pairs", type=int, default=1)
  ap.add_argument("--add-anchor-skip2", type=int, default=0)
  ap.add_argument("--add-anchor-skip3", type=int, default=0)
  ap.add_argument("--add-anchor-seq", type=int, default=0)
  ap.add_argument("--add-anchor-spans", type=int, default=0)
  ap.add_argument("--add-span-sigs", type=int, default=0)

  ap.add_argument("--cap-realizations-per-pattern", type=int, default=9999)
  ap.add_argument("--commit-every", type=int, default=5000)
  ap.add_argument("--max-lines", type=int, default=0, help="Debug: stop after N lines (0 = no limit)")
  args = ap.parse_args()

  corpus_path = Path(args.corpus)
  if not corpus_path.exists():
    raise FileNotFoundError(f"Corpus not found: {corpus_path}")

  anchors = load_anchors(args.anchors)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  conn = connect(out_path)
  init_prior_db(conn)

  # Store provenance + config
  meta_put(conn, "built_at_unix", int(time.time()))
  meta_put(conn, "corpus_path", str(corpus_path))
  meta_put(conn, "anchors_path", str(Path(args.anchors)))
  meta_put(conn, "anchors_count", len(anchors))

  meta_put(conn, "max_ngram_n", args.max_ngram_n)
  meta_put(conn, "span_max_gap", args.span_max_gap)
  meta_put(conn, "skip_max_jump", args.skip_max_jump)

  meta_put(conn, "add_tok_ngrams", int(bool(args.add_tok_ngrams)))
  meta_put(conn, "add_anchor_windows", int(bool(args.add_anchor_windows)))
  meta_put(conn, "add_skeleton", int(bool(args.add_skeleton)))
  meta_put(conn, "add_compressed_skeleton", int(bool(args.add_cskel)))
  meta_put(conn, "add_anchor_pairs", int(bool(args.add_anchor_pairs)))
  meta_put(conn, "add_anchor_skip2", int(bool(args.add_anchor_skip2)))
  meta_put(conn, "add_anchor_skip3", int(bool(args.add_anchor_skip3)))
  meta_put(conn, "add_anchor_sequence", int(bool(args.add_anchor_seq)))
  meta_put(conn, "add_anchor_spans", int(bool(args.add_anchor_spans)))
  meta_put(conn, "add_span_signatures", int(bool(args.add_span_sigs)))

  meta_put(conn, "cap_realizations_per_pattern", args.cap_realizations_per_pattern)
  conn.commit()

  # Aggregation:
  # - counts can be flushed incrementally
  # - realizations are capped; we track a small per-pattern set until full
  count_occ_batch: Counter[str] = Counter()
  count_sent_batch: Counter[str] = Counter()

  # Tracks how many UNIQUE realizations we have stored for each pattern (capped).
  # value = None means "already full, stop tracking"
  realizations_cache: Dict[str, Optional[Set[str]]] = {}
  realization_count: Dict[str, int] = {}  # monotonic up to cap

  total_lines = 0
  kept_lines = 0
  t0 = time.time()

  def ensure_row(pkey: str) -> None:
    # Make sure a stats row exists so later UPSERT works smoothly.
    conn.execute(
      """
      INSERT OR IGNORE INTO pattern_global_stats(
        pattern_key, family, count_sentences, count_occurrences, distinct_realization_count, p_global
      )
      VALUES(?, ?, 0, 0, 0, 0.0)
      """,
      (pkey, family_from_key(pkey)),
    )

  def maybe_add_realization(pkey: str, realization: str) -> None:
    cap = int(args.cap_realizations_per_pattern)
    if cap <= 0:
      return

    box = realizations_cache.get(pkey, None)

    # If we've already hit cap earlier, we store None as sentinel
    if box is None and pkey in realizations_cache:
      return

    if box is None:
      box = set()
      realizations_cache[pkey] = box

    if len(box) >= cap:
      realizations_cache[pkey] = None
      return

    if realization in box:
      return

    # Insert into DB (ignore duplicates)
    conn.execute(
      "INSERT OR IGNORE INTO pattern_global_realizations(pattern_key, realization) VALUES(?,?)",
      (pkey, realization),
    )

    # Track uniqueness only if it was new in our cache
    box.add(realization)
    realization_count[pkey] = min(cap, len(box))

    if len(box) >= cap:
      realizations_cache[pkey] = None

  def flush_counts() -> None:
    if not count_occ_batch and not count_sent_batch:
      return

    keys = set(count_occ_batch.keys()) | set(count_sent_batch.keys())

    # For speed: do stats inserts/updates in a tight loop.
    for pkey in keys:
      ensure_row(pkey)

      occ = int(count_occ_batch.get(pkey, 0))
      sent = int(count_sent_batch.get(pkey, 0))
      div = int(realization_count.get(pkey, 0))

      conn.execute(
        """
        UPDATE pattern_global_stats
        SET count_occurrences = count_occurrences + ?,
            count_sentences = count_sentences + ?,
            distinct_realization_count = CASE
              WHEN distinct_realization_count < ? THEN ? ELSE distinct_realization_count
            END
        WHERE pattern_key = ?
        """,
        (occ, sent, div, div, pkey),
      )

    count_occ_batch.clear()
    count_sent_batch.clear()

  with corpus_path.open("r", encoding="utf-8") as f:
    for line in f:
      total_lines += 1
      if args.max_lines and total_lines > args.max_lines:
        break

      s = (line or "").strip()
      if not s:
        continue

      kept_lines += 1
      toks = tokenize_words_jieba(s)
      if not toks:
        continue

      pats, _ = extract_patterns_from_tokens(
        toks,
        anchors=anchors,
        max_ngram_n=int(args.max_ngram_n),
        add_tok_ngrams=bool(args.add_tok_ngrams),
        add_anchor_windows=bool(args.add_anchor_windows),
        add_skeleton=bool(args.add_skeleton),
        add_compressed_skeleton=bool(args.add_cskel),
        add_anchor_pairs=bool(args.add_anchor_pairs),
        add_anchor_skip2=bool(args.add_anchor_skip2),
        add_anchor_skip3=bool(args.add_anchor_skip3),
        add_anchor_sequence=bool(args.add_anchor_seq),
        add_anchor_spans=bool(args.add_anchor_spans),
        add_span_signatures=bool(args.add_span_sigs),
        span_max_gap=int(args.span_max_gap),
        skip_max_jump=int(args.skip_max_jump),
      )

      # Update counts and store capped realizations
      sentence_keys: Set[str] = set()
      for p in pats:
        pkey = p.pattern_key
        count_occ_batch[pkey] += 1
        sentence_keys.add(pkey)
        # store realization sample (capped per pid)
        if p.realization:
          maybe_add_realization(pkey, p.realization)

      for pkey in sentence_keys:
        count_sent_batch[pkey] += 1

      if kept_lines % int(args.commit_every) == 0:
        flush_counts()
        meta_put(conn, "sentences_processed", kept_lines)
        conn.commit()

        dt = time.time() - t0
        rate = kept_lines / dt if dt > 0 else 0.0
        print(
          f"[prior-db] processed={kept_lines:,} lines | "
          f"patterns_batch={len(count_occ_batch):,} | rate={rate:,.1f} lines/s"
        )

  # Final flush
  flush_counts()
  meta_put(conn, "sentences_processed", kept_lines)
  meta_put(conn, "lines_seen", total_lines)
  meta_put(conn, "build_seconds", round(time.time() - t0, 2))

  # Compute p_global over all patterns (DF-weighted distribution)
  row = conn.execute("SELECT SUM(count_sentences) AS s FROM pattern_global_stats").fetchone()
  total_df_sum = int(row["s"] or 0)
  meta_put(conn, "total_df_sum", total_df_sum)
  if total_df_sum > 0:
    conn.execute(
      "UPDATE pattern_global_stats SET p_global = (1.0 * count_sentences / ?)",
      (total_df_sum,),
    )
  else:
    conn.execute("UPDATE pattern_global_stats SET p_global = 0.0")

  conn.commit()
  conn.close()

  print(f"✅ Built global grammar prior DB: {out_path}")
  print(f"   sentences processed: {kept_lines:,} (lines seen: {total_lines:,})")
  print(f"   anchors: {len(anchors):,}")
  print(f"   note: distinct_realizations are capped at {int(args.cap_realizations_per_pattern)} per pattern")


if __name__ == "__main__":
  main()
