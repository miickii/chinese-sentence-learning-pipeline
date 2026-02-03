"""
bootstrap/bootstrap.py

Orchestrates bootstrapping with HSK DB support:
  1) Init SQLite DB schema
  2) Load known Chinese sentences from CSV
  3) Load HSK lexicon from SQLite (chinese_words)
  4) For each sentence:
     - tokenize_jieba (grammar)
     - tokenize_hsk_first (stable vocab units)
     - tokenize_chars (fallback)
  5) Build anchors (GLOBAL or LOCAL; local uses DF by default)
  6) Extract Layer-B patterns from jieba tokens; update GrammarState
  7) Write sentences + vocab_stats + pattern_personal_stats to SQLite

New in this version:
- Stores anchors + extractor config into meta for reproducibility.
- Uses updated pattern extractor (compressed skeleton, spans, skip-grams).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ..store.db import connect, init_db
from ..utils.io import read_csv_column

from ..hsk.lexicon import HSKLexicon
from ..grammar.tokenize import (
    tokenize_words_jieba,
    tokenize_words_hsk_first,
    tokenize_chars,
)
from ..grammar.patterns import build_anchor_set, extract_patterns_from_tokens
from ..grammar.pattern_key import family_from_key
from ..grammar.state import GrammarState
from ..vocab.state import count_vocab

EXTRACTOR_VERSION = "patterns_v4_keyed_core"


def _load_global_anchors(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"global anchors file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = payload.get("anchors", [])
    if not isinstance(anchors, list) or not anchors:
        raise ValueError(f"Invalid global anchors payload in {path}: expected key 'anchors' as non-empty list.")
    return {str(a).strip() for a in anchors if str(a).strip()}


def _meta_put(conn, key: str, value: object) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
        (key, str(value)),
    )


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bootstrap(
    db_path: str | Path,
    csv_path: str | Path,
    zh_column: str,
    hsk_db_path: str | Path,
    hsk_max_level: int = 6,
    include_level7: bool = False,
    hsk_table: str = "chinese_words",
    source: str = "anki_bootstrap",

    # anchors
    anchors_top_k: int = 70,
    global_anchors_path: str | Path | None = None,
    anchor_method: str = "df",
    anchor_max_token_len: int = 4,
    debug_print_anchors: bool = False,

    # base pattern knobs
    max_ngram_n: int = 4,

    # extractor toggles (core families on by default)
    add_tok_ngrams: bool = False,
    add_anchor_windows: bool = True,
    add_skeleton: bool = True,
    add_compressed_skeleton: bool = True,
    add_anchor_pairs: bool = True,
    add_anchor_skip2: bool = False,
    add_anchor_skip3: bool = False,
    add_anchor_sequence: bool = False,
    add_anchor_spans: bool = False,
    add_span_signatures: bool = False,

    # extractor params
    span_max_gap: int = 20,
    skip_max_jump: int = 10,

    # grammar emergence thresholds (Layer C)
    pattern_min_count_seen: int = 3,
    pattern_min_distinct_sentences: int = 2,
) -> None:
    conn = connect(db_path)
    init_db(conn)

    sentences = read_csv_column(csv_path, zh_column)
    now = datetime.now(timezone.utc).isoformat()

    lex = HSKLexicon.from_sqlite(
        db_path=str(hsk_db_path),
        max_level=hsk_max_level,
        include_level7=include_level7,
        table=hsk_table,
    )

    tokens_jieba_all: list[list[str]] = []
    tokens_hsk_all: list[list[str]] = []
    tokens_char_all: list[list[str]] = []

    for s in sentences:
        tj = tokenize_words_jieba(s)
        th = tokenize_words_hsk_first(s, lex)
        tc = tokenize_chars(s)

        tokens_jieba_all.append(tj)
        tokens_hsk_all.append(th)
        tokens_char_all.append(tc)

    # -----------------------------
    # Anchors (GLOBAL vs LOCAL)
    # -----------------------------
    if global_anchors_path is not None:
        allowed_candidates = _load_global_anchors(global_anchors_path)

        # Activate anchors locally: pick top-K by DF (or TF), restricted to allowed candidates
        anchors = build_anchor_set(
            tokens_jieba_all,
            top_k=anchors_top_k,
            method=anchor_method,
            max_token_len=anchor_max_token_len,
            allowed=allowed_candidates,
        )

        anchors_source = "global_candidates+local_" + anchor_method
        _meta_put(conn, "anchors_source", anchors_source)
        _meta_put(conn, "global_anchors_path", str(global_anchors_path))
        _meta_put(conn, "global_candidates_count", len(allowed_candidates))
    else:
        anchors = build_anchor_set(
            tokens_jieba_all,
            top_k=anchors_top_k,
            method=anchor_method,
            max_token_len=anchor_max_token_len,
            allowed=None,
        )
        anchors_source = "local_" + anchor_method
        _meta_put(conn, "anchors_source", anchors_source)

    _meta_put(conn, "anchors_top_k", anchors_top_k)
    _meta_put(conn, "anchor_method", anchor_method)
    _meta_put(conn, "anchor_max_token_len", anchor_max_token_len)

    # Store anchors into meta (reproducibility / comparisons)
    anchors_sorted = sorted(anchors)
    anchors_json = json.dumps(anchors_sorted, ensure_ascii=False)
    anchors_hash = _sha256_text(anchors_json)
    _meta_put(conn, "anchors_count", len(anchors_sorted))
    _meta_put(conn, "anchors_json", anchors_json)
    _meta_put(conn, "anchors_hash", anchors_hash)

    if debug_print_anchors:
        print(f"[anchors_source={anchors_source}] n={len(anchors_sorted)}")
        print(anchors_sorted[:200])

    # Store extractor config into meta
    _meta_put(conn, "bootstrapped_at", now)
    _meta_put(conn, "hsk_db_path", str(hsk_db_path))
    _meta_put(conn, "hsk_table", hsk_table)
    _meta_put(conn, "hsk_max_level", hsk_max_level)
    _meta_put(conn, "include_level7", "1" if include_level7 else "0")

    _meta_put(conn, "max_ngram_n", max_ngram_n)
    _meta_put(conn, "add_tok_ngrams", int(add_tok_ngrams))
    _meta_put(conn, "add_anchor_windows", int(add_anchor_windows))
    _meta_put(conn, "add_skeleton", int(add_skeleton))
    _meta_put(conn, "add_compressed_skeleton", int(add_compressed_skeleton))
    _meta_put(conn, "add_anchor_pairs", int(add_anchor_pairs))
    _meta_put(conn, "add_anchor_skip2", int(add_anchor_skip2))
    _meta_put(conn, "add_anchor_skip3", int(add_anchor_skip3))
    _meta_put(conn, "add_anchor_sequence", int(add_anchor_sequence))
    _meta_put(conn, "add_anchor_spans", int(add_anchor_spans))
    _meta_put(conn, "add_span_signatures", int(add_span_signatures))
    _meta_put(conn, "span_max_gap", span_max_gap)
    _meta_put(conn, "skip_max_jump", skip_max_jump)
    _meta_put(conn, "pattern_min_count_seen", pattern_min_count_seen)
    _meta_put(conn, "pattern_min_distinct_sentences", pattern_min_distinct_sentences)
    _meta_put(conn, "extractor_version", EXTRACTOR_VERSION)

    config_obj = {
        "anchors_source": anchors_source,
        "anchors_top_k": anchors_top_k,
        "anchor_method": anchor_method,
        "anchor_max_token_len": anchor_max_token_len,
        "max_ngram_n": max_ngram_n,
        "add_tok_ngrams": add_tok_ngrams,
        "add_anchor_windows": add_anchor_windows,
        "add_skeleton": add_skeleton,
        "add_compressed_skeleton": add_compressed_skeleton,
        "add_anchor_pairs": add_anchor_pairs,
        "add_anchor_skip2": add_anchor_skip2,
        "add_anchor_skip3": add_anchor_skip3,
        "add_anchor_sequence": add_anchor_sequence,
        "add_anchor_spans": add_anchor_spans,
        "add_span_signatures": add_span_signatures,
        "span_max_gap": span_max_gap,
        "skip_max_jump": skip_max_jump,
        "pattern_min_count_seen": pattern_min_count_seen,
        "pattern_min_distinct_sentences": pattern_min_distinct_sentences,
        "hsk_db_path": str(hsk_db_path),
        "hsk_table": hsk_table,
        "hsk_max_level": hsk_max_level,
        "include_level7": include_level7,
    }
    config_json = json.dumps(config_obj, sort_keys=True, ensure_ascii=False)
    config_hash = _sha256_text(config_json)
    _meta_put(conn, "config_hash", config_hash)

    conn.execute(
        """
        INSERT INTO runs(kind, created_at, config_hash, anchors_hash, notes)
        VALUES(?,?,?,?,?)
        """,
        ("bootstrap", now, config_hash, anchors_hash, None),
    )

    conn.commit()

    grammar = GrammarState(
        min_count_seen=pattern_min_count_seen,
        min_distinct_sentence_count=pattern_min_distinct_sentences,
    )

    # Insert sentences + patterns, update grammar state
    for s, tj, th, tc in zip(sentences, tokens_jieba_all, tokens_hsk_all, tokens_char_all):
        pats, skel = extract_patterns_from_tokens(
            tj,
            anchors=anchors,
            max_ngram_n=max_ngram_n,
            add_tok_ngrams=add_tok_ngrams,
            add_anchor_windows=add_anchor_windows,
            add_skeleton=add_skeleton,
            add_compressed_skeleton=add_compressed_skeleton,
            add_anchor_pairs=add_anchor_pairs,
            add_anchor_skip2=add_anchor_skip2,
            add_anchor_skip3=add_anchor_skip3,
            add_anchor_sequence=add_anchor_sequence,
            add_anchor_spans=add_anchor_spans,
            add_span_signatures=add_span_signatures,
            span_max_gap=span_max_gap,
            skip_max_jump=skip_max_jump,
        )

        conn.execute(
            """
            INSERT OR IGNORE INTO sentences(
                zh_text,
                tokens_jieba_json,
                tokens_hsk_json,
                tokens_char_json,
                patterns_json,
                skeleton,
                source,
                created_at
            )
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                s,
                json.dumps(tj, ensure_ascii=False),
                json.dumps(th, ensure_ascii=False),
                json.dumps(tc, ensure_ascii=False),
                json.dumps([p.pattern_key for p in pats], ensure_ascii=False),
                skel,
                source,
                now,
            ),
        )

        grammar.observe_sentence([(p.pattern_key, p.realization) for p in pats])

    conn.commit()

    # Vocab stats (HSK tokens)
    vocab_counts = count_vocab(tokens_hsk_all)
    for word, cnt in vocab_counts.items():
        mastery = math.log1p(cnt)
        meta = lex.meta(word)
        hsk_level = meta.level if meta else None
        hsk_freq = meta.frequency if meta else None

        conn.execute(
            """
            INSERT OR REPLACE INTO vocab_stats(word, count, mastery, last_seen, hsk_level, hsk_frequency)
            VALUES(?,?,?,?,?,?)
            """,
            (word, cnt, mastery, now, hsk_level, hsk_freq),
        )

    # Pattern stats + realizations
    for pkey, st in grammar.patterns.items():
        emerged = 1 if st.emerged(grammar.min_count_seen, grammar.min_distinct_sentence_count) else 0

        conn.execute(
            """
            INSERT OR REPLACE INTO pattern_personal_stats(
                pattern_key, family, count_seen, distinct_sentence_count, emerged, last_seen_at
            )
            VALUES(?,?,?,?,?,?)
            """,
            (pkey, family_from_key(pkey), st.count_seen, st.distinct_sentence_count, emerged, now),
        )

        for r in st.realizations:
            conn.execute(
                "INSERT OR IGNORE INTO pattern_personal_realizations(pattern_key, realization) VALUES(?,?)",
                (pkey, r),
            )

    conn.commit()
    conn.close()
