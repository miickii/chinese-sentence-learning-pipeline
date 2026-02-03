"""
grammar/patterns.py

Goal
- Layer B pattern extraction (purely statistical; no preset grammar list).
- Anchors are discovered from corpus frequency/coverage (build_anchor_set),
  optionally constrained to an allowed candidate set (e.g. global POS-derived).
- Patterns are abstract, length-robust, and avoid collapsing everything into a few mega-patterns.

What this file implements
1) Anchor discovery (TF or DF) (+ optional allowed whitelist)

2) Pattern extraction families:

   A) Token n-grams containing anchors (anchor-specific slots: <A:的>)
      - local context signal; high recall; can be length-sensitive

   B) Anchor windows (±2 tokens around each anchor)
      - captures immediate grammar neighborhoods around anchors

   C) Skeleton patterns:
      - skel: coarse token-shape + anchors
      - cskel: compressed skeleton that collapses non-anchor runs into <SPAN>

   D) Span-based "next-anchor" patterns (NO hardcoded Chinese words):
      - anch_span: from an anchor to the next anchor (or END), with gap bucketing
      - span_sig: like anch_span but adds an interior "shape signature" (counts of anchors/non-anchors)

   E) Anchor skip-grams (NO hardcoded Chinese words):
      - a_skip2 / a_skip3 over the anchor-only sequence, bounded by max_jump

   F) NEW: Anchor-to-anchor pair patterns (NO hardcoded Chinese words):
      - anch_pair: from any anchor to any later anchor within max_gap, with gap bucketing
        This is the key feature for patterns like “因为 … 所以 …” / “虽然 … 但是 …”
        even when the content between them is long.

   G) NEW: Anchor sequence signature (NO hardcoded Chinese words):
      - anch_seq: the ordered sequence of anchors appearing in the sentence
        Captures constructions where anchor order matters, independent of content length.

Notes on controlling explosion:
- anchor-specific slots only in n-gram IDs (prevents mega-collapsing)
- gap bucketing in spans/pairs (prevents unique gap per length)
- bounded max_jump for skip-grams
- optional toggles to enable/disable families
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Iterable, List, Set, Tuple
import re

from .pattern_key import (
    key_anchor_pair,
    key_anchor_sequence,
    key_anchor_skip,
    key_anchor_span,
    key_anchor_window,
    key_compressed_skeleton,
    key_skeleton,
    key_span_signature,
    key_token_ngram,
)


@dataclass(frozen=True)
class Pattern:
    pattern_key: str
    realization: str


# -------------------------
# Anchors
# -------------------------

def build_anchor_set(
    corpus_tokens: List[List[str]],
    top_k: int = 60,
    method: str = "df",
    max_token_len: int = 2,
    allowed: Set[str] | None = None,
) -> Set[str]:
    """
    Anchors = short, high-coverage tokens that tend to be structural "glue".

    If allowed is provided, only tokens in allowed can become anchors.
    This supports: global POS-derived candidates + local DF activation.

    method:
    - "tf": total frequency across corpus
    - "df": document frequency (#sentences containing token) (recommended)
    """
    if method not in {"tf", "df"}:
        raise ValueError(f"Unknown method={method}. Use 'tf' or 'df'.")

    def ok(t: str) -> bool:
        if len(t) > max_token_len:
            return False
        if allowed is not None and t not in allowed:
            return False
        return True

    if method == "tf":
        tf = Counter(t for sent in corpus_tokens for t in sent if ok(t))
        return set(t for t, _ in tf.most_common(top_k))

    # method == "df"
    df = Counter()
    for sent in corpus_tokens:
        short_unique = {t for t in sent if ok(t)}
        df.update(short_unique)
    return set(t for t, _ in df.most_common(top_k))


# -------------------------
# Token utilities
# -------------------------

def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


_NUM_RE = re.compile(r"\d+(\.\d+)?")


# -------------------------
# Skeletons
# -------------------------

def skeletonize(tokens: List[str], anchors: Set[str]) -> str:
    out: List[str] = []
    for t in tokens:
        if t in anchors:
            out.append(t)
        elif _NUM_RE.fullmatch(t):
            out.append("<NUM>")
        elif len(t) == 1:
            out.append("<C1>")
        elif len(t) == 2:
            out.append("<C2>")
        else:
            out.append("<W>")
    return " ".join(out)


def skeletonize_compressed(tokens: List[str], anchors: Set[str]) -> str:
    """
    Length-robust skeleton:
    - keep anchors as-is
    - collapse any run of non-anchor tokens into <SPAN>
    """
    out: List[str] = []
    in_span = False

    for t in tokens:
        if t in anchors:
            out.append(t)
            in_span = False
        else:
            if not in_span:
                out.append("<SPAN>")
                in_span = True

    # trim leading/trailing <SPAN>
    while out and out[0] == "<SPAN>":
        out.pop(0)
    while out and out[-1] == "<SPAN>":
        out.pop()

    return " ".join(out)


# -------------------------
# Gap bucketing
# -------------------------

def _bucket_gap(g: int) -> str:
    # small buckets so lengths collapse into shared patterns
    if g == 0:
        return "0"
    if g == 1:
        return "1"
    if g <= 3:
        return "2-3"
    if g <= 7:
        return "4-7"
    return "8+"


def _anchor_sequence(tokens: List[str], anchors: Set[str]) -> List[Tuple[str, int]]:
    return [(t, i) for i, t in enumerate(tokens) if t in anchors]


# -------------------------
# Pattern families (helpers)
# -------------------------

def _anchor_skipgrams(
    tokens: List[str],
    anchors: Set[str],
    max_jump: int = 10,
    add_bigrams: bool = True,
    add_trigrams: bool = True,
) -> list[Pattern]:
    extracted: list[Pattern] = []
    seq = _anchor_sequence(tokens, anchors)

    # 2-anchor skip-grams
    if add_bigrams:
        for a in range(len(seq)):
            A1, i1 = seq[a]
            for b in range(a + 1, len(seq)):
                A2, i2 = seq[b]
                if i2 - i1 > max_jump:
                    break
                pkey = key_anchor_skip([A1, A2], max_jump=max_jump)
                extracted.append(Pattern(pkey, f"{A1} ... {A2}"))

    # 3-anchor skip-grams
    if add_trigrams:
        for a in range(len(seq)):
            A1, i1 = seq[a]
            for b in range(a + 1, len(seq)):
                A2, i2 = seq[b]
                if i2 - i1 > max_jump:
                    break
                for c in range(b + 1, len(seq)):
                    A3, i3 = seq[c]
                    if i3 - i2 > max_jump:
                        break
                    pkey = key_anchor_skip([A1, A2, A3], max_jump=max_jump)
                    extracted.append(Pattern(pkey, f"{A1} ... {A2} ... {A3}"))

    return extracted


def _anchor_span_patterns(
    tokens: List[str],
    anchors: Set[str],
    max_gap: int = 20,
) -> list[Pattern]:
    """
    For each anchor token t, look forward to the NEXT anchor within max_gap.
    This is a "local long-distance" family.
    """
    extracted: list[Pattern] = []
    n = len(tokens)

    for i, t in enumerate(tokens):
        if t not in anchors:
            continue

        tail = "<END>"
        tail_pos = None

        j_limit = min(n, i + 1 + max_gap)
        for j in range(i + 1, j_limit):
            if tokens[j] in anchors:
                tail = tokens[j]
                tail_pos = j
                break

        gap = (tail_pos - i - 1) if tail_pos is not None else min(max_gap, n - i - 1)
        gapb = _bucket_gap(gap)

        pkey = key_anchor_span(t, tail, gapb)
        end = (tail_pos + 1) if tail_pos is not None else j_limit
        realization = " ".join(tokens[i:end])
        extracted.append(Pattern(pkey, realization))

    return extracted


def _span_signature_patterns(
    tokens: List[str],
    anchors: Set[str],
    max_gap: int = 20,
) -> list[Pattern]:
    """
    Like anch_span, but includes a coarse interior signature:
    how many anchors (kA) and non-anchors (kX) are inside the span.
    """
    extracted: list[Pattern] = []
    n = len(tokens)

    for i, t in enumerate(tokens):
        if t not in anchors:
            continue

        tail = "<END>"
        tail_pos = None

        j_limit = min(n, i + 1 + max_gap)
        for j in range(i + 1, j_limit):
            if tokens[j] in anchors:
                tail = tokens[j]
                tail_pos = j
                break

        end = (tail_pos if tail_pos is not None else j_limit)
        inside = tokens[i + 1 : end]

        kA = sum(1 for x in inside if x in anchors)
        kX = len(inside) - kA
        gap = len(inside)
        gapb = _bucket_gap(gap)

        pkey = key_span_signature(t, tail, gapb, kA=kA, kX=kX)
        realization = " ".join(tokens[i : (tail_pos + 1 if tail_pos is not None else j_limit)])
        extracted.append(Pattern(pkey, realization))

    return extracted


def _anchor_pair_patterns(
    tokens: List[str],
    anchors: Set[str],
    max_gap: int = 20,
) -> list[Pattern]:
    """
    NEW:
    Any anchor -> any later anchor within max_gap (not just next anchor).
    This is the most direct "pattern grammar" extraction:
      因为 ... 所以 ...
      虽然 ... 但是 ...
      如果 ... 就 ...
    without hardcoding those words.
    """
    extracted: list[Pattern] = []
    seq = _anchor_sequence(tokens, anchors)  # [(anchor, index), ...]

    for a in range(len(seq)):
        A1, i1 = seq[a]
        for b in range(a + 1, len(seq)):
            A2, i2 = seq[b]
            gap = i2 - i1 - 1
            if gap > max_gap:
                break  # indices increase, so further anchors only increase gap
            pkey = key_anchor_pair(A1, A2, _bucket_gap(gap))
            realization = " ".join(tokens[i1 : i2 + 1])
            extracted.append(Pattern(pkey, realization))

    return extracted


def _anchor_sequence_signature(
    tokens: List[str],
    anchors: Set[str],
) -> list[Pattern]:
    """
    NEW:
    Ordered anchor signature of the sentence: anch_seq:a->b->c...
    Useful when anchor order defines the construction.
    """
    seq = [t for t in tokens if t in anchors]
    if len(seq) < 2:
        return []
    pkey = key_anchor_sequence(seq)
    return [Pattern(pkey, " ".join(seq))]


# -------------------------
# Main extractor
# -------------------------

def extract_patterns_from_tokens(
    tokens: List[str],
    anchors: Set[str],
    max_ngram_n: int = 4,
    # toggles
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
    # params
    span_max_gap: int = 20,
    skip_max_jump: int = 10,
) -> tuple[list[Pattern], str]:
    extracted: list[Pattern] = []

    # 1) token n-grams (anchor-specific slots to avoid mega collapse)
    if add_tok_ngrams:
        def slot(tok: str) -> str:
            return tok if tok in anchors else "<X>"

        for n in range(2, max_ngram_n + 1):
            for ng in ngrams(tokens, n):
                if any(t in anchors for t in ng):
                    sig = " ".join(slot(t) for t in ng)
                    anchors_in_order = [t for t in ng if t in anchors]
                    pkey = key_token_ngram(sig, anchors_in_order, n=n)
                    extracted.append(Pattern(pkey, " ".join(ng)))

    # 2) anchor window patterns (±2)
    if add_anchor_windows:
        def ph(x: str) -> str:
            return x if x in anchors else "<X>"

        for i, t in enumerate(tokens):
            if t in anchors:
                left = tokens[max(0, i - 2) : i]
                right = tokens[i + 1 : i + 3]
                sig_tokens = [ph(x) for x in left + [t] + right]
                sig = " ".join(sig_tokens)
                anchors_in_order = [x for x in sig_tokens if x in anchors]
                pkey = key_anchor_window(
                    window_sig=sig,
                    anchors_in_order=anchors_in_order,
                    left_len=len(left),
                    right_len=len(right),
                )
                extracted.append(Pattern(pkey, " ".join(left + [t] + right)))

    # 3) anchor-to-anchor pairs (any later anchor within max gap)
    if add_anchor_pairs:
        extracted.extend(_anchor_pair_patterns(tokens, anchors, max_gap=span_max_gap))

    # 4) anchor-sequence signature (ordered anchors)
    if add_anchor_sequence:
        extracted.extend(_anchor_sequence_signature(tokens, anchors))

    # 5) original skeleton
    skel = skeletonize(tokens, anchors)
    if add_skeleton:
        anchors_in_order = [t for t in tokens if t in anchors]
        pkey = key_skeleton(skel, anchors_in_order)
        extracted.append(Pattern(pkey, " ".join(tokens)))

    # 6) compressed skeleton
    if add_compressed_skeleton:
        cskel = skeletonize_compressed(tokens, anchors)
        if cskel:
            anchors_in_order = [t for t in tokens if t in anchors]
            pkey = key_compressed_skeleton(cskel, anchors_in_order)
            extracted.append(Pattern(pkey, " ".join(tokens)))

    # 7) anchor-span patterns (next-anchor)
    if add_anchor_spans:
        extracted.extend(_anchor_span_patterns(tokens, anchors, max_gap=span_max_gap))

    # 8) span signatures (next-anchor + interior shape)
    if add_span_signatures:
        extracted.extend(_span_signature_patterns(tokens, anchors, max_gap=span_max_gap))

    # 9) anchor skip-grams (anchor-only sequence)
    if add_anchor_skip2 or add_anchor_skip3:
        extracted.extend(
            _anchor_skipgrams(
                tokens,
                anchors,
                max_jump=skip_max_jump,
                add_bigrams=add_anchor_skip2,
                add_trigrams=add_anchor_skip3,
            )
        )

    return extracted, skel
