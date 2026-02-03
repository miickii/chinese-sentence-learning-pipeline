"""
grammar/metrics.py

Shared metrics:
- emergence checks
- global coverage mass
- IDF-weighted Jaccard similarity
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Set


def is_emerged(
    count_seen: int,
    distinct_sentence_count: int,
    min_count_seen: int = 3,
    min_distinct_sentence_count: int = 2,
) -> bool:
    return (
        count_seen >= min_count_seen
        and distinct_sentence_count >= min_distinct_sentence_count
    )


def coverage_mass(emerged_keys: Iterable[str], p_global_by_key: Mapping[str, float]) -> float:
    total = 0.0
    for key in emerged_keys:
        total += float(p_global_by_key.get(key, 0.0))
    return total


def idf_weighted_jaccard(
    a: Set[str],
    b: Set[str],
    df_by_key: Mapping[str, int],
    total_docs: int,
) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    denom = 0.0
    numer = 0.0
    union = a | b
    inter = a & b

    for key in union:
        df = max(1, int(df_by_key.get(key, 1)))
        w = math.log(float(total_docs) / float(df))
        denom += w
        if key in inter:
            numer += w

    if denom <= 0.0:
        return 0.0
    return numer / denom
