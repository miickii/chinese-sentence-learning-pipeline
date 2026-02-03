"""
vocab/state.py

What this file does:
- Counts vocabulary occurrences across tokenized sentences.
- Mastery is log-smoothed (log1p(count)).

How it fits:
- In this project, vocab_stats are computed from tokens_hsk (HSK-first tokens).
- This makes later "HSK queue" and "exactly one new word" consistent and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Iterable


@dataclass
class VocabItem:
  word: str
  count: int

  @property
  def mastery(self) -> float:
    return math.log1p(self.count)


def count_vocab(token_lists: Iterable[list[str]]) -> Counter[str]:
  c: Counter[str] = Counter()
  for toks in token_lists:
    c.update(toks)
  return c
