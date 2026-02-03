"""
grammar/state.py

Layer C: Track pattern counts + learner evidence.

Key idea:
- Layer B extracts abstract pattern keys per sentence.
- Layer C tracks which patterns have been observed in *your* learned data.

Emergence rule:
- emerged if count_seen >= min_count_seen AND distinct_sentence_count >= min_distinct_sentence_count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Iterable, Set


@dataclass
class PatternStats:
  count_seen: int = 0
  distinct_sentence_count: int = 0
  realizations: Set[str] = field(default_factory=set)

  def observe_occurrence(self, realization: str) -> None:
    self.count_seen += 1
    if realization:
      self.realizations.add(realization)

  def observe_sentence(self) -> None:
    self.distinct_sentence_count += 1

  def emerged(self, min_count_seen: int, min_distinct_sentence_count: int) -> bool:
    return (
      self.count_seen >= min_count_seen
      and self.distinct_sentence_count >= min_distinct_sentence_count
    )


class GrammarState:
  """
  Dynamic statistical grammar state.

  Patterns are discovered from data (Layer B) and tracked here (Layer C).
  """

  def __init__(
    self,
    min_count_seen: int = 3,
    min_distinct_sentence_count: int = 2,
  ) -> None:
    self.patterns: Dict[str, PatternStats] = defaultdict(PatternStats)
    self.min_count_seen = int(min_count_seen)
    self.min_distinct_sentence_count = int(min_distinct_sentence_count)

  def observe_sentence(self, patterns: Iterable[tuple[str, str]]) -> None:
    """
    Update counts for one sentence.
    - count_seen increments per occurrence
    - distinct_sentence_count increments once per pattern per sentence
    """
    seen_in_sentence: Set[str] = set()

    for key, realization in patterns:
      st = self.patterns[key]
      st.observe_occurrence(realization)
      seen_in_sentence.add(key)

    for key in seen_in_sentence:
      self.patterns[key].observe_sentence()

  def is_emerged(self, pattern_key: str) -> bool:
    st = self.patterns.get(pattern_key)
    if st is None:
      return False
    return st.emerged(self.min_count_seen, self.min_distinct_sentence_count)
