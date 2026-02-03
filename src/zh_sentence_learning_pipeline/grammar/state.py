"""
grammar/state.py

Layer C: Track pattern counts + diversity (distinct realizations).

Key idea:
- Layer B extracts abstract pattern IDs per sentence.
- Layer C tracks which patterns have been observed in *your* learned data.

Emergence rule:
- emerged if count >= min_count AND diversity >= min_diversity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Set
import math


@dataclass
class PatternStats:
  count: int = 0
  realizations: Set[str] = field(default_factory=set)

  @property
  def mastery(self) -> float:
    return math.log1p(self.count)

  @property
  def diversity(self) -> int:
    return len(self.realizations)

  def emerged(self, min_count: int, min_diversity: int) -> bool:
    return self.count >= min_count and self.diversity >= min_diversity


class GrammarState:
  """
  Dynamic statistical grammar state.

  Patterns are discovered from data (Layer B) and tracked here (Layer C).
  """

  def __init__(self, min_count: int = 3, min_diversity: int = 2) -> None:
    self.patterns: Dict[str, PatternStats] = defaultdict(PatternStats)
    self.min_count = int(min_count)
    self.min_diversity = int(min_diversity)

  def observe(self, pattern_id: str, realization: str) -> None:
    st = self.patterns[pattern_id]
    st.count += 1
    st.realizations.add(realization)

  def is_emerged(self, pattern_id: str) -> bool:
    st = self.patterns.get(pattern_id)
    if st is None:
      return False
    return st.emerged(self.min_count, self.min_diversity)
