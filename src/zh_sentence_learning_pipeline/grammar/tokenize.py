"""
grammar/tokenize.py

What this file does:
- Normalizes Chinese text (light normalization).
- Provides:
  1) jieba word tokenization (for grammar and anchors)
  2) character tokenization (fallback robustness)
  3) HSK-first tokenization using the HSKLexicon trie (stable vocab units)

How it fits:
- Bootstrapper stores all 3 tokenizations for each sentence.
- Later:
  - vocab growth constraints should use tokens_hsk
  - grammar patterns should use tokens_jieba
  - char tokens provide guardrails when segmentation is weird
"""

from __future__ import annotations

import re
from typing import List

import jieba

# Common punctuation + whitespace (single chars and sequences)
_PUNCT_RE = re.compile(
  r"[，。！？、；：…（）()“”\"'《》【】\[\]{}<>·—\-、\s"
  r",.!?;:]+"
)

def _is_keep_char(ch: str) -> bool:
  o = ord(ch)
  if 0x4E00 <= o <= 0x9FFF:  # CJK Unified Ideographs
    return True
  if ch.isdigit():
    return True
  return False

def normalize_zh(text: str) -> str:
  text = (text or "").strip()
  text = re.sub(r"\s+", " ", text)
  return text

def tokenize_words_jieba(text: str) -> List[str]:
  text = normalize_zh(text)
  tokens = [t.strip() for t in jieba.cut(text, cut_all=False)]
  return [t for t in tokens if t and not _PUNCT_RE.fullmatch(t)]

def tokenize_chars(text: str) -> List[str]:
  text = normalize_zh(text)
  text = _PUNCT_RE.sub("", text)
  return [ch for ch in text if _is_keep_char(ch)]

def tokenize_words_hsk_first(text: str, lexicon) -> List[str]:
  """
  Longest-match segmentation using the HSK trie.
  Non-matching characters become single-character tokens (if kept), punctuation is skipped.
  """
  text = normalize_zh(text)
  tokens: List[str] = []
  i = 0
  while i < len(text):
    ch = text[i]

    if _PUNCT_RE.fullmatch(ch):
      i += 1
      continue

    m = lexicon.longest_match(text, i)
    if m is not None:
      w, j = m
      tokens.append(w)
      i = j
      continue

    if _is_keep_char(ch):
      tokens.append(ch)
    i += 1

  return tokens
