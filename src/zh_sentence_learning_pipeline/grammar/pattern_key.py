"""
grammar/pattern_key.py

Centralized, stable PatternKey construction.

PatternKey fields:
  - family: string (e.g. skel, cskel, anch_pair, anch_win)
  - anchors: ordered list of anchor tokens in the pattern (may include duplicates)
  - params: small, normalized key/value pairs (sorted by key)

PatternKey string format (deterministic):
  family|a=<anchors_csv>|p=<params_csv>

Notes:
- We use light escaping for separators to keep keys readable.
- The first segment is always the family, so family extraction is trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


_ESCAPE_TABLE = {
    "\\": "\\\\",
    "|": "\\|",
    ",": "\\,",
    "=": "\\=",
}


def _escape_component(text: str) -> str:
    out = str(text)
    for raw, repl in _ESCAPE_TABLE.items():
        out = out.replace(raw, repl)
    return out


def _normalize_params(params: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
    if not params:
        return tuple()
    items = []
    for k, v in params.items():
        key = str(k).strip()
        if key == "":
            raise ValueError("PatternKey param key cannot be empty.")
        items.append((key, str(v)))
    items.sort(key=lambda kv: kv[0])
    return tuple(items)


@dataclass(frozen=True)
class PatternKey:
    family: str
    anchors: tuple[str, ...]
    params: tuple[tuple[str, str], ...]

    def to_string(self) -> str:
        if "|" in self.family:
            raise ValueError(f"PatternKey family contains illegal '|': {self.family}")
        anchors_csv = ",".join(_escape_component(a) for a in self.anchors)
        params_csv = ",".join(
            f"{_escape_component(k)}={_escape_component(v)}" for k, v in self.params
        )
        return f"{self.family}|a={anchors_csv}|p={params_csv}"


def make_key(
    family: str,
    anchors: Sequence[str] | None = None,
    params: Mapping[str, object] | None = None,
) -> str:
    pk = PatternKey(
        family=family,
        anchors=tuple(anchors or ()),
        params=_normalize_params(params),
    )
    return pk.to_string()


def family_from_key(pattern_key: str) -> str:
    if not pattern_key:
        return "unknown"
    return pattern_key.split("|", 1)[0]


# -------------------------
# Family-specific helpers
# -------------------------

def key_skeleton(skeleton_sig: str, anchors_in_order: Sequence[str]) -> str:
    return make_key(
        family="skel",
        anchors=anchors_in_order,
        params={"sig": skeleton_sig},
    )


def key_compressed_skeleton(skeleton_sig: str, anchors_in_order: Sequence[str]) -> str:
    return make_key(
        family="cskel",
        anchors=anchors_in_order,
        params={"sig": skeleton_sig},
    )


def key_anchor_pair(a1: str, a2: str, gap_bucket: str) -> str:
    return make_key(
        family="anch_pair",
        anchors=[a1, a2],
        params={"gap": gap_bucket},
    )


def key_anchor_window(
    window_sig: str,
    anchors_in_order: Sequence[str],
    left_len: int,
    right_len: int,
) -> str:
    return make_key(
        family="anch_win",
        anchors=anchors_in_order,
        params={"l": left_len, "r": right_len, "sig": window_sig},
    )


def key_anchor_skip(anchors_in_order: Sequence[str], max_jump: int) -> str:
    family = f"a_skip{len(anchors_in_order)}"
    return make_key(
        family=family,
        anchors=anchors_in_order,
        params={"max_jump": max_jump},
    )


def key_anchor_sequence(anchors_in_order: Sequence[str]) -> str:
    return make_key(
        family="anch_seq",
        anchors=anchors_in_order,
        params=None,
    )


def key_token_ngram(slot_sig: str, anchors_in_order: Sequence[str], n: int) -> str:
    return make_key(
        family="tok_ng",
        anchors=anchors_in_order,
        params={"n": n, "sig": slot_sig},
    )


def key_anchor_span(a1: str, tail: str, gap_bucket: str) -> str:
    return make_key(
        family="anch_span",
        anchors=[a1, tail],
        params={"gap": gap_bucket},
    )


def key_span_signature(a1: str, tail: str, gap_bucket: str, kA: int, kX: int) -> str:
    return make_key(
        family="span_sig",
        anchors=[a1, tail],
        params={"gap": gap_bucket, "kA": kA, "kX": kX},
    )
