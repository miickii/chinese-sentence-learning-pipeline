#!/usr/bin/env python3
"""
scripts/build_global_anchors.py

Build global_anchors.json from complete_hsk.json using POS tags.

Outputs:
  {"anchors": ["的","了","在","把","因为","所以", ...]}

Usage:
  PYTHONPATH=src python scripts/build_global_anchors.py \
    --json data/complete_hsk.json \
    --out  data/global_anchors.json \
    --pos  u,p,c,y,e \
    --max-len 4 \
    --min-level 1 \
    --max-level 7

Notes:
- You can include adverbs too by adding 'd' to --pos:
    --pos u,p,c,y,e,d
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

LEVEL_RE = re.compile(r"^(new|old)-(\d+)$")


def parse_level(level_list: List[str]) -> Optional[int]:
    new_levels: List[int] = []
    old_levels: List[int] = []
    for s in level_list or []:
        m = LEVEL_RE.match(str(s).strip())
        if not m:
            continue
        scheme, num = m.group(1), int(m.group(2))
        if scheme == "new":
            new_levels.append(num)
        else:
            old_levels.append(num)

    if new_levels:
        return min(new_levels)
    if old_levels:
        return min(old_levels)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to complete_hsk.json")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. data/global_anchors.json)")
    ap.add_argument(
        "--pos",
        default="u,p,c,y,e",
        help="Comma-separated POS tags to include as anchor candidates (default: u,p,c,y,e).",
    )
    ap.add_argument("--max-len", type=int, default=4, help="Max token length to include (default: 4)")
    ap.add_argument("--min-level", type=int, default=1, help="Min level to include (default: 1)")
    ap.add_argument("--max-level", type=int, default=7, help="Max level to include (default: 7)")
    ap.add_argument("--sort-by", choices=["frequency", "alpha"], default="frequency",
                    help="How to sort output anchors (default: frequency)")
    args = ap.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    wanted_pos = {p.strip() for p in args.pos.split(",") if p.strip()}
    max_len = args.max_len

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of entries.")

    anchors: List[Tuple[str, int]] = []  # (word, frequency_rank_or_big)
    seen: Set[str] = set()

    for e in data:
        if not isinstance(e, dict):
            continue

        word = str(e.get("simplified") or "").strip()
        if not word:
            continue
        if len(word) > max_len:
            continue

        lvl = parse_level(list(e.get("level") or []))
        if lvl is not None:
            if lvl < args.min_level or lvl > args.max_level:
                continue

        pos_list = e.get("pos") or []
        if not isinstance(pos_list, list) or not pos_list:
            continue

        if not any(str(p) in wanted_pos for p in pos_list):
            continue

        # frequency is a rank-like number in your JSON; smaller usually = more frequent
        freq_raw = e.get("frequency")
        try:
            freq = int(freq_raw)
        except Exception:
            freq = 10**9

        if word not in seen:
            anchors.append((word, freq))
            seen.add(word)

    if args.sort_by == "frequency":
        anchors.sort(key=lambda x: x[1])  # smaller freq rank first
    else:
        anchors.sort(key=lambda x: x[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"anchors": [w for w, _ in anchors]}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote: {out_path}")
    print(f"   anchors: {len(anchors)}")
    print(f"   pos: {sorted(list(wanted_pos))}")
    print(f"   max_len: {max_len}, level_range: [{args.min_level}, {args.max_level}]")
    print("   sample:", [w for w, _ in anchors[:25]])


if __name__ == "__main__":
    main()

'''
Current used command:
PYTHONPATH=src python scripts/build_global_anchors.py \
  --json data/complete_hsk.json \
  --out  data/global_anchors.core_candidates.json \
  --pos  u,p,c,y,e,d \
  --max-len 4 \
  --min-level 1 \
  --max-level 7 \
  --sort-by frequency
'''