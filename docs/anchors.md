# Anchors

Anchors are a **frozen set of function‑ish tokens** (particles, prepositions, conjunctions, etc.) used to extract grammar patterns in a stable way.

They make patterns **structural**, **length‑robust**, and **comparable** across datasets.

---

## Why anchors exist

1. **Stability**: lexical n‑grams explode in count and drift when tokenization changes.
2. **Structure**: in Chinese, function words carry most grammar structure.

---

## Examples of core anchors

You should expect many of these in the final list:

- 的 了 在 把 被 给 跟 和 也 就 都 还 又 很
- 因为 所以 虽然 但是 如果 就
- 着 过 呢 吗 吧

If you see too many topical nouns/adjectives, tighten the filters.

---

## Two‑stage anchor pipeline

### Stage A — Build candidate anchors (POS‑based)

**Input:** `data/complete_hsk.json` (or your dictionary JSON)

**Output:** `data/global_anchors.json`

```bash
PYTHONPATH=src python scripts/anchors/build_global_anchors.py \
  --json data/complete_hsk.json \
  --out  data/global_anchors.json \
  --pos  u,p,c,y,e,d \
  --max-len 4
```

Recommended POS set: `u,p,c,y,e,d`

---

### Stage B — Validate against the large corpus

**Input:**
- `data/global_anchors.json`
- `data/processed/global_prior.sentences.txt`

**Output:** `data/final_anchors.json`

```bash
PYTHONPATH=src python scripts/anchors/build_final_anchors_from_corpus.py \
  --candidates data/global_anchors.json \
  --corpus data/processed/global_prior.sentences.txt \
  --out data/final_anchors.json \
  --topk 600 \
  --max-len 4 \
  --min-df 50 \
  --min-df-rate 0.0005 \
  --min-entropy 1.8
```

**What it measures:**
- DF: number of sentences containing the token
- TF: total occurrences
- Neighbor entropy: diversity of left/right neighbors

**Tuning tips:**
- Too many content words → raise `--min-entropy`
- Missing useful anchors → lower `--min-df` or raise `--topk`

---

## How anchors are used

Each sentence is tokenized with **jieba**, then patterns are extracted using the frozen anchor set.

Core pattern families:
- `skel` / `cskel` — sentence shape
- `anch_pair` — long‑distance patterns (因为…所以…)
- `anch_win` — local anchor context (了/的/在 etc.)

Optional:
- `a_skip2` — anchor skip‑grams

---

## Sanity check

```bash
python -c "import json; d=json.load(open('data/final_anchors.json','r',encoding='utf-8')); print('anchors:', len(d['anchors'])); print('sample:', d['anchors'][:30])"
```

