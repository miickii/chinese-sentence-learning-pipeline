# Anchors (v2) — Global, Frozen, and Corpus‑Validated

This project uses **anchors** as a small, mostly-closed set of “function-ish” tokens (particles, prepositions, conjunctions, etc.) that provide a stable backbone for extracting **grammar patterns**.

The key update in v2 is:

- **Anchors are global and frozen** (same set used everywhere)
- Anchors are built in **two stages**:
  1) **Candidate list from a dictionary (POS-based)**
  2) **Validation + re-scoring against the large global corpus** (`data/processed/global_prior.sentences.txt`)

This makes your pattern IDs stable across:
- your personal learned-sentences DB
- the global prior DB
- future runs and future datasets

---

## Why anchors exist

Anchors solve two problems:

1) **Stability:** Token n-grams explode (too many unique patterns) and drift when tokenization shifts.
2) **Structure focus:** In Chinese, high-frequency “glue” words (的/了/在/把/因为/所以/虽然/但是/如果/就…) carry much of the sentence structure.

Anchors let you build patterns that are:
- **structural** (less topic/content sensitive)
- **length-robust** (works for short/long sentences)
- **joinable across DBs** (pattern_id is meaningful globally)

---

## The two-stage anchor pipeline

### Stage A — Build POS-derived candidate anchors (dictionary → candidates)

**Input**
- `data/complete_hsk.json` (or your dictionary JSON)

**Script**
- `scripts/build_global_anchors.py`

**Output**
- `data/global_anchors.json`  
  Format: `{"anchors": ["的","了","在", ...]}`

**Command**
```bash
PYTHONPATH=src python scripts/build_global_anchors.py   --json data/complete_hsk.json   --out  data/global_anchors.json   --pos  u,p,c,y,e,d   --max-len 4
```

Recommended POS (`--pos u,p,c,y,e,d`) is a good default:
- `u` particles
- `p` prepositions
- `c` conjunctions
- `y` modal/aux-ish (depends on dict)
- `e` exclamations
- `d` adverbs (optional; can add lots of candidates)

---

### Stage B — Validate + re-score candidates against the global corpus (candidates + corpus → final anchors)

This stage answers: *“Which candidate anchors are actually widely distributed and context-flexible in real text?”*

**Input**
- `data/global_anchors.json` (candidates)
- `data/processed/global_prior.sentences.txt` (your 1M-sentence corpus)

**Script**
- `scripts/build_final_anchors_from_corpus.py`

**Output (recommended name)**
- `data/final_anchors.json`  
  Format:
  ```json
  {
    "anchors": [...],
    "meta": {...},
    "stats": {
      "的": {"df":..., "tf":..., "df_rate":..., "H_lr":..., "score":...},
      ...
    }
  }
  ```

**Command (recommended)**
```bash
PYTHONPATH=src python scripts/build_final_anchors_from_corpus.py   --candidates data/global_anchors.json   --corpus data/processed/global_prior.sentences.txt   --out data/final_anchors.json   --topk 600   --max-len 4   --min-df 50   --min-df-rate 0.0005   --min-entropy 1.8
```

#### What the scoring means
For each candidate anchor, the script computes:

- **DF**: number of sentences containing the token  
- **TF**: total occurrences  
- **Neighbor entropy (H_lr)**: diversity of immediate left/right neighbors

“True anchors” tend to have:
- high DF (appear across many contexts)
- high neighbor entropy (combine with many neighbors)

#### Practical tuning tips
- If your final anchors include too many content-ish words: **increase `--min-entropy`**
- If you lose useful anchors (rare but grammatical): **lower `--min-df`** or increase `--topk`
- If you want a tighter anchor set: lower `--topk` (e.g. 300–500)

---

## How anchors are used at runtime

Anchors are used only for **pattern extraction and structure indexing**, not vocabulary teaching.

- Sentence is tokenized with **jieba** for grammar extraction
- Pattern extractor uses the frozen anchor set to build:
  - anchor windows
  - anchor-to-anchor pair patterns (e.g. 因为…所以…)
  - anchor sequences
  - skeletons / compressed skeletons
  - skip-grams over anchor sequences

The output is a set of `pattern_id`s per sentence that becomes your:
- grammar novelty tracker (personal state)
- structure similarity index (diversity control)
- global frequency lookup key (prior DB)

---

## Files you should commit (recommended)

Commit these so everyone uses the same anchor universe:

- `data/final_anchors.json`  ✅ (authoritative)
- optionally `data/global_anchors.json` (reproducibility)
- a short note in `meta` tables (stored automatically by bootstrap/prior build steps)

---

## Quick sanity checks

```bash
python -c "import json; d=json.load(open('data/final_anchors.json','r',encoding='utf-8')); print('anchors:', len(d['anchors'])); print('sample:', d['anchors'][:30])"
```

Look for lots of:
- 的 了 在 把 被 给 跟 和 也 就 都 还 又 很
- 因为 所以 虽然 但是 如果 就
- 着 过 呢 吗 吧

If you see many topical nouns/adjectives, raise `--min-entropy`.
