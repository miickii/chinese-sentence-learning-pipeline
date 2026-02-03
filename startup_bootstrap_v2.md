# Bootstrap (v2) — Personal DB + Global Prior DB

This project has two databases:

1) **Personal state DB** — what *you* know (vocab + grammar mastery)  
   Output: `data/state.db`

2) **Global prior DB** — what is common in Chinese in general (pattern frequencies)  
   Output: `data/chinese_prior.db`

The key change in v2:
- Both DBs use the **same frozen anchors** (`data/final_anchors.json`)
- Therefore pattern IDs are consistent, and you can join personal stats with global stats by `pattern_id`.

---

## Preconditions

You are in the project root and have:

- `data/complete_hsk.json`
- `data/bootstrap_known_chinese.csv` (your known sentences)
- `data/processed/global_prior.sentences.txt` (1M corpus)
- `data/final_anchors.json` (from `anchors_v2.md`)

Dependencies:
```bash
pip install jieba
```

---

## Step 1 — Build the HSK SQLite DB (one-time)

**Input:** `data/complete_hsk.json`  
**Output:** `data/hsk_vocabulary_v2.db`

```bash
PYTHONPATH=src python scripts/build_hsk_db_from_json.py   --json data/complete_hsk.json   --out  data/hsk_vocabulary_v2.db
```

Quick check:
```bash
python -c "import sqlite3; c=sqlite3.connect('data/hsk_vocabulary_v2.db'); print(c.execute('select count(*) from chinese_words').fetchone()[0])"
```

---

## Step 2 — Build the frozen anchors (one-time per corpus/version)

```bash
PYTHONPATH=src python scripts/build_global_anchors.py   --json data/complete_hsk.json   --out  data/global_anchors.json   --pos  u,p,c,y,e,d   --max-len 4

PYTHONPATH=src python scripts/build_final_anchors_from_corpus.py   --candidates data/global_anchors.json   --corpus data/processed/global_prior.sentences.txt   --out data/final_anchors.json   --topk 600   --max-len 4   --min-df 50   --min-df-rate 0.0005   --min-entropy 1.8
```

---

## Step 3 — Build the global prior DB (from 1M corpus)

**Goal:** Learn global pattern frequencies (grammar prior).

**Input**
- `data/processed/global_prior.sentences.txt`
- `data/final_anchors.json`

**Output**
- `data/chinese_prior.db`

### What gets stored in the prior DB
Typical tables (names may vary depending on your implementation):
- `meta` (anchors + extractor config)
- `pattern_global_stats(pattern_id, count, diversity, log_freq, ...)`
- `pattern_global_realizations(pattern_id, realization)` (sampled examples)

### Command
Run your prior builder script (the project’s “global pass”):

```bash
PYTHONPATH=src python scripts/build_prior_db.py   --corpus data/processed/global_prior.sentences.txt   --anchors data/anchors_v1.json   --out data/chinese_prior.db   --max-ngram-n 3   --span-max-gap 20   --skip-max-jump 10
```

> If your script name differs, search for `prior` under `scripts/`.

---

## Step 4 — Bootstrap the personal state DB (from your known sentences)

**Goal:** Build *your* vocabulary + grammar mastery baseline.

**Input**
- `data/bootstrap_known_chinese.csv`
- `data/hsk_vocabulary_v2.db`
- `data/final_anchors.json`

**Output**
- `data/state.db`

### Update `main.py` (recommended)
Make sure `main.py` points to the frozen anchors file:

- `global_anchors_path="data/final_anchors.json"` (or rename the argument to `anchors_path` if you refactor)

Then run:

```bash
PYTHONPATH=src python main.py
```

What bootstrap stores into `data/state.db`:
- `sentences`:
  - original zh text
  - tokenizations (jieba / hsk-first / char)
  - extracted `patterns_json`
  - `skeleton`
- `vocab_stats` (from HSK-first tokens)
- `pattern_stats` + `pattern_realizations` (personal grammar mastery)
- `meta` (anchors + extractor config + thresholds)

---

## Step 5 — Inspect and sanity-check

### Personal DB inspection
```bash
PYTHONPATH=src python scripts/inspect_bootstrap_state.py   --db data/state.db   --pairs 8
```

Key things to look at:
- Tables exist + reasonable row counts
- `meta` contains the anchors and extractor config
- emerged pattern ratio is not crazy-low
- vocab unknown ratio isn’t huge (HSK tokenization working)

### Prior DB inspection (recommended)
```bash
PYTHONPATH=src python scripts/inspect_prior_db.py   --db data/chinese_prior.db   --top 25
```

---

## How the two DBs are used together later

Once you start generation/filtering, each candidate sentence gets extracted into a set of `pattern_id`s.

You then:
- check **personal DB**: which patterns are already emerged/familiar?
- check **prior DB**: how common are those patterns globally?

This gives you:
- **grammar novelty control** (≤ 1 new personal pattern)
- **curriculum scheduling** (prefer globally common patterns earlier)
- **structure similarity control** (avoid too-similar sentences by pattern overlap)

---

## Full “from zero” command sequence (copy/paste)

```bash
# 1) HSK db
PYTHONPATH=src python scripts/build_hsk_db_from_json.py   --json data/complete_hsk.json   --out  data/hsk_vocabulary_v2.db

# 2) anchors
PYTHONPATH=src python scripts/build_global_anchors.py   --json data/complete_hsk.json   --out  data/global_anchors.json   --pos  u,p,c,y,e,d   --max-len 4

PYTHONPATH=src python scripts/build_final_anchors_from_corpus.py   --candidates data/global_anchors.json   --corpus data/processed/global_prior.sentences.txt   --out data/final_anchors.json   --topk 600

# 3) prior db (global grammar model)
PYTHONPATH=src python scripts/build_prior_db.py   --corpus data/processed/global_prior.sentences.txt   --anchors data/final_anchors.json   --out data/chinese_prior.db

# 4) personal bootstrap
PYTHONPATH=src python main.py

# 5) inspect
PYTHONPATH=src python scripts/inspect_bootstrap_state.py   --db data/state.db --pairs 8
```
