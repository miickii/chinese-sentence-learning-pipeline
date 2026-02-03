# Bootstrap the Project

This guide builds everything you need to **measure grammar/vocab progress**.

---

## Step 0 — Prerequisites

Install runtime deps:

```bash
pip install jieba opencc-python-reimplemented
```

---

## Step 1 — Build HSK DB

**Input:** `data/complete_hsk.json`

**Output:** `data/hsk_vocabulary_v2.db`

```bash
PYTHONPATH=src python scripts/build_hsk_db_from_json.py \
  --json data/complete_hsk.json \
  --out  data/hsk_vocabulary_v2.db
```

Quick check:

```bash
python -c "import sqlite3; c=sqlite3.connect('data/hsk_vocabulary_v2.db'); print('rows:', c.execute('select count(*) from chinese_words').fetchone()[0])"
```

---

## Step 2 — Build frozen anchors

```bash
PYTHONPATH=src python scripts/anchors/build_global_anchors.py \
  --json data/complete_hsk.json \
  --out  data/global_anchors.json \
  --pos  u,p,c,y,e,d \
  --max-len 4

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

---

## Step 3 — Build the global prior DB

**Input:**
- `data/processed/global_prior.sentences.txt`
- `data/final_anchors.json`

**Output:** `data/chinese_prior.db`

```bash
PYTHONPATH=src python scripts/build_prior_db.py \
  --corpus data/processed/global_prior.sentences.txt \
  --anchors data/final_anchors.json \
  --out data/chinese_prior.db
```

---

## Step 4 — Bootstrap personal state DB

**Input:**
- `data/bootstrap_known_chinese.csv`
- `data/hsk_vocabulary_v2.db`
- `data/final_anchors.json`

**Output:** `data/state.db`

Ensure `main.py` points to your CSV column and anchor file, then run:

```bash
PYTHONPATH=src python main.py
```

This writes:
- `sentences` (tokens + pattern keys)
- `vocab_stats`
- `pattern_personal_stats` and `pattern_personal_realizations`
- `meta` and `runs`

---

## Step 5 — Inspect and sanity‑check

**Unified check:**

```bash
PYTHONPATH=src python scripts/inspect_pipeline.py \
  --prior-db data/chinese_prior.db \
  --state-db data/state.db \
  --anchors data/final_anchors.json
```

**Optional focused checks:**

```bash
PYTHONPATH=src python scripts/inspect_bootstrap_state.py --db data/state.db --pairs 8
PYTHONPATH=src python scripts/inspect_prior_db.py --db data/chinese_prior.db --top 25
```

---

## Full command sequence (copy/paste)

```bash
PYTHONPATH=src python scripts/build_hsk_db_from_json.py \
  --json data/complete_hsk.json \
  --out  data/hsk_vocabulary_v2.db

PYTHONPATH=src python scripts/anchors/build_global_anchors.py \
  --json data/complete_hsk.json \
  --out  data/global_anchors.json \
  --pos  u,p,c,y,e,d \
  --max-len 4

PYTHONPATH=src python scripts/anchors/build_final_anchors_from_corpus.py \
  --candidates data/global_anchors.json \
  --corpus data/processed/global_prior.sentences.txt \
  --out data/final_anchors.json \
  --topk 600

PYTHONPATH=src python scripts/build_prior_db.py \
  --corpus data/processed/global_prior.sentences.txt \
  --anchors data/final_anchors.json \
  --out data/chinese_prior.db

PYTHONPATH=src python main.py

PYTHONPATH=src python scripts/inspect_pipeline.py \
  --prior-db data/chinese_prior.db \
  --state-db data/state.db \
  --anchors data/final_anchors.json
```

