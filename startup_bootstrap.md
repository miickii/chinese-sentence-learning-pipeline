````markdown
# Chinese Sentence Curriculum Generator — Quickstart (Bootstrap)

This guide shows the **exact commands** to set up the project and **finish the bootstrap load** using your current code and folder structure.

---

## 0) Preconditions

- You are in the **project root** (the folder that contains `src/`, `scripts/`, `main.py`).
- You have Python 3.10+ recommended.
- Your data files are (or will be) here:
  - `data/complete_hsk.json`
  - `data/bootstrap_known_chinese.csv`

---

## 1) Create a virtual environment + install dependencies

> If you already manage dependencies elsewhere, you can skip this section.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jieba
````

---

## 2) Create required folders

```bash
mkdir -p data
```

---

## 3) Build the HSK SQLite DB (from `complete_hsk.json`) — one-time

Input:

* `data/complete_hsk.json`

Output:

* `data/hsk_vocabulary_v2.db`

```bash
PYTHONPATH=src python scripts/build_hsk_db_from_json.py \
  --json data/complete_hsk.json \
  --out  data/hsk_vocabulary_v2.db
```

Quick sanity check:

```bash
python -c "import sqlite3; c=sqlite3.connect('data/hsk_vocabulary_v2.db'); print('rows:', c.execute('select count(*) from chinese_words').fetchone()[0])"
```

---

## 4) Build global anchor candidates JSON (POS-based) — one-time

Input:

* `data/complete_hsk.json`

Output:

* `data/global_anchors.json`

Recommended POS set (matches your script docs):

* `u,p,c,y,e,d`  (particles, prepositions, conjunctions, modal-ish, exclamations, adverbs)

```bash
PYTHONPATH=src python scripts/build_global_anchors.py \
  --json data/complete_hsk.json \
  --out  data/global_anchors.json \
  --pos  u,p,c,y,e,d \
  --max-len 4
```

---

## 5) Prepare your bootstrap CSV

Input:

* `data/bootstrap_known_chinese.csv`

Requirement:

* It must have a header column matching what `main.py` uses:

  * default: `sentence_zh`

If your CSV column name is different, edit `main.py`:

```py
zh_column="sentence_zh"  # change to your real column name
```

---

## 6) Run the bootstrap (writes `data/state.db`)

This will:

* initialize the SQLite schema
* load sentences from your CSV
* tokenize (jieba / HSK-first / chars)
* build anchors (from global candidates + local activation)
* extract patterns, build grammar stats
* write sentences + vocab_stats + pattern_stats + pattern_realizations + meta + runs

```bash
PYTHONPATH=src python main.py
```

Expected output:

* prints anchor debug list (because `debug_print_anchors=True`)
* ends with: `✅ Bootstrap complete: data/state.db`

---

## 7) Inspect and validate the bootstrap DB

This prints a “pasteable” report:

* DB table counts
* meta keys (anchors source, config, thresholds)
* activated anchors vs global candidates
* random sentence spot-checks
* pattern health + emerged ratio
* vocab health (unknown/fallback ratio)
* length robustness demo (pattern overlap)

```bash
PYTHONPATH=src python scripts/inspect_bootstrap_state.py \
  --db data/state.db \
  --global-anchors data/global_anchors.json \
  --pairs 8
```

Optional (compare stored anchors vs recomputed anchors):

```bash
PYTHONPATH=src python scripts/inspect_bootstrap_state.py \
  --db data/state.db \
  --global-anchors data/global_anchors.json \
  --pairs 8 \
  --recompute-anchors \
  --anchors-top-k 120 \
  --anchor-method df \
  --anchor-max-len 4
```

---

## 8) Summary: the full command sequence (copy/paste)

```bash
# (optional) env
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jieba

# folders
mkdir -p data

# build HSK db
PYTHONPATH=src python scripts/build_hsk_db_from_json.py \
  --json data/complete_hsk.json \
  --out  data/hsk_vocabulary_v2.db

# build global anchors candidates
PYTHONPATH=src python scripts/build_global_anchors.py \
  --json data/complete_hsk.json \
  --out  data/global_anchors.json \
  --pos  u,p,c,y,e,d \
  --max-len 4

# bootstrap -> creates data/state.db
PYTHONPATH=src python main.py

# inspect report
PYTHONPATH=src python scripts/inspect_bootstrap_state.py \
  --db data/state.db \
  --global-anchors data/global_anchors.json \
  --pairs 8
```

```
```
