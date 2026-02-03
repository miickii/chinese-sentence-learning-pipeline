# Project Architecture — Chinese Sentence Curriculum Generator (v2)

This repository builds a **Chinese-first, self-updating sentence curriculum system**.

Core promise:
- Each accepted sentence introduces **exactly one new vocabulary item**
- At most **one new grammar pattern**
- Low similarity to previously accepted sentences (structure diversity)

Grammar is not a hand-written syllabus. Instead it is **discovered statistically** as “patterns” extracted from text using a frozen set of **anchors**.

---

## Big picture: the three layers

### Layer A — Vocabulary state (personal)
Tracks what words you have seen and how well you know them.

- tokenization for vocab is **HSK-first longest match**
- stored in `data/state.db` table `vocab_stats`

Why HSK-first:
- stable word units (no weird segmentation drift)
- consistent “exactly one new word” constraint

### Layer B — Pattern extraction (statistical grammar features)
Given jieba tokens and a frozen anchor set, extract pattern IDs such as:
- anchor windows
- anchor-to-anchor pairs (because…so…)
- anchor sequences (ordered anchors)
- skeletons / compressed skeletons
- skip-grams over anchor-only sequences

Patterns are designed to be:
- structural (anchor-driven)
- robust to length/content
- comparable across datasets

### Layer C — Grammar mastery (personal)
Tracks which pattern IDs have “emerged” for you.

Default emergence rule (tunable):
- count ≥ `min_count`
- diversity (distinct realizations) ≥ `min_diversity`

Stored in `data/state.db`:
- `pattern_stats`
- `pattern_realizations`

---

## Two databases (and why you want both)

### 1) Personal DB — `data/state.db`
Built from your known sentences (Anki export).

Contains:
- sentences + tokenizations + extracted patterns
- vocab stats (HSK-first)
- personal pattern stats (grammar mastery)
- meta (anchors + extractor config)

This DB answers:
- “What do I already know?”
- “Which patterns are familiar vs novel?”
- “What sentence structures have I already seen a lot?”

### 2) Global prior DB — `data/chinese_prior.db`
Built from a large corpus (e.g. your 1M sentences).

Contains:
- global pattern frequencies and diversity
- optional sampled realizations per pattern
- meta (anchors + extractor config)

This DB answers:
- “How common is this pattern in Chinese overall?”
- “Is this pattern rare/weird or frequent/standard?”
- “Which structures should be introduced earlier?”

**Important:** Both DBs must use the same frozen anchors to keep `pattern_id` stable.

---

## Data flow (what you run)

### A) Build the global corpus (one sentence per line)
See `Getting_started.md` for:
- downloading
- extracting
- cleaning
- mixing into:
  - `data/processed/global_prior.sentences.txt`

### B) Build anchors (frozen)
See `anchors_v2.md` for:
- `data/global_anchors.json` candidates (POS-based)
- `data/final_anchors.json` final (corpus-validated)

### C) Build the global prior DB
Run:
- `scripts/build_prior_db.py`  
Inputs:
- `data/processed/global_prior.sentences.txt`
- `data/final_anchors.json`

Output:
- `data/chinese_prior.db`

### D) Bootstrap the personal DB
Run:
- `main.py` (calls `bootstrap()`)

Inputs:
- your known sentences CSV
- HSK DB
- frozen anchors

Output:
- `data/state.db`

---

## How pattern IDs are used later

Patterns become a **structure index**.

Given a sentence, we can represent it as a set of pattern IDs.
Then we can:

1) **Control grammar novelty**
- allow ≤ 1 not-yet-emerged pattern

2) **Control structure similarity**
- compute overlap / Jaccard between candidate patterns and previous patterns
- reject near-duplicates even if the vocabulary is different

3) **Prior-guided curriculum**
- prioritize introducing patterns that are globally frequent
- avoid rare patterns early (unless you explicitly want them)

This is the mechanism that makes the system “tailored but varied”.

---

## Repository map (what’s where)

### `scripts/`
One-off, reproducible pipeline steps:
- corpus creation: `wiki_to_sentences.py`, `clean_leipzig.py`, `mix_corpora.py`
- anchors: `build_global_anchors.py`, `build_final_anchors_from_corpus.py`
- bootstrapping + inspection: `inspect_bootstrap_state.py`
- (recommended) prior build + inspection: `build_prior_db.py`, `inspect_prior_db.py`

### `src/zh_sentence_learning_pipeline/`
Core library code:
- `bootstrap/` bootstraps personal DB
- `grammar/` tokenization + pattern extraction + grammar state
- `hsk/` HSK lexicon + trie segmentation
- `store/` SQLite schema helpers
- `vocab/` vocab counting/mastery

---

## “If you’re new to the project, what should you do?”

1) Read `Getting_started.md` (global corpus)
2) Read `anchors_v2.md` (frozen anchor pipeline)
3) Follow `startup_bootstrap_v2.md`:
   - build HSK DB
   - build prior DB
   - bootstrap personal DB
   - run inspections

After that, you can implement the generator loop:
- propose candidate sentences (LLM or rule-based)
- extract patterns + vocab
- score with (personal DB + prior DB + similarity)
- accept best and update state
