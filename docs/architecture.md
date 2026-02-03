# Architecture

This document describes the **current architecture** of the project and how the pieces fit together.

---

## Goal (current scope)

The system is designed to **control novelty** in Chinese sentences.
Right now, the pipeline builds:

- a **global grammar prior** from a large corpus
- a **personal grammar/vocab state** from known sentences

These are the required foundations for the later **generation + judging** loop.

---

## Core concepts

### Anchors (frozen)
A small, stable set of high‑coverage function-ish tokens (particles, prepositions, conjunctions). The same anchor set is used everywhere.

### PatternKey (stable ID)
Every extracted pattern produces a stable **PatternKey**.
It must be **identical across the global prior and personal DBs**.

Format:
```
family|a=<anchors_csv>|p=<params_csv>
```

Examples:
- `skel|a=虽然,但是|p=sig=<SPAN>虽然<SPAN>但是<SPAN>`
- `anch_pair|a=因为,所以|p=gap=4-7`

### Pattern families (core)
These are enabled by default:
- `skel`: full skeleton shape with anchors
- `cskel`: compressed skeleton (collapses spans)
- `anch_pair`: ordered anchor pairs with gap bucket
- `anch_win`: local anchor window (±2 tokens)

Optional families (off by default):
- `a_skip2`: anchor skip‑grams of length 2
- `tok_ng`, `anch_seq`, `anch_span`, `span_sig`, `a_skip3`

---

## Databases

### 1) Global prior DB (`data/chinese_prior.db`)
Built from ~1M sentences. Stores **what exists globally** and **how frequent** it is.

Tables (main):
- `pattern_global_stats`
  - `pattern_key` (PRIMARY)
  - `family`
  - `count_sentences` (document frequency)
  - `count_occurrences` (token frequency)
  - `distinct_realization_count`
  - `p_global` (normalized DF)

- `pattern_global_realizations`
  - `pattern_key`
  - `realization`

### 2) Personal state DB (`data/state.db`)
Built from your known sentences (Anki export). Stores **what the learner has seen**.

Tables (main):
- `pattern_personal_stats`
  - `pattern_key` (PRIMARY)
  - `family`
  - `count_seen`
  - `distinct_sentence_count`
  - `emerged`
  - `last_seen_at`

- `pattern_personal_realizations`
  - `pattern_key`
  - `realization`

---

## Emergence + coverage

**Emergence rule (default):**
```
count_seen >= 3 AND distinct_sentence_count >= 2
```

**Global weighted coverage:**
```
coverage_mass = Σ_{emerged patterns} p_global(pattern)
```

This gives a single number like “you’ve covered X% of grammar mass.”

---

## Similarity (diversity control)

For each sentence, represent it as a set of PatternKeys.
Similarity uses **IDF‑weighted Jaccard** computed from prior DB DF:

```
w(p) = log(N / df(p))
J(A,B) = Σ w(p in A∩B) / Σ w(p in A∪B)
```

This lets you reject sentences that are structurally too similar to recent ones.

---

## Pipeline stages

1. **Build corpus** (1M sentences)
2. **Build frozen anchors** (candidates → corpus validation)
3. **Build global prior DB** (pattern frequencies)
4. **Bootstrap personal DB** (patterns + vocab from known sentences)
5. **Inspect** (sanity checks + coverage)
6. **Generation loop (future)**

---

## Where things live

- Pattern extraction: `src/zh_sentence_learning_pipeline/grammar/patterns.py`
- PatternKey construction: `src/zh_sentence_learning_pipeline/grammar/pattern_key.py`
- Global prior schema: `src/zh_sentence_learning_pipeline/store/prior_db.py`
- Personal state schema: `src/zh_sentence_learning_pipeline/store/db.py`
- Builders:
  - `scripts/build_prior_db.py`
  - `main.py` (bootstrap personal DB)
- Inspect:
  - `scripts/inspect_pipeline.py`
  - `scripts/inspect_prior_db.py`
  - `scripts/inspect_bootstrap_state.py`

