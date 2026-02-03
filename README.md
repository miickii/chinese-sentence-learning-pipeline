# Chinese Sentence Curriculum Generator

This project builds a **Chinese-first, self-updating sentence curriculum system** for language learning.

**Overall goal:** generate new Chinese sentences that introduce exactly one new vocabulary item, at most one new grammar pattern, and avoid structural repetition. This lets the curriculum adapt to what the learner already knows.

**Current status (what’s implemented):**
- Global **anchor system** (frozen anchor list derived from a 1M‑sentence corpus)
- Pattern extraction with a stable **PatternKey** format
- **Global prior DB** with pattern frequencies (`data/chinese_prior.db`)
- **Personal state DB** with emerged patterns (`data/state.db`)
- Inspection tools that validate anchors, patterns, and coverage

**Not yet implemented:**
- Candidate sentence generation
- Ensemble judging/verification
- The acceptance loop

---

## Quickstart (short)

```bash
# 1) Build HSK DB
PYTHONPATH=src python scripts/build_hsk_db_from_json.py \
  --json data/complete_hsk.json \
  --out  data/hsk_vocabulary_v2.db

# 2) Build anchors (candidates -> final)
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

# 3) Build global prior DB (grammar frequencies)
PYTHONPATH=src python scripts/build_prior_db.py \
  --corpus data/processed/global_prior.sentences.txt \
  --anchors data/final_anchors.json \
  --out data/chinese_prior.db

# 4) Bootstrap personal state DB
PYTHONPATH=src python main.py

# 5) Inspect pipeline health
PYTHONPATH=src python scripts/inspect_pipeline.py \
  --prior-db data/chinese_prior.db \
  --state-db data/state.db \
  --anchors data/final_anchors.json
```

---

## How it works (short version)

1. **Anchors**: a frozen set of high-coverage function-ish tokens (的/了/在/把/因为/所以/虽然/但是/如果/就…).
2. **Pattern extraction**: each sentence yields a set of **PatternKeys** based on anchors and sentence shape.
3. **Two DBs**:
   - **Global prior DB**: how frequent each pattern is in real Chinese.
   - **Personal state DB**: which patterns the learner has seen and which have emerged.
4. **Selection logic (planned)**: only accept sentences with ≤1 new pattern, favor common patterns, and avoid structural duplicates.

---

## Docs (recommended order)

1. `docs/architecture.md` — system design + PatternKey spec + DB schemas
2. `docs/anchors.md` — anchor pipeline + examples + sanity checks
3. `docs/corpus.md` — build the 1M‑sentence corpus (exact commands)
4. `docs/bootstrap.md` — end‑to‑end build: HSK → anchors → prior DB → state DB

---

## Repo structure

- `src/zh_sentence_learning_pipeline/` — core library
- `scripts/` — one‑off build/inspection tools
- `data/` — generated artifacts (corpus, anchors, SQLite DBs)

