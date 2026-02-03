# Anchor building (global anchors)

This step creates a **global anchor list** (function‑ish “grammar words”) that your pipeline uses to extract patterns from any sentence.

## Why anchors exist

Anchors are a small, mostly‑closed set of words that:

- occur in many different sentences (high distribution)
- strongly influence grammar / structure (e.g. particles, prepositions, conjunctions)
- make patterns more stable than using “any random n‑gram”

This is aligned with how function words are central to Chinese grammatical analysis and meaning shifts. fileciteturn3file14

## Inputs

1. **Candidate anchors** (dictionary-derived):
   - produced by `scripts/build_global_anchors.py` from your HSK/dictionary JSON using POS tags. fileciteturn3file1
   - output looks like `{"anchors": [...]}`. fileciteturn3file4

2. **Large corpus** (distribution evidence):
   - `combined_corpus_simplified.txt` (one sentence per line).

## How final anchors are produced

We refine candidates using the large corpus:

1. Tokenize each sentence (jieba).
2. For each candidate token, compute:
   - **DF**: number of sentences containing the token
   - **TF**: total occurrences
   - **Neighbor entropy**: diversity of left/right neighboring tokens  
     (function words tend to combine with many different neighbors)

3. Filter out tokens that are too rare or too “context-fixed”:
   - `df >= min_df`
   - `df_rate >= min_df_rate`
   - `H_lr >= min_entropy`

4. Score the remaining tokens (distribution + entropy + log(TF)), keep top‑K.

This yields anchors that are both **widely used** and **structurally useful**.

## Output

`data/global_anchors.final.json`:

- `anchors`: the list used at runtime
- `stats`: per-anchor df/tf/entropy/score (for debugging & tuning)

## How anchors are used later

During bootstrapping and the learning loop:

- Every sentence is tokenized.
- The pattern extractor finds anchor windows / anchor n‑grams / skeletons.
- Those pattern IDs become an index for:
  - **structure similarity** retrieval (“find sentences with similar grammar”)
  - **diversity control** (“avoid generating too-similar sentences”)
  - **grammar novelty scheduling** (limit new patterns per accepted sentence)

## Recommended commands

Build candidates (dictionary → initial list):

```bash
PYTHONPATH=src python scripts/build_global_anchors.py   --json data/complete_hsk.json   --out  data/global_anchors.json   --pos  u,p,c,y,e,d   --max-len 4
```

Refine with corpus (candidates + corpus → final list):

```bash
PYTHONPATH=src python scripts/build_final_anchors_from_corpus.py   --candidates data/global_anchors.json   --corpus data/combined_corpus_simplified.txt   --out data/global_anchors.final.json   --topk 600   --min-df 50   --min-df-rate 0.0005   --min-entropy 2.0
```

Tune `min_df`, `min_entropy`, and `topk` by inspecting the `stats` section.
