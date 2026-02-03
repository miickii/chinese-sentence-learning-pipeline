# Build the Global Corpus

This guide builds a **large, clean, simplified‑only** Chinese corpus used for:
- global grammar frequency (`data/chinese_prior.db`)
- anchor validation (`data/final_anchors.json`)

---

## Prerequisites

Install Python deps for corpus cleaning:

```bash
pip install opencc-python-reimplemented
```

---

## 1) Create folders

```bash
mkdir -p data/raw/wiki data/raw/leipzig data/processed
```

---

## 2) Download datasets

### Wikipedia dump
```bash
cd data/raw/wiki
curl -L -o zhwiki-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
```

### Leipzig news (1M)
```bash
cd ../leipzig
curl -L -o zho_news_2007-2009_1M.tar.gz \
  https://downloads.wortschatz-leipzig.de/corpora/zho_news_2007-2009_1M.tar.gz

tar -xzf zho_news_2007-2009_1M.tar.gz
```

---

## 3) Extract Wikipedia text

If the repo already includes `wikiextractor/`, use it. Otherwise:

```bash
git clone https://github.com/attardi/wikiextractor.git
```

Run extraction:

```bash
python -m wikiextractor.WikiExtractor \
  -o data/raw/wiki/extracted \
  --json \
  --no-templates \
  --processes 1 \
  -b 250M \
  -q \
  data/raw/wiki/zhwiki-latest-pages-articles.xml.bz2 \
  2> data/raw/wiki/wikiextractor.log
```

---

## 4) Convert to sentence files

### Wikipedia → sentences
```bash
PYTHONPATH=src python scripts/corpora/wiki_to_sentences.py \
  data/raw/wiki/extracted \
  data/processed/wiki.sentences.txt
```

### Leipzig → sentences
Find the extracted sentences file (usually `*-sentences.txt`):

```bash
ls data/raw/leipzig | grep sentences
```

Then:

```bash
PYTHONPATH=src python scripts/corpora/clean_leipzig.py \
  data/raw/leipzig/<LEIPZIG_SENTENCES_FILE>.txt \
  data/processed/leipzig_news.sentences.txt
```

---

## 5) Mix into the global prior corpus

Example mix (60% wiki / 40% news):

```bash
PYTHONPATH=src python scripts/corpora/mix_corpora.py \
  --wiki data/processed/wiki.sentences.txt \
  --news data/processed/leipzig_news.sentences.txt \
  --out  data/processed/global_prior.sentences.txt \
  --wiki_n 600000 \
  --news_n 400000 \
  --seed 7
```

---

## 6) Quick checks

```bash
wc -l data/processed/*.sentences.txt
shuf -n 5 data/processed/global_prior.sentences.txt
```

