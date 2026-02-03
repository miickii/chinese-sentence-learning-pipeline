````markdown
# Getting Started — Build the Global Chinese Corpus

We build a **global sentence corpus** to learn **grammar-pattern frequencies** and **anchor statistics** (e.g., 的/了/在/因为/所以).  
It is **not** used for vocabulary teaching—only for a language-wide grammar + frequency prior and structure-diversity control.

## 1) Create folders

```bash
mkdir -p data/raw/wiki data/raw/leipzig data/processed
````

## 2) Download datasets (curl)

### Wikipedia dump (articles)

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

## 3) Extract Wikipedia text (WikiExtractor)

```bash
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
python -m wikiextractor.WikiExtractor \
  -o ../data/raw/wiki/extracted \
  --json \
  --no-templates \
  --processes 1 \
  -b 250M \
  -q \
  ../data/raw/wiki/zhwiki-latest-pages-articles.xml.bz2 \
  2> ../data/raw/wiki/wikiextractor.log
```

## Why these flags
--no-templates: stops the recursion spam + reduces CPU/memory.
--processes 1: avoids your machine getting hammered (you can try 2 later).
-b 250M: fewer files than 1M default, but not too huge.
-q: suppresses progress output (still logs errors to the log file).
2> ...log: keeps terminal responsive.
If you want a bit more speed and your laptop can take it, try --processes 2.

> If WikiExtractor errors on Python 3.11, use your patched version or run it with Python 3.10.

## 4) Prepare sentence files (our scripts)

These scripts produce **UTF-8, one sentence per line**, **strict simplified-only** (OpenCC check), and **deduplicated** outputs.

### Wikipedia → sentences

```bash
python scripts/wiki_to_sentences.py \
  data/raw/wiki/extracted \
  data/processed/wiki.sentences.txt
```

### Leipzig → sentences

Find the extracted sentences file (usually `*-sentences.txt`):

```bash
ls data/raw/leipzig | grep sentences
```

Then run:

```bash
python scripts/clean_leipzig.py \
  data/raw/leipzig/<LEIPZIG_SENTENCES_FILE>.txt \
  data/processed/leipzig_news.sentences.txt
```

## 5) Mix into the global prior

Example: 1,000,000 sentences total (60% wiki / 40% news)

```bash
python scripts/mix_corpora.py \
  --wiki data/processed/wiki.sentences.txt \
  --news data/processed/leipzig_news.sentences.txt \
  --out  data/processed/global_prior.sentences.txt \
  --wiki_n 600000 \
  --news_n 400000 \
  --seed 7
```

## Output

You should end with:

* `data/processed/wiki.sentences.txt`
* `data/processed/leipzig_news.sentences.txt`
* `data/processed/global_prior.sentences.txt`

Quick check:

```bash
wc -l data/processed/*.sentences.txt
shuf -n 5 data/processed/global_prior.sentences.txt
```

```
::contentReference[oaicite:0]{index=0}
```
