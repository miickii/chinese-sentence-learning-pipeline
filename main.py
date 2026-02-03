"""
main.py

What this file does:
- Runs the bootstrapper once to initialize your project state DB.

How to run:
- From project root:
  PYTHONPATH=src python main.py
or
  PYTHONPATH=src python -m zh_sentence_learning_pipeline.bootstrap.bootstrap
"""

from __future__ import annotations

from pathlib import Path
from zh_sentence_learning_pipeline.bootstrap.bootstrap import bootstrap

if __name__ == "__main__":
    bootstrap(
        db_path=Path("data/state.db"),
        csv_path=Path("data/bootstrap_known_chinese.csv"),
        zh_column="sentence_zh",               # <-- change if needed
        hsk_db_path=Path("data/hsk_vocabulary_v2.db"),
        hsk_max_level=6,
        include_level7=False,
        anchors_top_k=120,
        max_ngram_n=3,
        global_anchors_path="data/global_anchors.json",
        debug_print_anchors=True,
    )
    print("✅ Bootstrap complete: data/state.db")
