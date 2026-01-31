import pandas as pd
import re
from bs4 import BeautifulSoup
from pathlib import Path


def clean_html(text: str) -> str:
    """Remove HTML, sound tags, and extra whitespace."""
    if not isinstance(text, str):
        return ""

    # Remove [sound:...] tags
    text = re.sub(r"\[sound:[^\]]+\]", "", text)

    # Parse HTML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_chinese_sentence(text: str) -> str:
    """
    Extract the first Chinese sentence.
    Assumes Chinese appears before pinyin / English.
    """
    # Keep only CJK + punctuation
    matches = re.findall(r"[一-龯，。！？、；：“”‘’（）…—]+", text)
    if matches:
        return matches[0].strip()
    return ""


def txt_to_csv(txt_path, csv_path):
    txt_path = Path(txt_path)
    csv_path = Path(csv_path)

    # Read tab-separated file
    df = pd.read_csv(
        txt_path,
        sep="\t",
        header=None,
        names=["front", "back"],
        encoding="utf-8",
    )

    records = []
    for _, row in df.iterrows():
        combined = f"{row['front']} {row['back']}"
        cleaned = clean_html(combined)
        zh = extract_chinese_sentence(cleaned)

        if zh:
            records.append({"sentence_zh": zh})

    out = pd.DataFrame(records).drop_duplicates()
    out.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"✅ Extracted {len(out)} Chinese sentences → {csv_path}")


if __name__ == "__main__":
    txt_to_csv(
        txt_path="Bootstrap_Known_Chinese.txt",
        csv_path="bootstrap_known_chinese.csv",
    )
