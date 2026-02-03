import sqlite3
import csv

conn = sqlite3.connect("data/state.db")
cur = conn.cursor()

cur.execute("""
SELECT word, count, hsk_level, hsk_frequency
FROM vocab_stats
ORDER BY count DESC
""")

with open("data/vocab_stats_full.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "count", "hsk_level", "hsk_frequency"])
    writer.writerows(cur.fetchall())

conn.close()
print("✅ Wrote data/vocab_stats_full.csv")
