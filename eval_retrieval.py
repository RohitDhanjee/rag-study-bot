import json
from pathlib import Path

from search import chunks, keyword_search, search, semantic_search

items = [q for q in json.loads(Path("eval_set.json").read_text(encoding="utf-8")) if q["page"] is not None]


def hit_rate(get_pages):
    hits = 0
    for q in items:
        ok = q["page"] in get_pages(q["question"])
        hits += ok
        if not ok:
            print(f"   MISS: {q['question']}")
    return f"{hits}/{len(items)} = {100 * hits / len(items):.0f}%"


print("Semantic only:")
print("  hit@3 =", hit_rate(lambda q: [chunks[i]["page"] for i in semantic_search(q, 3)]))
print("Keyword (BM25) only:")
print("  hit@3 =", hit_rate(lambda q: [chunks[i]["page"] for i in keyword_search(q, 3)]))
print("Hybrid (RRF):")
print("  hit@3 =", hit_rate(lambda q: [c["page"] for _, c in search(q, k=3)]))