"""EEM experience library: (record, lesson) items, retrieved before every investment decision."""
import json
import os
from pathlib import Path

from . import core

LIB = Path(os.environ.get("EXPOS_RESEARCH", core.ROOT / "research")) / "experience" / "library.jsonl"
TAGS = ("mechanism", "feature", "data", "image", "error", "setting", "causality", "evaluation", "hpo", "ensemble",
        "negative", "positive", "physics", "infrastructure")


def _items():
    return [json.loads(l) for l in LIB.read_text().splitlines() if l.strip()] if LIB.exists() else []


def add(record, applicability, finding, implication, tags, author):
    bad = [t for t in tags if t not in TAGS]
    if bad:
        raise core.ExposError(f"unknown tags {bad}; allowed {TAGS}")
    if not record:
        raise core.ExposError("record (evidence path or experiment id) is required")
    item = {"id": f"X{len(_items()) + 1:04d}", "ts": core.now(), "author": author, "record": record,
            "lesson": {"applicability": applicability, "finding": finding, "implication": implication},
            "tags": list(tags)}
    LIB.parent.mkdir(parents=True, exist_ok=True)
    with open(LIB, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return item


def query(text="", tags=(), k=8):
    items = [i for i in _items() if not tags or set(tags) & set(i["tags"])]
    if not items or not text:
        return items[-k:]
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join([*i["lesson"].values(), " ".join(i["tags"]), i["record"]]) for i in items]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(docs + [text])  # works for Korean and English
    sims = (vec.transform(docs) @ vec.transform([text]).T).toarray().ravel()
    return [items[i] | {"score": round(float(sims[i]), 3)} for i in sims.argsort()[::-1][:k]]
