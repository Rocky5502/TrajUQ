from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def existing_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(p):
        item_id = str(row.get("example_id", row.get("id", "")))
        if item_id:
            if item_id in ids:
                raise ValueError(f"Duplicate id already present in {p}: {item_id}")
            ids.add(item_id)
    return ids
