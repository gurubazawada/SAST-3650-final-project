from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


SOURCE_PREFIX = {
    "Bhagavad Gita": "BG",
    "Yoga Sutras": "YS",
    "Upanishads": "UP",
    "Mahabharata": "MB",
}


@dataclass(frozen=True)
class Verse:
    verse_id: str
    source: str
    chapter: int
    verse_num: int
    speaker: str | None
    sanskrit: str | None
    english: str
    translator: str


def make_verse_id(source: str, chapter: int, verse_num: int) -> str:
    prefix = SOURCE_PREFIX.get(source)
    if prefix is None:
        prefix = "".join(w[0] for w in source.split()).upper()
    return f"{prefix}.{chapter}.{verse_num}"


def dump_corpus(verses: list[Verse], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(v) for v in verses]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus(path: str | Path) -> list[Verse]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Verse(**entry) for entry in raw]
