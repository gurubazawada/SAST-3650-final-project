from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import warnings

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning
from ebooklib import epub, ITEM_DOCUMENT

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from .corpus import Verse, make_verse_id


_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _parse_versepara(p, current_speaker: str | None) -> tuple[list[tuple[int, str | None, str]], str | None]:
    """Return (verses, trailing_orphan).

    verses is a list of (verse_num, speaker, english_text).
    trailing_orphan is text in this <p> that never got a verseno — caller may append
    it to the previously-emitted verse (or discard).
    """
    tokens: list[tuple[str, object]] = []
    for node in p.descendants:
        if isinstance(node, NavigableString):
            parent = node.parent
            if parent is not None and parent.name == "span" and "verseno" in (parent.get("class") or []):
                continue
            tokens.append(("text", str(node)))
        elif node.name == "span" and "verseno" in (node.get("class") or []):
            raw = node.get_text(strip=True)
            try:
                tokens.append(("vn", int(raw)))
            except ValueError:
                pass

    verses: list[tuple[int, str | None, str]] = []
    leading: list[str] = []
    current_vn: int | None = None
    buf: list[str] = []

    for kind, val in tokens:
        if kind == "vn":
            assert isinstance(val, int)
            if current_vn is None:
                buf = leading + buf
                leading = []
            else:
                text = _clean("".join(buf))
                if text:
                    verses.append((current_vn, current_speaker, text))
                buf = []
            current_vn = val
        else:
            target = buf if current_vn is not None else leading
            target.append(str(val))

    if current_vn is not None:
        text = _clean("".join(buf))
        if text:
            verses.append((current_vn, current_speaker, text))

    trailing_orphan: str | None = None
    if current_vn is None:
        orphan_text = _clean("".join(leading))
        if orphan_text:
            trailing_orphan = orphan_text

    return verses, trailing_orphan


def extract_verses_easwaran(html: str, chapter: int) -> list[Verse]:
    soup = BeautifulSoup(html, "lxml")

    header = soup.find("h2", class_="sigilNotInTOC")
    if header is None:
        return []

    verses: list[Verse] = []
    current_speaker: str | None = None
    last_builder: list | None = None

    for node in header.find_all_next(["p", "h2"]):
        if node is header:
            continue
        classes = node.get("class") or []

        if node.name == "h2":
            break

        if "speaker" in classes:
            current_speaker = node.get_text(strip=True) or None
            continue

        if "versepara" not in classes:
            continue

        parsed, orphan = _parse_versepara(node, current_speaker)

        for vn, speaker, text in parsed:
            v = Verse(
                verse_id=make_verse_id("Bhagavad Gita", chapter, vn),
                source="Bhagavad Gita",
                chapter=chapter,
                verse_num=vn,
                speaker=speaker,
                sanskrit=None,
                english=text,
                translator="Eknath Easwaran",
            )
            verses.append(v)
            last_builder = [v]

        if orphan and last_builder:
            v = last_builder[0]
            new_v = Verse(
                verse_id=v.verse_id,
                source=v.source,
                chapter=v.chapter,
                verse_num=v.verse_num,
                speaker=v.speaker,
                sanskrit=v.sanskrit,
                english=_clean(v.english + " " + orphan),
                translator=v.translator,
            )
            verses[-1] = new_v
            last_builder[0] = new_v

    return verses


PARSERS: dict[str, Callable[[str, int], list[Verse]]] = {
    "easwaran": extract_verses_easwaran,
}


_CHAPTER_FILE_RE = re.compile(r"^(\d{2})\.xhtml$", re.IGNORECASE)


def ingest_epub(epub_path: str | Path, parser: str = "easwaran") -> list[Verse]:
    parse = PARSERS[parser]
    book = epub.read_epub(str(epub_path))

    collected: list[Verse] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        name = Path(item.get_name()).name
        m = _CHAPTER_FILE_RE.match(name)
        if not m:
            continue
        chapter = int(m.group(1))
        html = item.get_content().decode("utf-8", errors="replace")
        verses = parse(html, chapter)
        collected.extend(verses)

    collected.sort(key=lambda v: (v.chapter, v.verse_num))
    return collected


def dump_toc(epub_path: str | Path, out_path: str | Path, snippet_chars: int = 200) -> None:
    book = epub.read_epub(str(epub_path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# TOC dump for {Path(epub_path).name}\n")
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        name = item.get_name()
        html = item.get_content().decode("utf-8", errors="replace")
        text = _clean(BeautifulSoup(html, "lxml").get_text(" "))
        snippet = text[:snippet_chars]
        lines.append(f"## {name}")
        lines.append(f"    {snippet}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
