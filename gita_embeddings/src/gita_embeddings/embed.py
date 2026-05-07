from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types


MODEL = "gemini-embedding-2-preview"
DIM = 3072

# Free tier is 100 RPM for gemini-embedding-2; pace at ~75 RPM for headroom.
DEFAULT_MIN_INTERVAL = 0.8


_RETRY_DELAY_PATTERNS = (
    re.compile(r"retry in ([\d.]+)\s*s"),
    re.compile(r"'retryDelay':\s*'([\d.]+)s'"),
    re.compile(r'"retryDelay":\s*"([\d.]+)s"'),
)


def _parse_retry_delay(msg: str) -> float | None:
    for pat in _RETRY_DELAY_PATTERNS:
        m = pat.search(msg)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def fmt_similarity(text: str) -> str:
    return f"task: sentence similarity | query: {text}"


def fmt_clustering(text: str) -> str:
    return f"task: clustering | query: {text}"


def fmt_classification(text: str) -> str:
    return f"task: classification | query: {text}"


def fmt_search_query(text: str) -> str:
    return f"task: search result | query: {text}"


def fmt_doc(title: str | None, text: str) -> str:
    return f"title: {title or 'none'} | text: {text}"


def _cache_key(model: str, dim: int, formatted_text: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{dim}|{formatted_text}".encode("utf-8"))
    return h.hexdigest()


class EmbeddingClient:
    """Thin wrapper over google-genai with disk caching and retry."""

    def __init__(
        self,
        cache_path: str | Path = "data/.embedding_cache.json",
        model: str = MODEL,
        dim: int = DIM,
        api_key: str | None = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> None:
        self.model = model
        self.dim = dim
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=key)
        self._api_calls = 0
        self._cache_hits = 0
        self._pending_flush = 0
        self._min_interval = min_interval
        self._last_api_time = 0.0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "api_calls": self._api_calls,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
        }

    def _flush_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self.cache_path)
        self._pending_flush = 0

    def _pace(self) -> None:
        elapsed = time.time() - self._last_api_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _embed_one_api(self, formatted_text: str, max_retries: int = 10) -> list[float]:
        for attempt in range(max_retries):
            try:
                self._pace()
                self._last_api_time = time.time()
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=formatted_text,
                    config=types.EmbedContentConfig(output_dimensionality=self.dim),
                )
                self._api_calls += 1
                emb = result.embeddings[0].values
                return list(emb)
            except Exception as e:
                msg = str(e)
                lower = msg.lower()
                is_429 = "429" in msg or "resource_exhausted" in lower or "quota" in lower
                is_transient_server = any(s in lower for s in ("unavailable", "timeout", "503", "502", "500"))
                transient = is_429 or is_transient_server
                if attempt == max_retries - 1 or not transient:
                    try:
                        self._flush_cache()
                    finally:
                        raise
                if is_429:
                    delay = _parse_retry_delay(msg) or 60.0
                    delay = max(delay, 30.0) + random.uniform(1.0, 3.0)
                    sys.stderr.write(
                        f"\n[rate limit hit — sleeping {delay:.1f}s before retry {attempt + 2}/{max_retries}]\n"
                    )
                    sys.stderr.flush()
                    self._flush_cache()
                else:
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    sys.stderr.write(f"\n[transient error — retry in {delay:.1f}s]\n")
                    sys.stderr.flush()
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def embed_one(self, formatted_text: str) -> np.ndarray:
        key = _cache_key(self.model, self.dim, formatted_text)
        if key in self._cache:
            self._cache_hits += 1
            return np.asarray(self._cache[key], dtype=np.float32)
        values = self._embed_one_api(formatted_text)
        self._cache[key] = values
        self._pending_flush += 1
        if self._pending_flush >= 20:
            self._flush_cache()
        return np.asarray(values, dtype=np.float32)

    def embed_many(self, formatted_texts: list[str], progress_cb=None) -> np.ndarray:
        out = np.empty((len(formatted_texts), self.dim), dtype=np.float32)
        try:
            for i, t in enumerate(formatted_texts):
                out[i] = self.embed_one(t)
                if progress_cb is not None:
                    progress_cb(i + 1, len(formatted_texts))
        finally:
            self._flush_cache()
        return out

    def close(self) -> None:
        self._flush_cache()
