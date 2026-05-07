# Gītā Embedding Analysis — Project Spec

## 1. Goal

Build a small Python project that takes an ePub of the Bhagavad Gītā (and related texts, if the ePub contains them) and produces a queryable embedding index of its verses, using Google's Gemini embedding API. The index supports four analyses: semantic search, term-in-context retrieval, concept-axis projection, and cross-text centroid comparison. A single CLI drives ingestion, embedding, and analysis. Output is a set of reproducible artifacts (JSON catalog, parquet embedding store, HTML visualization), not a web app.

The audience is one user doing exploratory philological/philosophical analysis. Prioritize clarity, reproducibility, and the ability to re-run a single stage without redoing the others. Do not over-engineer.

## 2. Inputs

- One ePub file at a path supplied by the user. Structure is publisher-dependent and will need inspection; do **not** assume a specific TOC layout or chapter naming scheme.
- `GEMINI_API_KEY` in the environment.
- (Optional, later) additional ePub or plaintext files for related texts (Yoga Sūtras, Upaniṣads, Mahābhārata excerpts).

## 3. Outputs

- `data/corpus.json` — parsed, normalized verse catalog (see schema §6).
- `data/embeddings.parquet` — embedding store keyed by `verse_id`.
- `data/.embedding_cache.json` — raw API-response cache, to avoid re-billing on re-runs.
- `reports/projection.html` — interactive UMAP scatter of verse embeddings.
- `reports/<analysis>.csv` — per-analysis result dumps for the user to inspect.

## 4. Tech stack

- Python 3.11+
- `google-genai` (Gemini embeddings — model `gemini-embedding-001`, task `SEMANTIC_SIMILARITY`, output dim 1536 via Matryoshka truncation)
- `ebooklib` + `beautifulsoup4` + `lxml` for ePub parsing
- `numpy`, `pandas`, `pyarrow` for data handling
- `scikit-learn` for clustering
- `umap-learn` + `plotly` for visualization
- `typer` for the CLI
- `pytest` for tests

No database. No web framework. No Docker.

## 5. Phased implementation

Implement in this order. Each phase has a human-in-the-loop checkpoint before moving on.

### Phase 1 — Ingestion (**this is where the project lives or dies**)

ePub layouts vary wildly between publishers. Do not silently pattern-match — write an ingestion pipeline that is explicit, inspectable, and easy to iterate on.

1. Open the ePub, enumerate its `ITEM_DOCUMENT` items, and dump a `debug/toc.txt` listing each internal document with its first ~200 chars of plain text. The user will use this to understand their file.
2. Write a verse-extraction function that takes an HTML document and returns `list[RawVerse]` (chapter, verse_num, sanskrit_if_present, english, speaker_if_inferable). Start with a conservative heuristic (look for common verse-number patterns like `2.47`, `BG 2.47`, `॥ ४७ ॥`); make it trivial to replace with a publisher-specific parser.
3. After extraction, write `data/corpus.json` and **stop for user review**. Do not proceed to embedding until the user confirms verse counts and spot-checks a handful of verses.

Deliverable: `gita ingest <path-to-epub>` produces `data/corpus.json` and `debug/toc.txt`.

### Phase 2 — Embedding

1. Thin wrapper over `google-genai`: batch size 50, exponential backoff, on-disk JSON cache keyed by `sha256(model|task|dim|text)`. The cache is essential — a re-run must be free.
2. Embed the English field of each verse. (Sanskrit as a secondary field is stretch; pretrained models are much stronger on English and the corpus is well-represented in English on the open web.)
3. Store embeddings as a parquet file with columns `[verse_id, source, chapter, verse_num, speaker, english, embedding]` where `embedding` is a fixed-length `float32` list.

Deliverable: `gita embed` turns `corpus.json` into `embeddings.parquet`.

### Phase 3 — Analyses

Implement all four as pure functions over `(verses, embeddings, client)`, and expose each via the CLI.

- **Semantic search.** Embed a query string; return top-k verses by cosine similarity.
  - `gita search "the steady mind, undisturbed by desire" --k 10`
- **Term-in-context.** Given a term (case-insensitive substring on `english`), return the matching verses and their embeddings; additionally run KMeans (`k=3..6`, user-specifiable) and print per-cluster exemplars. This is the tool for polysemy studies (*yoga*, *dharma*, *karma*).
  - `gita term yoga --clusters 4`
- **Concept axis.** Embed two pole phrases, compute `axis = normalize(embed(pole_b) - embed(pole_a))`, project every verse onto it, return verses sorted by signed projection. This is how to handle "opposition" — raw cosine does not.
  - `gita axis "attachment and desire" "detachment and equanimity"`
- **Centroids.** Group by `source`, compute mean embedding per source, print pairwise cosine similarity matrix.
  - `gita centroids`

Each analysis writes its result to `reports/<name>_<timestamp>.csv` in addition to stdout.

### Phase 4 — Visualization

- `gita viz` runs UMAP (`n_neighbors=15`, `min_dist=0.1`) on all verse embeddings and writes `reports/projection.html` — a Plotly scatter colored by `source` with `speaker`, `verse_id`, and `english` on hover.
- Stretch: a second plot colored by the user's most recent concept-axis projection.

## 6. Schemas

```python
@dataclass(frozen=True)
class Verse:
    verse_id: str      # "BG.2.47", "YS.1.2", etc. — stable, human-readable
    source: str        # "Bhagavad Gita", "Yoga Sutras", ...
    chapter: int
    verse_num: int
    speaker: str | None
    sanskrit: str | None
    english: str
    translator: str    # track this — it affects every downstream result
```

Parquet store columns: `verse_id (str), source (str), chapter (int), verse_num (int), speaker (str|null), english (str), embedding (list[float32])`. One row per verse.

## 7. Repository layout

```
gita_embeddings/
├── pyproject.toml
├── README.md
├── src/gita_embeddings/
│   ├── __init__.py
│   ├── cli.py              # typer entrypoint: ingest, embed, search, term, axis, centroids, viz
│   ├── corpus.py           # Verse dataclass, JSON (de)serialization
│   ├── ingest.py           # ePub → list[Verse]
│   ├── embed.py            # EmbeddingClient (batch + cache + retry)
│   ├── store.py            # parquet read/write of the embedding index
│   ├── analysis.py         # semantic_search, term_in_context, concept_axis, centroids
│   └── viz.py              # UMAP + plotly
├── tests/
│   ├── test_ingest.py      # fixture: a tiny synthetic ePub
│   ├── test_embed.py       # mock the Gemini client; verify caching + batching
│   ├── test_analysis.py    # correctness of cosine, axis projection, centroid math
│   └── fixtures/
└── data/                   # gitignored
```

## 8. Acceptance criteria

The project is "done" when all of the following are true on a clean clone with `GEMINI_API_KEY` set and an ePub provided:

1. `gita ingest book.epub` produces `data/corpus.json` and `debug/toc.txt`. Corpus has ≥90% of expected verses for the supplied text (user validates).
2. `gita embed` populates `data/embeddings.parquet` with one row per verse, embedding dim = 1536.
3. Re-running `gita embed` after the cache is populated makes zero API calls (verified by mocking `genai.Client` in tests).
4. `gita search "renunciation of the fruits of action"` returns BG 2.47 in the top 5 on a standard Gītā ePub.
5. `gita axis "action" "renunciation"` produces a signed ranking; the top and bottom 3 verses are visibly consistent with the axis.
6. `gita viz` produces a viewable HTML file where distinct sources (if multiple are present) separate into visually identifiable regions.
7. `pytest` passes. Analyses have unit tests with synthetic inputs — cosine similarity is computed correctly, axis projection signs are correct, centroid math matches a hand-computed case.

## 9. Non-goals

- Sanskrit morphological analysis, sandhi-splitting, or compound-splitting. (If the ePub has a Sanskrit field, pass it through; do not process it.)
- Training custom embeddings.
- A web UI, a notebook interface, or a chat interface.
- Supporting multiple embedding providers behind an abstraction. Code against Gemini directly; refactor only if a second provider is actually needed.
- Multi-user concerns (auth, persistence, concurrency).

## 10. Open questions for the user

Claude Code should ask the user these at the start of planning, not guess:

1. Which translator / edition is in the ePub? (Stored in every `Verse.translator` field; affects interpretation of results.)
2. Does the ePub contain only the Gītā, or also commentary / introductions / other texts? If mixed, how should non-verse content be excluded?
3. Initial concept axes of interest? (Examples: *action ↔ renunciation*, *attachment ↔ detachment*, *knowledge ↔ devotion*, *Self ↔ world*.)
4. Preferred CLI name? (Default: `gita`.)

## 11. Notes on quality bars specific to this project

- **Embeddings are not philosophy.** Every analysis CSV should include the raw `english` text in a column so the user can validate results against the actual verses. Do not present projections or cluster assignments without the underlying text next to them.
- **Isolated-word embeddings are misleading.** When embedding a term for `term_in_context` clustering, embed the full containing verse, not the term on its own. "Yoga" embedded alone is dominated by modern-wellness distributional signal.
- **Translator choice leaks into every result.** Log the translator in every output CSV header.
