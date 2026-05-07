# gita-embeddings

Embedding-based exploratory analysis of the Bhagavad Gītā (Eknath Easwaran edition) via Gemini Embedding 2.

## Quickstart

```bash
# 1. Install
python3 -m venv .venv
.venv/bin/pip install -e .

# 2. Set your API key (either method works)
#    (a) .env file — project auto-loads this
echo 'GEMINI_API_KEY=your-key-here' > .env
#    (b) shell export
export GEMINI_API_KEY="your-key-here"

# 3. Ingest the ePub → data/corpus.json (one-time, offline)
.venv/bin/gita ingest path/to/your.epub

# 4. Embed every verse (two stores: similarity + retrieval; ~1400 API calls, cached)
.venv/bin/gita embed

# 5. Generate the full presentation set under reports/
.venv/bin/gita demo --set full
```

## CLI surface

```
gita ingest <epub>          Parse ePub → corpus.json
gita embed                  Embed all verses (similarity + retrieval stores)
gita search "<query>"       Semantic search → CSV
gita term <word>            Polysemy clustering of matching verses
gita axis --preset <key>    Project verses onto a named concept axis
gita axes list              Show all 17 preset axes
gita centroids --by <col>   Centroid cosine matrix (source | chapter | speaker)
gita neighbors <verse_id>   Nearest-neighbor verses
gita demo --set full        One-shot: generate the whole reports/ dir
gita viz <subcommand>       Individual visualizations (umap / heatmap / arc /
                            radar / small-multiples / extremes / polysemy /
                            freq / neighbors / trajectory)
```

## What's in reports/ after `gita demo --set full`

Roughly:
- 3 base UMAP scatters (by source, chapter, speaker)
- 1 narrative-trajectory view (reading order through semantic space)
- 2 structural heatmaps (chapter 18×18, speaker 3×3)
- 1 random-sample similarity heatmap
- 17 axis-colored UMAP views (one per preset)
- 17 chapter-arc line plots (one per preset)
- 16 per-verse arc plots (4 chapters × 4 key axes)
- 1 speaker voice radar (all 17 axes)
- 1 axis small-multiples histogram grid
- 1 axis-extremes HTML table (top/bottom 5 per axis)
- 10 famous-passage search result plots + CSVs
- 7 polysemy UMAPs + term-frequency bars (yoga, dharma, karma, Self, sacrifice, desire, meditation)
- 5 nearest-neighbor graphs (seed verses BG 2.47, 2.62, 11.32, 6.5, 18.66)
- 3 centroid CSVs (by chapter, speaker, source)

That's ~90 artifacts. Use it as the raw material for a slide deck.

## Preset axes (17)

| Key | Poles | Theme |
|---|---|---|
| `action-renunciation` | action ↔ renunciation | karma-yoga vs. sannyāsa |
| `attachment-detachment` | attachment ↔ detachment | rāga vs. vairāgya |
| `knowledge-devotion` | knowledge ↔ devotion | jñāna vs. bhakti |
| `self-world` | Self ↔ world | Ātman vs. prakṛti |
| `pleasure-pain` | pleasure ↔ pain | dualities |
| `fear-fearlessness` | fear ↔ fearlessness | Arjuna's transformation |
| `war-peace` | war ↔ peace | narrative frame |
| `desire-contentment` | desire ↔ contentment | kāma vs. santoṣa |
| `ignorance-wisdom` | ignorance ↔ wisdom | avidyā vs. vidyā |
| `body-spirit` | body ↔ spirit | dehin vs. deha |
| `ego-surrender` | ego ↔ surrender | ahaṃkāra vs. śaraṇāgati |
| `doubt-faith` | doubt ↔ faith | saṃśaya vs. śraddhā |
| `sorrow-equanimity` | sorrow ↔ equanimity | śoka vs. samatva |
| `multiplicity-unity` | multiplicity ↔ unity | dvaita vs. advaita |
| `senses-selfcontrol` | senses ↔ self-control | indriya-nigraha |
| `activity-stillness` | activity ↔ stillness | the guṇas |
| `worldly-divine` | worldly ↔ divine | pravṛtti vs. nivṛtti |

Add your own via `gita axis "<pole_a>" "<pole_b>"`.

## Architecture notes

- **Model:** `gemini-embedding-2-preview`, 3072-dim (self-normalized).
- **Task prefixes:** Gemini 2 uses in-text prefixes instead of a `task_type` parameter. We embed each verse twice:
  - `task: sentence similarity | query: <verse>` — drives axes, clusters, UMAP, heatmaps, arcs.
  - `title: <verse_id> | text: <verse>` — drives search (queries use `task: search result | query: <q>`).
- **Cache:** `data/.embedding_cache.json` keyed by `sha256(model|dim|formatted_text)`. Includes the prefix, so the same text under two task prefixes caches independently. Re-runs are free.
- **No tests:** manually verified via `gita demo` spot-checks.

## Demo recipes

```bash
gita search "renunciation of the fruits of action"      # expect BG 2.47 in top 5
gita search "I am time, destroyer of worlds"            # expect BG 11.32
gita search "the cosmic form of God"                    # expect chapter 11
gita term yoga --clusters 4                             # karma / jñāna / bhakti / dhyāna senses
gita term dharma --clusters 4                           # duty / cosmic law / righteousness
gita axis --preset action-renunciation
gita axis --preset fear-fearlessness
gita axis --preset ego-surrender
gita centroids --by chapter
gita neighbors BG.2.47 --k 15
gita viz arc --axis sorrow-equanimity
gita viz arc --axis action-renunciation --chapter 2
gita viz radar
gita viz polysemy yoga --clusters 4
gita viz extremes
```
