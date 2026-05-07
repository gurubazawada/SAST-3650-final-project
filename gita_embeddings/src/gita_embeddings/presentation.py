from __future__ import annotations

from pathlib import Path

from rich.console import Console

from . import analysis
from . import store as store_mod
from . import viz as viz_mod
from .axes import PRESET_AXES
from .embed import EmbeddingClient


DEMO_SEARCHES = [
    "the steady mind, undisturbed by desire",
    "renunciation of the fruits of action",
    "yoga is skill in action",
    "the cosmic form of God",
    "devotion with love to me",
    "the three gunas",
    "I am the beginning, the middle, and the end",
    "the imperishable Self cannot be slain",
    "when dharma declines I take birth",
    "surrender all dharmas to me",
]

DEMO_TERMS = [
    ("yoga", 4),
    ("dharma", 4),
    ("karma", 3),
    ("Self", 3),
    ("sacrifice", 3),
    ("desire", 3),
    ("meditation", 3),
]

DEMO_NEIGHBORS = [
    "BG.2.47",
    "BG.2.62",
    "BG.11.32",
    "BG.6.5",
    "BG.18.66",
]

CHAPTERS_OF_INTEREST = [2, 11, 12, 18]


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")[:40] or "query"


def _write_csv(df, path: Path, translator: str = "Eknath Easwaran") -> None:
    import datetime
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# translator: {translator}\n# generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        df.to_csv(f, index=False)


def generate_all(console: Console, demo_set: str = "full") -> None:
    sim_path = Path("data/embeddings_sim.parquet")
    ret_path = Path("data/embeddings_ret.parquet")
    if not sim_path.exists() or not ret_path.exists():
        console.print(f"[red]Missing embedding stores. Run `gita embed` first.[/red]")
        raise SystemExit(1)

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading stores[/bold]")
    df_sim, M_sim = store_mod.read_store(sim_path)
    df_ret, M_ret = store_mod.read_store(ret_path)
    client = EmbeddingClient()

    console.print("[bold]Computing UMAP (shared across panels)[/bold]")
    coords = viz_mod.compute_umap(M_sim)

    console.print("[bold]Core UMAP panels[/bold]")
    console.print(f"  {viz_mod.viz_umap_by_source(df_sim, coords)}")
    console.print(f"  {viz_mod.viz_umap_by_chapter(df_sim, coords)}")
    console.print(f"  {viz_mod.viz_umap_by_speaker(df_sim, coords)}")
    console.print(f"  {viz_mod.viz_narrative_trajectory(df_sim, coords)}")

    console.print("[bold]Structural heatmaps[/bold]")
    console.print(f"  {viz_mod.viz_chapter_centroid_heatmap(df_sim, M_sim)}")
    console.print(f"  {viz_mod.viz_speaker_centroid_heatmap(df_sim, M_sim)}")
    console.print(f"  {viz_mod.viz_similarity_matrix_sample(df_sim, M_sim)}")

    console.print("[bold]All-axis scores (embed pole phrases; cached after first run)[/bold]")
    axis_scores = viz_mod.compute_all_axis_scores(df_sim, M_sim, client)
    axis_scores.to_parquet(Path("data/axis_scores.parquet"))
    console.print(f"  data/axis_scores.parquet  (verses × {len(PRESET_AXES)} axes)")

    console.print("[bold]Per-axis visualizations[/bold]")
    for ax in PRESET_AXES.values():
        scores = axis_scores[ax.key].to_numpy()
        p1 = viz_mod.viz_umap_by_axis(df_sim, coords, scores, ax)
        p2 = viz_mod.viz_chapter_arc(df_sim, scores, ax)
        console.print(f"  [dim]{ax.key}[/dim] → {p1.name}, {p2.name}")

    if demo_set == "full":
        console.print("[bold]Per-verse arcs (key chapters)[/bold]")
        for ch in CHAPTERS_OF_INTEREST:
            for ax_key in ("action-renunciation", "fear-fearlessness", "ego-surrender", "sorrow-equanimity"):
                ax = PRESET_AXES[ax_key]
                scores = axis_scores[ax_key].to_numpy()
                p = viz_mod.viz_verse_arc(df_sim, scores, ax, ch)
                console.print(f"  [dim]ch{ch} {ax_key}[/dim] → {p.name}")

    console.print("[bold]Cross-axis comparison panels[/bold]")
    console.print(f"  {viz_mod.viz_speaker_voice_radar(df_sim, axis_scores)}")
    console.print(f"  {viz_mod.viz_axis_small_multiples(axis_scores)}")
    console.print(f"  {viz_mod.viz_axis_extremes_table(df_sim, axis_scores, n=5)}")

    console.print("[bold]Demo searches[/bold]")
    for q in DEMO_SEARCHES:
        res = analysis.semantic_search(q, df_ret, M_ret, client, k=10)
        csv_p = reports / f"search_{_slug(q)}.csv"
        _write_csv(res, csv_p)
        html_p = viz_mod.viz_query_results(q, res)
        console.print(f"  [dim]{q!r}[/dim] → {csv_p.name}, {html_p.name}")

    console.print("[bold]Polysemy / term-in-context[/bold]")
    for term, k in DEMO_TERMS:
        try:
            p = viz_mod.viz_polysemy(term, df_sim, M_sim, k=k)
            p_freq = viz_mod.viz_term_frequency_by_chapter(term, df_sim)
            hits, exemplars = analysis.term_in_context(term, df_sim, M_sim, n_clusters=k)
            _write_csv(hits, reports / f"term_{_slug(term)}_hits.csv")
            _write_csv(exemplars, reports / f"term_{_slug(term)}_exemplars.csv")
            console.print(f"  [dim]{term}[/dim] → {p.name}, {p_freq.name}, exemplars.csv")
        except ValueError as e:
            console.print(f"  [yellow]skip {term}: {e}[/yellow]")

    console.print("[bold]Seed-verse neighborhoods[/bold]")
    for vid in DEMO_NEIGHBORS:
        try:
            p = viz_mod.viz_neighbors_graph(vid, df_sim, M_sim, k=15)
            console.print(f"  [dim]{vid}[/dim] → {p.name}")
        except ValueError as e:
            console.print(f"  [yellow]skip {vid}: {e}[/yellow]")

    console.print("[bold]Group centroids[/bold]")
    for by in ("chapter", "speaker", "source"):
        sim_df, _ = analysis.centroids(df_sim, M_sim, group_col=by)
        p = reports / f"centroids_by_{by}.csv"
        sim_df.to_csv(p)
        console.print(f"  {p}")

    stats = client.stats
    console.print(
        f"\n[bold green]Done.[/bold green]  API calls: {stats['api_calls']}   "
        f"cache hits: {stats['cache_hits']}   cache size: {stats['cache_size']}"
    )
    client.close()
