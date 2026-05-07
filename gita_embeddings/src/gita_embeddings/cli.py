from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from . import corpus as corpus_mod
from . import ingest as ingest_mod


def _slug(s: str, maxlen: int = 40) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower())
    s = s.strip("_")
    return s[:maxlen] or "query"


def _timestamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_csv(df, path: Path, translator: str = "Eknath Easwaran") -> None:
    import datetime
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# translator: {translator}\n# generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        df.to_csv(f, index=False)


def _require_store(path: Path) -> None:
    if not path.exists():
        console.print(f"[red]Missing store:[/red] {path}")
        console.print("[yellow]Run `gita embed` first to generate the embedding stores.[/yellow]")
        raise typer.Exit(code=1)


def _print_result_table(df, title: str, cols: list[str], max_english: int = 120) -> None:
    tbl = Table(title=title, show_lines=False)
    for c in cols:
        tbl.add_column(c, overflow="fold" if c == "english" else None)
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row.get(c, "")
            if c == "english" and isinstance(val, str) and len(val) > max_english:
                val = val[: max_english - 1] + "…"
            if c == "score" and isinstance(val, (float, int)):
                val = f"{val:+.4f}"
            cells.append(str(val))
        tbl.add_row(*cells)
    console.print(tbl)


app = typer.Typer(help="Gītā embedding analysis toolkit.", no_args_is_help=True)
console = Console()


@app.callback()
def _root() -> None:
    """Gītā embedding analysis toolkit."""
    load_dotenv()


@app.command("ingest")
def cmd_ingest(
    epub_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the ePub file."),
    out: Path = typer.Option(Path("data/corpus.json"), "--out", "-o", help="Where to write corpus JSON."),
    toc: Path = typer.Option(Path("debug/toc.txt"), "--toc", help="Where to write a TOC dump for inspection."),
    parser: str = typer.Option("easwaran", "--parser", help="Parser to use (registered in ingest.PARSERS)."),
) -> None:
    """Ingest an ePub into a verse corpus JSON file."""
    console.print(f"[bold]Ingesting[/bold] {epub_path}")

    console.print(f"  writing TOC dump → {toc}")
    ingest_mod.dump_toc(epub_path, toc)

    verses = ingest_mod.ingest_epub(epub_path, parser=parser)
    corpus_mod.dump_corpus(verses, out)

    per_chapter = Counter(v.chapter for v in verses)
    per_speaker = Counter(v.speaker or "(none)" for v in verses)

    tbl = Table(title=f"Ingested {len(verses)} verses → {out}")
    tbl.add_column("Chapter", justify="right")
    tbl.add_column("Verses", justify="right")
    for ch in sorted(per_chapter):
        tbl.add_row(str(ch), str(per_chapter[ch]))
    console.print(tbl)

    stbl = Table(title="Verses by speaker")
    stbl.add_column("Speaker")
    stbl.add_column("Verses", justify="right")
    for sp, n in per_speaker.most_common():
        stbl.add_row(sp, str(n))
    console.print(stbl)

    console.print(f"[green]✓ Wrote[/green] {out}  ({len(verses)} verses)")
    console.print(f"[green]✓ Wrote[/green] {toc}")
    console.print("\n[bold yellow]Review corpus.json before running `gita embed`.[/bold yellow]")


@app.command("embed")
def cmd_embed(
    corpus_path: Path = typer.Option(Path("data/corpus.json"), "--corpus", help="Corpus JSON input."),
    sim_out: Path = typer.Option(Path("data/embeddings_sim.parquet"), "--sim-out", help="Symmetric (sentence-similarity) store."),
    ret_out: Path = typer.Option(Path("data/embeddings_ret.parquet"), "--ret-out", help="Asymmetric retrieval-document store."),
    cache_path: Path = typer.Option(Path("data/.embedding_cache.json"), "--cache", help="On-disk embedding cache."),
) -> None:
    """Embed every verse with Gemini (two stores: similarity + retrieval).

    Re-runs are free: the cache keys on (model|dim|formatted_text), so the same text
    embedded with different task prefixes cache independently.
    """
    from .embed import EmbeddingClient, fmt_similarity, fmt_doc

    verses = corpus_mod.load_corpus(corpus_path)
    console.print(f"[bold]Loaded[/bold] {len(verses)} verses from {corpus_path}")

    client = EmbeddingClient(cache_path=cache_path)
    console.print(f"  model=[cyan]{client.model}[/cyan]  dim=[cyan]{client.dim}[/cyan]  cache_size=[cyan]{client.stats['cache_size']}[/cyan]")

    sim_texts = [fmt_similarity(v.english) for v in verses]
    ret_texts = [fmt_doc(v.verse_id, v.english) for v in verses]

    def _embed_with_progress(label: str, texts: list[str]):
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]{label}[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            t = progress.add_task(label, total=len(texts))

            def cb(done: int, total: int) -> None:
                progress.update(t, completed=done)

            return client.embed_many(texts, progress_cb=cb)

    sim_emb = _embed_with_progress("similarity", sim_texts)
    ret_emb = _embed_with_progress("retrieval ", ret_texts)

    from . import store as store_mod

    store_mod.write_store(sim_out, verses, sim_emb)
    console.print(f"[green]✓ Wrote[/green] {sim_out}  shape=({sim_emb.shape[0]}, {sim_emb.shape[1]})")
    store_mod.write_store(ret_out, verses, ret_emb)
    console.print(f"[green]✓ Wrote[/green] {ret_out}  shape=({ret_emb.shape[0]}, {ret_emb.shape[1]})")

    stats = client.stats
    console.print(
        f"\n[bold]API calls:[/bold] {stats['api_calls']}   "
        f"[bold]Cache hits:[/bold] {stats['cache_hits']}   "
        f"[bold]Cache size:[/bold] {stats['cache_size']}"
    )
    client.close()


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Natural-language query."),
    k: int = typer.Option(10, "--k", help="Number of hits to return."),
    ret_path: Path = typer.Option(Path("data/embeddings_ret.parquet"), "--ret"),
) -> None:
    """Semantic search over verses (uses the retrieval store)."""
    from . import analysis, store as store_mod
    from .embed import EmbeddingClient

    _require_store(ret_path)
    df_ret, M_ret = store_mod.read_store(ret_path)
    client = EmbeddingClient()
    results = analysis.semantic_search(query, df_ret, M_ret, client, k=k)

    _print_result_table(
        results,
        title=f'search: "{query}"  (top {k})',
        cols=["score", "verse_id", "speaker", "english"],
    )
    out = Path(f"reports/search_{_slug(query)}_{_timestamp()}.csv")
    _write_csv(results, out)
    console.print(f"[green]✓[/green] {out}")
    client.close()


@app.command("term")
def cmd_term(
    term: str = typer.Argument(..., help="Term to find (substring, case-insensitive)."),
    clusters: int = typer.Option(4, "--clusters", help="Number of KMeans clusters."),
    sim_path: Path = typer.Option(Path("data/embeddings_sim.parquet"), "--sim"),
) -> None:
    """Term-in-context: find matching verses and cluster their embeddings."""
    from . import analysis, store as store_mod

    _require_store(sim_path)
    df_sim, M_sim = store_mod.read_store(sim_path)
    hits, exemplars = analysis.term_in_context(term, df_sim, M_sim, n_clusters=clusters)

    if len(hits) == 0:
        console.print(f"[yellow]No verses contain '{term}'.[/yellow]")
        return

    console.print(f"[bold]{len(hits)}[/bold] verses contain '[cyan]{term}[/cyan]', partitioned into {len(exemplars)} clusters.\n")
    _print_result_table(
        exemplars,
        title=f'exemplars for "{term}"',
        cols=["cluster", "cluster_size", "verse_id", "speaker", "english"],
        max_english=180,
    )

    hits_out = Path(f"reports/term_{_slug(term)}_hits_{_timestamp()}.csv")
    exem_out = Path(f"reports/term_{_slug(term)}_exemplars_{_timestamp()}.csv")
    _write_csv(hits, hits_out)
    _write_csv(exemplars, exem_out)
    console.print(f"[green]✓[/green] {hits_out}")
    console.print(f"[green]✓[/green] {exem_out}")


@app.command("axis")
def cmd_axis(
    pole_a: str = typer.Argument(None, help="Pole A phrase (or omit when using --preset)."),
    pole_b: str = typer.Argument(None, help="Pole B phrase (or omit when using --preset)."),
    preset: str = typer.Option(None, "--preset", help="Preset axis key. See `gita axes list`."),
    top: int = typer.Option(5, "--top", help="How many verses to show at each pole."),
    sim_path: Path = typer.Option(Path("data/embeddings_sim.parquet"), "--sim"),
) -> None:
    """Project every verse onto a concept axis."""
    from . import analysis, store as store_mod
    from .axes import PRESET_AXES
    from .embed import EmbeddingClient

    if preset:
        if preset not in PRESET_AXES:
            console.print(f"[red]Unknown preset: {preset}[/red]")
            raise typer.Exit(code=1)
        ax = PRESET_AXES[preset]
        pole_a, pole_b = ax.pole_a, ax.pole_b
        label_a, label_b = ax.label_a, ax.label_b
        axis_name = preset
    else:
        if not pole_a or not pole_b:
            console.print("[red]Must supply both pole_a and pole_b, or use --preset.[/red]")
            raise typer.Exit(code=1)
        label_a, label_b = pole_a[:30], pole_b[:30]
        axis_name = _slug(f"{label_a}_vs_{label_b}")

    _require_store(sim_path)
    df_sim, M_sim = store_mod.read_store(sim_path)
    client = EmbeddingClient()
    ranked, _axis = analysis.concept_axis(pole_a, pole_b, df_sim, M_sim, client)

    console.print(f'[bold]axis:[/bold] {label_a}  ←→  {label_b}')
    _print_result_table(ranked.head(top), title=f'most "{label_a}"', cols=["score", "verse_id", "speaker", "english"])
    _print_result_table(ranked.tail(top).iloc[::-1], title=f'most "{label_b}"', cols=["score", "verse_id", "speaker", "english"])

    out = Path(f"reports/axis_{axis_name}_{_timestamp()}.csv")
    _write_csv(ranked, out)
    console.print(f"[green]✓[/green] {out}")
    client.close()


@app.command("axes")
def cmd_axes(
    action: str = typer.Argument("list", help="'list' prints preset axes."),
) -> None:
    """List preset axes."""
    from .axes import PRESET_AXES

    if action != "list":
        console.print("[red]Unknown action.[/red]")
        raise typer.Exit(code=1)

    tbl = Table(title=f"Preset axes ({len(PRESET_AXES)})")
    tbl.add_column("Key")
    tbl.add_column("Pole A")
    tbl.add_column("Pole B")
    tbl.add_column("Theme", overflow="fold")
    for a in PRESET_AXES.values():
        tbl.add_row(a.key, a.label_a, a.label_b, a.description)
    console.print(tbl)


@app.command("centroids")
def cmd_centroids(
    by: str = typer.Option("source", "--by", help="Group column: source | chapter | speaker."),
    sim_path: Path = typer.Option(Path("data/embeddings_sim.parquet"), "--sim"),
) -> None:
    """Group-centroid cosine similarity matrix."""
    from . import analysis, store as store_mod

    _require_store(sim_path)
    df_sim, M_sim = store_mod.read_store(sim_path)
    sim_df, _C = analysis.centroids(df_sim, M_sim, group_col=by)

    tbl = Table(title=f"centroid cosine similarity (by {by})")
    tbl.add_column("")
    for c in sim_df.columns:
        tbl.add_column(str(c))
    for idx, row in sim_df.iterrows():
        tbl.add_row(str(idx), *[f"{v:+.3f}" for v in row])
    console.print(tbl)

    out = Path(f"reports/centroids_by_{by}_{_timestamp()}.csv")
    sim_df.to_csv(out)
    console.print(f"[green]✓[/green] {out}")


@app.command("neighbors")
def cmd_neighbors(
    verse_id: str = typer.Argument(..., help='e.g. "BG.2.47"'),
    k: int = typer.Option(15, "--k"),
    sim_path: Path = typer.Option(Path("data/embeddings_sim.parquet"), "--sim"),
) -> None:
    """Nearest neighbors to a seed verse."""
    from . import analysis, store as store_mod

    _require_store(sim_path)
    df_sim, M_sim = store_mod.read_store(sim_path)
    out = analysis.neighbors(verse_id, df_sim, M_sim, k=k)
    _print_result_table(
        out,
        title=f"nearest {k} to {verse_id}",
        cols=["score", "verse_id", "speaker", "english"],
    )
    csv = Path(f"reports/neighbors_{_slug(verse_id)}_{_timestamp()}.csv")
    _write_csv(out, csv)
    console.print(f"[green]✓[/green] {csv}")


viz_app = typer.Typer(help="Visualization subcommands.", no_args_is_help=True)
app.add_typer(viz_app, name="viz")


def _load_both_stores():
    from . import store as store_mod

    sim_path = Path("data/embeddings_sim.parquet")
    _require_store(sim_path)
    df_sim, M_sim = store_mod.read_store(sim_path)
    return df_sim, M_sim


@viz_app.command("umap")
def cmd_viz_umap(
    by: str = typer.Option("source", "--by", help="source | chapter | speaker | axis"),
    axis_key: str = typer.Option(None, "--axis", help="axis key (required when --by axis)"),
) -> None:
    """UMAP scatter plot colored by the chosen attribute."""
    from . import viz as viz_mod

    df, M = _load_both_stores()
    coords = viz_mod.compute_umap(M)

    if by == "source":
        p = viz_mod.viz_umap_by_source(df, coords)
    elif by == "chapter":
        p = viz_mod.viz_umap_by_chapter(df, coords)
    elif by == "speaker":
        p = viz_mod.viz_umap_by_speaker(df, coords)
    elif by == "axis":
        from .axes import PRESET_AXES
        from .embed import EmbeddingClient

        if not axis_key or axis_key not in PRESET_AXES:
            console.print("[red]--axis <key> required (see `gita axes list`).[/red]")
            raise typer.Exit(code=1)
        ax = PRESET_AXES[axis_key]
        client = EmbeddingClient()
        scores = viz_mod.compute_axis_scores(df, M, ax, client)
        p = viz_mod.viz_umap_by_axis(df, coords, scores, ax)
        client.close()
    else:
        console.print(f"[red]unknown --by {by}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] {p}")


@viz_app.command("heatmap")
def cmd_viz_heatmap(
    what: str = typer.Option("chapter", "--what", help="chapter | speaker | similarity-sample"),
) -> None:
    """Heatmap of centroid similarities or a random verse sample."""
    from . import viz as viz_mod

    df, M = _load_both_stores()
    if what == "chapter":
        p = viz_mod.viz_chapter_centroid_heatmap(df, M)
    elif what == "speaker":
        p = viz_mod.viz_speaker_centroid_heatmap(df, M)
    elif what == "similarity-sample":
        p = viz_mod.viz_similarity_matrix_sample(df, M)
    else:
        console.print(f"[red]unknown --what {what}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]✓[/green] {p}")


@viz_app.command("arc")
def cmd_viz_arc(
    axis_key: str = typer.Option(..., "--axis"),
    chapter: int = typer.Option(None, "--chapter", help="If set, per-verse arc within that chapter."),
) -> None:
    """Axis projection along chapters (or verses within one chapter)."""
    from . import viz as viz_mod
    from .axes import PRESET_AXES
    from .embed import EmbeddingClient

    if axis_key not in PRESET_AXES:
        console.print(f"[red]unknown axis {axis_key}[/red]")
        raise typer.Exit(code=1)
    ax = PRESET_AXES[axis_key]
    df, M = _load_both_stores()
    client = EmbeddingClient()
    scores = viz_mod.compute_axis_scores(df, M, ax, client)
    if chapter is not None:
        p = viz_mod.viz_verse_arc(df, scores, ax, chapter)
    else:
        p = viz_mod.viz_chapter_arc(df, scores, ax)
    console.print(f"[green]✓[/green] {p}")
    client.close()


@viz_app.command("radar")
def cmd_viz_radar() -> None:
    """Speaker voice radar across all preset axes."""
    from . import viz as viz_mod
    from .embed import EmbeddingClient

    df, M = _load_both_stores()
    client = EmbeddingClient()
    axis_scores = viz_mod.compute_all_axis_scores(df, M, client)
    p = viz_mod.viz_speaker_voice_radar(df, axis_scores)
    console.print(f"[green]✓[/green] {p}")
    client.close()


@viz_app.command("small-multiples")
def cmd_viz_small_multiples() -> None:
    """Histogram grid of axis projection distributions."""
    from . import viz as viz_mod
    from .embed import EmbeddingClient

    df, M = _load_both_stores()
    client = EmbeddingClient()
    axis_scores = viz_mod.compute_all_axis_scores(df, M, client)
    p = viz_mod.viz_axis_small_multiples(axis_scores)
    console.print(f"[green]✓[/green] {p}")
    client.close()


@viz_app.command("extremes")
def cmd_viz_extremes(n: int = typer.Option(5, "--n")) -> None:
    """HTML table of top/bottom N verses for every axis."""
    from . import viz as viz_mod
    from .embed import EmbeddingClient

    df, M = _load_both_stores()
    client = EmbeddingClient()
    axis_scores = viz_mod.compute_all_axis_scores(df, M, client)
    p = viz_mod.viz_axis_extremes_table(df, axis_scores, n=n)
    console.print(f"[green]✓[/green] {p}")
    client.close()


@viz_app.command("polysemy")
def cmd_viz_polysemy(term: str = typer.Argument(...), k: int = typer.Option(4, "--clusters")) -> None:
    """UMAP of just-the-matching verses for a term, colored by KMeans cluster."""
    from . import viz as viz_mod

    df, M = _load_both_stores()
    p = viz_mod.viz_polysemy(term, df, M, k=k)
    console.print(f"[green]✓[/green] {p}")


@viz_app.command("freq")
def cmd_viz_freq(term: str = typer.Argument(...)) -> None:
    """Per-chapter bar chart of a term's occurrences."""
    from . import viz as viz_mod

    df, _M = _load_both_stores()
    p = viz_mod.viz_term_frequency_by_chapter(term, df)
    console.print(f"[green]✓[/green] {p}")


@viz_app.command("neighbors")
def cmd_viz_neighbors(verse_id: str = typer.Argument(...), k: int = typer.Option(15, "--k")) -> None:
    """Force-directed graph of the k nearest neighbors to a seed verse."""
    from . import viz as viz_mod

    df, M = _load_both_stores()
    p = viz_mod.viz_neighbors_graph(verse_id, df, M, k=k)
    console.print(f"[green]✓[/green] {p}")


@viz_app.command("trajectory")
def cmd_viz_trajectory() -> None:
    """Reading-order trajectory of all verses through UMAP space."""
    from . import viz as viz_mod

    df, M = _load_both_stores()
    coords = viz_mod.compute_umap(M)
    p = viz_mod.viz_narrative_trajectory(df, coords)
    console.print(f"[green]✓[/green] {p}")


@app.command("demo")
def cmd_demo(
    demo_set: str = typer.Option("full", "--set", help="full | minimal"),
) -> None:
    """Generate the full set of presentation artifacts under reports/."""
    from . import presentation

    presentation.generate_all(console, demo_set=demo_set)


if __name__ == "__main__":
    app()
