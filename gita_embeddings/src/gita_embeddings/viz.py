from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from umap import UMAP

from . import analysis
from .axes import Axis, PRESET_AXES
from .embed import EmbeddingClient, fmt_similarity


REPORTS = Path("reports")


def _ensure_reports() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)


def _hover_text(df: pd.DataFrame, max_en: int = 180) -> list[str]:
    out = []
    for _, r in df.iterrows():
        eng = str(r.get("english", ""))
        if len(eng) > max_en:
            eng = eng[: max_en - 1] + "…"
        out.append(f"<b>{r['verse_id']}</b> ({r.get('speaker') or '-'})<br>{eng}")
    return out


def compute_umap(M: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42) -> np.ndarray:
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(M).astype(np.float32)


# ---------------------------------------------------------------------------
# Core UMAP panels
# ---------------------------------------------------------------------------


def viz_umap_colored(
    df: pd.DataFrame,
    coords: np.ndarray,
    color_col: str,
    title: str,
    out: Path,
    continuous: bool = False,
    colorscale: str | None = None,
) -> Path:
    hovers = _hover_text(df)
    data = df.copy()
    data["x"] = coords[:, 0]
    data["y"] = coords[:, 1]

    if continuous:
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color=color_col,
            color_continuous_scale=colorscale or "RdBu",
            hover_name="verse_id",
            title=title,
            height=700,
        )
    else:
        fig = px.scatter(
            data,
            x="x",
            y="y",
            color=data[color_col].astype(str),
            hover_name="verse_id",
            title=title,
            height=700,
        )

    fig.update_traces(hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>", marker=dict(size=7, opacity=0.8))
    fig.update_layout(xaxis_title="UMAP-1", yaxis_title="UMAP-2", legend_title=color_col)
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_umap_by_source(df: pd.DataFrame, coords: np.ndarray) -> Path:
    return viz_umap_colored(df, coords, "source", "UMAP projection — colored by source", REPORTS / "projection_by_source.html")


def viz_umap_by_chapter(df: pd.DataFrame, coords: np.ndarray) -> Path:
    return viz_umap_colored(df, coords, "chapter", "UMAP projection — colored by chapter", REPORTS / "projection_by_chapter.html", continuous=True, colorscale="Turbo")


def viz_umap_by_speaker(df: pd.DataFrame, coords: np.ndarray) -> Path:
    return viz_umap_colored(df, coords, "speaker", "UMAP projection — colored by speaker", REPORTS / "projection_by_speaker.html")


def viz_umap_by_axis(df: pd.DataFrame, coords: np.ndarray, scores: np.ndarray, axis: Axis) -> Path:
    data = df.copy()
    data["axis_score"] = scores
    return viz_umap_colored(
        data,
        coords,
        "axis_score",
        f"UMAP — colored by axis score ({axis.label_a} ←→ {axis.label_b})",
        REPORTS / f"projection_by_axis_{axis.key}.html",
        continuous=True,
        colorscale="RdBu",
    )


# ---------------------------------------------------------------------------
# Structural panels
# ---------------------------------------------------------------------------


def viz_chapter_centroid_heatmap(df: pd.DataFrame, M: np.ndarray) -> Path:
    sim, _C = analysis.centroids(df, M, group_col="chapter")
    fig = px.imshow(
        sim.values,
        x=[str(c) for c in sim.columns],
        y=[str(c) for c in sim.index],
        color_continuous_scale="Viridis",
        aspect="equal",
        title="Chapter centroid cosine similarity",
    )
    fig.update_layout(xaxis_title="chapter", yaxis_title="chapter", height=700)
    out = REPORTS / "chapter_centroid_heatmap.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_speaker_centroid_heatmap(df: pd.DataFrame, M: np.ndarray) -> Path:
    sim, _C = analysis.centroids(df, M, group_col="speaker")
    fig = px.imshow(
        sim.values,
        x=[str(c) for c in sim.columns],
        y=[str(c) for c in sim.index],
        color_continuous_scale="Viridis",
        aspect="equal",
        title="Speaker centroid cosine similarity",
    )
    fig.update_layout(height=550)
    out = REPORTS / "speaker_centroid_heatmap.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_similarity_matrix_sample(df: pd.DataFrame, M: np.ndarray, n: int = 40, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    idx.sort()
    sub = df.iloc[idx]
    Msub = M[idx]
    Mn = Msub / np.linalg.norm(Msub, axis=1, keepdims=True).clip(min=1e-9)
    sim = Mn @ Mn.T
    labels = sub["verse_id"].tolist()
    fig = px.imshow(sim, x=labels, y=labels, color_continuous_scale="Viridis", aspect="equal", title=f"Pairwise cosine similarity (random sample of {len(idx)})")
    fig.update_layout(height=800)
    out = REPORTS / "similarity_sample.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ---------------------------------------------------------------------------
# Axis score helpers
# ---------------------------------------------------------------------------


def compute_axis_scores(df: pd.DataFrame, M: np.ndarray, axis: Axis, client: EmbeddingClient) -> np.ndarray:
    va = client.embed_one(fmt_similarity(axis.pole_a))
    vb = client.embed_one(fmt_similarity(axis.pole_b))
    axis_vec = vb - va
    axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-9)
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-9)
    return (Mn @ axis_vec).astype(np.float32)


def compute_all_axis_scores(df: pd.DataFrame, M: np.ndarray, client: EmbeddingClient) -> pd.DataFrame:
    cols = {"verse_id": df["verse_id"].tolist()}
    for a in PRESET_AXES.values():
        cols[a.key] = compute_axis_scores(df, M, a, client)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Arc / trajectory panels
# ---------------------------------------------------------------------------


def viz_chapter_arc(df: pd.DataFrame, scores: np.ndarray, axis: Axis) -> Path:
    d = df.copy()
    d["score"] = scores
    agg = d.groupby("chapter")["score"].agg(["mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    agg.columns = ["mean", "q25", "q75"]
    agg = agg.reset_index().sort_values("chapter")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["chapter"].tolist() + agg["chapter"][::-1].tolist(),
            y=agg["q75"].tolist() + agg["q25"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            hoverinfo="skip",
            name="IQR",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["chapter"],
            y=agg["mean"],
            mode="lines+markers",
            name="mean score",
            line=dict(width=3),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f'Chapter arc: "{axis.label_a}" ←→ "{axis.label_b}"',
        xaxis_title="chapter",
        yaxis_title=f"axis projection  (+ = {axis.label_b})",
        height=500,
        xaxis=dict(dtick=1),
    )
    out = REPORTS / f"chapter_arc_{axis.key}.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_verse_arc(df: pd.DataFrame, scores: np.ndarray, axis: Axis, chapter: int) -> Path:
    mask = df["chapter"].to_numpy() == chapter
    d = df[mask].copy().reset_index(drop=True)
    d["score"] = scores[mask]
    fig = px.bar(
        d,
        x="verse_num",
        y="score",
        hover_data=["verse_id", "speaker", "english"],
        title=f'Chapter {chapter} verse-by-verse: "{axis.label_a}" ←→ "{axis.label_b}"',
        color="score",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )
    fig.update_layout(height=500, xaxis_title="verse", yaxis_title="axis projection")
    out = REPORTS / f"verse_arc_ch{chapter:02d}_{axis.key}.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_narrative_trajectory(df: pd.DataFrame, coords: np.ndarray) -> Path:
    d = df.copy()
    d["x"] = coords[:, 0]
    d["y"] = coords[:, 1]
    d["ord"] = range(len(d))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["x"],
            y=d["y"],
            mode="lines",
            line=dict(color="rgba(100,100,100,0.25)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["x"],
            y=d["y"],
            mode="markers",
            marker=dict(size=6, color=d["chapter"], colorscale="Turbo", showscale=True, colorbar=dict(title="chapter")),
            text=_hover_text(df),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(title="Narrative trajectory through semantic space (verse reading order)", height=700, xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    out = REPORTS / "narrative_trajectory.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ---------------------------------------------------------------------------
# Comparison panels
# ---------------------------------------------------------------------------


def viz_speaker_voice_radar(df: pd.DataFrame, axis_scores: pd.DataFrame) -> Path:
    merged = df[["verse_id", "speaker"]].merge(axis_scores, on="verse_id")
    speakers = [s for s in merged["speaker"].dropna().unique() if s]
    axis_keys = [k for k in axis_scores.columns if k != "verse_id"]
    axis_labels = [f"{PRESET_AXES[k].label_a}→{PRESET_AXES[k].label_b}" for k in axis_keys]

    fig = go.Figure()
    for sp in speakers:
        sub = merged[merged["speaker"] == sp]
        means = [float(sub[k].mean()) for k in axis_keys]
        means_closed = means + [means[0]]
        labels_closed = axis_labels + [axis_labels[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=means_closed,
                theta=labels_closed,
                fill="toself",
                name=f"{sp} (n={len(sub)})",
                opacity=0.5,
            )
        )
    fig.update_layout(
        title="Speaker voice fingerprint across all preset axes",
        polar=dict(radialaxis=dict(visible=True)),
        height=700,
    )
    out = REPORTS / "speaker_voice_radar.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_axis_small_multiples(axis_scores: pd.DataFrame) -> Path:
    axis_keys = [k for k in axis_scores.columns if k != "verse_id"]
    ncols = 4
    nrows = (len(axis_keys) + ncols - 1) // ncols
    titles = [f"{PRESET_AXES[k].label_a} ←→ {PRESET_AXES[k].label_b}" for k in axis_keys]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, horizontal_spacing=0.06, vertical_spacing=0.08)
    for i, k in enumerate(axis_keys):
        r = i // ncols + 1
        c = i % ncols + 1
        fig.add_trace(go.Histogram(x=axis_scores[k], nbinsx=40, showlegend=False, marker_color="#636efa"), row=r, col=c)
        fig.add_vline(x=0, line_dash="dot", line_color="red", row=r, col=c)
    fig.update_layout(title="Axis projection distributions (all 17 axes)", height=250 * nrows, showlegend=False)
    out = REPORTS / "axis_small_multiples.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_axis_extremes_table(df: pd.DataFrame, axis_scores: pd.DataFrame, n: int = 5) -> Path:
    merged = df[["verse_id", "speaker", "english"]].merge(axis_scores, on="verse_id")
    blocks_html = ['<style>body{font-family:system-ui,sans-serif;margin:2em;max-width:1200px;}h2{margin-top:2em;border-bottom:2px solid #333;padding-bottom:.25em}h3{color:#555;margin-top:1em}table{border-collapse:collapse;width:100%;margin:.5em 0 1em 0}td{padding:.5em;border-bottom:1px solid #eee;vertical-align:top}.vid{white-space:nowrap;font-family:monospace;color:#888;padding-right:1em}.sp{font-weight:bold;color:#555;padding-right:1em;white-space:nowrap}.score{font-family:monospace;color:#c44;padding-right:1em;white-space:nowrap;text-align:right}</style>']
    blocks_html.append(f"<h1>Axis extremes — top/bottom {n} verses per axis</h1>")
    for k in [c for c in axis_scores.columns if c != "verse_id"]:
        ax = PRESET_AXES[k]
        ranked = merged.sort_values(k)
        bottom = ranked.head(n)
        top = ranked.tail(n).iloc[::-1]
        blocks_html.append(f"<h2>{ax.key} — {ax.description}</h2>")

        blocks_html.append(f"<h3>Most '{ax.label_b}' (+)</h3><table>")
        for _, r in top.iterrows():
            blocks_html.append(
                f"<tr><td class='score'>{r[k]:+.3f}</td><td class='vid'>{r['verse_id']}</td><td class='sp'>{r['speaker'] or '-'}</td><td>{r['english']}</td></tr>"
            )
        blocks_html.append("</table>")

        blocks_html.append(f"<h3>Most '{ax.label_a}' (−)</h3><table>")
        for _, r in bottom.iterrows():
            blocks_html.append(
                f"<tr><td class='score'>{r[k]:+.3f}</td><td class='vid'>{r['verse_id']}</td><td class='sp'>{r['speaker'] or '-'}</td><td>{r['english']}</td></tr>"
            )
        blocks_html.append("</table>")
    out = REPORTS / "axis_extremes.html"
    _ensure_reports()
    out.write_text("\n".join(blocks_html), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Term / polysemy
# ---------------------------------------------------------------------------


def viz_polysemy(term: str, df: pd.DataFrame, M: np.ndarray, k: int = 4, seed: int = 42) -> Path:
    mask = df["english"].str.contains(term, case=False, regex=False).to_numpy()
    if mask.sum() < k:
        k = max(2, mask.sum())

    sub = df[mask].copy().reset_index(drop=True)
    Msub = M[mask]
    if len(sub) < 4:
        raise ValueError(f"not enough matches for '{term}' (found {len(sub)})")

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Msub)
    sub["cluster"] = labels

    reducer = UMAP(n_components=2, n_neighbors=min(15, max(2, len(sub) - 1)), min_dist=0.1, metric="cosine", random_state=seed)
    coords = reducer.fit_transform(Msub)
    sub["x"], sub["y"] = coords[:, 0], coords[:, 1]

    fig = px.scatter(
        sub,
        x="x",
        y="y",
        color=sub["cluster"].astype(str),
        hover_name="verse_id",
        title=f"Polysemy of '{term}' — {len(sub)} verses, {k} clusters",
        height=700,
    )
    fig.update_traces(hovertext=_hover_text(sub), hovertemplate="%{hovertext}<extra></extra>", marker=dict(size=9))
    fig.update_layout(legend_title="cluster")
    out = REPORTS / f"polysemy_{term.lower()}.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_term_frequency_by_chapter(term: str, df: pd.DataFrame) -> Path:
    mask = df["english"].str.contains(term, case=False, regex=False)
    counts = df[mask].groupby("chapter").size().reindex(range(1, 19), fill_value=0)
    fig = px.bar(x=counts.index, y=counts.values, title=f"Occurrences of '{term}' per chapter", labels={"x": "chapter", "y": "count"})
    fig.update_layout(height=400, xaxis=dict(dtick=1))
    out = REPORTS / f"term_freq_{term.lower()}.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ---------------------------------------------------------------------------
# Neighborhood
# ---------------------------------------------------------------------------


def viz_neighbors_graph(verse_id: str, df: pd.DataFrame, M: np.ndarray, k: int = 15) -> Path:
    import networkx as nx

    idx_arr = df.index[df["verse_id"] == verse_id].to_numpy()
    if len(idx_arr) == 0:
        raise ValueError(f"verse_id {verse_id!r} not found")
    i = int(idx_arr[0])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-9)
    scores = Mn @ Mn[i]
    order = np.argsort(-scores)
    order = order[order != i][:k]

    G = nx.Graph()
    G.add_node(verse_id, text=df.iloc[i]["english"], speaker=df.iloc[i].get("speaker") or "-", seed=True)
    for j in order:
        vid = df.iloc[j]["verse_id"]
        G.add_node(vid, text=df.iloc[j]["english"], speaker=df.iloc[j].get("speaker") or "-", seed=False)
        G.add_edge(verse_id, vid, weight=float(scores[j]))

    pos = nx.spring_layout(G, seed=42, weight="weight")

    edge_x, edge_y = [], []
    for a, b in G.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="rgba(120,120,120,0.5)"), hoverinfo="none", showlegend=False)

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for n, data in G.nodes(data=True):
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        text = data["text"]
        if len(text) > 200:
            text = text[:199] + "…"
        node_text.append(f"<b>{n}</b> ({data['speaker']})<br>{text}")
        node_color.append("#e74c3c" if data.get("seed") else "#3498db")
        node_size.append(22 if data.get("seed") else 14)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=f"Nearest {k} neighbors of {verse_id}", height=700, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False))
    out = REPORTS / f"neighbors_{verse_id.replace('.', '_')}.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def viz_query_results(query: str, results: pd.DataFrame) -> Path:
    d = results.copy()
    d["label"] = d.apply(lambda r: f"{r['verse_id']} [{r.get('speaker') or '-'}]", axis=1)
    fig = px.bar(
        d.iloc[::-1],
        x="score",
        y="label",
        orientation="h",
        hover_data=["english"],
        title=f'search: "{query}"',
    )
    fig.update_layout(height=500, xaxis_title="cosine similarity", yaxis_title="")
    out = REPORTS / f"search_{_slugify(query)}.html"
    _ensure_reports()
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def _slugify(s: str, maxlen: int = 40) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")
    return s[:maxlen] or "query"
