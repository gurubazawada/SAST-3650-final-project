from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .embed import EmbeddingClient, fmt_similarity, fmt_search_query


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0 else v / n


def semantic_search(
    query: str,
    df_ret: pd.DataFrame,
    M_ret: np.ndarray,
    client: EmbeddingClient,
    k: int = 10,
) -> pd.DataFrame:
    q_vec = _normalize(client.embed_one(fmt_search_query(query)))
    Mn = _normalize_rows(M_ret)
    scores = Mn @ q_vec
    idx = np.argsort(-scores)[:k]

    out = df_ret.iloc[idx].copy()
    out.insert(0, "score", scores[idx])
    return out.reset_index(drop=True)


def term_in_context(
    term: str,
    df_sim: pd.DataFrame,
    M_sim: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df_sim["english"].str.contains(term, case=False, regex=False)
    hits = df_sim[mask].copy().reset_index(drop=True)
    if len(hits) == 0:
        return hits, pd.DataFrame()

    idx = df_sim.index[mask].to_numpy()
    X = M_sim[idx]

    k = min(n_clusters, len(hits))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    hits["cluster"] = labels

    Xn = _normalize_rows(X)
    Cn = _normalize_rows(km.cluster_centers_)
    sims = Xn @ Cn.T

    exemplar_rows = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        best_local = members[np.argmax(sims[members, c])]
        row = hits.iloc[best_local].copy()
        row["cluster"] = c
        row["cluster_size"] = int((labels == c).sum())
        exemplar_rows.append(row)
    exemplars = pd.DataFrame(exemplar_rows).reset_index(drop=True)

    return hits, exemplars


def concept_axis(
    pole_a: str,
    pole_b: str,
    df_sim: pd.DataFrame,
    M_sim: np.ndarray,
    client: EmbeddingClient,
) -> tuple[pd.DataFrame, np.ndarray]:
    va = client.embed_one(fmt_similarity(pole_a))
    vb = client.embed_one(fmt_similarity(pole_b))
    axis = _normalize(vb - va)
    Mn = _normalize_rows(M_sim)
    scores = Mn @ axis
    out = df_sim.copy()
    out["score"] = scores
    out = out.sort_values("score", kind="mergesort").reset_index(drop=True)
    return out, axis


def centroids(
    df: pd.DataFrame,
    M: np.ndarray,
    group_col: str = "source",
) -> tuple[pd.DataFrame, np.ndarray]:
    groups = df[group_col].fillna("(none)")
    keys = sorted(groups.unique().tolist(), key=lambda x: (isinstance(x, str), x))
    centroid_rows = []
    for k in keys:
        idx = np.where(groups.to_numpy() == k)[0]
        centroid_rows.append(M[idx].mean(axis=0))
    C = np.stack(centroid_rows).astype(np.float32)
    Cn = _normalize_rows(C)
    sim = Cn @ Cn.T
    sim_df = pd.DataFrame(sim, index=keys, columns=keys)
    return sim_df, C


def neighbors(
    verse_id: str,
    df_sim: pd.DataFrame,
    M_sim: np.ndarray,
    k: int = 15,
) -> pd.DataFrame:
    idx_arr = df_sim.index[df_sim["verse_id"] == verse_id].to_numpy()
    if len(idx_arr) == 0:
        raise ValueError(f"verse_id {verse_id!r} not found")
    i = int(idx_arr[0])
    Mn = _normalize_rows(M_sim)
    scores = Mn @ Mn[i]
    order = np.argsort(-scores)
    order = order[order != i][:k]

    out = df_sim.iloc[order].copy()
    out.insert(0, "score", scores[order])
    return out.reset_index(drop=True)
