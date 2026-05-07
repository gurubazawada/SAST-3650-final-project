from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .corpus import Verse


COLUMNS = ["verse_id", "source", "chapter", "verse_num", "speaker", "english", "embedding"]


def _verses_to_frame(verses: list[Verse]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "verse_id": [v.verse_id for v in verses],
            "source": [v.source for v in verses],
            "chapter": [v.chapter for v in verses],
            "verse_num": [v.verse_num for v in verses],
            "speaker": [v.speaker for v in verses],
            "english": [v.english for v in verses],
        }
    )


def write_store(path: str | Path, verses: list[Verse], embeddings: np.ndarray) -> None:
    assert embeddings.dtype == np.float32
    assert len(verses) == embeddings.shape[0]
    dim = embeddings.shape[1]

    df = _verses_to_frame(verses)
    emb_list = [row.tolist() for row in embeddings]

    schema = pa.schema(
        [
            pa.field("verse_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("chapter", pa.int32()),
            pa.field("verse_num", pa.int32()),
            pa.field("speaker", pa.string()),
            pa.field("english", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=dim)),
        ]
    )

    table = pa.table(
        {
            "verse_id": df["verse_id"].to_list(),
            "source": df["source"].to_list(),
            "chapter": df["chapter"].astype("int32").to_list(),
            "verse_num": df["verse_num"].astype("int32").to_list(),
            "speaker": df["speaker"].to_list(),
            "english": df["english"].to_list(),
            "embedding": emb_list,
        },
        schema=schema,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def read_store(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    table = pq.read_table(path)
    df = table.to_pandas()
    matrix = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    df_meta = df.drop(columns=["embedding"]).reset_index(drop=True)
    return df_meta, matrix
