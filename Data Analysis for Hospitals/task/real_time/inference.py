from __future__ import annotations

import pandas as pd

from real_time.streaming import stream_dataframe


def run_streaming_inference(X: pd.DataFrame, model, chunk_size: int) -> pd.DataFrame:
    frames = []
    for chunk in stream_dataframe(X, chunk_size=chunk_size):
        probs = model.predict_proba(chunk)[:, 1]
        chunk_output = pd.DataFrame(
            {
                "risk_probability": probs,
                "risk_label": (probs >= 0.5).astype(int),
            },
            index=chunk.index,
        )
        frames.append(chunk_output)

    return pd.concat(frames).sort_index()
