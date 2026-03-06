from __future__ import annotations

import pandas as pd


def generate_daily_alpha(
    feature_df: pd.DataFrame, model, feature_columns: list[str], top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = feature_df.copy()
    scored["score"] = model.predict(scored[feature_columns].to_numpy())
    latest_date = scored["date"].max()
    latest = scored[scored["date"] == latest_date].copy()
    top = latest.nlargest(top_k, "score")[["date", "symbol", "score", "label"]]
    return scored, top
