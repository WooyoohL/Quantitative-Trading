from __future__ import annotations

import pandas as pd


def _rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=False, pct=True)


def _rank_asc(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=True, pct=True)


def _latest_continuous_tail_days(raw_df: pd.DataFrame, all_dates: list[pd.Timestamp]) -> pd.Series:
    if raw_df.empty or not all_dates:
        return pd.Series(dtype="int64")

    calendar_index = {pd.Timestamp(date).normalize(): idx for idx, date in enumerate(pd.to_datetime(all_dates))}
    tail_lengths: dict[str, int] = {}

    for symbol, symbol_dates in raw_df.groupby("symbol")["date"]:
        ordered_dates = pd.to_datetime(symbol_dates).drop_duplicates().sort_values()
        date_indices = [calendar_index[pd.Timestamp(date).normalize()] for date in ordered_dates if pd.Timestamp(date).normalize() in calendar_index]
        if not date_indices:
            tail_lengths[str(symbol)] = 0
            continue

        tail_days = 1
        for idx in range(len(date_indices) - 1, 0, -1):
            if date_indices[idx] - date_indices[idx - 1] != 1:
                break
            tail_days += 1
        tail_lengths[str(symbol)] = tail_days

    return pd.Series(tail_lengths, dtype="int64")


def select_training_universe(raw_df: pd.DataFrame, config: dict) -> tuple[list[str], pd.DataFrame]:
    lookback_days = int(config.get("lookback_days", 60))

    filters = config.get("filters", {})
    min_avg_turnover = float(filters.get("min_avg_turnover", 2e8))
    min_latest_price = float(filters.get("min_latest_price", 2.0))
    max_avg_intraday_range = float(filters.get("max_avg_intraday_range", 0.08))
    max_avg_abs_ret1 = float(filters.get("max_avg_abs_ret1", 0.05))
    min_n_days = int(filters.get("min_n_days", max(30, lookback_days - 10)))
    min_continuous_tail_days = int(filters.get("min_continuous_tail_days", 0))
    training_max_latest_price = filters.get("training_max_latest_price")

    latest_dates = sorted(pd.to_datetime(raw_df["date"]).unique())
    if not latest_dates:
        return [], pd.DataFrame()

    active_dates = set(latest_dates[-lookback_days:])
    recent = raw_df[raw_df["date"].isin(active_dates)].copy()
    recent = recent.sort_values(["symbol", "date"]).reset_index(drop=True)
    recent["ret_1"] = recent.groupby("symbol")["close"].pct_change().fillna(0.0)
    recent["intraday_range"] = (recent["high"] - recent["low"]) / recent["close"].replace(0.0, pd.NA)
    recent["intraday_range"] = recent["intraday_range"].fillna(0.0)

    latest_close = recent.groupby("symbol")["close"].last()
    agg = (
        recent.groupby("symbol", as_index=False)
        .agg(
            avg_turnover=("turnover", "mean"),
            avg_intraday_range=("intraday_range", "mean"),
            avg_abs_ret1=("ret_1", lambda x: x.abs().mean()),
            median_turnover=("turnover", "median"),
            n_days=("date", "nunique"),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    agg["latest_close"] = agg["symbol"].map(latest_close)

    continuous_tail_days = _latest_continuous_tail_days(raw_df, latest_dates)
    agg["continuous_tail_days"] = agg["symbol"].map(continuous_tail_days).fillna(0).astype(int)
    agg["required_continuous_tail_days"] = min_continuous_tail_days
    agg["continuous_tail_pass"] = True
    if min_continuous_tail_days > 0:
        agg["continuous_tail_pass"] = agg["continuous_tail_days"] >= min_continuous_tail_days

    exclude_symbols = set(config.get("exclude_symbols", []))
    if exclude_symbols:
        agg = agg[~agg["symbol"].isin(exclude_symbols)].copy()
    st_symbol_set = {str(symbol) for symbol in config.get("st_symbol_set", [])}
    agg["is_st_symbol"] = agg["symbol"].astype(str).isin(st_symbol_set)

    base_filter = agg["continuous_tail_pass"].copy()
    agg["passes_min_turnover"] = agg["avg_turnover"] >= min_avg_turnover
    agg["passes_intraday_range"] = agg["avg_intraday_range"] <= max_avg_intraday_range
    agg["passes_abs_ret1"] = agg["avg_abs_ret1"] <= max_avg_abs_ret1
    agg["passes_min_n_days"] = agg["n_days"] >= min_n_days
    agg["passes_min_latest_price"] = agg["latest_close"] >= min_latest_price
    agg["passes_training_max_latest_price"] = True
    if training_max_latest_price is not None:
        agg["passes_training_max_latest_price"] = agg["latest_close"] <= float(training_max_latest_price)

    # Training samples now use a broad market universe and only remove bottom-tier symbols.
    eligibility_filter = (
        base_filter
        & ~agg["is_st_symbol"]
        & agg["passes_min_turnover"]
        & agg["passes_intraday_range"]
        & agg["passes_abs_ret1"]
        & agg["passes_min_latest_price"]
        & agg["passes_training_max_latest_price"]
        & agg["passes_min_n_days"]
    )
    filtered = agg[eligibility_filter].copy()

    if filtered.empty:
        filtered = agg[base_filter].copy()

    filtered["liq_rank"] = _rank_desc(filtered["avg_turnover"])
    filtered["stability_rank"] = _rank_asc(filtered["avg_abs_ret1"])
    filtered["range_rank"] = _rank_asc(filtered["avg_intraday_range"])
    filtered["price_rank"] = _rank_asc(filtered["latest_close"])
    filtered["pool_score"] = (
        0.45 * filtered["liq_rank"]
        + 0.20 * filtered["stability_rank"]
        + 0.15 * filtered["range_rank"]
        + 0.20 * filtered["price_rank"]
    )

    selected = filtered.sort_values(["pool_score", "avg_turnover"], ascending=[False, False]).reset_index(drop=True)
    return selected["symbol"].tolist(), selected
